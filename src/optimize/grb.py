"""gurobipy backend for src.optimize.model: same model, one Gurobi model object for the whole pipeline.

  1. relaxation (continuous, no cardinality), as the cvxpy heuristic's stage 1
  2. keep the k_max largest weights, fix the rest to 0 and lift the kept ones to the minimum lot, re-solve in place
     (warm basis, bounds change only). This is the incumbent, the same heuristic as the cvxpy path.
  3. method "misocp" only: restore the bounds, add the binaries z to the same model, load the incumbent as the MIP
     start and let Gurobi improve it (MIPFocus, MIPGap, TimeLimit from Settings).
Units and scaling are the same as the cvxpy path: solver weights X = W x in percent, objective times OBJ_SCALE with the
scale folded into auxiliary variables, so every squared term enters as a plain sum of squares with coefficient 1.
"""
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .model import OBJ_SCALE, W, Solution, _solution, _theta_out

_OK = (GRB.OPTIMAL, GRB.SUBOPTIMAL)
_LADDER = [(1, False), (1, True), (2, True), (4, True), (8, True)]
_ENV = None


def _env():
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


class _Model:
    """Holds the Gurobi model and the handles needed to change tolerances, drop the return goal, fix the support."""

    def __init__(self, P, S, x_init, s):
        H, m = s.H, len(S)
        self.P, self.S, self.s, self.H, self.m = P, S, s, H, m
        om = np.asarray(s.omega[:H], float)
        om = om / om.sum()
        mod = self.mod = gp.Model(env=_env())
        self.X = [mod.addMVar(m, lb=0.0, name=f"x{h}") for h in range(H)]
        self.up = [mod.addMVar(m, lb=0.0, name=f"up{h}") for h in range(H)]
        self.um = [mod.addMVar(m, lb=0.0, name=f"um{h}") for h in range(H)]
        self.delta_rows = []    # (constr, center, half_width, sign): rhs = center + sign * scale * half_width
        self.ret_rows = []      # return goal constraints, removed together when the goal is dropped
        self.ub0, self.ell = [], []
        c = P.c[S]
        obj = None
        for h in range(H):
            q = P.periods[h]
            bS = q.b_hat[S]
            X, up, um = self.X[h], self.up[h], self.um[h]
            k = np.sqrt(OBJ_SCALE * s.gamma * om[h])
            Fb = q.F.T @ q.b_hat
            kf = q.F.shape[1]
            fs = mod.addMVar(kf, lb=-GRB.INFINITY)
            mod.addConstr(fs == (k / W) * (q.F[S].T @ X) - k * Fb)
            g = mod.addMVar(m, lb=-GRB.INFINITY)
            sp = k * np.sqrt(P.model.psi[S])
            mod.addConstr(g == (sp / W) * X - sp * bS)
            # epigraph: tq >= |fs|^2 + |g|^2 as one convex quadratic constraint, objective stays linear. A quadratic
            # objective in the MIP gave a much weaker bound than this form (which is also what cvxpy sends Gurobi).
            tq = mod.addVar(lb=0.0)
            qe = gp.QuadExpr()
            fv, gv = fs.tolist(), g.tolist()
            qe.addTerms(np.ones(len(fv) + len(gv)), fv + gv, fv + gv)
            qe.addTerms(-1.0, tq)
            mod.addQConstr(qe <= 0)
            term = tq + (OBJ_SCALE / W) * (c @ up) + (OBJ_SCALE / W) * (c @ um)   # up, um are in percent
            obj = term if obj is None else obj + term
            xt = W * x_init if h == 0 else (q.drift_gross[S] / (1 + q.r_b)) * self.X[h - 1]
            mod.addConstr(X - xt == up - um)
            mod.addConstr(X.sum() == W)
            ub = np.full(m, GRB.INFINITY)
            if s.max_active_w is not None:
                ub = W * (bS + s.max_active_w)
            ub = np.where(q.exits[S], 0.0, ub)
            X.ub = ub
            self.ub0.append(ub)
            self.ell.append(q.ell[S])
            for arr, d in ((q.D, s.delta_D), (q.D_spr, s.delta_S), (q.C, s.delta_C)):
                self._delta(arr[S], W * (arr @ q.b_hat), W * d, X)
            Kb, KS = q.K.T @ q.b_hat, q.K[S]
            for j in range(KS.shape[1]):
                self._delta(KS[:, j], W * Kb[j], W * s.delta_K, X)
            # return goal: mu'x - eps2 * ||Theta^(1/2) (x - b_hat)|| >= r_req, scaled by W (sv = W * ||.||).
            # The penalty is on the active weights so holding the benchmark costs nothing; bonds outside S add th_out.
            r_req = q.mu_hat @ q.b_hat - s.ret_slack
            thF = mod.addMVar(P.theta_F.shape[1], lb=-GRB.INFINITY)
            thP = mod.addMVar(m, lb=-GRB.INFINITY)
            sv = mod.addVar(lb=0.0)
            mod.addConstr(thF == P.theta_F[S].T @ X - W * (P.theta_F.T @ q.b_hat))
            mod.addConstr(thP == np.sqrt(P.theta_psi[S]) * X - W * np.sqrt(P.theta_psi[S]) * bS)
            qc = gp.QuadExpr()
            tv = thF.tolist() + thP.tolist()
            qc.addTerms(np.ones(len(tv)), tv, tv)
            qc.addConstant((W * _theta_out(P, S, q.b_hat)) ** 2)
            qc.addTerms(-1.0, sv, sv)
            self.ret_rows += [mod.addQConstr(qc <= 0),
                              mod.addConstr(q.mu_hat[S] @ X - s.eps2 * sv >= W * r_req)]
        mod.setObjective(obj, GRB.MINIMIZE)
        mod.update()
        self.scale, self.dropped = 1.0, False

    def _delta(self, a, center, half, X):
        self.delta_rows.append((self.mod.addConstr(a @ X <= center + half), center, half, 1))
        self.delta_rows.append((self.mod.addConstr(a @ X >= center - half), center, half, -1))

    def set_tolerance(self, scale, dropped):
        if dropped and not self.dropped:
            self.mod.remove(self.ret_rows)
            self.dropped = True
        self.scale = scale
        for con, center, half, sign in self.delta_rows:
            con.RHS = center + sign * scale * half

    def optimize(self, **params):
        for k, v in params.items():
            self.mod.setParam(k, v)
        self.mod.optimize()
        return self.mod.Status

    def values(self):
        xs = [v.X / W for v in self.X]
        return xs, self.up[0].X / W, self.um[0].X / W


def _solve_ladder(M, ladder):
    for scale, dropped in ladder:
        M.set_tolerance(scale, dropped)
        if M.optimize() in _OK:
            return scale, dropped
    raise RuntimeError(f"gurobi relaxation failed (status {M.mod.Status})")


def solve_grb(P, S, x_init, s) -> Solution:
    t0 = time.time()
    M = _Model(P, S, x_init, s)
    mod, H, m = M.mod, M.H, M.m
    # stage 1: relaxation
    scale, dropped = _solve_ladder(M, _LADDER)
    w = np.max([v.X for v in M.X], axis=0)
    w[np.any([P.periods[h].exits[S] for h in range(H)], axis=0)] = -1
    support = np.zeros(m, bool)
    support[np.argsort(-w)[: s.k_max]] = True
    # stage 2: fix the support in place. If the minimum lots make it infeasible, drop the goal, then widen the deltas.
    off, on = np.flatnonzero(~support), np.flatnonzero(support)
    for h in range(H):
        ub = M.ub0[h].copy()
        ub[off] = 0.0
        M.X[h].ub = ub
        lb = np.zeros(m)
        lb[on] = W * M.ell[h][on]
        M.X[h].lb = lb
    ladder2 = [(scale, dropped), (scale, True), (2 * scale, True), (4 * scale, True), (8 * scale, True)]
    for sc, dr in ladder2:
        M.set_tolerance(sc, dr)
        if M.optimize() in _OK:
            scale, dropped = sc, dr
            break
    else:
        raise RuntimeError(f"gurobi stage 2 failed (status {mod.Status})")
    xs, up0, um0 = M.values()
    heur_obj = mod.ObjVal / OBJ_SCALE
    if s.method == "heuristic":
        return _solution(P, S, xs, up0, um0, support, "optimal" if mod.Status == GRB.OPTIMAL else "optimal_inaccurate",
                         scale, dropped, t0, {"method": "heuristic", "backend": "gurobipy", "objective": heur_obj})

    # MISOCP on the same model: restore bounds, add z, start from the stage 2 solution
    allowed = support.copy()
    if s.mip_cand is not None:
        allowed[np.argsort(-w)[: s.mip_cand]] = True
    starts = [(v.X.copy(), a.X.copy(), b.X.copy()) for v, a, b in zip(M.X, M.up, M.um)]
    for h in range(H):
        ub = M.ub0[h].copy()
        ub[~allowed] = 0.0
        M.X[h].lb = 0.0
        M.X[h].ub = ub
        q = P.periods[h]
        bS = q.b_hat[S]
        big = np.minimum(1.0, bS + s.max_active_w) if s.max_active_w is not None else np.ones(m)
        z = mod.addMVar(m, vtype=GRB.BINARY)
        z.ub = allowed.astype(float)
        mod.addConstr(M.X[h] <= (W * big) * z)
        mod.addConstr(M.X[h] >= (W * M.ell[h]) * z)
        mod.addConstr(z.sum() <= s.k_max)
        z.Start = (starts[h][0] > 1e-9 * W).astype(float)
        M.X[h].Start, M.up[h].Start, M.um[h].Start = starts[h]
        if h == 0:
            z0 = z
    status = M.optimize(MIPFocus=s.mip_focus, MIPGap=s.mip_gap, TimeLimit=s.time_limit)
    if mod.SolCount == 0:
        raise RuntimeError(f"gurobi misocp failed (status {status})")
    xs, up0, um0 = M.values()
    st = "optimal" if status == GRB.OPTIMAL else "user_limit" if status == GRB.TIME_LIMIT else "optimal_inaccurate"
    return _solution(P, S, xs, up0, um0, z0.X > 0.5, st, scale, dropped, t0,
                     {"method": "misocp", "backend": "gurobipy", "objective": mod.ObjVal / OBJ_SCALE,
                      "heuristic_objective": heur_obj, "mip_gap": float(mod.MIPGap), "nodes": int(mod.NodeCount)})
