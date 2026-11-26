"""D3 Section 3 multi-period tracking model for one rebalance date, solved with cvxpy.

Two solve paths, chosen by Settings.method:
  * "heuristic" (default, any conic solver such as CLARABEL). Stage 1 solves the continuous relaxation (no z, no k_max).
    Stage 2 keeps the k_max names with the largest stage-1 weight, forces them to at least the minimum lot, and
    re-solves. This is a heuristic for the cardinality and minimum-lot constraints, not an optimal MISOCP solution.
  * "misocp" (needs a MIP-capable conic solver, i.e. GUROBI). One solve with a binary z_i per bond and period:
    x_i <= z_i, x_i >= ell_i z_i, sum z <= k_max. Same objective and other constraints as the heuristic.
Settings.objective picks what is optimized, with the same constraint set:
  * "tracking" (default): min gamma * tracking variance + cost, subject to the return goal. The goal is a hard
    constraint (goal_penalty None, dropped by the ladder when infeasible) or a soft one (goal_penalty set, the
    shortfall below the goal is charged in the objective, so the model is always feasible on the goal).
  * "return": max robust active return mu'a - eps2 * ||Theta^(1/2) a|| minus cost, gamma * tracking variance as a
    tie-breaker, no return constraint. Needs te_cap, otherwise nothing limits the active risk.
te_cap (annualized, None = off) bounds the ex-ante tracking error at every planning date in either mode.
The universe is cut to a candidate set S (largest benchmark weights plus current holdings). Tracking is still measured
against the full benchmark b over all n bonds, so a = x - b has support outside S and the missing part is a constant.

Units follow src.params: monthly, decimal returns, weights sum to 1.
"""
import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

W = 100.0           # solver weights are in percent
BP = 1e4            # goal shortfall variables are in bp
OBJ_SCALE = 1e3     # solver objective = OBJ_SCALE * (gamma * variance + trading cost)


@dataclass
class Settings:
    H: int = 2
    omega: tuple = (0.5, 0.5)       # weights on tracking variance per planning date, sum 1 (first H used)
    gamma: float = 3e3              # multiplier on tracking variance relative to proportional trading cost
    k_max: int = 150
    max_active_w: float = 0.02      # x_i <= b_i + max_active_w (no shorting, so underweights are bounded by b_i)
    psi_floor: bool = False         # lift each bond's psi to its sector median (backtest.floor_psi). Off: src.params
                                    # already shrinks psi toward a duration-scaled sector prior
    delta_D: float = 0.25           # |D'a| in years
    delta_K: float = 0.20           # |K_j'a| per key rate, years
    delta_DTS: float = 0.0025       # |dts'a|, years x decimal spread (0.0025 = 0.25 yrs at 100 bp)
    delta_C: float = 15.0           # |C'a|
    eps2: float = 0.5               # robustness multiplier in the return goal
    ret_slack: float = 0.0025       # r_req = benchmark expected return - slack, per month (placeholder goal)
    solver: str = "CLARABEL"        # used by the heuristic path
    method: str = "heuristic"       # "heuristic" (relax, truncate, re-solve) or "misocp" (binary z, needs GUROBI)
    backend: str = "cvxpy"          # "cvxpy" (CLARABEL heuristic, GUROBI via cvxpy) or "gurobipy" (src.optimize.grb, both methods)
    mip_solver: str = "GUROBI"
    mip_gap: float = 1e-2           # relative MIP gap at which Gurobi stops
    mip_focus: int = 2              # Gurobi MIPFocus, 2 = prove the bound (the incumbent is found early, the bound is slow)
    time_limit: float = 120.0       # Gurobi seconds per solve, the best incumbent is used if it stops early
    mip_cand: int | None = 150      # misocp: binaries only for the mip_cand largest stage-1 weights (plus the heuristic's
                                    # support), every other bond is fixed to 0. None keeps a binary on every bond.
    mip_warm: bool = True           # misocp: solve the heuristic first and pass it to Gurobi as the MIP start
    mip_skip_shortfall: bool = True # misocp, gurobipy, soft goal only: if the heuristic misses the goal, return it
                                    # without the MIP. Proving the gap is slow in those months and gained < 0.5%
    objective: str = "tracking"     # "tracking" (min gamma * TE variance + cost, return goal) or "return" (max robust
                                    # active return - cost, gamma * TE variance as a tie-breaker, no goal constraint)
    te_cap: float | None = None     # annualized ex-ante tracking error cap (decimal, 0.005 = 50 bp), None = off.
                                    # Widened with the deltas by the tolerance ladder
    cost_weight: float = 1.0        # multiplier on the trading cost in the objective. Below 1 amortizes the one-off cost
                                    # over that many months (1/12 = a year), which "return" mode needs: it otherwise
                                    # weighs a single month of extra return against the full cost and barely trades
    goal_penalty: float | None = None   # tracking only. None: hard return goal. Else soft: objective += goal_penalty
                                        # * monthly shortfall (decimal), and the ladder never drops the goal


@dataclass
class Solution:
    x: list                         # per planning date, length |S|
    trades_plus: np.ndarray         # h = 0 buys, length |S|
    trades_minus: np.ndarray        # h = 0 sells
    support: np.ndarray             # bool |S|, names kept by stage 2
    status: str
    delta_scale: float              # tolerance multiplier needed for feasibility (1 = as specified)
    return_goal_dropped: bool
    track_var: float                # forecast h = 0 tracking variance of the executed portfolio
    seconds: float
    info: dict = field(default_factory=dict)
    robust_active: float = float("nan")   # forecast h = 0 robust active return mu'a - eps2 ||Theta^(1/2) a||, monthly
    shortfall: float = 0.0                # max(0, goal - robust_active), monthly. 0 when the goal is met


def _out_norm(w, S, b):
    """sqrt(sum of w b^2 over bonds outside S): the idiosyncratic part of a norm of the active weights that S cannot
    change (bonds outside S have active weight -b)."""
    out = np.ones(len(b), bool)
    out[S] = False
    return float(np.sqrt(np.sum(w[out] * b[out] ** 2)))


def _theta_out(P, S, b):
    return _out_norm(P.theta_psi, S, b)


def _soft(s: Settings) -> bool:
    """True when the return goal is never a hard constraint, so the fallback ladder only widens tolerances."""
    return s.objective == "return" or s.goal_penalty is not None


def _ladder(s: Settings, scale=1.0, dropped=False) -> list:
    """(tolerance scale, goal dropped) steps to try in order when the model is infeasible. Hard goal: drop the goal
    first, then widen the deltas (and te_cap) x2, x4, x8. Soft or no goal: only widen."""
    if _soft(s):
        return [(scale * f, False) for f in (1, 2, 4, 8)]
    return [(scale, dropped), (scale, True), (2 * scale, True), (4 * scale, True), (8 * scale, True)]


def _check(s: Settings):
    if s.objective not in ("tracking", "return"):
        raise ValueError(f"unknown objective {s.objective!r}")
    if s.objective == "return" and s.te_cap is None:
        raise ValueError('objective "return" needs te_cap, otherwise the active risk is unbounded')


def _build(P, S, x_init, s: Settings, scale: float, support=None, drop_return=False, mip=False, allowed=None):
    """Returns (problem, x, up, um, z). z is None unless mip=True, then one boolean vector per planning date.
    allowed (bool |S|, mip only): bonds outside it are fixed to 0 so Gurobi presolve drops their binaries."""
    H, m = s.H, len(S)
    om = np.asarray(s.omega[:H], float)
    om = om / om.sum()
    # Solver variables are in percent (X = W x) and the objective is multiplied by OBJ_SCALE. Raw weights are about
    # 4e-3 and the objective about 5e-4, well inside the solver tolerances (1e-6), which made Gurobi's answers noisy.
    # x, up, um below are affine expressions in the true units, so the rest of the model is written as before.
    Xr = [cp.Variable(m, nonneg=True) for _ in range(H)]
    x = [v / W for v in Xr]
    up = [cp.Variable(m, nonneg=True) / W for _ in range(H)]
    um = [cp.Variable(m, nonneg=True) / W for _ in range(H)]
    z = [cp.Variable(m, boolean=True) for _ in range(H)] if mip else None
    soft = s.objective == "tracking" and s.goal_penalty is not None and not drop_return
    sh = [cp.Variable(nonneg=True) / BP for _ in range(H)] if soft else None   # goal shortfall, monthly decimal
    cons, obj = [], 0
    c = P.c[S]
    for h in range(H):
        q = P.periods[h]
        bS = q.b_hat[S]
        # active weights a = x - b_hat; the part of a outside S is -b_hat there (constant)
        Fb = q.F.T @ q.b_hat
        f = q.F[S].T @ x[h] - Fb
        # OBJ_SCALE * gamma * om * variance, with the scale inside the squares so the epigraph variables keep coefficient 1
        k = np.sqrt(OBJ_SCALE * s.gamma * om[h])
        obj += cp.sum_squares(k * f) + cp.sum_squares(cp.multiply(k * np.sqrt(P.model.psi[S]), x[h] - bS))
        obj += OBJ_SCALE * s.cost_weight * (c @ (up[h] + um[h]))
        xt = x_init if h == 0 else cp.multiply(q.drift_gross[S] / (1 + q.r_b), x[h - 1])
        cons += [x[h] - xt == up[h] - um[h], cp.sum(x[h]) == 1]
        if s.max_active_w is not None:
            cons += [x[h] <= bS + s.max_active_w]
        for arr, d in ((q.D, s.delta_D), (q.dts, s.delta_DTS), (q.C, s.delta_C)):
            e = arr[S] @ x[h] - arr @ q.b_hat
            cons += [cp.abs(e) <= d * scale]
        cons += [cp.abs(q.K[S].T @ x[h] - q.K.T @ q.b_hat) <= s.delta_K * scale]
        if s.te_cap is not None:
            # ex-ante TE: ||(F'a, psi^(1/2) a)|| with the constant part from bonds outside S, in percent like X
            te = cp.hstack([W * f, cp.multiply(W * np.sqrt(P.model.psi[S]), x[h] - bS),
                            np.array([W * _out_norm(P.model.psi, S, q.b_hat)])])
            cons += [cp.norm(te, 2) <= W * s.te_cap / np.sqrt(12) * scale]
        if s.objective == "return" or not drop_return:
            # robustness on the active return: ||Theta^(1/2) (x - b_hat)||, zero when x = b_hat.
            # Bonds outside S have active weight -b_hat, which adds the constant th_out to the norm.
            th_out = _theta_out(P, S, q.b_hat)
            theta = cp.hstack([P.theta_F[S].T @ x[h] - P.theta_F.T @ q.b_hat,
                               cp.multiply(np.sqrt(P.theta_psi[S]), x[h] - bS), np.array([th_out])])
            robust = q.mu_hat[S] @ x[h] - s.eps2 * cp.norm(theta, 2)   # robust return of x, the active part plus b'mu
            if s.objective == "return":
                obj += OBJ_SCALE * om[h] * (q.mu_hat @ q.b_hat - robust)
            else:
                r_req = q.mu_hat @ q.b_hat - s.ret_slack
                if soft:
                    cons += [robust + sh[h] >= r_req]
                    obj += OBJ_SCALE * s.goal_penalty * om[h] * sh[h]
                else:
                    cons += [robust >= r_req]
        ex = q.exits[S]
        if ex.any():
            cons += [x[h][np.flatnonzero(ex)] == 0]
        if mip:
            # z_i = 1 iff bond i is held: x_i <= u_i z_i, x_i >= ell_i z_i, at most k_max held. u_i is the tightest known
            # upper bound (weights sum to 1, and the active weight cap), a much stronger big-M than 1.
            ub = np.minimum(1.0, bS + s.max_active_w) if s.max_active_w is not None else np.ones(m)
            cons += [x[h] <= cp.multiply(ub, z[h]), x[h] >= cp.multiply(q.ell[S], z[h]), cp.sum(z[h]) <= s.k_max]
            if allowed is not None and (~allowed).any():
                cons += [z[h][np.flatnonzero(~allowed)] == 0]
        if support is not None:
            off = np.flatnonzero(~support)
            on = np.flatnonzero(support)
            if len(off):
                cons += [x[h][off] == 0]
            cons += [x[h][on] >= q.ell[S][on]]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.raw_x = Xr   # solver variables, for warm starts (set .value = W * x)
    return prob, x, up, um, z


def _solve(prob, solver, **opts):
    try:
        prob.solve(solver=solver, **opts)
    except cp.error.SolverError:
        return "solver_error"
    return prob.status


def _solution(P, S, xs, up, um, support, status, scale, dropped, t0, info=None, s: Settings | None = None) -> Solution:
    xs = [np.clip(xi, 0, None) for xi in xs]
    q = P.periods[0]
    a = -q.b_hat.copy()     # active weights over every benchmark bond
    a[S] += xs[0]
    fa = q.F.T @ a
    tv = float(fa @ fa + np.sum(P.model.psi * a**2))
    sol = Solution(xs, np.clip(up, 0, None), np.clip(um, 0, None), support & (xs[0] > 1e-9), status, float(scale),
                   dropped, tv, time.time() - t0, info or {})
    if s is not None:
        th = np.sqrt(np.sum((P.theta_F.T @ a) ** 2) + np.sum(P.theta_psi * a**2))
        sol.robust_active = float(q.mu_hat @ a - s.eps2 * th)
        sol.shortfall = max(0.0, -s.ret_slack - sol.robust_active)
    return sol


def _solve_misocp(P, S, x_init, s: Settings) -> Solution:
    """Single MISOCP solve. If infeasible, drop the return goal first, then widen the delta tolerances (x2, x4, x8).
    With mip_warm the heuristic runs first: its solution is the MIP start, its tolerance scale is where the fallback
    ladder begins, and its stage-1 weights pick the bonds that get a binary (mip_cand)."""
    t0, ok = time.time(), ("optimal", "optimal_inaccurate", "user_limit")   # user_limit: time limit, incumbent used if any
    ladder = _ladder(s)
    warm, allowed = None, None
    if s.mip_warm or s.mip_cand is not None:
        warm = _solve_heuristic(P, S, x_init, s)
        ladder = [l for l in ladder if l >= (warm.delta_scale, warm.return_goal_dropped)] or ladder[-1:]
        if s.mip_cand is not None:
            allowed = warm.support.copy()
            allowed[np.argsort(-warm.info["stage1_w"])[: s.mip_cand]] = True
    for scale, dropped in ladder:
        prob, x, up, um, z = _build(P, S, x_init, s, scale, drop_return=dropped, mip=True, allowed=allowed)
        if s.mip_warm and warm is not None:
            for h in range(s.H):
                prob.raw_x[h].value = W * warm.x[h]
                z[h].value = (warm.x[h] > 1e-9).astype(float)
        status = _solve(prob, s.mip_solver, warm_start=s.mip_warm, TimeLimit=s.time_limit, MIPGap=s.mip_gap,
                       MIPFocus=s.mip_focus)
        if status in ok and x[0].value is not None:
            break
    if status not in ok or x[0].value is None:
        raise RuntimeError(f"misocp failed: {status}")
    support = z[0].value > 0.5
    return _solution(P, S, [xi.value for xi in x], up[0].value, um[0].value, support, status, scale, dropped, t0,
                     {"method": "misocp", "objective": float(prob.value) / OBJ_SCALE}, s)


def _solve_heuristic(P, S, x_init, s: Settings) -> Solution:
    t0 = time.time()
    scale, dropped = 1.0, False
    ok = ("optimal", "optimal_inaccurate")
    # if infeasible: drop a hard return goal first, then widen the tolerances (x2, x4, x8)
    for scale, dropped in _ladder(s):
        prob, x, up, um, _ = _build(P, S, x_init, s, scale, drop_return=dropped)
        status = _solve(prob, s.solver)
        if status in ok:
            break
    if status not in ok:
        raise RuntimeError(f"stage 1 failed: {status}")
    w = np.max([xi.value for xi in x], axis=0)
    w[np.any([P.periods[h].exits[S] for h in range(s.H)], axis=0)] = -1   # a bond leaving the index at any h cannot be a core holding
    support = np.zeros(len(S), bool)
    support[np.argsort(-w)[: s.k_max]] = True
    # stage 2 keeps the k_max cardinality: if the minimum lots make it infeasible, drop a hard goal, then widen
    for sc, dr in _ladder(s, scale, dropped):
        prob, x, up, um, _ = _build(P, S, x_init, s, sc, support=support, drop_return=dr)
        status = _solve(prob, s.solver)
        if status in ok:
            scale, dropped = sc, dr
            break
    if status not in ok:
        raise RuntimeError(f"stage 2 failed: {status}")
    return _solution(P, S, [xi.value for xi in x], up[0].value, um[0].value, support, status, scale, dropped, t0,
                     {"method": "heuristic", "stage1_w": w, "objective": float(prob.value) / OBJ_SCALE}, s)


def solve_rebalance(P, S: np.ndarray, x_init: np.ndarray, s: Settings = Settings()) -> Solution:
    """Plan H dates from the state in P (a src.params.Params), trade date 0. S indexes P.cusips, x_init is the current
    drifted holdings on S. s.method picks the heuristic (default) or the MISOCP."""
    if s.method not in ("heuristic", "misocp"):
        raise ValueError(f"unknown method {s.method!r}")
    _check(s)
    if s.backend == "gurobipy":
        from .grb import solve_grb
        return solve_grb(P, S, x_init, s)
    if s.backend != "cvxpy":
        raise ValueError(f"unknown backend {s.backend!r}")
    if s.method == "misocp":
        return _solve_misocp(P, S, x_init, s)
    return _solve_heuristic(P, S, x_init, s)
