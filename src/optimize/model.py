"""D3 Section 3 multi-period tracking model for one rebalance date, solved with cvxpy.

Two solve paths, chosen by Settings.method:
  * "heuristic" (default, any conic solver such as CLARABEL). Stage 1 solves the continuous relaxation (no z, no k_max).
    Stage 2 keeps the k_max names with the largest stage-1 weight, forces them to at least the minimum lot, and
    re-solves. This is a heuristic for the cardinality and minimum-lot constraints, not an optimal MISOCP solution.
  * "misocp" (needs a MIP-capable conic solver, i.e. GUROBI). One solve with a binary z_i per bond and period:
    x_i <= z_i, x_i >= ell_i z_i, sum z <= k_max. Same objective and other constraints as the heuristic.
The universe is cut to a candidate set S (largest benchmark weights plus current holdings). Tracking is still measured
against the full benchmark b over all n bonds, so a = x - b has support outside S and the missing part is a constant.

Units follow src.params: monthly, decimal returns, weights sum to 1.
"""
import time
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np


@dataclass
class Settings:
    H: int = 2
    omega: tuple = (0.5, 0.5)       # weights on tracking variance per planning date, sum 1 (first H used)
    gamma: float = 3e3              # multiplier on tracking variance relative to proportional trading cost
    k_max: int = 150
    max_active_w: float = 0.15      # x_i <= b_i + max_active_w (no shorting, so underweights are bounded by b_i)
    psi_floor: bool = True          # lift each bond's psi to its sector median (see backtest.floor_psi)
    delta_D: float = 0.25           # |D'a| in years
    delta_K: float = 0.20           # |K_j'a| per key rate, years
    delta_S: float = 0.25           # |D_spr'a|
    delta_C: float = 15.0           # |C'a|
    eps2: float = 0.5               # robustness multiplier in the return goal
    ret_slack: float = 0.0025       # r_req = benchmark expected return - slack, per month (placeholder goal)
    solver: str = "CLARABEL"        # used by the heuristic path
    method: str = "heuristic"       # "heuristic" (relax, truncate, re-solve) or "misocp" (binary z, needs GUROBI)
    mip_solver: str = "GUROBI"
    mip_gap: float = 1e-3           # relative MIP gap at which Gurobi stops
    time_limit: float = 120.0       # Gurobi seconds per solve, the best incumbent is used if it stops early


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


def _build(P, S, x_init, s: Settings, scale: float, support=None, drop_return=False, mip=False):
    """Returns (problem, x, up, um, z). z is None unless mip=True, then one boolean vector per planning date."""
    H, m = s.H, len(S)
    om = np.asarray(s.omega[:H], float)
    om = om / om.sum()
    x = [cp.Variable(m, nonneg=True) for _ in range(H)]
    up = [cp.Variable(m, nonneg=True) for _ in range(H)]
    um = [cp.Variable(m, nonneg=True) for _ in range(H)]
    z = [cp.Variable(m, boolean=True) for _ in range(H)] if mip else None
    cons, obj = [], 0
    c = P.c[S]
    for h in range(H):
        q = P.periods[h]
        bS = q.b_hat[S]
        # active weights a = x - b_hat; the part of a outside S is -b_hat there (constant)
        Fb = q.F.T @ q.b_hat
        f = q.F[S].T @ x[h] - Fb
        var = cp.sum_squares(f) + cp.sum_squares(cp.multiply(np.sqrt(P.model.psi[S]), x[h] - bS))
        obj += s.gamma * om[h] * var + c @ (up[h] + um[h])
        xt = x_init if h == 0 else cp.multiply(q.drift_gross[S] / (1 + q.r_b), x[h - 1])
        cons += [x[h] - xt == up[h] - um[h], cp.sum(x[h]) == 1]
        if s.max_active_w is not None:
            cons += [x[h] <= bS + s.max_active_w]
        for arr, d in ((q.D, s.delta_D), (q.D_spr, s.delta_S), (q.C, s.delta_C)):
            e = arr[S] @ x[h] - arr @ q.b_hat
            cons += [cp.abs(e) <= d * scale]
        cons += [cp.abs(q.K[S].T @ x[h] - q.K.T @ q.b_hat) <= s.delta_K * scale]
        if not drop_return:
            r_req = q.mu_hat @ q.b_hat - s.ret_slack
            theta = cp.hstack([P.theta_F[S].T @ x[h], cp.multiply(np.sqrt(P.theta_psi[S]), x[h])])
            cons += [q.mu_hat[S] @ x[h] - s.eps2 * cp.norm(theta, 2) >= r_req]
        ex = q.exits[S]
        if ex.any():
            cons += [x[h][np.flatnonzero(ex)] == 0]
        if mip:
            # z_i = 1 iff bond i is held: x_i <= z_i (x <= 1 since weights sum to 1), x_i >= ell_i z_i, at most k_max held
            cons += [x[h] <= z[h], x[h] >= cp.multiply(q.ell[S], z[h]), cp.sum(z[h]) <= s.k_max]
        if support is not None:
            off = np.flatnonzero(~support)
            on = np.flatnonzero(support)
            if len(off):
                cons += [x[h][off] == 0]
            cons += [x[h][on] >= q.ell[S][on]]
    return cp.Problem(cp.Minimize(obj), cons), x, up, um, z


def _solve(prob, solver, **opts):
    try:
        prob.solve(solver=solver, **opts)
    except cp.error.SolverError:
        return "solver_error"
    return prob.status


def _solution(P, S, xs, up, um, support, status, scale, dropped, t0, info=None) -> Solution:
    xs = [np.clip(xi, 0, None) for xi in xs]
    q = P.periods[0]
    a = xs[0] - q.b_hat[S]
    fa = q.F[S].T @ xs[0] - q.F.T @ q.b_hat
    rest = np.ones(len(q.b_hat), bool)
    rest[S] = False
    tv = float(fa @ fa + np.sum(P.model.psi[S] * a**2) + np.sum(P.model.psi[rest] * q.b_hat[rest] ** 2))
    return Solution(xs, np.clip(up, 0, None), np.clip(um, 0, None), support & (xs[0] > 1e-9), status, float(scale),
                    dropped, tv, time.time() - t0, info or {})


def _solve_misocp(P, S, x_init, s: Settings) -> Solution:
    """Single MISOCP solve. If infeasible, drop the return goal first, then widen the delta tolerances (x2, x4, x8)."""
    t0, ok = time.time(), ("optimal", "optimal_inaccurate", "user_limit")   # user_limit: time limit, incumbent used if any
    for scale, dropped in [(1, False), (1, True), (2, True), (4, True), (8, True)]:
        prob, x, up, um, z = _build(P, S, x_init, s, scale, drop_return=dropped, mip=True)
        status = _solve(prob, s.mip_solver, TimeLimit=s.time_limit, MIPGap=s.mip_gap)
        if status in ok and x[0].value is not None:
            break
    if status not in ok or x[0].value is None:
        raise RuntimeError(f"misocp failed: {status}")
    support = z[0].value > 0.5
    return _solution(P, S, [xi.value for xi in x], up[0].value, um[0].value, support, status, scale, dropped, t0,
                     {"method": "misocp", "objective": float(prob.value)})


def solve_rebalance(P, S: np.ndarray, x_init: np.ndarray, s: Settings = Settings()) -> Solution:
    """Plan H dates from the state in P (a src.params.Params), trade date 0. S indexes P.cusips, x_init is the current
    drifted holdings on S. s.method picks the heuristic (default) or the MISOCP."""
    if s.method == "misocp":
        return _solve_misocp(P, S, x_init, s)
    if s.method != "heuristic":
        raise ValueError(f"unknown method {s.method!r}")
    t0 = time.time()
    scale, dropped = 1.0, False
    ok = ("optimal", "optimal_inaccurate")
    # if infeasible: drop the return goal first, then widen the delta tolerances (x2, x4, x8)
    for scale, dropped in [(1, False), (1, True), (2, True), (4, True), (8, True)]:
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
    # stage 2 keeps the k_max cardinality: if the minimum lots make it infeasible, drop the return goal, then widen deltas
    for sc, dr in [(scale, dropped), (scale, True), (2 * scale, True), (4 * scale, True), (8 * scale, True)]:
        prob, x, up, um, _ = _build(P, S, x_init, s, sc, support=support, drop_return=dr)
        status = _solve(prob, s.solver)
        if status in ok:
            scale, dropped = sc, dr
            break
    if status not in ok:
        raise RuntimeError(f"stage 2 failed: {status}")
    return _solution(P, S, [xi.value for xi in x], up[0].value, um[0].value, support, status, scale, dropped, t0,
                     {"method": "heuristic"})
