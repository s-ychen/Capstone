"""Monthly roll-forward over a user horizon, as driven by workbook/goal_optimizer.ipynb.

At each rebalance date: estimate (src.params) with data up to that date, sell bonds that left the benchmark at half
the bid-ask, solve from the drifted holdings, pay c'|trades|, and measure the realised return to the next month-end.

Catch-up (tracking mode only): g is a goal over the whole horizon, so each month the required monthly active return
is recomputed from the cumulative active growth A so far and the months left,
    g_req = ((1 + g_m)^N / A)^(1 / months_left) - 1,  clipped to [floor, cap * g_m]
where g_m is the original monthly goal. Behind the goal asks for more (at most cap times g_m), ahead of it eases off
(down to floor, 0 = never aim below the benchmark).
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from src.params import build_inputs

from .backtest import exit_cost, realised_returns
from .model import Settings, solve_rebalance

CATCH_UP = dict(on=True, cap=2.0, floor=0.0)


def candidates(P, hold: dict, k_max: int, H: int, n_cand: int):
    """Candidate index set S and x_init (current holdings on S). S is the n_cand largest benchmark weights among
    eligible bonds (not stale, not exiting, minimum lot small enough that k_max lots fit) plus current holdings."""
    q = P.periods[0]
    pos = {c: i for i, c in enumerate(P.cusips)}
    lot_ok = np.all([pe.ell <= 0.5 / k_max for pe in P.periods[:H]], axis=0)
    eligible = ~P.stale & ~q.exits & np.isfinite(q.mu_hat) & lot_ok
    top = np.argsort(-np.where(eligible, q.b_hat, -1))[:n_cand]   # ineligible bonds get -1, sorting to the end
    S = np.unique(np.concatenate([top, [pos[c] for c in hold]])).astype(int)
    x_init = np.array([hold.get(P.cusips[i], 0.0) for i in S])
    return S, x_init


def goal_rate(g_m: float, n: int, k: int, active_growth: float, cap: float = 2.0, floor: float = 0.0) -> float:
    """Required monthly active return at rebalance k (0-based) of n, given cumulative active growth so far."""
    req = ((1 + g_m) ** n / active_growth) ** (1 / (n - k)) - 1
    return float(np.clip(req, floor, cap * g_m))


def run(s: Settings, dates, ends, data, data_for, table, W: float, params: dict, n_cand: int,
        catch_up: dict = CATCH_UP, verbose: bool = True) -> pd.DataFrame:
    """One row per rebalance date. params caches build_inputs by (date, H) and is filled in place.
    data_for(d) returns data with the universe cut to the benchmark at d. s.ret_slack holds the original goal."""
    g_m, n = -s.ret_slack, len(dates)
    adapt = catch_up["on"] and s.objective == "tracking"
    hold, c_prev, rows = {}, {}, []
    growth = 1.0    # cumulative (1 + portfolio) / (1 + benchmark)
    for k, d in enumerate(dates):
        if (d, s.H) not in params:
            params[(d, s.H)] = build_inputs(d, W, H=s.H, data=data_for(d))
        P = params[(d, s.H)]
        g_req = goal_rate(g_m, n, k, growth, catch_up["cap"], catch_up["floor"]) if adapt else g_m
        sk = replace(s, ret_slack=-g_req)
        q, d1 = P.periods[0], ends[ends.get_loc(d) + 1]
        pos = {c: i for i, c in enumerate(P.cusips)}
        drop = {c: w for c, w in hold.items() if c not in pos}
        exit_c = exit_cost(drop, d, data.prices, c_prev)
        hold = {c: w / (1 - exit_c) for c, w in hold.items() if c in pos}
        S, x_init = candidates(P, hold, s.k_max, s.H, n_cand)
        sol = solve_rebalance(P, S, x_init, sk)
        x0 = sol.x[0] / sol.x[0].sum()
        d_x = x0 - x_init
        cost = float(P.c[S] @ np.abs(d_x))
        r, share_fb, bad, p0 = realised_returns(P, d, d1, data, table)
        gross = float(x0 @ r[S])
        net = (1 - exit_c) * (1 - cost) * (1 + gross) - 1
        cost = 1 - (1 - exit_c) * (1 - cost)
        bench = float(q.b_hat @ r)
        growth *= (1 + net) / (1 + bench)
        hold = {P.cusips[i]: w * (1 + r[i]) / (1 + gross) for i, w in zip(S, x0) if w > 1e-9}
        c_prev = {P.cusips[i]: P.c[i] for i, w in zip(S, x0) if w > 1e-9}
        a = -q.b_hat.copy()
        a[S] += x0
        rows.append(dict(
            date=d,
            ret_port=net,
            ret_bench=bench,
            active=net - bench,
            cost=cost,
            exit_cost=exit_c,
            turnover=float(d_x.clip(0).sum()),
            n_hold=int((x0 > 1e-6).sum()),
            n_bonds=len(P.cusips),
            te_fcst_ann=np.sqrt(sol.track_var * 12),
            te_cap_ann=s.te_cap * sol.delta_scale if s.te_cap is not None else np.nan,
            act_D=float(q.D @ a),
            goal_req=g_req,                 # monthly goal this solve aimed for (g_m without catch-up)
            robust_active=sol.robust_active,
            shortfall=sol.shortfall,        # against goal_req
            goal_dropped=sol.return_goal_dropped,
            mip_skipped=sol.info.get("mip_skipped", False),
            delta_scale=sol.delta_scale,
            status=sol.status,
            seconds=sol.seconds,
        ))
        if verbose:
            print(f"{d.date()} active {1e4 * (net - bench):+6.1f} bp  cost {1e4 * cost:5.1f} bp  hold {rows[-1]['n_hold']:3d}  "
                  f"TE {1e4 * rows[-1]['te_fcst_ann']:5.1f} bp/yr  goal {1e4 * g_req:5.2f}  shortfall {1e4 * sol.shortfall:5.2f} bp  "
                  f"scale {sol.delta_scale:g}  solve {sol.seconds:4.1f}s")
    return pd.DataFrame(rows).set_index("date")
