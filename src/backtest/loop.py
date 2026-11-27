"""Monthly roll-forward over a user horizon.

At each rebalance date: estimate (src.params) with data up to that date, sell bonds that left the benchmark at half
the bid-ask, solve from the drifted holdings, pay c'|trades|, and measure the realised return to the next month-end.

Catch-up (tracking mode only): g is a goal over the whole horizon, so each month the required monthly active return
is recomputed from the cumulative active growth A so far and the months left,
    g_req = ((1 + g_m)^N / A)^(1 / months_left) - 1,  clipped to [floor, cap * g_m]
where g_m is the original monthly goal. Behind the goal asks for more (at most cap times g_m), ahead of it eases off
(down to floor, 0 = never aim below the benchmark).
"""
import logging
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pandas as pd

from data_cleaning import config as C
from src.optimize.model import Settings, solve_rebalance

from .accounting import exit_cost, realised_returns

log = logging.getLogger(__name__)
CATCH_UP = dict(on=True, cap=2.0, floor=0.0)
KEY_RATES = [k.lower() for k in C.KEY_RATES]


class Frames(NamedTuple):
    monthly: pd.DataFrame       # one row per rebalance date
    holdings: pd.DataFrame      # (date, cusip) for every bond held before or after the trade
    sectors: pd.DataFrame       # (date, sector) weights and gross return contributions, portfolio and benchmark
    risk: pd.DataFrame          # (date, factor) ex-ante active variance split, monthly units


def candidates(P, hold: dict, k_max: int, H: int, n_cand: int, max_active_w: float | None = None):
    """Candidate index set S and x_init (current holdings on S). S is the n_cand largest benchmark weights among
    eligible bonds (not stale, not exiting, minimum lot small enough that k_max lots fit and, with max_active_w,
    within the per-bond cap b_i + max_active_w) plus current holdings."""
    q = P.periods[0]
    pos = {c: i for i, c in enumerate(P.cusips)}
    lot_ok = np.all([(pe.ell <= 0.5 / k_max) & (max_active_w is None or pe.ell <= pe.b_hat + max_active_w)
                     for pe in P.periods[:H]], axis=0)
    eligible = ~P.stale & ~q.exits & np.isfinite(q.mu_hat) & lot_ok
    el = np.flatnonzero(eligible)
    top = el[np.argsort(-q.b_hat[el])[:n_cand]]
    S = np.unique(np.concatenate([top, [pos[c] for c in hold]])).astype(int)
    x_init = np.array([hold.get(P.cusips[i], 0.0) for i in S])
    return S, x_init


def goal_rate(g_m: float, n: int, k: int, active_growth: float, cap: float = 2.0, floor: float = 0.0) -> float:
    """Required monthly active return at rebalance k (0-based) of n, given cumulative active growth so far."""
    req = ((1 + g_m) ** n / active_growth) ** (1 / (n - k)) - 1
    lo, hi = min(floor, g_m), max(floor, cap * g_m)   # g_m < 0 would otherwise invert the bounds
    return float(np.clip(req, lo, hi))


def _risk_split(P, q, a) -> dict:
    """Ex-ante active variance a' Q a split into one term per curve PC, credit factor and residual sector."""
    f = q.F.T @ a
    p = len(P.model.Lambda)
    out = {f"pc{j + 1}" if j < p else f"credit_{P.model.credit_groups[j - p]}": float(v ** 2) for j, v in enumerate(f)}
    res = pd.Series(P.model.psi * a ** 2, index=P.sector).groupby(level=0).sum()
    out.update({f"resid_{k}": float(v) for k, v in res.items()})
    return out


def run(s: Settings, dates, ends, data, data_for, table, params, n_cand: int,
        catch_up: dict = CATCH_UP) -> Frames:
    """One run over dates. params.get(d, data_for) returns Params at d (see cache.ParamsCache). data_for(d) returns
    data with the universe cut to the benchmark at d. s.ret_slack holds the original goal."""
    g_m, n = -s.ret_slack, len(dates)
    adapt = catch_up["on"] and s.objective == "tracking"
    hold, c_prev = {}, {}
    rows, hold_f, sec_f, risk_f = [], [], [], []
    growth = 1.0    # cumulative (1 + portfolio) / (1 + benchmark)
    for k, d in enumerate(dates):
        P = params.get(d, data_for)
        g_req = goal_rate(g_m, n, k, growth, catch_up["cap"], catch_up["floor"]) if adapt else g_m
        sk = replace(s, ret_slack=-g_req)
        q, d1 = P.periods[0], ends[ends.get_loc(d) + 1]
        pos = {c: i for i, c in enumerate(P.cusips)}
        drop = {c: w for c, w in hold.items() if c not in pos}
        exit_c = exit_cost(drop, d, data.prices, c_prev)
        hold = {c: w / (1 - exit_c) for c, w in hold.items() if c in pos}
        S, x_init = candidates(P, hold, s.k_max, s.H, n_cand, s.max_active_w)
        sol = solve_rebalance(P, S, x_init, sk)
        x0 = sol.x[0] / sol.x[0].sum()
        d_x = x0 - x_init
        trade_c = float(P.c[S] @ np.abs(d_x))
        r, share_fb, bad, _ = realised_returns(P, d, d1, data, table)
        gross = float(x0 @ r[S])
        net = (1 - exit_c) * (1 - trade_c) * (1 + gross) - 1
        cost = 1 - (1 - exit_c) * (1 - trade_c)
        bench = float(q.b_hat @ r)
        growth *= (1 + net) / (1 + bench)
        hold = {P.cusips[i]: w * (1 + r[i]) / (1 + gross) for i, w in zip(S, x0) if w > 1e-9}
        c_prev = {P.cusips[i]: P.c[i] for i, w in zip(S, x0) if w > 1e-9}
        a = -q.b_hat.copy()
        a[S] += x0
        x_full = np.zeros(len(P.cusips))
        x_full[S] = x0

        row = dict(
            date=d,
            ret_port=net,
            ret_gross=gross,
            ret_bench=bench,
            active=net - bench,
            growth_active=growth,
            cost=cost,
            exit_cost=exit_c,
            n_exits=len(drop),
            turnover=float(d_x.clip(0).sum()),
            n_hold=int((x0 > 1e-6).sum()),
            n_cand=len(S),
            n_bonds=len(P.cusips),
            share_fallback=share_fb,
            active_share=float(0.5 * np.abs(a).sum()),
            te_fcst_ann=np.sqrt(sol.track_var * 12),
            te_cap_ann=s.te_cap * sol.delta_scale if s.te_cap is not None else np.nan,
            port_D=float(q.D @ x_full),
            bench_D=float(q.D @ q.b_hat),
            act_D=float(q.D @ a),
            act_DTS=float(q.dts @ a),
            act_C=float(q.C @ a),
            **{f"act_krd_{t}": float(v) for t, v in zip(KEY_RATES, q.K.T @ a)},
            goal_req=g_req,                 # monthly goal this solve aimed for (g_m without catch-up)
            robust_active=sol.robust_active,
            shortfall=sol.shortfall,        # against goal_req
            goal_dropped=sol.return_goal_dropped,
            mip_skipped=sol.info.get("mip_skipped", False),
            delta_scale=sol.delta_scale,
            status=sol.status,
            seconds=sol.seconds,
        )
        rows.append(row)

        xi = np.zeros(len(P.cusips))
        xi[S] = x_init
        keep = (x_full > 1e-9) | (xi > 1e-9)
        hold_f.append(pd.DataFrame(dict(date=d, cusip=P.cusips[keep], sector=P.sector[keep],
                                        asset_type=P.asset_type[keep], x_init=xi[keep], x=x_full[keep],
                                        b=q.b_hat[keep], r=r[keep], fallback=bad[keep], c=P.c[keep],
                                        mu_hat=q.mu_hat[keep], D=q.D[keep])))
        sec_f.append(pd.DataFrame(dict(w_port=x_full, w_bench=q.b_hat, contrib_port=x_full * r,
                                       contrib_bench=q.b_hat * r), index=pd.Index(P.sector, name="sector"))
                     .groupby(level=0).sum().reset_index().assign(date=d))
        risk_f.append(pd.Series(_risk_split(P, q, a), name="var").rename_axis("factor").reset_index().assign(date=d))

        log.info("%s active %+6.1f bp  cost %5.1f bp  hold %3d  TE %5.1f bp/yr  goal %5.2f  shortfall %5.2f bp  "
                 "scale %g  solve %4.1fs", d.date(), 1e4 * row["active"], 1e4 * cost, row["n_hold"],
                 1e4 * row["te_fcst_ann"], 1e4 * g_req, 1e4 * sol.shortfall, sol.delta_scale, sol.seconds)

    def cat(fs, cols):
        return pd.concat(fs, ignore_index=True)[cols] if fs else pd.DataFrame(columns=cols)

    return Frames(
        monthly=pd.DataFrame(rows).set_index("date"),
        holdings=cat(hold_f, ["date", "cusip", "sector", "asset_type", "x_init", "x", "b", "r", "fallback", "c",
                              "mu_hat", "D"]),
        sectors=cat(sec_f, ["date", "sector", "w_port", "w_bench", "contrib_port", "contrib_bench"]),
        risk=cat(risk_f, ["date", "factor", "var"]),
    )
