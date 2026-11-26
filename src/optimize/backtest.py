"""Rolling monthly backtest of the tracking model with a train / validation / test split.

At each month-end t only data up to t is used (src.params estimates everything as of t), the model is solved, the
date-t trades are executed, and the realised return to the next month-end is measured from actual month-end prices
plus cash paid. Bonds with a stale price at either end take their model return mu_hat instead.

Split (24 month-ends of curve data, 2024-01 to 2025-12; the first 6 are estimation warm-up for Sigma and psi):
  train  2024-07 .. 2024-12   fit nothing beyond warm-up, used to sanity-check configs
  val    2025-01 .. 2025-06   pick the config (gamma, H)
  test   2025-07 .. 2025-11   reported once for the chosen config
"""
import json
import logging
import pickle
import subprocess
import time
from dataclasses import asdict, replace
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.params import build_cashflows, build_inputs, load_data
from src.params.frictions import COST_LOOKBACK_DAYS
from src.params.krd import cash_between

from .model import Settings, solve_rebalance

log = logging.getLogger(__name__)
RESULTS = Path(__file__).resolve().parents[2] / "results"
FIRST, LAST = "2024-07-31", "2025-11-30"
SPLITS = {"train": ("2024-07-31", "2024-12-31"), "val": ("2025-01-31", "2025-06-30"),
          "test": ("2025-07-31", "2025-11-30")}
WEALTH = 1e6
N_CAND = 400


def split_of(d) -> str:
    d = pd.Timestamp(d)
    return next(k for k, (a, b) in SPLITS.items() if pd.Timestamp(a) <= d <= pd.Timestamp(b))


def rebalance_dates(data) -> list:
    ends = data.panel.index.get_level_values("month_end").unique().sort_values()
    return [d for d in ends if pd.Timestamp(FIRST) <= d <= pd.Timestamp(LAST)]


def load_params(dates, data, H=2, cache=RESULTS / "params_cache.pkl"):
    """build_inputs for each date (about 11 s each), cached on disk."""
    cached = pickle.loads(cache.read_bytes()) if cache.exists() else {}
    for d in dates:
        if d not in cached:
            t = time.time()
            cached[d] = build_inputs(d, WEALTH, H=H, data=data)
            log.info("params %s: %d bonds, %.0fs", d.date(), len(cached[d].cusips), time.time() - t)
            cache.parent.mkdir(exist_ok=True)
            cache.write_bytes(pickle.dumps(cached))
    return {d: cached[d] for d in dates}


def floor_psi(P):
    """Copy of P with every bond's psi raised to its sector median (per date). Bond level psi from a handful of
    residuals can be near zero (Treasuries fit the curve almost exactly), which makes single-bond bets look free."""
    psi = P.model.psi
    med = pd.Series(psi).groupby(P.sector).transform("median").to_numpy()
    new = np.maximum(psi, med)
    return replace(P, model=replace(P.model, psi=new), theta_psi=new / P.model.n_obs)


def realised_returns(P, d0, d1, data, table) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Realised one-month return of every bond in P from d0 to d1, the share that fell back to mu_hat, the mask of
    fallbacks and the d0 dirty price."""
    wide = data.panel[["dirty_price", "stale"]].unstack("month_end")
    p0 = wide["dirty_price"][d0].reindex(P.cusips).to_numpy(float)
    p1 = wide["dirty_price"][d1].reindex(P.cusips).to_numpy(float)
    st = (wide["stale"][d0].reindex(P.cusips) != False) | (wide["stale"][d1].reindex(P.cusips) != False)  # noqa: E712
    cash = pd.Series(cash_between(table, d0, d1), index=table.cusips).reindex(P.cusips).fillna(0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p1 + cash) / p0 - 1
    bad = st.to_numpy() | ~np.isfinite(r)
    r = np.where(bad, P.periods[0].mu_hat, r)
    return r, float(bad.mean()), bad, p0


def exit_cost(drop: dict, d, prices: pd.DataFrame, c_prev: dict) -> float:
    """Share of wealth lost selling `drop` (cusip -> weight) at d for half the bid-ask: each bond's last quote in the
    lookback window to d, else its cost at the previous rebalance."""
    if not drop:
        return 0.0
    d = pd.Timestamp(d)
    p = prices[(prices["date"] > d - pd.Timedelta(days=COST_LOOKBACK_DAYS)) & (prices["date"] <= d)
               & prices["cusip"].isin(list(drop))]
    half = (p.dropna(subset=["bid_ask_spread_bps"]).sort_values("date")
             .groupby("cusip")["bid_ask_spread_bps"].last() / 2 * 1e-4)
    return float(sum(w * half.get(c, c_prev.get(c, 0.0)) for c, w in drop.items()))


def run(settings: Settings, params: dict, data, table, n_cand: int = N_CAND, store: dict | None = None) -> pd.DataFrame:
    """One backtest. Returns one row per rebalance date. If store is given, it collects the pieces the dashboard needs
    for a daily wealth path: store["holdings"] (list of date, cusip, x frames, this config) and store["bond_month"]
    (date, cusip, sector, b, r, bad, p0 for every index bond, config independent, set once)."""
    dates = list(params)
    ends = data.panel.index.get_level_values("month_end").unique().sort_values()
    hold: dict[str, float] = {}
    c_prev: dict[str, float] = {}
    rows = []
    for d in dates:
        P = floor_psi(params[d]) if settings.psi_floor else params[d]
        d1 = ends[ends.get_loc(d) + 1]
        q = P.periods[0]
        pos = {c: i for i, c in enumerate(P.cusips)}
        # held bonds that left the index (or have no price at d) are sold at half the bid-ask; the proceeds stay in
        # cash, so x_init sums to under 1 and the optimizer pays to reinvest them
        drop = {c: w for c, w in hold.items() if c not in pos}
        exit_c = exit_cost(drop, d, data.prices, c_prev)
        hold = {c: w / (1 - exit_c) for c, w in hold.items() if c in pos}
        # a bond whose minimum lot exceeds half an equal share (0.5 / k_max) is excluded, so k_max minimum lots always fit the budget
        lot_ok = np.all([pe.ell <= 0.5 / settings.k_max for pe in P.periods[: settings.H]], axis=0)
        eligible = ~P.stale & ~q.exits & np.isfinite(q.mu_hat) & lot_ok
        order = np.argsort(-np.where(eligible, q.b_hat, -1))[:n_cand]
        S = np.unique(np.concatenate([order, [pos[c] for c in hold]])).astype(int)
        x_init = np.zeros(len(S))
        for j, i in enumerate(S):
            x_init[j] = hold.get(P.cusips[i], 0.0)
        t0 = time.time()
        sol = solve_rebalance(P, S, x_init, settings)
        x0 = sol.x[0] / sol.x[0].sum()
        d_x = x0 - x_init
        trade = np.abs(d_x)
        cost = float(P.c[S] @ trade)
        r, share_fb, bad, p0 = realised_returns(P, d, d1, data, table)
        gross = float(x0 @ r[S])
        net = (1 - exit_c) * (1 - cost) * (1 + gross) - 1
        cost = 1 - (1 - exit_c) * (1 - cost)
        bench = float(q.b_hat @ r)
        hold = {P.cusips[i]: w * (1 + r[i]) / (1 + gross) for i, w in zip(S, x0) if w > 1e-9}
        c_prev = {P.cusips[i]: P.c[i] for i, w in zip(S, x0) if w > 1e-9}
        a = np.zeros(len(P.cusips))
        a[:] = -q.b_hat
        a[S] += x0
        sec = pd.Series(a, index=P.sector)
        wp = pd.Series(np.zeros(len(P.cusips)), index=P.sector)
        wp.iloc[S] = x0
        wp = wp.groupby(level=0).sum()
        wb = pd.Series(q.b_hat, index=P.sector).groupby(level=0).sum()
        # ex-ante variance split (monthly units, same Q = F F' + diag(psi) the model uses, over all n bonds):
        # one term per curve principal component, one per credit factor and one residual term per sector
        fa_all = q.F.T @ a
        res = pd.Series(P.model.psi * a**2, index=P.sector).groupby(level=0).sum()
        if store is not None:
            store.setdefault("holdings", []).append(pd.DataFrame({"date": d, "cusip": P.cusips[S], "x": x0})
                                                    .query("x > 1e-9"))
            store.setdefault("bond_month", {})
            if d not in store["bond_month"]:
                store["bond_month"][d] = pd.DataFrame({"date": d, "cusip": P.cusips, "sector": P.sector, "b": q.b_hat,
                                                       "r": r, "bad": bad, "p0": p0})
        row = dict(date=d, split=split_of(d), ret_port=net, ret_gross=gross, ret_bench=bench, active=net - bench,
                   turnover=float(d_x.clip(0).sum()), cost=cost, exit_cost=exit_c, n_hold=int((x0 > 1e-6).sum()),
                   n_support=int(sol.support.sum()), n_bonds=len(P.cusips), track_var_fcst=sol.track_var,
                   act_D=float(q.D @ a), act_C=float(q.C @ a), delta_scale=sol.delta_scale,
                   return_goal_dropped=sol.return_goal_dropped, status=sol.status, share_fallback=share_fb,
                   solve_s=time.time() - t0, port_D=float(q.D[S] @ x0), bench_D=float(q.D @ q.b_hat))
        p_curve = len(P.model.Lambda)
        for j, v in enumerate(fa_all):
            key = f"var_pc{j}" if j < p_curve else f"var_credit_{P.model.credit_groups[j - p_curve]}"
            row[key] = float(v**2)
        for k, v in res.items():
            row[f"var_res_{k}"] = float(v)
        for j, v in enumerate(q.K[S].T @ x0 - q.K.T @ q.b_hat):
            row[f"act_krd{j}"] = float(v)
        for k in sorted(set(wp.index) | set(wb.index)):
            row[f"w_port_{k}"], row[f"w_bench_{k}"] = float(wp.get(k, 0)), float(wb.get(k, 0))
        rows.append(row)
        log.info("%s %s active %+.1fbp cost %.1fbp hold %d scale %g", d.date(), row["split"], 1e4 * row["active"],
                 1e4 * cost, row["n_hold"], sol.delta_scale)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per split statistics of one backtest."""
    out = {}
    for k, g in df.groupby("split"):
        out[k] = dict(months=len(g), cum_port=(1 + g.ret_port).prod() - 1, cum_bench=(1 + g.ret_bench).prod() - 1,
                      mean_active_bp=1e4 * g.active.mean(), te_ann_bp=1e4 * g.active.std(ddof=1) * np.sqrt(12),
                      rms_active_bp=1e4 * np.sqrt((g.active**2).mean()), turnover_pct=100 * g.turnover.mean(),
                      cost_bp=1e4 * g.cost.mean(), avg_holdings=g.n_hold.mean(), act_dur=g.act_D.mean())
    return pd.DataFrame(out).T.reindex([k for k in SPLITS if k in out])


GRID = dict(gamma=[3e2, 3e3, 3e4], H=[1, 2])


def grid_search(params, data, table, base: Settings = Settings(), grid=GRID) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Backtest every config. Returns (all rows with config columns, one summary row per config and split,
    {"holdings": ..., "bond_month": ...} for the dashboard)."""
    frames, sums = [], []
    book = {"holdings": [], "bond_month": {}}
    for vals in product(*grid.values()):
        cfg = dict(zip(grid, vals))
        s = replace(base, **cfg)
        log.info("config %s", cfg)
        st = {"holdings": [], "bond_month": book["bond_month"]}
        df = run(s, params, data, table, store=st)
        book["holdings"].append(pd.concat(st["holdings"]).assign(**cfg))
        frames.append(df.assign(**{k: v for k, v in cfg.items()}))
        sm = summarize(df)
        for k, v in cfg.items():
            sm[k] = v
        sums.append(sm.reset_index(names="split"))
    return pd.concat(frames, ignore_index=True), pd.concat(sums, ignore_index=True), book


def pick_config(summary: pd.DataFrame, grid=GRID) -> dict:
    """Config with the lowest RMS active return on the validation split."""
    v = summary[summary.split == "val"].sort_values("rms_active_bp").iloc[0]
    return {k: v[k] for k in grid}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="uv run python -m src.optimize")
    ap.add_argument("--quick", action="store_true", help="one config instead of the grid")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_data()
    dates = rebalance_dates(data)
    params = load_params(dates, data)
    u = data.universe
    table = build_cashflows(u[u.maturity_date.notna() & u.coupon_rate.notna()].reset_index(drop=True),
                            pd.Timestamp(dates[0]) - pd.Timedelta(days=400))
    started = datetime.now()
    if args.quick:
        base = Settings()
        st = {"holdings": [], "bond_month": {}}
        df = run(base, params, data, table, store=st).assign(gamma=base.gamma, H=base.H)
        book = {"holdings": [pd.concat(st["holdings"]).assign(gamma=base.gamma, H=base.H)], "bond_month": st["bond_month"]}
        sm = summarize(df).assign(gamma=base.gamma, H=base.H).reset_index(names="split")
        grid = {"gamma": [base.gamma], "H": [base.H]}
    else:
        grid = GRID
        df, sm, book = grid_search(params, data, table)
    best = {k: v.item() if hasattr(v, "item") else v for k, v in pick_config(sm, grid).items()}
    out = RESULTS / "runs" / started.strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True)
    df.to_parquet(out / "backtest.parquet")
    sm.to_csv(out / "summary.csv", index=False)
    pd.concat(book["holdings"], ignore_index=True).to_parquet(out / "holdings.parquet")
    pd.concat(book["bond_month"].values(), ignore_index=True).to_parquet(out / "bond_month.parquet")
    chosen = sm[(sm.gamma == best["gamma"]) & (sm.H == best["H"])].set_index("split")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=RESULTS.parent).stdout.strip()
    except OSError:
        commit = ""
    meta = {
        "run": {"started": started.isoformat(timespec="seconds"), "seconds": round((datetime.now() - started).total_seconds()),
                "git_commit": commit, "quick": args.quick},
        "data": {"rebalance_dates": [str(d.date()) for d in dates], "n_rebalances": len(dates),
                 "splits": SPLITS, "wealth": WEALTH, "candidate_set_size": N_CAND,
                 "bonds_per_date": {str(d.date()): len(params[d].cusips) for d in dates}},
        "estimation": {"sigma_method": "ledoit_wolf", "pca_components": 3, "curve_change_freq": "M", "psi_min_obs": 6,
                       "theta": "Q / T_s", "realised_return_fallback": "mu_hat where either price is stale"},
        "settings_base": asdict(Settings()),
        "grid": grid,
        "selection": {"rule": "lowest RMS active return on validation", "chosen": best,
                      "settings_chosen": asdict(replace(Settings(), **best))},
        "solver_notes": "cvxpy + Clarabel; two-stage relax-and-restrict for cardinality and minimum lots; "
                        "lot cap 0.5 / k_max; return goal then delta tolerances relaxed if infeasible",
        "performance_chosen": json.loads(chosen.drop(columns=["gamma", "H"]).to_json(orient="index")),
        "performance_all": json.loads(sm.to_json(orient="records")),
        "diagnostics_chosen": {
            "return_goal_dropped_months": int(df[(df.gamma == best["gamma"]) & (df.H == best["H"])].return_goal_dropped.sum()),
            "max_delta_scale": float(df[(df.gamma == best["gamma"]) & (df.H == best["H"])].delta_scale.max())},
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    print("run saved to", out)
    print("chosen on validation:", best)
    print(chosen.round(4).to_string())


if __name__ == "__main__":
    main()
