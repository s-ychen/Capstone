"""Daily wealth of the portfolio and the benchmark between monthly rebalances.

Weights are held between rebalances, so bond i's return from the rebalance d0 to day t is

    R_i(t) = (P_i(t) + cash_i(d0, t)) / P_i(d0) - 1

with P the dirty price (last quote carried forward) and cash the coupons and principal paid in (d0, t]. Bonds the
backtest priced at mu_hat (stale endpoint or no price) move linearly over the month to that return. Portfolio and
benchmark wealth are W(t) = W(d0) (1 + R(t))'w with w the post-trade weights x or the benchmark b_hat, and the
portfolio pays its trading cost on the rebalance day. Each month therefore ends exactly on the backtest's monthly
return. The full benchmark weights come from the Params cache the run used.
"""
import hashlib
import json
import pickle

import numpy as np
import pandas as pd

from src.backtest.accounting import realised_returns
from src.backtest.cache import ROOT as CACHE_ROOT
from src.backtest.results import Run
from src.params import build_cashflows, load_data
from src.params.krd import cash_between


def _params_dir(run: Run):
    key = run.meta["params_cache"]
    d = CACHE_ROOT / hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    if not d.exists():
        raise SystemExit(f"params cache {d} not found; rerun the backtest to rebuild it")
    return d


def daily_paths(run: Run, out_dir) -> pd.DataFrame:
    """date, wealth_port, wealth_bench, month (the rebalance date the day belongs to). The first row is the first
    rebalance at wealth 1. Cached in out_dir/daily.parquet."""
    f = out_dir / "daily.parquet"
    if f.exists():
        return pd.read_parquet(f)
    pdir = _params_dir(run)
    data = load_data()
    ends = data.panel.index.get_level_values("month_end").unique().sort_values()
    cal = pd.DatetimeIndex(data.curve["date"])
    m = run.monthly
    dates = pd.DatetimeIndex(m.index)
    u = data.universe
    table = build_cashflows(u[u.asset_type.isin(run.meta["inputs"]["asset_types"]) & u.maturity_date.notna()
                              & u.coupon_rate.notna()].reset_index(drop=True), dates[0] - pd.Timedelta(days=400))
    tpos = pd.Series(np.arange(len(table.cusips)), index=table.cusips)
    px = data.prices[["date", "cusip", "dirty_price"]].sort_values("date", kind="stable")
    pdates = px["date"].to_numpy()
    hold = run.holdings

    wp, wb = 1.0, 1.0
    out = [pd.DataFrame({"date": [dates[0]], "wealth_port": [1.0], "wealth_bench": [1.0], "month": [dates[0]]})]
    for d0 in dates:
        P = pickle.loads((pdir / f"{d0.date()}.pkl").read_bytes())
        b = P.periods[0].b_hat
        d1 = ends[ends.get_loc(d0) + 1]
        r, _, bad, p0 = realised_returns(P, d0, d1, data, table)
        days = cal[(cal > d0) & (cal <= d1)]
        x = hold[hold.date == d0].set_index("cusip").x.reindex(P.cusips).fillna(0.0).to_numpy()

        lo, hi = np.searchsorted(pdates, np.datetime64(d0), "right"), np.searchsorted(pdates, np.datetime64(d1), "right")
        sl = px.iloc[lo:hi]
        sl = sl[sl.cusip.isin(P.cusips)].drop_duplicates(["cusip", "date"], keep="last")
        p = sl.pivot(index="date", columns="cusip", values="dirty_price").reindex(index=days, columns=P.cusips)
        p = pd.concat([pd.DataFrame([p0], index=[d0], columns=P.cusips), p]).ffill().iloc[1:].to_numpy()

        ti = tpos.reindex(P.cusips)
        cash = np.stack([cash_between(table, d0, t) for t in days])[:, ti.fillna(0).astype(int).to_numpy()]
        cash[:, ti.isna().to_numpy()] = 0
        with np.errstate(divide="ignore", invalid="ignore"):
            R = (p + cash) / p0 - 1
        frac = (np.arange(1, len(days) + 1) / len(days))[:, None]
        R = np.where(bad[None, :] | ~np.isfinite(R), r[None, :] * frac, R)
        gp = (1 + R) @ x * (1 - m.loc[d0, "cost"])
        gb = (1 + R) @ b
        out.append(pd.DataFrame({"date": days, "wealth_port": wp * gp, "wealth_bench": wb * gb, "month": d0}))
        wp, wb = wp * gp[-1], wb * gb[-1]

    daily = pd.concat(out, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(f)
    return daily


def check(daily: pd.DataFrame, monthly: pd.DataFrame) -> float:
    """Largest gap (bp) between the daily path compounded over each month and the backtest's monthly returns."""
    end = daily.groupby("month").last()
    worst = 0.0
    for col, ret in (("wealth_port", "ret_port"), ("wealth_bench", "ret_bench")):
        w = end[col].to_numpy()
        prev = np.r_[1.0, w[:-1]]
        worst = max(worst, float(np.max(np.abs(w / prev - 1 - monthly[ret].to_numpy()))))
    return 1e4 * worst
