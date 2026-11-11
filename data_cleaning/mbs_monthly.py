"""Month-end MBS panel from the sparse daily pool prices:  uv run python -m data_cleaning.mbs_monthly

Reads data/processed/mbs_daily.parquet (+ bond_universe, yield_curve_daily for the trading calendar) and writes
data/processed/mbs_monthly.parquet: one row per (pool, month-end) with no gaps between a pool's first trade and the
last month in the data.

Month-end price (Deliverable 3, step 5): the last trade in the month. If it falls in the final WINDOW_DAYS trading days
it is a proper month-end observation (price_source = traded_window); a trade earlier in the month is still used but
flagged (traded_month, is_stale = True). Months without any trade are filled from the pool's coupon cohort
(same program GNMA/conventional, term and coupon, the way pools are priced off TBAs): the pool's last traded price is
moved by the cohort's median month-end price change since that trade (cohort_proxy), so the pool-specific pay-up is
held constant and the market move comes from pools that did trade. With no cohort index the price is carried flat.

Bid-ask spread is a cost feature kept separate from the price: the month's mean observed spread, else the pool's last
observed spread carried forward, else the cohort median that month.
"""
import logging
import re

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger("data_cleaning.mbs_monthly")

WINDOW_DAYS = 5            # last trade within the final 5 trading days = month-end observation
MIN_COHORT_POOLS = 3       # pools that must trade in a month for the cohort index to count
MAX_CARRY_MONTHS = None    # drop rows more than this many months after the last trade (None = keep all)


def cohort_key(universe: pd.DataFrame) -> pd.Series:
    """'gnma|30|5.5'-style cohort from the AGG name (program, term) and coupon (rounded to the 0.5 coupon grid)."""
    name = universe["name"].str.upper()
    program = np.where(name.str.contains("GNMA|GINNIE"), "gnma", "conv")
    term = name.str.extract(r"(\d+)\s*YR")[0].fillna("30")
    arm = np.where(name.str.contains("ARM"), "arm", "")
    coupon = (universe["coupon_rate"] * 2).round() / 2
    return pd.Series(program, index=universe.index) + "|" + term + arm + "|" + coupon.map("{:.1f}".format)


def month_ends(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Per calendar month: the last trading day and the first day of the final WINDOW_DAYS trading days."""
    s = pd.Series(calendar, index=calendar)
    g = s.groupby(s.dt.to_period("M"))
    return pd.DataFrame({"last_trading_day": g.max(), "window_start": g.apply(lambda x: x.iloc[-WINDOW_DAYS] if len(x) >= WINDOW_DAYS else x.iloc[0])})


def monthly_observations(daily: pd.DataFrame, me: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily rows to one row per (cusip, month): last trade price plus monthly volume / spread aggregates."""
    d = daily.sort_values(["cusip", "date"]).copy()
    d["month"] = d["date"].dt.to_period("M")
    g = d.groupby(["cusip", "month"])
    last = g.last()[["date", "clean_price", "pool_factor"]].rename(columns={"date": "price_date", "pool_factor": "pool_factor_obs"})
    agg = g.agg(bid_ask_spread_obs=("bid_ask_spread", "mean"), par_volume_month=("par_volume", "sum"),
                dollar_volume_month=("dollar_volume", "sum"), trade_days=("date", "size"), trade_count=("trade_count", "sum"))
    obs = last.join(agg).reset_index()
    obs["in_window"] = obs["price_date"] >= obs["month"].map(me["window_start"])
    return obs


def cohort_index(obs: pd.DataFrame) -> pd.DataFrame:
    """Median month-end price of the pools that traded, per (cohort, month); NaN where fewer than MIN_COHORT_POOLS."""
    g = obs.groupby(["cohort", "month"])["clean_price"]
    idx = g.median().where(g.size() >= MIN_COHORT_POOLS).rename("cohort_index")
    return idx.reset_index()


def build_monthly(daily: pd.DataFrame, universe: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    u = universe[universe["asset_type"] == "mbs"].copy()
    u["cohort"] = cohort_key(u)
    me = month_ends(calendar)
    obs = monthly_observations(daily, me).merge(u[["cusip", "cohort"]], on="cusip")
    cidx = cohort_index(obs[~obs["cusip"].isin(u.loc[u["is_tba"], "cusip"])])     # TBAs are not pools: keep them out of the index

    # full grid: each pool from its first traded month to the last month with any trade in the data
    last_month = obs["month"].max()
    if daily["date"].max() < me.loc[last_month, "window_start"]:          # data stops before the final month's window
        log.info("dropping incomplete last month %s (data ends %s)", last_month, daily["date"].max().date())
        obs = obs[obs["month"] < last_month]
    months = pd.period_range(obs["month"].min(), obs["month"].max(), freq="M")
    first = obs.groupby("cusip")["month"].min().rename("first_month").reset_index()
    grid = first.merge(pd.DataFrame({"month": months}), how="cross")
    grid = grid[grid["month"] >= grid["first_month"]].drop(columns="first_month")
    p = grid.merge(obs.drop(columns="cohort"), on=["cusip", "month"], how="left").merge(u[["cusip", "cohort", "coupon_rate", "is_tba"]], on="cusip")
    p = p.merge(cidx, on=["cohort", "month"], how="left").sort_values(["cusip", "month"]).reset_index(drop=True)

    # price: traded this month, else last trade scaled by the cohort move since that trade, else carried flat
    traded = p["clean_price"].notna()
    gp = p.groupby("cusip")
    p["last_trade_month"] = p["month"].where(traded).groupby(p["cusip"]).ffill()
    p["last_trade_price"] = gp["clean_price"].ffill()
    p["price_date"] = gp["price_date"].ffill()
    idx_at_trade = p["cohort_index"].where(traded).groupby(p["cusip"]).ffill()
    proxy = p["last_trade_price"] * p["cohort_index"] / idx_at_trade
    p["clean_price"] = p["clean_price"].fillna(proxy).fillna(p["last_trade_price"])
    p["price_source"] = np.select([traded & p["in_window"].eq(True), traded, proxy.notna()],
                                  ["traded_window", "traded_month", "cohort_proxy"], "carried")
    p["is_stale"] = p["price_source"] != "traded_window"
    p["months_since_trade"] = (p["month"] - p["last_trade_month"]).map(lambda x: x.n if pd.notna(x) else np.nan).astype(int)

    # bid-ask spread: observed this month, else the pool's last observed, else the cohort median this month
    cohort_ba = p.groupby(["cohort", "month"])["bid_ask_spread_obs"].transform("median")
    carried_ba = gp["bid_ask_spread_obs"].ffill()
    p["bid_ask_spread"] = p["bid_ask_spread_obs"].fillna(carried_ba).fillna(cohort_ba)
    p["bid_ask_source"] = np.select([p["bid_ask_spread_obs"].notna(), carried_ba.notna(), cohort_ba.notna()],
                                    ["observed", "carried", "cohort"], "missing")
    p["bid_ask_spread_bps"] = p["bid_ask_spread"] / p["clean_price"] * 1e4

    # pool factor: only known on trade days, carried forward (it really declines every month with paydowns)
    p["pool_factor"] = gp["pool_factor_obs"].ffill()
    p["pool_factor_date"] = p["price_date"].where(p["pool_factor_obs"].notna()).groupby(p["cusip"]).ffill()

    # month-end accrued (30/360 from the 1st: 29 days at the 30th) and dirty price
    p["month_end_date"] = p["month"].map(me["last_trading_day"])
    p["accrued_interest"] = p["coupon_rate"] / 12 * 29 / 30
    p["dirty_price"] = p["clean_price"] + p["accrued_interest"]
    for c in ["par_volume_month", "dollar_volume_month", "trade_days", "trade_count"]:
        p[c] = p[c].fillna(0)
    if MAX_CARRY_MONTHS is not None:
        p = p[p["months_since_trade"] <= MAX_CARRY_MONTHS]

    cols = ["cusip", "month_end_date", "cohort", "is_tba", "coupon_rate", "clean_price", "accrued_interest", "dirty_price",
            "price_source", "is_stale", "price_date", "months_since_trade", "cohort_index",
            "bid_ask_spread", "bid_ask_spread_bps", "bid_ask_source",
            "par_volume_month", "dollar_volume_month", "trade_days", "trade_count", "pool_factor", "pool_factor_date"]
    out = p[cols].reset_index(drop=True)
    log.info("mbs monthly: %d pool-months, %d pools, %s to %s", len(out), out["cusip"].nunique(),
             out["month_end_date"].min().date(), out["month_end_date"].max().date())
    log.info("price source: %s", out["price_source"].value_counts(normalize=True).round(3).to_dict())
    log.info("bid-ask source: %s", out["bid_ask_source"].value_counts(normalize=True).round(3).to_dict())
    return out


def main():
    daily = pd.read_parquet(C.PROCESSED_DIR / "mbs_daily.parquet")
    universe = pd.read_parquet(C.PROCESSED_DIR / "bond_universe.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(C.PROCESSED_DIR / "yield_curve_daily.parquet")["date"])
    out = build_monthly(daily, universe, calendar)
    out.to_parquet(C.PROCESSED_DIR / "mbs_monthly.parquet", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    main()
