"""
Cleaned daily prices -> the panels that get saved: one row per (bond, month) for prices_monthly, and the observed
bond-days for prices_daily, only in months the bond was in AGG's start-of-month portfolio.

Month-end price (Deliverable 3, step 5): the last observation of the month. If it falls in the final WINDOW_DAYS
trading days it is a proper month-end price (price_source = traded_window); an observation earlier in the month is
still used but flagged (traded_month). A held month with no observation carries the last price forward (carried).
is_stale is True unless traded_window, so stale months can be dropped when estimating residual variance.
For Treasuries "traded" means quoted by CRSP. MBS months without trades are filled from the pool's coupon cohort
instead (cohort_proxy, see build_mbs_monthly).

Monthly liquidity features come from every day of the month, not just month-end: mean of the observed daily bid-ask
spreads (carried from the bond's last observed month when there is none), and total par volume, dollar volume, trade
count and number of days with a trade.

A month is only built for a source if its data reaches that month's final window, so an incomplete last month at the
end of a source (e.g. TRACE corporates ending 2026-03-17) is dropped rather than filled with stale prices.
"""
import logging

import numpy as np
import pandas as pd

from . import config as C
from .bond_math import accrued_interest, solve_yield_and_spread

log = logging.getLogger(__name__)
WINDOW_DAYS = 5            # last observation within the final 5 trading days = month-end observation
MIN_COHORT_POOLS = 3       # MBS: pools that must trade in a month for the cohort index to count


def month_ends(calendar):
    """Per calendar month: the last trading day and the first day of the final WINDOW_DAYS trading days."""
    s = pd.Series(calendar, index=calendar)
    g = s.groupby(s.dt.to_period("M"))
    return pd.DataFrame({"last_trading_day": g.max(),
                         "window_start": g.apply(lambda x: x.iloc[-WINDOW_DAYS] if len(x) >= WINDOW_DAYS else x.iloc[0])})


def monthly_observations(daily, me, calendar):
    """Collapse daily rows to one row per (cusip, month): the last observation plus monthly volume / spread aggregates.
    Observations on non-trading days (rare weekend or Good Friday TRACE prints) are dropped: they would fall after the
    month's last trading day and have no zero curve to value against."""
    d = daily[daily["date"].isin(calendar)].sort_values(["cusip", "date"]).copy()
    d["month"] = d["date"].dt.to_period("M")
    for c in ["par_volume", "dollar_volume", "trade_count", "bid_ask_spread"]:
        if c not in d:
            d[c] = np.nan
    g = d.groupby(["cusip", "month"])
    last_cols = [c for c in ["date", "clean_price", "yield", "accrued_interest", "pool_factor"] if c in d]
    last = g[last_cols].last().rename(columns={"date": "price_date", "yield": "yield_obs", "accrued_interest": "accrued_obs",
                                                "pool_factor": "pool_factor_obs"})
    agg = g.agg(bid_ask_spread_obs=("bid_ask_spread", "mean"), par_volume=("par_volume", "sum"),
                dollar_volume=("dollar_volume", "sum"), trade_count=("trade_count", "sum"), trade_days=("date", "size"))
    obs = last.join(agg).reset_index()
    obs["in_window"] = obs["price_date"] >= obs["month"].map(me["window_start"])
    return obs


def carry_panel(daily, membership, calendar):
    """Month-end panel for one source (Treasuries or corporates): observed or carried price in every held month from
    the bond's first observation up to the source's last complete month."""
    me = month_ends(calendar)
    obs = monthly_observations(daily, me, calendar)
    # last month the source fully covers
    last = me.index[me["window_start"] <= daily["date"].max()].max()      
    obs = obs[obs["month"] <= last]

    # make a grid: every bond in every month from its first observation to the last complete month
    months = pd.period_range(obs["month"].min(), last, freq="M")
    first = obs.groupby("cusip")["month"].min().rename("first_month").reset_index()
    grid = first.merge(pd.DataFrame({"month": months}), how="cross")
    grid = grid[grid["month"] >= grid["first_month"]].drop(columns="first_month")
    # fill the grid with actual data 
    p = grid.merge(obs, on=["cusip", "month"], how="left").sort_values(["cusip", "month"]).reset_index(drop=True)

    # forward fill motnhs that dont have any data but i dont think this is a big deal bc our data should have values
    # for at least some day of the month
    # also tag the day which the data is ffilled from e.g the data from final 5 days of month, earlier in the month, or carried over from prev month
    observed = p["clean_price"].notna()
    gp = p.groupby("cusip")
    last_obs_month = p["month"].where(observed).groupby(p["cusip"]).ffill()
    p["clean_price"] = gp["clean_price"].ffill()
    p["price_date"] = gp["price_date"].ffill()
    p["price_source"] = np.select([observed & p["in_window"].eq(True), observed], ["traded_window", "traded_month"], "carried")
    p["is_stale"] = p["price_source"] != "traded_window"
    p["months_since_trade"] = [(m - lm).n for m, lm in zip(p["month"], last_obs_month)]
    carried_ba = gp["bid_ask_spread_obs"].ffill()
    p["bid_ask_spread"] = p["bid_ask_spread_obs"].fillna(carried_ba)
    p["bid_ask_source"] = np.select([p["bid_ask_spread_obs"].notna(), carried_ba.notna()], ["observed", "carried"], "missing")
    for c in ["par_volume", "dollar_volume", "trade_count", "trade_days"]:
        p[c] = p[c].fillna(0)

    # keep only the months the bond was in AGG's start-of-month portfolio
    held = pd.MultiIndex.from_arrays([membership["cusip"], membership["month"]])
    p = p[pd.MultiIndex.from_arrays([p["cusip"], p["month"]]).isin(held)].reset_index(drop=True)
    p["month_end_date"] = p["month"].map(me["last_trading_day"])
    # observed rows are valued on the day the price was seen; carried rows at month-end
    p["valuation_date"] = p["price_date"].where(~p["price_source"].eq("carried"), p["month_end_date"])

    return p


def value(p, terms, curve):
    """Accrued interest (the source's own where observed, else from the coupon schedule), dirty price, reported yield
    on observed rows, solved yield to maturity and spread over the same-day zero curve, all at valuation_date.
    We just return p with the new cols
    """
    tm = terms.set_index("cusip")
    # initialize an array for the accrued int
    acc = np.full(len(p), np.nan)

    for cusip, idx in p.groupby("cusip").indices.items():
        # for each bond's data, we fill the slots of acc in order with the calculated accrued interest
        b = tm.loc[cusip]
        acc[idx] = accrued_interest(p["valuation_date"].values[idx], b["maturity_date"], b["coupon_rate"],
                                    int(b["coupon_frequency"]), b["day_count"])
    # we wanna use the accrued interest if the data source reported it, and also is not ffilled
    # else we use as calculated above in acc
    own = p["accrued_obs"].where(p["price_source"] != "carried") if "accrued_obs" in p else pd.Series(np.nan, index=p.index)
    p["accrued_interest"] = own.fillna(pd.Series(acc, index=p.index))
    p["dirty_price"] = p["clean_price"] + p["accrued_interest"]
    # we wanna use the yield if reported as well, else we use the solved yield and spreads 
    p["yield"] = p["yield_obs"].where(p["price_source"] != "carried") if "yield_obs" in p else np.nan
    solved = solve_yield_and_spread(p.assign(date=p["valuation_date"]), terms, curve)
    p["ytm_solved"], p["spread_bps"] = solved["ytm_solved"], solved["spread_bps"]
    p["years_to_maturity"] = (p["cusip"].map(tm["maturity_date"]) - p["month_end_date"]).dt.days / 365.25
    p["bid_ask_spread_bps"] = p["bid_ask_spread"] / p["clean_price"] * 1e4
    return p


def build_mbs_monthly(daily, universe, calendar, membership=None):
    """MBS month-end panel from the sparse daily pool prices. With membership (holdings.build_holdings_history), rows
    are kept only for months the pool was in AGG's start-of-month portfolio; every pool's trades still feed its own
    price history and the cohort index.

    Months without any trade are filled from the pool's coupon cohort (same program GNMA/conventional, term and
    coupon, the way pools are priced off TBAs): the pool's last traded price is moved by the cohort's median month-end
    price change since that trade (cohort_proxy), so the pool-specific pay-up is held constant and the market move
    comes from pools that did trade. With no cohort index the price is carried flat.
    Bid-ask: the month's mean observed spread, else the pool's last observed spread, else the cohort median that month.
    """
    u = universe[universe["asset_type"] == "mbs"].copy()
    # cohort, e.g. 'gnma|30|5.5': program and term from the AGG name, coupon rounded to the 0.5 grid
    name = u["name"].str.upper()
    program = np.where(name.str.contains("GNMA|GINNIE"), "gnma", "conv")
    term = name.str.extract(r"(\d+)\s*YR")[0].fillna("30")
    arm = np.where(name.str.contains("ARM"), "arm", "")
    coupon = (u["coupon_rate"] * 2).round() / 2
    u["cohort"] = pd.Series(program, index=u.index) + "|" + term + arm + "|" + coupon.map("{:.1f}".format)

    me = month_ends(calendar)
    obs = monthly_observations(daily, me, calendar).merge(u[["cusip", "cohort"]], on="cusip")
    # cohort index: median month-end price of the pools that traded (TBAs are not pools: keep them out), per
    # (cohort, month); NaN where fewer than MIN_COHORT_POOLS traded
    pools = obs[~obs["cusip"].isin(u.loc[u["is_tba"], "cusip"])]
    g = pools.groupby(["cohort", "month"])["clean_price"]
    cidx = g.median().where(g.size() >= MIN_COHORT_POOLS).rename("cohort_index").reset_index()

    # full grid: each pool from its first traded month to the last month with any trade in the data
    obs = obs[obs["month"] <= me.index[me["window_start"] <= daily["date"].max()].max()]    # no stale months past the data's end
    months = pd.period_range(obs["month"].min(), obs["month"].max(), freq="M")
    first = obs.groupby("cusip")["month"].min().rename("first_month").reset_index()
    grid = first.merge(pd.DataFrame({"month": months}), how="cross")
    grid = grid[grid["month"] >= grid["first_month"]].drop(columns="first_month")
    held = None
    if membership is not None:
        held = pd.MultiIndex.from_arrays([membership["cusip"], membership["month"]])
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
    p["valuation_date"] = p["month_end_date"]
    for c in ["par_volume", "dollar_volume", "trade_days", "trade_count"]:
        p[c] = p[c].fillna(0)
    if held is not None:
        p = p[pd.MultiIndex.from_arrays([p["cusip"], p["month"]]).isin(held)].copy()

    p["years_to_maturity"] = (p["cusip"].map(u.set_index("cusip")["maturity_date"]) - p["month_end_date"]).dt.days / 365.25
    cols = ["cusip", "month", "month_end_date", "valuation_date", "cohort", "is_tba", "clean_price", "accrued_interest",
            "dirty_price", "price_source", "is_stale", "price_date", "months_since_trade", "cohort_index", "years_to_maturity",
            "bid_ask_spread", "bid_ask_spread_bps", "bid_ask_source",
            "par_volume", "dollar_volume", "trade_days", "trade_count", "pool_factor", "pool_factor_date"]
    out = p[cols].reset_index(drop=True)
    log.info("mbs monthly: %d pool-months, %d pools, %s to %s", len(out), out["cusip"].nunique(),
             out["month_end_date"].min().date(), out["month_end_date"].max().date())
    log.info("price source: %s", out["price_source"].value_counts(normalize=True).round(3).to_dict())
    log.info("bid-ask source: %s", out["bid_ask_source"].value_counts(normalize=True).round(3).to_dict())
    return out


def held_days(daily, membership, calendar):
    """Daily rows for prices_daily: observed bond-days (no filling) on trading days from C.OUTPUT_START, in months the
    bond was in AGG's start-of-month portfolio."""
    d = daily[daily["date"].isin(calendar) & (daily["date"] >= pd.Timestamp(C.OUTPUT_START))]
    keys = pd.MultiIndex.from_arrays([d["cusip"], d["date"].dt.to_period("M")])
    held = pd.MultiIndex.from_arrays([membership["cusip"], membership["month"]])
    return d[pd.Series(keys.isin(held), index=d.index)].reset_index(drop=True)


def value_daily(d, terms, curve):
    """Daily version of value() for Treasuries and corporates: accrued interest (CRSP's own, else from the coupon
    schedule), dirty price, solved yield to maturity and spread over the same-day zero curve, bid-ask in bps."""
    if "accrued_interest" not in d:
        tm = terms.set_index("cusip")
        acc = np.full(len(d), np.nan)
        for cusip, idx in d.groupby("cusip").indices.items():
            if cusip in tm.index:
                b = tm.loc[cusip]
                acc[idx] = accrued_interest(d["date"].values[idx], b["maturity_date"], b["coupon_rate"],
                                            int(b["coupon_frequency"]), b["day_count"])
        d["accrued_interest"] = acc
        d["dirty_price"] = d["clean_price"] + d["accrued_interest"]
    solved = solve_yield_and_spread(d[["cusip", "date", "dirty_price"]], terms, curve)
    d["ytm_solved"], d["spread_bps"] = solved["ytm_solved"].to_numpy(), solved["spread_bps"].to_numpy()
    d["years_to_maturity"] = solved["years_to_maturity"].to_numpy()
    d["bid_ask_spread_bps"] = d["bid_ask_spread"] / d["clean_price"] * 1e4
    return d
