"""Step 5: month-end prices (last of the final 5 trading days), stale carry-forward, accrued interest, spreads, yields, costs."""
import numpy as np
import pandas as pd

from . import config as C
from .curve import params_asof
from .finmath import accrued, cashflows, coupon_dates, solve_batch


def window_dates(cal, n):
    """The last `n` trading days of every month in the calendar."""
    s = pd.Series(pd.DatetimeIndex(cal))
    return pd.DatetimeIndex(s.groupby(s.dt.to_period("M")).tail(n))


def complete_months(cal):
    """Month-end dates of months whose last trading day is within 4 days of the true month-end (drops the unfinished month)."""
    s = pd.Series(pd.DatetimeIndex(cal))
    last = s.groupby(s.dt.to_period("M")).max()
    me = last.index.to_timestamp(how="end").normalize()
    return pd.DatetimeIndex(me[(me - last.values) <= pd.Timedelta(days=4)])


def month_end(daily, cal):
    """Last observation per (cusip, month) inside the final WINDOW_DAYS trading days. `price_date` = day actually observed."""
    d = daily[daily["price_date"].isin(window_dates(cal, C.WINDOW_DAYS))].copy()
    d["month_end_date"] = d["price_date"] + pd.offsets.MonthEnd(0)
    return d.sort_values(["cusip", "price_date"]).drop_duplicates(["cusip", "month_end_date"], keep="last")


def carry_forward(obs, terms, months):
    """
    Full (cusip x month-end) grid from each bond's first price until it matures (or its source's last month). Missing months get the
    previous clean price and is_stale=True. Reported yields/spreads are never carried (they belong to the old date).
    """
    first = obs.groupby("cusip").agg(first_month=("month_end_date", "min"), src=("source", "first")).reset_index()
    first["last_month"] = first["src"].map(obs.groupby("source")["month_end_date"].max())
    grid = first.merge(terms[["cusip", "maturity_date"]], on="cusip", how="left").merge(pd.DataFrame({"month_end_date": months}), how="cross")
    grid = grid[(grid["month_end_date"] >= grid["first_month"]) & (grid["month_end_date"] <= grid["last_month"])
                & (grid["maturity_date"].isna() | (grid["maturity_date"] > grid["month_end_date"]))]
    grid = grid[["cusip", "month_end_date"]].merge(obs, on=["cusip", "month_end_date"], how="left").sort_values(["cusip", "month_end_date"]).reset_index(drop=True)
    grid["is_stale"] = grid["clean_price"].isna()
    last_obs = grid["month_end_date"].where(~grid["is_stale"]).groupby(grid["cusip"]).ffill()
    carried = ["price_date", "clean_price", "source", "half_spread_frac"]
    grid[carried] = grid.groupby("cusip")[carried].ffill()
    grid["months_since_observed"] = (grid["month_end_date"].dt.year - last_obs.dt.year) * 12 + grid["month_end_date"].dt.month - last_obs.dt.month
    grid.loc[grid["is_stale"], ["yield_pct_reported", "spread_bps_reported"]] = np.nan
    if C.MAX_STALE_MONTHS is not None:
        grid = grid[grid["months_since_observed"] <= C.MAX_STALE_MONTHS]
    return grid.reset_index(drop=True)


def value(rows, terms, curves):
    """
    Add valuation_date (price_date, or month_end_date if stale), accrued_interest, dirty_price, spread_bps (over the fitted curve at
    month-end) and yield_pct. BondRet-reported spread/yield are used when present, otherwise solved from the cash flows in batch.
    """
    rows = rows.reset_index(drop=True)
    for c in ["yield_pct_reported", "spread_bps_reported", "accrued_interest_crsp"]:
        if c not in rows:
            rows[c] = np.nan
    rows["valuation_date"] = rows["price_date"].where(~rows["is_stale"], rows["month_end_date"])
    need_solve = rows["spread_bps_reported"].isna() | rows["yield_pct_reported"].isna()
    acc, times, amts = np.full(len(rows), np.nan), [np.array([])] * len(rows), [np.array([])] * len(rows)
    start, term = rows["valuation_date"].min() - pd.DateOffset(years=2), terms.set_index("cusip")
    for cusip, g in rows.groupby("cusip", sort=False):
        b = term.loc[cusip]
        if pd.isna(b["maturity_date"]) or pd.isna(b["coupon_pct"]):
            continue
        s = coupon_dates(b["maturity_date"], b["coupon_frequency"], start)
        acc[g.index] = accrued(g["valuation_date"].values, s, b["coupon_pct"], b["coupon_frequency"], b["day_count"])
        for i in g.index[need_solve[g.index]]:
            times[i], amts[i] = cashflows(rows.at[i, "valuation_date"], s, b["coupon_pct"], b["coupon_frequency"], b["is_amortising"])

    crsp = (rows["source"] == "crsp") & ~rows["is_stale"] & rows["accrued_interest_crsp"].notna()
    rows["accrued_interest_diff_vs_crsp"] = np.where(crsp, acc - rows["accrued_interest_crsp"], np.nan)   # sanity check
    rows["accrued_interest"] = np.where(crsp, rows["accrued_interest_crsp"], acc)                        # spec: CRSP accrued for Treasuries
    rows["dirty_price"] = rows["clean_price"] + rows["accrued_interest"]

    p = params_asof(curves, rows["month_end_date"])
    sp = np.where(need_solve & rows["spread_bps_reported"].isna(), solve_batch(rows["dirty_price"], times, amts, "spread", p), np.nan)
    yl = np.where(need_solve & rows["yield_pct_reported"].isna(), solve_batch(rows["dirty_price"], times, amts, "ytm"), np.nan)
    rows["spread_source"] = np.where(rows["spread_bps_reported"].notna(), "reported", "solved")
    rows["yield_source"] = np.where(rows["yield_pct_reported"].notna(), "reported", "solved")
    rows["spread_bps"] = rows["spread_bps_reported"].fillna(pd.Series(sp, index=rows.index))
    rows["yield_pct"] = rows["yield_pct_reported"].fillna(pd.Series(yl, index=rows.index))
    return rows


def build_prices(tsy_daily, bondret, trace_daily, terms, cal, curves):
    """
    Combine the three price sources, carry forward stale prices, value them, and add transaction costs.
    Returns the Prices table: one row per bond per complete month-end, with is_stale and usable_for_residual_variance flags.
    """
    months = complete_months(cal)
    br = bondret.assign(date=pd.to_datetime(bondret["date"]))
    br = pd.DataFrame({"cusip": br["cusip"], "month_end_date": br["date"] + pd.offsets.MonthEnd(0), "price_date": br["date"],
                       "clean_price": pd.to_numeric(br["price_l5m"], errors="coerce"),
                       "yield_pct_reported": pd.to_numeric(br["t_yld_pt"], errors="coerce"),
                       "spread_bps_reported": pd.to_numeric(br["t_spread"], errors="coerce") * C.BONDRET_SPREAD_TO_BPS,
                       "volume": pd.to_numeric(br["t_dvolume"], errors="coerce"), "source": "bondret"}).dropna(subset=["clean_price"])
    obs = pd.concat([month_end(tsy_daily, cal).assign(source="crsp"), br, month_end(trace_daily, cal)], ignore_index=True)
    obs = obs[obs["month_end_date"].isin(months)]
    print(obs.groupby("source")["cusip"].nunique().rename("bonds with a price"))

    p = value(carry_forward(obs, terms, months), terms, curves).sort_values(["cusip", "month_end_date"])
    p["is_stale_prev"] = p.groupby("cusip")["is_stale"].shift(1).fillna(True).astype(bool)
    p["usable_for_residual_variance"] = ~(p["is_stale"] | p["is_stale_prev"])
    default = p["cusip"].map(terms.set_index("cusip")["model_sector"]).map(C.HALF_SPREAD_BPS) / 1e4     # placeholder where unobserved
    p["transaction_cost_frac"] = p["half_spread_frac"].fillna(default)
    p["cost_source"] = np.where(p["half_spread_frac"].notna(), "observed", "sector_default")
    keep = ["cusip", "month_end_date", "price_date", "valuation_date", "source", "is_stale", "is_stale_prev", "usable_for_residual_variance",
            "clean_price", "accrued_interest", "dirty_price", "spread_bps", "spread_source", "yield_pct", "yield_source",
            "transaction_cost_frac", "cost_source", "volume", "accrued_interest_diff_vs_crsp"]
    return p[[c for c in keep if c in p]].reset_index(drop=True)
