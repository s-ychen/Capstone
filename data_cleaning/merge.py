"""Stack the per-source panels into the output tables: prices_monthly, prices_daily, bond_universe and
agg_holdings_monthly (saved by run_pipeline to data/processed/)."""
import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

# columns of the combined prices_monthly table, in this order (missing ones are NaN for that asset type)
MONTHLY_COLS = ["cusip", "month", "month_end_date", "asset_type", "sector", "source", "price_source", "is_stale",
                "price_date", "valuation_date", "months_since_trade", "clean_price", "accrued_interest", "dirty_price",
                "yield", "ytm_solved", "spread_bps", "years_to_maturity", "bid_ask_spread", "bid_ask_spread_bps",
                "bid_ask_source", "par_volume", "dollar_volume", "trade_count", "trade_days",
                "pool_factor", "pool_factor_date", "is_tba", "cohort", "cohort_index"]


def build_bond_universe(universe, corp_terms, tsy, prices):
    """One row per bond held by AGG in any month of the window: the AGG fields from the latest holdings file it
    appears in (snapshot_date), when it was held, bond terms (CRSP / FISD / AGG file) and how many held months have
    an observed price (prices_monthly rows with price_source traded_window / traded_month).
    Time-varying benchmark weights are in agg_holdings_monthly, not here."""
    u = universe.rename(columns={"holdings_asof_date": "snapshot_date"}).copy()
    for c in ["first_held_month", "last_held_month"]:
        u[c] = u[c].dt.to_timestamp()

    t = tsy.drop_duplicates("cusip").set_index("cusip")
    ct = corp_terms.set_index("cusip")
    is_t, is_c = u["asset_type"] == "treasury", u["asset_type"] == "corporate"
    u["maturity_date"] = u["agg_maturity_date"]
    u["coupon_rate"] = u["agg_coupon_pct"]
    u.loc[is_t, "maturity_date"] = u.loc[is_t, "cusip"].map(t["maturity_date"]).fillna(u.loc[is_t, "maturity_date"])
    u.loc[is_t, "coupon_rate"] = u.loc[is_t, "cusip"].map(t["coupon_rate"]).fillna(u.loc[is_t, "coupon_rate"])
    u.loc[is_c, "maturity_date"] = u.loc[is_c, "cusip"].map(ct["maturity_date"])
    u.loc[is_c, "coupon_rate"] = u.loc[is_c, "cusip"].map(ct["coupon_rate"])
    u["coupon_frequency"] = u["asset_type"].map(C.COUPON_FREQ)
    u.loc[is_c, "coupon_frequency"] = u.loc[is_c, "cusip"].map(ct["coupon_frequency"])
    u["day_count"] = u["asset_type"].map(C.DAY_COUNT)
    u.loc[is_c, "day_count"] = u.loc[is_c, "cusip"].map(ct["day_count"])
    u["terms_source"] = np.select([is_t & u["cusip"].isin(t.index), is_c], ["crsp", u["cusip"].map(ct["terms_source"])], "agg_file")
    u["dated_date"] = pd.NaT
    u.loc[is_t, "dated_date"] = u.loc[is_t, "cusip"].map(t["dated_date"])
    u.loc[is_c, "dated_date"] = u.loc[is_c, "cusip"].map(ct["dated_date"])
    for c in ["coupon_type", "min_denomination", "amount_outstanding", "bond_type", "security_level", "is_callable", "is_144a"]:
        u[c] = u["cusip"].map(ct[c])
    u.loc[is_t, "min_denomination"] = 100.0

    obs = prices[prices["price_source"].str.startswith("traded")]
    cov = obs.groupby("cusip").agg(n_price_months=("month", "size"), first_price_month=("month", "min"), last_price_month=("month", "max"))
    u = u.merge(cov, on="cusip", how="left")
    u["n_price_months"] = u["n_price_months"].fillna(0).astype(int)
    u["has_prices"] = u["n_price_months"] > 0
    cols = ["cusip", "name", "sector", "asset_type", "agg_sector", "is_tba", "first_held_month", "last_held_month",
            "months_held", "snapshot_date", "agg_market_value", "agg_par_value", "has_prices", "n_price_months",
            "first_price_month", "last_price_month", "agg_duration", "agg_mod_duration", "agg_ytm_pct",
            "maturity_date", "coupon_rate", "coupon_frequency", "coupon_type", "day_count", "dated_date", "terms_source",
            "min_denomination", "amount_outstanding", "bond_type", "security_level", "is_callable", "is_144a"]
    out = u[cols].sort_values(["asset_type", "cusip"]).reset_index(drop=True)
    log.info("bonds with at least one price while held: %s", out.groupby("asset_type")["has_prices"].mean().round(3).to_dict())
    return out


def build_holdings_monthly(membership, prices):
    """The benchmark per month: one row per (month, cusip) in AGG's start-of-month portfolio.

    month           first day of the month the holdings apply to; holdings_date is the file's date (prior month-end)
    agg_weight      market value / the fund's total bond market value on holdings_date (sums to ~0.95 per month:
                    the excluded sectors make up the rest). The file's Weight (%) is rounded to 2 dp, hence recomputed
    price_status    observed (a quote or clean trade during the month), filled (no trade that month: an earlier price
                    carried forward, or an MBS cohort proxy; a usable but stale month-end price) or none
    model_weight    agg_weight renormalised over bonds with price_status != none, so it sums to 1 each month (b_t)
    """
    m = membership.copy()
    status = prices.assign(month=prices["month"].dt.to_period("M")).set_index(["cusip", "month"])["price_source"]
    src = pd.MultiIndex.from_frame(m[["cusip", "month"]]).map(status.to_dict().get)
    m["price_status"] = np.select([src.isin(["traded_window", "traded_month"]), src.isin(["carried", "cohort_proxy"])],
                                  ["observed", "filled"], "none")
    priced = m["price_status"] != "none"
    m["model_weight"] = np.where(priced, m["agg_weight"] / m["agg_weight"].where(priced).groupby(m["month"]).transform("sum"), np.nan)
    m["month"] = m["month"].dt.to_timestamp()
    cov = m.groupby("asset_type").apply(lambda g: g.loc[g["price_status"] != "none", "agg_weight"].sum() / g["agg_weight"].sum(),
                                        include_groups=False)
    log.info("benchmark weight with a price in the month (all months pooled): %s", cov.round(3).to_dict())
    return m.sort_values(["month", "asset_type", "cusip"]).reset_index(drop=True)


def build_prices_monthly(panels, universe):
    """Stack the per-asset-type month-end panels (panels.carry_panel + value, panels.build_mbs_monthly) on MONTHLY_COLS."""
    sec = universe.set_index("cusip")["sector"]
    frames = []
    for asset_type, d in panels.items():
        if d is None or not len(d):
            continue
        d = d.copy()
        d["asset_type"] = asset_type
        d["sector"] = d["cusip"].map(sec)
        if asset_type == "mbs":
            d["source"] = np.where(d["is_tba"], "trace_tba", "trace_mbs")
        for c in MONTHLY_COLS:
            if c not in d:
                d[c] = np.nan
        d["month"] = d["month"].dt.to_timestamp() if isinstance(d["month"].dtype, pd.PeriodDtype) else d["month"]
        frames.append(d[MONTHLY_COLS])
    p = pd.concat(frames, ignore_index=True).sort_values(["asset_type", "cusip", "month"]).reset_index(drop=True)
    assert not p.duplicated(["cusip", "month"]).any(), "duplicate (cusip, month) rows in prices_monthly"
    log.info("prices_monthly: %d bond-months; price source %s", len(p), p["price_source"].value_counts(normalize=True).round(3).to_dict())
    return p


DAILY_COLS = ["cusip", "date", "month", "asset_type", "sector", "source", "clean_price", "accrued_interest",
              "dirty_price", "yield", "ytm_solved", "spread_bps", "years_to_maturity", "bid_ask_spread",
              "bid_ask_spread_bps", "par_volume", "dollar_volume", "trade_count", "pool_factor", "is_tba"]


def build_prices_daily(frames, universe):
    """Stack the per-asset-type daily rows (panels.held_days, valued) on DAILY_COLS. Only observed days: Treasuries
    are quoted every trading day, corporates and MBS appear on days they traded. Volumes are 0 for CRSP quotes."""
    sec = universe.set_index("cusip")["sector"]
    out = []
    for asset_type, d in frames.items():
        if d is None or not len(d):
            continue
        d = d.copy()
        d["asset_type"] = asset_type
        d["sector"] = d["cusip"].map(sec)
        d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
        if asset_type == "mbs":
            d["source"] = np.where(d["is_tba"], "trace_tba", "trace_mbs")
            d["years_to_maturity"] = (d["maturity_date"] - d["date"]).dt.days / 365.25
        if asset_type == "treasury":
            d[["par_volume", "dollar_volume", "trade_count"]] = 0.0
        for c in DAILY_COLS:
            if c not in d:
                d[c] = np.nan
        out.append(d[DAILY_COLS])
    p = pd.concat(out, ignore_index=True).sort_values(["asset_type", "cusip", "date"]).reset_index(drop=True)
    p["is_tba"] = p["is_tba"].astype("boolean")
    assert not p.duplicated(["cusip", "date"]).any(), "duplicate (cusip, date) rows in prices_daily"
    log.info("prices_daily: %d bond-days, %s", len(p), p["asset_type"].value_counts().to_dict())
    return p

