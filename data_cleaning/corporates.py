"""Step 4 (corporates): FISD bond terms, enhanced TRACE trades -> clean daily prices, Dickerson analytics merged in."""
import logging

import numpy as np
import pandas as pd

from . import config as C
from . import trace_clean
from .finmath import accrued_interest
from .wrds_pull import cached, chunks, connect, sql

log = logging.getLogger(__name__)

FISD_QUERY = """
    select complete_cusip as cusip, issue_id, maturity as maturity_date, coupon as coupon_rate, coupon_type,
           interest_frequency, dated_date, first_interest_date, last_interest_date, day_count_basis,
           offering_date, offering_amt as offering_amount, amount_outstanding, principal_amt as principal_amount,
           denomination, bond_type, security_level, redeemable as is_callable, convertible as is_convertible,
           rule_144a as is_144a
    from {table}
    where complete_cusip in %(cusips)s
"""

DICKERSON_QUERY = """
    select cusip_id as cusip, trd_exctn_dt as date, prc_vw_par as dk_price_vw, pr as dk_clean_price,
           prfull as dk_dirty_price, acclast as dk_accrued_interest, ytm as dk_yield, credit_spread as dk_credit_spread,
           mod_dur as dk_modified_duration, mac_dur as dk_macaulay_duration, convexity as dk_convexity,
           prc_bid as dk_price_bid, prc_ask as dk_price_ask, qvolume as dk_par_volume_mm, dvolume as dk_dollar_volume_mm,
           trade_count as dk_trade_count, bond_amt_outstanding as dk_amount_outstanding,
           sp_rating as dk_sp_rating_num, mdy_rating as dk_moodys_rating_num
    from {table}
    where cusip_id in %(cusips)s and trd_exctn_dt between %(start)s and %(end)s
"""


def pull_fisd_terms(cusips: list[str], refresh: bool = False) -> pd.DataFrame:
    return cached("fisd_corporate_terms",
                  lambda: sql(FISD_QUERY.format(table=C.FISD_ISSUES), {"cusips": tuple(cusips)},
                              date_cols=["maturity_date", "dated_date", "first_interest_date", "last_interest_date", "offering_date"]),
                  refresh)


def pull_dickerson(cusips: list[str], refresh: bool = False) -> pd.DataFrame:
    return cached("dickerson_corporate_daily",
                  lambda: sql(DICKERSON_QUERY.format(table=C.DICKERSON_DAILY),
                              {"cusips": tuple(cusips), "start": C.START_DATE, "end": C.END_DATE}, date_cols=["date"]),
                  refresh)


def pull_trace(cusips: list[str], table: str, name: str, refresh: bool = False):
    """Yield raw trade-level TRACE frames, one per CUSIP chunk, cached as data/raw/<name>/chunk_NNN_<window>.parquet."""
    have = set(connect().get_table(library=table.split(".")[0], table=table.split(".")[1], obs=1).columns)
    cols = ", ".join(c for c in trace_clean.SELECT_COLS if c in have)
    query = f"select {cols} from {table} where cusip_id in %(cusips)s and trd_exctn_dt between %(start)s and %(end)s"
    for i, batch in enumerate(chunks(sorted(cusips), C.TRACE_CHUNK_SIZE)):
        yield cached(f"{name}/chunk_{i:03d}",
                     lambda: sql(query, {"cusips": tuple(batch), "start": C.START_DATE, "end": C.END_DATE}), refresh)


def clean_terms(fisd: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """One row per corporate CUSIP: coupon, maturity, frequency, day count, min denomination, and a terms_source flag.
    FISD is used when it has the bond; otherwise the AGG file (coupon, maturity) with sector default conventions."""
    f = fisd.drop_duplicates("cusip").copy()
    f["coupon_frequency"] = pd.to_numeric(f["interest_frequency"], errors="coerce")
    f["day_count"] = f["day_count_basis"].fillna(C.DAY_COUNT["corporate"]).str.upper()
    # FISD denomination is "minimum/increment" in thousands of dollars (e.g. "2/1" = $2,000 minimum)
    f["min_denomination"] = pd.to_numeric(f["denomination"].str.split("/").str[0], errors="coerce") * 1000
    f["min_denomination"] = f["min_denomination"].fillna(f["principal_amount"])
    keep = ["cusip", "issue_id", "maturity_date", "coupon_rate", "coupon_type", "coupon_frequency", "day_count",
            "dated_date", "first_interest_date", "last_interest_date", "offering_date", "offering_amount",
            "amount_outstanding", "min_denomination", "bond_type", "security_level", "is_callable", "is_convertible", "is_144a"]
    u = universe.loc[universe["asset_type"] == "corporate", ["cusip", "agg_coupon_pct", "agg_maturity_date"]]
    t = u.merge(f[keep], on="cusip", how="left")
    t["terms_source"] = np.where(t["maturity_date"].notna() & t["coupon_rate"].notna(), "fisd", "agg_file")
    t["maturity_date"] = t["maturity_date"].fillna(t["agg_maturity_date"])
    t["coupon_rate"] = t["coupon_rate"].fillna(t["agg_coupon_pct"])
    t["coupon_frequency"] = t["coupon_frequency"].where(t["coupon_frequency"] > 0).fillna(C.COUPON_FREQ["corporate"]).astype(int)
    t["day_count"] = t["day_count"].fillna(C.DAY_COUNT["corporate"])
    log.info("corporate terms: %d bonds, %d from FISD", len(t), (t["terms_source"] == "fisd").sum())
    return t.drop(columns=["agg_coupon_pct", "agg_maturity_date"])


def build_corporate_daily(universe: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (corporate_daily, terms, cleaning_log). One row per (cusip, date) with a clean trade."""
    cusips = universe.loc[universe["asset_type"] == "corporate", "cusip"].tolist()
    terms = clean_terms(pull_fisd_terms(cusips, refresh), universe)

    daily, logs = [], []
    for raw in pull_trace(cusips, C.TRACE_CORP, "trace_corporate", refresh):
        trades, step_log = trace_clean.clean_trades(raw, product="corporate")
        logs.append(step_log)
        if len(trades):
            daily.append(trace_clean.daily_prices(trades))
    d = pd.concat(daily, ignore_index=True)
    clean_log = pd.concat(logs).groupby("step", sort=False)["trades_left"].sum().reset_index()

    # accrued interest from the FISD coupon schedule (30/360 for nearly all corporates), dirty price
    d = d.sort_values(["cusip", "date"]).reset_index(drop=True)
    acc = np.full(len(d), np.nan)
    tm = terms.set_index("cusip")
    for cusip, idx in d.groupby("cusip").indices.items():
        b = tm.loc[cusip]
        acc[idx] = accrued_interest(d["date"].values[idx], b["maturity_date"], b["coupon_rate"],
                                    int(b["coupon_frequency"]), b["day_count"])
    d["accrued_interest"] = acc
    d["dirty_price"] = d["clean_price"] + d["accrued_interest"]
    d["bid_ask_spread_bps"] = d["bid_ask_spread"] / d["clean_price"] * 1e4

    dk = pull_dickerson(cusips, refresh)
    d = d.merge(dk, on=["cusip", "date"], how="left")
    d["source"] = "trace_enhanced"
    log.info("corporate daily: %d rows, %d bonds, %s to %s; Dickerson analytics on %.0f%% of rows",
             len(d), d["cusip"].nunique(), d["date"].min().date(), d["date"].max().date(), 100 * d["dk_price_vw"].notna().mean())
    return d, terms, clean_log
