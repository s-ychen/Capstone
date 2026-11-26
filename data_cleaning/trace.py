"""
TRACE trades -> cleaned daily prices, for the corporates (enhanced TRACE) and the MBS pass-throughs (SPDS pool and
TBA tables), plus the corporates' bond terms from FISD.

Trade cleaning (from deliverable 3):
  1. drop cancelled / corrected / reversed trades together with the reports they refer to (per Dick-Nielsen 2009, 2014)
  2. drop the buy-side copy of inter-dealer trades, which TRACE reports twice (keep the sell-side report)
  3. drop when-issued, locked-in, special-condition and long-settlement (> 2 business days) trades ( per Dickerson et al.)
  4. drop price outliers with the median and reversal filters of Edwards, Harris and Piwowar (used in WRDS's methodology)
  5. MBS: convert original-face quantities to current face with the pool factor on file
Then one row per (cusip, date): face-weighted clean price, par and dollar volume, trade count,
dealer-side bid/ask prices and the bid-ask spread implied by customer trades.

Status codes as they appear in the post-2012 enhanced tables:
  T trade, R corrected trade (replaces the report it names in orig_msg_seq_nb),
  C/X correction/cancellation markers carrying the ORIGINAL report's msg_seq_nb (so the original T row shares it),
  Y reversal of an earlier day's trade (asof_cd = 'R'), with the original identified by its economics.
"""
import logging

import numpy as np
import pandas as pd

from . import config as C
from .pull import TRACE_COLS, pull_fisd_terms, pull_trace

log = logging.getLogger(__name__)


PRICE_DEVIATION = .10       # from Edwards et al.: 10% from the daily / 9-day median, or from neighbouring trades
MEDIAN_WINDOW_DAYS = 9       # centred window of trading days for the rolling median
MAX_SETTLE_DAYS = 2          # corporates: drop trades settling after T+2
OK_SALE_CONDITIONS = ("", "@")   # "@" = regular-way trade in TRACE sale-condition codes
TRADE_KEY = ["cusip", "date", "time", "price", "quantity", "side", "contra"]   # good cols to match reversed trades


def clean_trades(raw, product="corporate"):
    """Apply the cleaning steps. product is "corporate", "mbs_pool" or "tba" (tba mbs's). Returns (clean trades, [(step, trades left)]).
    Pools and TBAs settle on forward SIFMA dates, so the T+2 settlement filter applies to corporates only;
    only pools need the pool-factor conversion to current face."""
    """
    Apply the cleaning steps outlined above and also in our deliverable 3
    The products are either corporate, mbs_pool or tba (which are just TBA mbas). 
    """
    assert product in ("corporate", "mbs_pool", "tba"), product

    # first we clean the data 
    # standardize trace col names with rest of data, then convert to standard data types
    master_df = raw.rename(columns=TRACE_COLS)
    for c in TRACE_COLS.values():
        if c not in master_df.columns:
            master_df[c] = np.nan
    master_df = master_df[list(TRACE_COLS.values())].copy()
    master_df["cusip"] = master_df["cusip"].astype("string").str.strip().str.upper()
    master_df["date"] = pd.to_datetime(master_df["date"])
    master_df["settle_date"] = pd.to_datetime(master_df["settle_date"], errors="coerce")
    for c in ["price", "quantity", "yield_pct", "settle_days", "pool_factor", "pool_factor_alt"]:
        master_df[c] = pd.to_numeric(master_df[c], errors="coerce").astype(float)
    for c in ["side", "contra", "status", "when_issued", "locked_in", "sale_cond", "sale_cond2", "msg_seq", "orig_msg_seq", "asof"]:
        master_df[c] = master_df[c].astype("string").str.strip().str.upper().fillna("")
    master_df["time"] = master_df["time"].astype("string").fillna("")
    master_df = master_df.dropna(subset=["cusip", "date", "price", "quantity"])
    master_df = master_df[(master_df["price"] > 0) & (master_df["quantity"] > 0)]
    steps = [("0 valid raw trades", len(master_df))]

    # 1. we want to remove cancelled trades (msg_seq/orig_msg_seq=C or X) and we want to keep corrected/reversed trades (R).
    marks = master_df[master_df["status"].isin(["C", "X"])]
    bad_ids = pd.concat([marks[["cusip", "date", "msg_seq"]], marks[["cusip", "date", "orig_msg_seq"]].rename(columns={"orig_msg_seq": "msg_seq"})])
    bad_ids = bad_ids[bad_ids["msg_seq"] != ""].drop_duplicates().assign(_cancelled=True)
    reversals = master_df.loc[master_df["status"] == "Y", TRADE_KEY].copy()
    master_df = master_df.merge(bad_ids, on=["cusip", "date", "msg_seq"], how="left")
    master_df = master_df[master_df["_cancelled"].isna() & master_df["status"].isin(["T", "R"])].drop(columns="_cancelled")
     # one original per reversal
    if len(reversals):                                  
        reversals["_n"] = reversals.groupby(TRADE_KEY).cumcount()
        master_df["_n"] = master_df.groupby(TRADE_KEY).cumcount()
        # merge master_df with our reversals, on the cols in TRADE_KEY and _n
        master_df = master_df.merge(reversals.assign(_rev=True), on=TRADE_KEY + ["_n"], how="left")
        master_df = master_df[master_df["_rev"].isna()].drop(columns=["_n", "_rev"])
    master_df = master_df[master_df["asof"] != "R"]
    steps.append(("1 cancels/corrections/reversals", len(master_df)))

    # 2. since trades between dealers have them register both a buy and sell side, just keep the buy side trade logs
    master_df = master_df[~((master_df["contra"] == "D") & (master_df["side"] == "B"))]
    steps.append(("2 inter-dealer duplicates", len(master_df)))

    # 3. non-standard trades
    # we filter out those that were traded before issuance, locked-in trades, or those with no good sale conditions as defined at top of script
    bad = (master_df["when_issued"] == "Y") | (master_df["locked_in"] == "Y") | ~master_df["sale_cond"].isin(OK_SALE_CONDITIONS) | ~master_df["sale_cond2"].isin(OK_SALE_CONDITIONS)
    if product == "corporate":                          
        settle = master_df["settle_days"].copy()
        need = settle.isna() & master_df["settle_date"].notna()
        settle[need] = np.busday_count(master_df.loc[need, "date"].values.astype("datetime64[D]"),
                                       master_df.loc[need, "settle_date"].values.astype("datetime64[D]"))
        bad |= settle > MAX_SETTLE_DAYS # add condition that the settle is past the max settle days        
    master_df = master_df[~bad]
    steps.append(("3 non-standard trades", len(master_df)))
    if master_df.empty:
        return master_df, steps

    # 4a. median filter: within 10% of the day's median and of the 9-trading-day centred median of daily medians 
    daily_med = master_df.groupby(["cusip", "date"])["price"].median().rename("daily_median").reset_index()
    wide = daily_med.pivot(index="date", columns="cusip", values="daily_median")
    grid = wide.reindex(pd.bdate_range(wide.index.min(), wide.index.max()).union(wide.index))
    window_med = (grid.rolling(MEDIAN_WINDOW_DAYS, center=True, min_periods=1).median()
                  .loc[wide.index].stack().rename("window_median").reset_index())
    master_df = master_df.merge(daily_med, on=["cusip", "date"]).merge(window_med, on=["cusip", "date"])
    far = lambda ref: (master_df["price"] - ref).abs() / ref > PRICE_DEVIATION
    master_df = master_df[~(far(master_df["daily_median"]) | far(master_df["window_median"]))].drop(columns=["daily_median", "window_median"])
    steps.append(("4a median filter", len(master_df)))

    # 4b. reversal filter: >= 10% from the previous trade, the next trade and their average
    master_df = master_df.sort_values(["cusip", "date", "time"], kind="stable")
    g = master_df.groupby("cusip")["price"]
    prev, nxt = g.shift(1), g.shift(-1)
    far = lambda ref: (master_df["price"] - ref).abs() / ref >= PRICE_DEVIATION
    master_df = master_df[~(far(prev) & far(nxt) & far((prev + nxt) / 2))].copy()
    steps.append(("4b reversal filter", len(master_df)))

    # 5. face: corporates and TBAs report face in dollars; MBS pools report original face -> scale by the pool factor
    master_df["face"] = master_df["quantity"]
    if product == "mbs_pool":
        factor = master_df["pool_factor"].where(master_df["pool_factor"] > 0).fillna(master_df["pool_factor_alt"].where(master_df["pool_factor_alt"] > 0))
        master_df["pool_factor"] = factor
        master_df["face"] = master_df["quantity"] * factor             
        master_df = master_df[master_df["face"].notna()]
        steps.append(("5 pool factor available", len(master_df)))
    return master_df.reset_index(drop=True), steps


def daily_prices(trades): 
    """Collapse clean trades to one row per (cusip, date).

    clean_price        face-weighted average price (TRACE prices are clean, per 100 face)
    yield              face-weighted average of reported yields, decimal (NaN if none reported)
    par_volume         face traded in dollars (current face for MBS); dollar_volume = par_volume * price / 100
    price_bid/ask      face-weighted price at which dealers bought from / sold to customers
    bid_ask_spread     price_ask - price_bid in price points, NaN unless customers traded on both sides that day
    """
    t = trades.assign(pw=trades["price"] * trades["face"],
                      yw=trades["yield_pct"] * trades["face"],
                      yface=trades["face"].where(trades["yield_pct"].notna(), 0.0))
    g = t.groupby(["cusip", "date"])
    d = pd.DataFrame({
        "clean_price": g["pw"].sum() / g["face"].sum(),
        "yield": g["yw"].sum() / g["yface"].sum().replace(0, np.nan) / 100,
        "price_low": g["price"].min(), 
        "price_high": g["price"].max(),
        "par_volume": g["face"].sum(), 
        "trade_count": g.size(),
    })
    if "pool_factor" in t and t["pool_factor"].notna().any():
        d["pool_factor"] = g["pool_factor"].last()
    d["dollar_volume"] = d["par_volume"] * d["clean_price"] / 100
    cust = t[t["contra"] == "C"].groupby(["cusip", "date", "side"]).agg(pw=("pw", "sum"), face=("face", "sum"))
    side_px = (cust["pw"] / cust["face"]).unstack("side").reindex(d.index)
    d["price_bid"] = side_px["B"] if "B" in side_px else np.nan      # dealer buys from customer
    d["price_ask"] = side_px["S"] if "S" in side_px else np.nan      # dealer sells to customer
    d["bid_ask_spread"] = (d["price_ask"] - d["price_bid"]).clip(lower=0)
    return d.reset_index()


def clean_terms(fisd, universe):
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


def log_cleaning(name, step_lists):
    """Print how many trades are left after each cleaning step, summed over the TRACE buckets"""
    left = {}
    for steps in step_lists:
        for step, n in steps:
            left[step] = left.get(step, 0) + n
    first = next(iter(left.values()), 0) or 1
    lines = [f"  {step:34s} {n:>14,}  ({n / first:.1%})" for step, n in left.items()]
    log.info("%s cleaning, trades left after each step:\n%s", name, "\n".join(lines))


def build_corporate_daily(universe, refresh=False):
    """Returns (corporate_daily, terms). One row per cusip x date with a clean trade."""
    cusips = universe.loc[universe["asset_type"] == "corporate", "cusip"].tolist()
    terms = clean_terms(pull_fisd_terms(cusips, refresh), universe)

    # clean each bucket of raw trades and collapse it to daily prices 
    daily, logs = [], []
    for raw in pull_trace(cusips, C.TRACE_CORP, "trace_corporate", refresh):
        trades, steps = clean_trades(raw, product="corporate")
        logs.append(steps)
        if len(trades):
            daily.append(daily_prices(trades))
    d = pd.concat(daily, ignore_index=True).sort_values(["cusip", "date"]).reset_index(drop=True)
    d["source"] = "trace_enhanced"
    log_cleaning("trace_corporate", logs)

    log.info("corporate daily: %d bond-days, %d bonds, %s to %s", len(d), d["cusip"].nunique(),
             d["date"].min().date(), d["date"].max().date())
    return d, terms


def build_mbs_daily(universe, refresh=False):
    """Returns mbs_daily. Pools come from the SPDS MBS table (quantities scaled to current face by the
    pool factor on file); AGG's TBA positions come from the TBA table (face already current, no factor)."""
    u = universe[universe["asset_type"] == "mbs"]
    groups = [(u.loc[~u["is_tba"], "cusip"].tolist(), C.TRACE_MBS, "trace_mbs", "mbs_pool"),
              (u.loc[u["is_tba"], "cusip"].tolist(), C.TRACE_TBA, "trace_tba", "tba")]
    daily = []
    for cusips, table, name, product in groups:
        if not cusips:
            continue
        logs = []
        for raw in pull_trace(cusips, table, name, refresh):
            trades, steps = clean_trades(raw, product=product)
            logs.append(steps)
            if len(trades):
                daily.append(daily_prices(trades).assign(source=name))
        log_cleaning(name, logs)
    d = pd.concat(daily, ignore_index=True).sort_values(["cusip", "date"]).reset_index(drop=True)

    # terms come from the AGG file (coupon, stated maturity)
    terms = u.set_index("cusip")
    d["coupon_rate"] = d["cusip"].map(terms["agg_coupon_pct"])
    d["maturity_date"] = d["cusip"].map(terms["agg_maturity_date"])
    d["is_tba"] = d["cusip"].map(terms["is_tba"])
    # agency pass-throughs accrue 30/360 from the 1st of the month: coupon/12 * (day - 1)/30 per 100 face
    day = pd.DatetimeIndex(d["date"]).day.values
    d["accrued_interest"] = d["coupon_rate"].to_numpy() / 12 * (np.minimum(day, 30) - 1) / 30
    d["dirty_price"] = d["clean_price"] + d["accrued_interest"]
    d["bid_ask_spread_bps"] = d["bid_ask_spread"] / d["clean_price"] * 1e4
    if "pool_factor" not in d:
        d["pool_factor"] = np.nan

    # lalala manual part
    log.info("mbs daily: %d rows, %d pools, %d TBAs, %s to %s", len(d), d.loc[~d["is_tba"], "cusip"].nunique(),
             d.loc[d["is_tba"], "cusip"].nunique(), d["date"].min().date(), d["date"].max().date())
    return d