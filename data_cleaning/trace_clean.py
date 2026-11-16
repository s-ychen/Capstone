"""Trade-level TRACE cleaning (enhanced corporate, agency-MBS and TBA tables) and aggregation to daily prices.

Steps follow the Deliverable 3 spec:
  1. drop cancelled / corrected / reversed trades together with the reports they refer to (Dick-Nielsen 2009, 2014)
  2. drop the buy-side copy of inter-dealer trades, which TRACE reports twice (keep the sell-side report)
  3. drop when-issued, locked-in, special-condition and long-settlement (> 2 business days) trades (Dickerson et al.)
  4. drop price outliers with the median and reversal filters of Edwards, Harris and Piwowar (2007)
  5. MBS: convert original-face quantities to current face with the pool factor on file
Then one row per (cusip, date): face-weighted clean price, par and dollar volume, trade count,
dealer-side bid/ask prices and the bid-ask spread implied by customer trades.

Status codes as they appear in the post-2012 enhanced tables (checked on WRDS, Nov 2026):
  T trade, R corrected trade (replaces the report it names in orig_msg_seq_nb),
  C/X correction/cancellation markers carrying the ORIGINAL report's msg_seq_nb (so the original T row shares it),
  Y reversal of an earlier day's trade (asof_cd = 'R'), with the original identified by its economics.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PRICE_DEVIATION = 0.10       # Edwards et al.: 10% from the daily / 9-day median, or from neighbouring trades
MEDIAN_WINDOW_DAYS = 9       # centred window of trading days for the rolling median
MAX_SETTLE_DAYS = 2          # corporates: drop trades settling after T+2
OK_SALE_CONDITIONS = ("", "@")   # "@" = regular-way trade in TRACE sale-condition codes

# raw TRACE column -> our name. Columns absent from a table (e.g. yld_pt in the MBS table) are created as NaN.
RAW_COLS = {
    "cusip_id": "cusip", "trd_exctn_dt": "date", "trd_exctn_tm": "time", "rptd_pr": "price",
    "entrd_vol_qt": "quantity", "yld_pt": "yield_pct", "rpt_side_cd": "side", "cntra_mp_id": "contra",
    "trc_st": "status", "msg_seq_nb": "msg_seq", "orig_msg_seq_nb": "orig_msg_seq", "asof_cd": "asof",
    "wis_fl": "when_issued", "lckd_in_ind": "locked_in", "sale_cndtn_cd": "sale_cond", "sale_cndtn2_cd": "sale_cond2",
    "days_to_sttl_ct": "settle_days", "stlmnt_dt": "settle_date", "factor_on_file": "pool_factor", "factor": "pool_factor_alt",
}
SELECT_COLS = list(RAW_COLS)                       # what the pull functions ask WRDS for (filtered to the table's columns)
TRADE_KEY = ["cusip", "date", "time", "price", "quantity", "side", "contra"]   # identifies a trade for reversal matching


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=RAW_COLS)
    for c in RAW_COLS.values():
        if c not in df.columns:
            df[c] = np.nan
    df = df[list(RAW_COLS.values())].copy()
    df["cusip"] = df["cusip"].astype("string").str.strip().str.upper()
    df["date"] = pd.to_datetime(df["date"])
    df["settle_date"] = pd.to_datetime(df["settle_date"], errors="coerce")
    for c in ["price", "quantity", "yield_pct", "settle_days", "pool_factor", "pool_factor_alt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ["side", "contra", "status", "when_issued", "locked_in", "sale_cond", "sale_cond2", "msg_seq", "orig_msg_seq", "asof"]:
        df[c] = df[c].astype("string").str.strip().str.upper().fillna("")
    df["time"] = df["time"].astype("string").fillna("")
    df = df.dropna(subset=["cusip", "date", "price", "quantity"])
    return df[(df["price"] > 0) & (df["quantity"] > 0)]


def clean_trades(raw: pd.DataFrame, product: str = "corporate") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the cleaning steps. product is "corporate", "mbs_pool" or "tba". Returns (clean trades, step log).
    Pools and TBAs settle on forward SIFMA dates, so the T+2 settlement filter applies to corporates only;
    only pools need the pool-factor conversion to current face."""
    assert product in ("corporate", "mbs_pool", "tba"), product
    df = _prepare(raw)
    steps = [("0 valid raw trades", len(df))]

    # 1. C/X markers name the cancelled report: by their own msg_seq (usual) or by orig_msg_seq (rare).
    #    Drop those reports and the markers; keep R (corrected) reports unless they are cancelled in turn.
    marks = df[df["status"].isin(["C", "X"])]
    bad_ids = pd.concat([marks[["cusip", "date", "msg_seq"]],
                         marks[["cusip", "date", "orig_msg_seq"]].rename(columns={"orig_msg_seq": "msg_seq"})])
    bad_ids = bad_ids[bad_ids["msg_seq"] != ""].drop_duplicates().assign(_cancelled=True)
    reversals = df.loc[df["status"] == "Y", TRADE_KEY].copy()
    df = df.merge(bad_ids, on=["cusip", "date", "msg_seq"], how="left")
    df = df[df["_cancelled"].isna() & df["status"].isin(["T", "R"])].drop(columns="_cancelled")
    if len(reversals):                                   # one original per reversal, matched on economics
        reversals["_n"] = reversals.groupby(TRADE_KEY).cumcount()
        df["_n"] = df.groupby(TRADE_KEY).cumcount()
        df = df.merge(reversals.assign(_rev=True), on=TRADE_KEY + ["_n"], how="left")
        df = df[df["_rev"].isna()].drop(columns=["_n", "_rev"])
    df = df[df["asof"] != "R"]
    steps.append(("1 cancels/corrections/reversals", len(df)))

    # 2. inter-dealer trades appear once from each dealer: keep the sell side
    df = df[~((df["contra"] == "D") & (df["side"] == "B"))]
    steps.append(("2 inter-dealer duplicates", len(df)))

    # 3. non-standard trades
    bad = (df["when_issued"] == "Y") | (df["locked_in"] == "Y") \
        | ~df["sale_cond"].isin(OK_SALE_CONDITIONS) | ~df["sale_cond2"].isin(OK_SALE_CONDITIONS)
    if product == "corporate":                           # pass-throughs and TBAs settle on the monthly SIFMA date
        settle = df["settle_days"].copy()
        need = settle.isna() & df["settle_date"].notna()
        settle[need] = np.busday_count(df.loc[need, "date"].values.astype("datetime64[D]"),
                                       df.loc[need, "settle_date"].values.astype("datetime64[D]"))
        bad |= settle > MAX_SETTLE_DAYS                  # unknown settlement is kept
    df = df[~bad]
    steps.append(("3 non-standard trades", len(df)))
    if df.empty:
        return df, pd.DataFrame(steps, columns=["step", "trades_left"])

    # 4a. median filter: within 10% of the day's median and of the 9-trading-day centred median of daily medians
    daily_med = df.groupby(["cusip", "date"])["price"].median().rename("daily_median").reset_index()
    wide = daily_med.pivot(index="date", columns="cusip", values="daily_median")
    grid = wide.reindex(pd.bdate_range(wide.index.min(), wide.index.max()).union(wide.index))
    window_med = (grid.rolling(MEDIAN_WINDOW_DAYS, center=True, min_periods=1).median()
                  .loc[wide.index].stack().rename("window_median").reset_index())
    df = df.merge(daily_med, on=["cusip", "date"]).merge(window_med, on=["cusip", "date"])
    far = lambda ref: (df["price"] - ref).abs() / ref > PRICE_DEVIATION
    df = df[~(far(df["daily_median"]) | far(df["window_median"]))].drop(columns=["daily_median", "window_median"])
    steps.append(("4a median filter", len(df)))

    # 4b. reversal filter: >= 10% from the previous trade, the next trade and their average
    df = df.sort_values(["cusip", "date", "time"], kind="stable")
    g = df.groupby("cusip")["price"]
    prev, nxt = g.shift(1), g.shift(-1)
    far = lambda ref: (df["price"] - ref).abs() / ref >= PRICE_DEVIATION
    df = df[~(far(prev) & far(nxt) & far((prev + nxt) / 2))].copy()
    steps.append(("4b reversal filter", len(df)))

    # 5. face: corporates and TBAs report face in dollars; MBS pools report original face -> scale by the pool factor
    df["face"] = df["quantity"]
    if product == "mbs_pool":
        factor = df["pool_factor"].where(df["pool_factor"] > 0).fillna(df["pool_factor_alt"].where(df["pool_factor_alt"] > 0))
        df["pool_factor"] = factor
        df["face"] = df["quantity"] * factor             # NaN when no factor is on file
        df = df[df["face"].notna()]
        steps.append(("5 pool factor available", len(df)))
    return df.reset_index(drop=True), pd.DataFrame(steps, columns=["step", "trades_left"])


def daily_prices(trades: pd.DataFrame) -> pd.DataFrame:
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
        "price_low": g["price"].min(), "price_high": g["price"].max(),
        "par_volume": g["face"].sum(), "trade_count": g.size(),
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
