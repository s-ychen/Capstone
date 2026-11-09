"""Step 4: clean trade-level TRACE data (Dick-Nielsen, Dickerson et al., Edwards et al.) and aggregate to daily prices."""
import numpy as np
import pandas as pd

from . import config as C

KEY = ["cusip", "trade_date", "trade_time", "price", "quantity", "side", "contra"]


def clean_trades(raw, is_mbs=False, pool_factors=None):
    """
    Returns (clean trades, log of trades removed per step). Steps:
      1 drop cancelled/corrected/reversed trades and their originals     2 keep only the sell-side report of inter-dealer trades
      3 drop when-issued, locked-in, special-condition, long-settlement   4 median filter and reversal filter (10%)
      5 MBS: quantity -> current face via pool_factors [cusip, factor_date, pool_factor] (optional)
    """
    if raw.empty:
        return (pd.DataFrame(columns=["cusip", "trade_date", "trade_time", "price", "yield_pct", "side", "contra", "qty_face"]),
                pd.DataFrame({"step": ["0. valid raw trades"], "trades_left": [0]}))
    df = raw.rename(columns={v: k for k, v in C.TRACE_COLS.items()}).copy()
    for c in C.TRACE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df["cusip"] = df["cusip"].astype("string").str.strip().str.upper()
    for c in ["trade_date", "report_date", "settle_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["report_date"] = df["report_date"].fillna(df["trade_date"])
    for c in ["price", "quantity", "yield_pct", "settle_days", "pool_factor"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)      # plain float: nullable NA would poison the boolean masks below
    for c in ["side", "contra", "status", "wi", "locked", "cond1", "cond2", "msg_seq", "orig_msg_seq"]:
        df[c] = df[c].astype("string").str.strip().str.upper()
    for c in ["side", "contra", "wi", "locked", "cond1", "cond2"]:     # blanks become "" (NA would make comparisons drop rows)
        df[c] = df[c].fillna("")
    df["status"] = df["status"].fillna("T")
    df["trade_time"] = df["trade_time"].astype("string")
    df = df.dropna(subset=["cusip", "trade_date", "price", "quantity"])
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    log = [("0. valid raw trades", len(df))]

    # 1) cancels/corrections point at their original via (cusip, trade_date, orig_msg_seq); reversals match on identical economics
    cancels = df.loc[df["status"].isin(["X", "C"]), ["cusip", "trade_date", "orig_msg_seq"]].dropna().rename(columns={"orig_msg_seq": "msg_seq"})
    df = df.merge(cancels.assign(_cancelled=True).drop_duplicates(), on=["cusip", "trade_date", "msg_seq"], how="left")
    df = df[df["_cancelled"].isna() & ~df["status"].isin(["X", "C"])].drop(columns="_cancelled")
    rev, orig = df["status"] == "Y", df["status"] != "Y"
    if rev.any():
        o = df[orig].assign(_n=df[orig].groupby(KEY, dropna=False).cumcount())
        r = df.loc[rev, KEY].assign(_n=df.loc[rev, KEY].groupby(KEY, dropna=False).cumcount(), _rev=True)
        df = o.merge(r, on=KEY + ["_n"], how="left")
        df = df[df["_rev"].isna()].drop(columns=["_n", "_rev"])
    df = df[df["status"].isin(["T", "W"])]
    log.append(("1. cancelled/corrected/reversed", len(df)))

    # 2) inter-dealer trades are reported twice: keep the sell side
    df = df[~((df["contra"] == "D") & (df["side"] == "B"))]
    log.append(("2. inter-dealer duplicates", len(df)))

    # 3) non-standard trades
    bad = (df["wi"] == "Y") | (df["locked"] == "Y") | ~df["cond1"].isin(C.TRACE_OK_CONDITIONS) | ~df["cond2"].isin(C.TRACE_OK_CONDITIONS)
    if C.TRACE_MAX_SETTLE_DAYS is not None and not is_mbs:          # pass-throughs settle on the monthly SIFMA date, so no T+n rule
        ok = df["trade_date"].notna() & df["settle_date"].notna()      # days_to_sttl_ct is blank in enhanced TRACE: derive it from stlmnt_dt
        derived = pd.Series(np.nan, index=df.index)
        derived[ok] = np.busday_count(df.loc[ok, "trade_date"].values.astype("datetime64[D]"), df.loc[ok, "settle_date"].values.astype("datetime64[D]"))
        bad |= df["settle_days"].fillna(derived) > C.TRACE_MAX_SETTLE_DAYS   # NaN > n is False: unknown settlement is kept
    df = df[~bad.fillna(False).astype(bool)]
    log.append(("3. non-standard trades", len(df)))

    if df.empty:
        return df.reindex(columns=["cusip", "trade_date", "trade_time", "price", "yield_pct", "side", "contra", "qty_face"]), pd.DataFrame(log, columns=["step", "trades_left"])

    # 4a) median filter: price within 10% of the daily median and of the 9-trading-day centred median (median of daily medians)
    daily = df.groupby(["cusip", "trade_date"])["price"].median().rename("daily_med").reset_index()
    wide = daily.pivot(index="trade_date", columns="cusip", values="daily_med")
    grid = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="B").union(wide.index))
    roll = grid.rolling(2 * C.TRACE_MARGIN_DAYS + 1, center=True, min_periods=1).median().loc[wide.index].stack().rename("win_med").reset_index()
    df = df.merge(daily, on=["cusip", "trade_date"]).merge(roll, on=["cusip", "trade_date"])
    dev = lambda m: (df["price"] - m).abs() / m
    df = df[~((dev(df["daily_med"]) > C.TRACE_PRICE_DEV) | (dev(df["win_med"]) > C.TRACE_PRICE_DEV))]
    log.append(("4a. median filter", len(df)))

    # 4b) reversal filter: >=10% away from previous trade, next trade and their average
    df = df.sort_values(["cusip", "trade_date", "trade_time"], kind="stable")
    g = df.groupby("cusip")["price"]
    prev, nxt = g.shift(1), g.shift(-1)
    far = lambda ref: (df["price"] - ref).abs() / ref >= C.TRACE_PRICE_DEV
    df = df[~(far(prev) & far(nxt) & far((prev + nxt) / 2))].copy()
    log.append(("4b. reversal filter", len(df)))

    # 5) MBS: TRACE reports original face; scale by the pool factor (TRACE factor_on_file, else the optional pool_factors file, else 1)
    df["qty_face"] = df["quantity"]
    if is_mbs:
        if pool_factors is not None:
            pf = pool_factors.rename(columns={"pool_factor": "file_factor"}).assign(factor_date=lambda x: pd.to_datetime(x["factor_date"]))
            df = pd.merge_asof(df.sort_values("trade_date"), pf.sort_values("factor_date"), left_on="trade_date", right_on="factor_date", by="cusip")
            df["pool_factor"] = df["pool_factor"].where(df["pool_factor"] > 0).fillna(df["file_factor"])
        df["qty_face"] = df["quantity"] * df["pool_factor"].where(df["pool_factor"] > 0).fillna(1.0)
    log_df = pd.DataFrame(log, columns=["step", "trades_left"])
    return df[["cusip", "trade_date", "trade_time", "price", "yield_pct", "side", "contra", "qty_face"]].reset_index(drop=True), log_df


def daily_prices(t):
    """
    One row per (cusip, price_date): clean_price (volume-weighted), yield_pct_reported (volume-weighted), volume, and
    half_spread_frac = (VWAP dealer sells - VWAP dealer buys)/2/price on customer trades (NaN unless both sides traded).
    """
    t = t.assign(pw=t["price"] * t["qty_face"], yw=t["yield_pct"] * t["qty_face"], yq=t["qty_face"].where(t["yield_pct"].notna(), 0.0))
    g = t.groupby(["cusip", "trade_date"])
    d = pd.DataFrame({"clean_price": g["pw"].sum() / g["qty_face"].sum(),
                      "yield_pct_reported": g["yw"].sum() / g["yq"].sum().replace(0, np.nan),
                      "volume": g["qty_face"].sum()})
    c = t[t["contra"] == "C"].groupby(["cusip", "trade_date", "side"]).agg(pw=("pw", "sum"), q=("qty_face", "sum"))
    v = (c["pw"] / c["q"]).unstack("side")
    d["half_spread_frac"] = ((v.get("S") - v.get("B")) / 2).clip(lower=0).reindex(d.index) / d["clean_price"] if {"S", "B"} <= set(v.columns) else np.nan
    return d.reset_index().rename(columns={"trade_date": "price_date"})
