"""End-to-end pipeline: run() returns (and saves) the three model tables plus supporting tables."""
import functools

import numpy as np
import pandas as pd

from . import config as C
from . import curve, features, prices, trace, wrds_data as W
from .finmath import cashflows, coupon_dates, solve_batch
from .universe import build_universe


def run(use_cache=False, download=True, auctions=True):
    """
    Build the model inputs. use_cache=True reloads data/raw/*.pkl instead of querying WRDS; download=False reuses the saved AGG CSV.
    Returns a dict: bond_features, prices, yield_curve (the 3 model tables) + key_rate_changes_bps, curve_params, universe,
    dropped_bonds, trace_log, auctions.
    """
    db = functools.cache(W.connect)                          # connects lazily, only if something must be pulled
    pull = lambda name, fn: W.cached(name, fn, use_cache)

    print("1) AGG universe")
    universe, asof = build_universe(download)
    universe["holdings_asof_date"] = asof
    by_sector = universe.groupby("model_sector")["cusip"].apply(list).to_dict()
    corp = sorted(c for s in C.CORP_SECTORS for c in by_sector.get(s, []))
    mbs = sorted(by_sector.get("mbs_passthrough", []))

    print("2) WRDS pulls")
    cal = pd.DatetimeIndex(pull("calendar", lambda: pd.DataFrame({"d": W.trading_calendar(db())}))["d"])
    window = prices.window_dates(cal, C.WINDOW_DAYS)
    bondret = pull("bondret", lambda: W.pull_bondret(db(), corp))
    crsp_iss = pull("crsp_issues", lambda: W.pull_crsp_issues(db()))
    crsp_dly = pull("crsp_quotes", lambda: W.pull_crsp_quotes(db(), window if C.CURVE_FREQ == "month_end" else None))
    mbs_master = pull("mbs_master", lambda: W.pull_mbs_master(db(), mbs))
    trace_only = sorted(set(corp) - set(bondret["cusip"]))        # corporates BondRet does not cover
    pf_path = C.DATA_DIR / "raw" / "mbs_pool_factors.csv"          # optional: columns cusip, factor_date, pool_factor
    pool = pd.read_csv(pf_path) if pf_path.exists() else None
    trace_dates, logs = prices.window_dates(cal, C.WINDOW_DAYS + C.TRACE_MARGIN_DAYS), []

    def trace_daily(cusips, is_mbs):
        """Pull, clean and collapse TRACE in small CUSIP batches so raw trades never pile up in memory."""
        daily = []
        for raw in W.trace_chunks(db(), cusips, trace_dates, C.TRACE_MBS_TABLE if is_mbs else C.TRACE_TABLE):
            t, log = trace.clean_trades(raw, is_mbs, pool if is_mbs else None)
            logs.append(log.assign(group="mbs" if is_mbs else "corporates"))
            if len(t):
                daily.append(trace.daily_prices(t))
        return pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=["cusip", "price_date", "clean_price"])

    print("3) Treasuries and zero curve")
    quotes = curve.clean_crsp(crsp_iss, crsp_dly).merge(universe.loc[universe["model_sector"] == "treasury", ["cusip", "cusip8"]], on="cusip8", how="left")
    params = curve.fit_curves(quotes)
    yield_curve = curve.key_rate_table(params)
    me = yield_curve.groupby(yield_curve.index.to_period("M")).tail(1)
    me = me[me.index.to_period("M").isin(prices.complete_months(cal).to_period("M"))]
    changes = (me[list(curve.KEY_COLS.values())].diff() * 100).dropna()
    changes.columns = [f"d_{k.lower()}_bps" for k in C.KEY_RATES]

    print("4) TRACE (pull + clean in batches)")
    d_corp = pull("trace_daily_corp", lambda: trace_daily(trace_only, False)).assign(source="trace_corp")
    d_mbs = pull("trace_daily_mbs", lambda: trace_daily(mbs, True)).assign(source="trace_mbs")
    trace_daily_df = pd.concat([d_corp, d_mbs], ignore_index=True)
    trace_log = pd.concat(logs).groupby(["group", "step"], sort=False)["trades_left"].sum().reset_index() if logs else pd.DataFrame()

    print("5) Terms, month-end prices")
    terms = features.build_terms(universe, bondret, quotes, features.clean_mbs_master(mbs_master, asof))
    tsy_daily = quotes.dropna(subset=["cusip"]).rename(columns={"quote_date": "price_date", "mid_clean_price": "clean_price",
                                                                "accrued_interest": "accrued_interest_crsp"})
    tsy_daily = tsy_daily[["cusip", "price_date", "clean_price", "accrued_interest_crsp", "half_spread_frac"]]
    price_table = prices.build_prices(tsy_daily, bondret, trace_daily_df, terms, cal, params)

    print("6) Merge and renormalise")
    bond_features, dropped = features.build_features(universe, terms, price_table)
    price_table = price_table[price_table["cusip"].isin(bond_features["cusip"])].reset_index(drop=True)
    out = {"bond_features": bond_features, "prices": price_table, "yield_curve": yield_curve, "key_rate_changes_bps": changes,
           "curve_params": params, "universe": universe, "dropped_bonds": dropped, "trace_log": trace_log,
           "auctions": features.fetch_treasury_auctions() if auctions else pd.DataFrame()}
    save(out)
    return out


def save(tables):
    """Write each table to data/processed/<name>.csv."""
    folder = C.DATA_DIR / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(folder / f"{name}.csv", index=name in ("yield_curve", "key_rate_changes_bps", "curve_params"))


def checks(t):
    """Print PASS/FAIL sanity checks, the solved-vs-AGG yield gap on the holdings date, and a warning if prices lag the holdings."""
    f, p, yc = t["bond_features"], t["prices"], t["yield_curve"]
    tests = {"weights sum to 1": np.isclose(f["agg_weight"].sum(), 1), "unique CUSIPs": f["cusip"].is_unique,
             "one row per (cusip, month-end)": not p.duplicated(["cusip", "month_end_date"]).any(),
             "no missing clean prices": p["clean_price"].notna().all(), "dirty >= clean": (p["dirty_price"] >= p["clean_price"] - 1e-9).all(),
             "all 8 key rates present": yc[list(curve.KEY_COLS.values())].notna().all().all()}
    for k, ok in tests.items():
        print("PASS " if ok else "FAIL ", k)
    asof, last_me = t["universe"]["holdings_asof_date"].iloc[0], p["month_end_date"].max()
    print(f"\nSolved yield minus AGG-file YTM (bps), both at the holdings date {asof.date()} (tests the yield maths, not the data):")
    print(agg_yield_gap(t["universe"]).groupby("model_sector")["gap_bps"].describe()[["count", "mean", "50%"]].round(1))
    end = pd.Timestamp(C.END_DATE or pd.Timestamp.today()) + pd.offsets.MonthEnd(0)
    print(f"\nSample: {p['month_end_date'].min().date()} to {last_me.date()} (END_DATE {C.END_DATE}); universe = AGG holdings of {asof.date()}")
    print(f"{'PASS ' if last_me >= end else 'FAIL '} prices reach the END_DATE month-end ({end.date()})")
    print("Last priced month-end by source:", p.groupby("source")["month_end_date"].max().dt.date.to_dict())
    print("\nStale share by source:\n", p.groupby("source")["is_stale"].mean().round(3))


def agg_yield_gap(universe):
    """Re-solve each bond's yield from the iShares price (market value / par, which includes accrued) on the holdings date and compare
    to the iShares YTM. Callables differ (iShares uses yield-to-worst logic; we use bullet cash flows), so look at the median."""
    u = universe.dropna(subset=["agg_maturity_date", "agg_coupon_pct", "agg_ytm_pct"])
    u = u[(u["agg_par_value"] > 0) & (u["model_sector"] != "mbs_passthrough")]     # MBS YTM depends on iShares' prepayment model
    settle = u["holdings_asof_date"].iloc[0] + pd.offsets.BDay(1)
    times, amts = [], []
    for _, b in u.iterrows():
        f = C.COUPON_FREQ[b["model_sector"]]
        tt, aa = cashflows(settle, coupon_dates(b["agg_maturity_date"], f, settle - pd.DateOffset(years=1)), b["agg_coupon_pct"], f)
        times.append(tt), amts.append(aa)
    solved = solve_batch((u["agg_market_value"] / u["agg_par_value"] * 100).values, times, amts, "ytm")
    return pd.DataFrame({"model_sector": u["model_sector"].values, "gap_bps": (solved - u["agg_ytm_pct"].values) * 100})
