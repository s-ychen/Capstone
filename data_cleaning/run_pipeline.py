"""
Main file to run the data collection, processing, engineering pipeline.

Outputs are parquet files for the agg_holdings, bond_universe, prices_monthly, prices_daily and yield_curve_daily, from config.OUTPUT_START on. Raw data is pulled for [START_DATE, END_DATE] (cached in data/raw).

run with uv run python -m data_cleaning.run_pipeline with params --refresh, --skip-mbs, --skip-corporates (to make run faster)

  holdings.py        AGG's start-of-month holdings for every month (the bond universe)
  pull.py            WRDS connection, cache and queries
  treasury_curve.py  clean CRSP Treasury quotes, fit the daily zero curve
  trace.py           TRACE trades -> clean daily prices (corporates, MBS pools, TBAs), FISD terms
  panels.py          daily prices -> month-end panel and daily rows, in held months only
  bond_math.py       coupon schedules, accrued interest, yield and spread solvers
  merge.py           stack the panels into the output tables

"""
import argparse
import logging

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from . import config as C
from .holdings import build_holdings_history, restrict_to_output
from .merge import build_bond_universe, build_holdings_monthly, build_prices_daily, build_prices_monthly
from .panels import build_mbs_monthly, carry_panel, held_days, value, value_daily
from .pull import pull_treasuries
from .trace import build_corporate_daily, build_mbs_daily
from .treasury_curve import clean_treasuries, fit_curves

log = logging.getLogger("data_cleaning") #make a log for data cleaning, for progress checks at each step of the pipeline


def run(refresh=False, corporates=True, mbs=True):
    '''
    refresh enables if we are using cached data or re-pulling from wrds
    corporates enables if we are running to pull and process corporates
    mbs enbales if we are runnign to pull and process mbs pass through bonds
    '''
    # TREASURIES
    # holdings.py 
    log.info("1) Get AGG holdings history as of the prev month's last trading day")
    universe, membership = build_holdings_history()
    out_universe, membership = restrict_to_output(universe, membership)
    by_type = {t: membership[membership["asset_type"] == t] for t in ["treasury", "corporate", "mbs"]}
    # treasury_curve.py
    log.info("2) CRSP Treasuries")
    tsy_all = clean_treasuries(pull_treasuries(refresh))            # every nominal Treasury: curve input
    log.info("3) daily zero curve")
    curve = fit_curves(tsy_all)

    # make the trading calendar from the curve's dates, extended to the present/end date, without holidays
    crsp = pd.DatetimeIndex(curve["date"]).sort_values().unique()
    calendar = crsp.union(pd.bdate_range(crsp.max() + pd.Timedelta(days=1), C.END_DATE, freq="C",holidays=USFederalHolidayCalendar().holidays(crsp.max(), C.END_DATE)))

    # make a set of the treasury cusips from our ideal universe, and then filter our actual list of treasuries to 
    # this set + make cols for coupon freq (treasury notes and bonds are 2x a year)
    tsy_cusips = set(universe.loc[universe["asset_type"] == "treasury", "cusip"])
    tsy = tsy_all[tsy_all["cusip"].isin(tsy_cusips)]
    tsy_terms = tsy.drop_duplicates("cusip")[["cusip", "maturity_date", "coupon_rate"]].assign(coupon_frequency=2, day_count=C.DAY_COUNT["treasury"])

    #panels.py
    # so under the key treasury, we have for each bond, all the dates, and the data for it (solved if necessary in bond_math.py if the data source doesnt say)
    panels = {"treasury": value(carry_panel(tsy, by_type["treasury"], calendar), tsy_terms, curve)}
    panels["treasury"]["source"] = "crsp"
    daily = {"treasury": value_daily(held_days(tsy, by_type["treasury"], calendar), tsy_terms, curve).assign(source="crsp")}
    log.info("treasuries: %d of %d held by AGG in the window found in CRSP", tsy["cusip"].nunique(), len(tsy_cusips))
    

    #CORPS
    corp_terms = pd.DataFrame(columns=["cusip"])
    # For the simpler models, look at only the treasuries, but we add in corps and mbs if wanted
    if corporates:
        log.info("4) corporates (TRACE enhanced + FISD)")
        # Since we use TRACE data, we need to process from trade level data ...
        corp_daily, corp_terms = build_corporate_daily(universe, refresh)
        panels["corporate"] = value(carry_panel(corp_daily, by_type["corporate"], calendar), corp_terms, curve)
        panels["corporate"]["source"] = "trace_enhanced"
        daily["corporate"] = value_daily(held_days(corp_daily, by_type["corporate"], calendar), corp_terms, curve)
        del corp_daily

    #MBS
    if mbs:
        log.info("5) MBS pass-throughs (TRACE SPDS enhanced), month-end panel with cohort fill")
        mbs_daily = build_mbs_daily(universe, refresh)
        mbs_terms = universe.assign(coupon_rate=universe["agg_coupon_pct"], maturity_date=universe["agg_maturity_date"])
        panels["mbs"] = build_mbs_monthly(mbs_daily, mbs_terms, calendar, by_type["mbs"])
        daily["mbs"] = held_days(mbs_daily, by_type["mbs"], calendar)
        del mbs_daily

    log.info("6) merge all together into dfs for monthly holdings, the bond universe constructe")
    prices = build_prices_monthly(panels, out_universe)
    del panels
    prices_daily = build_prices_daily(daily, out_universe)
    del daily
    bond_universe = build_bond_universe(out_universe, corp_terms, tsy_all, prices)
    holdings_monthly = build_holdings_monthly(membership, prices)
    tables = {"agg_holdings_monthly": holdings_monthly, "bond_universe": bond_universe, "prices_monthly": prices,
              "prices_daily": prices_daily,
              "yield_curve_daily": curve[curve["date"] >= C.OUTPUT_START].reset_index(drop=True)}

    # save every table to data/processed/<name>.parquet
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = C.PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote %s: %d rows x %d cols", path.name, len(df), df.shape[1])
    log.info("done")
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-pull everything from WRDS instead of using data/raw caches")
    ap.add_argument("--skip-corporates", action="store_true")
    ap.add_argument("--skip-mbs", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    run(refresh=a.refresh, corporates=not a.skip_corporates, mbs=not a.skip_mbs)
