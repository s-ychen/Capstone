"""Run the whole pipeline:  uv run python -m data_cleaning.run_pipeline [--download] [--refresh] [--skip-mbs] [--skip-corporates]

--download fetches the latest holdings CSV from iShares into data/raw and uses it instead of data/AGG_holdings_clean.csv.

Outputs (data/processed/*.parquet): bond_universe, prices_daily, treasury_daily, corporate_daily, mbs_daily,
yield_curve_daily, plus cleaning logs. See README.md in this folder for the table definitions.
"""
import argparse
import logging
import time

import pandas as pd

from . import config as C
from .corporates import build_corporate_daily
from .curve import fit_curves
from .features import solve_yield_and_spread
from .mbs import build_mbs_daily
from .merge import build_bond_universe, build_prices_daily, save
from .treasuries import clean_treasuries, pull_treasuries
from .universe import build_universe, download_holdings

log = logging.getLogger("data_cleaning")


def run(refresh: bool = False, corporates: bool = True, mbs: bool = True, download: bool = False) -> dict[str, pd.DataFrame]:
    t0 = time.time()
    log.info("1) AGG universe")
    universe = build_universe(download_holdings() if download else C.AGG_HOLDINGS_CSV)
    tsy_cusips = set(universe.loc[universe["asset_type"] == "treasury", "cusip"])

    log.info("2) CRSP Treasuries")
    tsy_all = clean_treasuries(pull_treasuries(refresh))            # every nominal Treasury: curve input
    log.info("3) zero curve")
    curve = fit_curves(tsy_all)
    tsy = tsy_all[tsy_all["cusip"].isin(tsy_cusips)].copy()           # AGG's Treasuries only
    tsy_terms = tsy.drop_duplicates("cusip")[["cusip", "maturity_date", "coupon_rate"]].assign(coupon_frequency=2)
    tsy = solve_yield_and_spread(tsy, tsy_terms, curve)
    tsy["bid_ask_spread_bps"] = tsy["bid_ask_spread"] / tsy["clean_price"] * 1e4
    tsy["source"] = "crsp"
    log.info("treasury daily: %d rows, %d of %d AGG Treasuries found in CRSP", len(tsy), tsy["cusip"].nunique(), len(tsy_cusips))

    empty = pd.DataFrame(columns=["cusip", "date"])
    corp, corp_terms, corp_log = (empty, pd.DataFrame(columns=["cusip"]), pd.DataFrame())
    if corporates:
        log.info("4) corporates (TRACE enhanced + FISD + Dickerson)")
        corp, corp_terms, corp_log = build_corporate_daily(universe, refresh)
        corp = solve_yield_and_spread(corp, corp_terms, curve)
    mbs_daily, mbs_log = empty, pd.DataFrame()
    if mbs:
        log.info("5) MBS pass-throughs (TRACE SPDS enhanced)")
        mbs_daily, mbs_log = build_mbs_daily(universe, refresh)
        mbs_daily["years_to_maturity"] = (mbs_daily["maturity_date"] - mbs_daily["date"]).dt.days / 365.25

    log.info("6) merge")
    prices = build_prices_daily(tsy, corp, mbs_daily, universe)
    bond_universe = build_bond_universe(universe, corp_terms, tsy_all, prices)
    prices = prices[prices["cusip"].isin(bond_universe.loc[bond_universe["has_prices"], "cusip"])].reset_index(drop=True)
    tables = {"bond_universe": bond_universe, "prices_daily": prices, "treasury_daily": tsy,
              "corporate_daily": corp, "mbs_daily": mbs_daily, "yield_curve_daily": curve,
              "trace_cleaning_log": pd.concat([corp_log.assign(group="corporate"), mbs_log], ignore_index=True)}
    save(tables)
    log.info("done in %.0f s", time.time() - t0)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download", action="store_true", help="fetch the latest holdings CSV from iShares instead of the committed snapshot")
    ap.add_argument("--refresh", action="store_true", help="re-pull everything from WRDS instead of using data/raw caches")
    ap.add_argument("--skip-corporates", action="store_true")
    ap.add_argument("--skip-mbs", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    run(refresh=a.refresh, corporates=not a.skip_corporates, mbs=not a.skip_mbs, download=a.download)
