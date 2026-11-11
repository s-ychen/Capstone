"""Step 1: the ideal bond universe = iShares AGG holdings file, filtered to the sectors the model covers."""
import logging
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

# iShares column -> our column. Everything else in the file (ISIN, SEDOL, FX, "Real" fields, ...) is dropped.
COLS = {
    "CUSIP": "cusip", "Name": "name", "Sector": "agg_sector", "Asset Class": "asset_class",
    "Market Value": "agg_market_value", "Weight (%)": "agg_weight_pct", "Par Value": "agg_par_value",
    "Maturity": "agg_maturity_date", "Coupon (%)": "agg_coupon_pct", "Accrual Date": "agg_accrual_date",
    "Duration": "agg_duration", "Mod. Duration": "agg_mod_duration", "YTM (%)": "agg_ytm_pct",
}


def read_holdings(path: Path = C.AGG_HOLDINGS_CSV) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Parse the iShares CSV (preamble rows skipped automatically). Returns (holdings, holdings_as_of_date)."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = next(i for i, l in enumerate(lines) if l.startswith("Name,") and "CUSIP" in l)
    asof_line = next(l for l in lines[:header] if "Holdings as of" in l)
    asof = pd.to_datetime(asof_line.split(",", 1)[1].strip().strip('"'))

    df = pd.read_csv(path, skiprows=header, thousands=",", dtype={"CUSIP": "string"}, na_values=["-"])
    df = df.rename(columns=COLS)[list(COLS.values())]
    df["cusip"] = df["cusip"].str.strip().str.upper()
    for c in ["agg_maturity_date", "agg_accrual_date"]:
        df[c] = pd.to_datetime(df[c], format="%b %d, %Y", errors="coerce")
    return df, asof


def build_universe(path: Path = C.AGG_HOLDINGS_CSV) -> pd.DataFrame:
    """One row per CUSIP in the modelled sectors (C.SECTOR_MAP), with the AGG snapshot fields and the holdings date.

    Weights are kept as the raw AGG percentage of the whole fund; renormalisation to the bonds we actually
    have data for happens in merge.py once we know which bonds have prices.
    """
    df, asof = read_holdings(path)
    total_weight = df["agg_weight_pct"].sum()
    df = df[df["cusip"].str.fullmatch(r"[A-Z0-9]{9}", na=False)]              # drops cash / collateral rows
    df = df[df["agg_sector"].isin(C.SECTOR_MAP)].copy()
    df["sector"] = df["agg_sector"].map(C.SECTOR_MAP)
    df["asset_type"] = df["sector"].map(C.ASSET_TYPE)
    df["is_tba"] = (df["sector"] == "mbs_passthrough") & df["name"].str.contains("TBA", na=False)
    df["holdings_asof_date"] = asof
    dup = df["cusip"].duplicated(keep=False)
    if dup.any():
        log.warning("%d duplicate CUSIPs in holdings file; keeping the first", dup.sum() // 2)
        df = df.drop_duplicates("cusip")
    log.info("AGG holdings as of %s: fund weight %.1f%%, modelled sectors %.1f%% in %d bonds",
             asof.date(), total_weight, df["agg_weight_pct"].sum(), len(df))
    return df.reset_index(drop=True)
