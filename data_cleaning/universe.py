"""Step 1: the ideal bond universe = iShares AGG holdings file, filtered to the sectors the model covers."""
import logging
import re
from pathlib import Path
from urllib.parse import urljoin

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


def download_holdings(dest_dir: Path = C.RAW_DIR) -> Path:
    """Download the latest AGG holdings CSV from the iShares fund page into dest_dir. Returns the saved path,
    named by the holdings date in the file (e.g. ishares_AGG_holdings_20261112.csv)."""
    import requests
    from bs4 import BeautifulSoup
    with requests.Session() as s:
        s.headers.update(C.HTTP_HEADERS)
        page = s.get(C.AGG_URL, timeout=30)
        page.raise_for_status()
        links = BeautifulSoup(page.text, "html.parser").find_all("a", href=True)
        pattern = re.compile(C.AGG_LINK_TEXT, re.I)
        href = next((a["href"] for a in links if pattern.search(a.get_text(" ", strip=True))), None) \
            or next(a["href"] for a in links if "holdings" in a["href"].lower() and "csv" in a["href"].lower())
        resp = s.get(urljoin(page.url, href), headers={"Referer": page.url}, timeout=60)
        resp.raise_for_status()
    if "html" in resp.headers.get("Content-Type", "").lower():
        raise ValueError("iShares returned HTML instead of the CSV (blocked or page layout changed)")
    tmp = Path(dest_dir) / "ishares_AGG_holdings_latest.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(resp.content)
    _, asof = read_holdings(tmp)
    path = tmp.with_name(f"ishares_AGG_holdings_{asof:%Y%m%d}.csv")
    tmp.replace(path)
    log.info("downloaded iShares AGG holdings as of %s -> %s", asof.date(), path)
    return path


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
