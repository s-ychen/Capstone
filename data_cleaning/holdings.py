"""
Get AGG's holdings as of the last trading day of the previous month (to prevent forward leakage)
The download link is formatted as https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/api/v1/
get-fund-document?appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares&locale=en_US&portfolioId=239458&userType=individual&asOfDate=
{date}&component=holdings

outputted dfs:
membership  one row per (month, cusip): the bonds held at the start of `month`, with AGG market value, par, weight
              and holdings_date (the file's date, the last trading day of the previous month)
universe    one row per cusip held in any month of the window (fields from the latest file it appears in);
              this is the set of bonds pulled from WRDS
"""
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import config as C

log = logging.getLogger(__name__)
 # just move backwards from last day of month (can make a trading day schedule but i dont really want to)
MAX_STEP_BACK_DAYS = 7     
# iShares column -> our column. Everything else in the file (ISIN, SEDOL, FX, "Real" fields, ...) is dropped.
COLS = {
    "CUSIP": "cusip", "Name": "name", "Sector": "agg_sector", "Asset Class": "asset_class",
    "Market Value": "agg_market_value", "Weight (%)": "agg_weight_pct", "Par Value": "agg_par_value",
    "Maturity": "agg_maturity_date", "Coupon (%)": "agg_coupon_pct", "Accrual Date": "agg_accrual_date",
    "Duration": "agg_duration", "Mod. Duration": "agg_mod_duration", "YTM (%)": "agg_ytm_pct",
}


def read_holdings(path):
    """Parse one iShares holdings CSV (the preamble rows above the header are skipped).
    Returns (holdings df, the "Holdings as of" date in the file)."""
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


def modelled_holdings(df, asof):
    """Keep the bonds in the sectors we model (C.SECTOR_MAP) from one parsed holdings file.
    agg_weight = market value / market value of all the fund's bonds that day (cash and derivatives excluded),
    so it sums to ~0.95 over the modelled sectors. Duplicate CUSIP rows are summed.
    """
    bonds = df[df["cusip"].str.fullmatch(r"[A-Z0-9]{9}", na=False) & (df["asset_class"] == "Fixed Income")]
    total_mv = bonds["agg_market_value"].sum()
    out = bonds[bonds["agg_sector"].isin(C.SECTOR_MAP)].copy()
    if out["cusip"].duplicated().any():
        log.info("%s: %d duplicate CUSIP rows summed", asof.date(), out["cusip"].duplicated().sum())
        sums = out.groupby("cusip")[["agg_market_value", "agg_par_value", "agg_weight_pct"]].sum()
        out = out.drop_duplicates("cusip").set_index("cusip")
        out[sums.columns] = sums
        out = out.reset_index()
    out["agg_weight"] = out["agg_market_value"] / total_mv
    out["sector"] = out["agg_sector"].map(C.SECTOR_MAP)
    out["asset_type"] = out["sector"].map(C.ASSET_TYPE)
    out["is_tba"] = (out["sector"] == "mbs_passthrough") & out["name"].str.contains("TBA", na=False)
    out["holdings_asof_date"] = asof
    return out.reset_index(drop=True)


def download_month_end(month, session=None):
    """Path of the cached holdings file for the last trading day of `month`, downloading it if needed. None if the
    API returns nothing for the final MAX_STEP_BACK_DAYS days of the month (e.g. the month has not ended yet)."""
    # store the raw AGG holding files in this directory, so we don't have to start from scratch everytime
    C.HOLDINGS_DIR.mkdir(parents=True, exist_ok=True)
    cached = sorted(C.HOLDINGS_DIR.glob(f"AGG_holdings_{month.strftime('%Y%m')}*.csv"))
    if cached:
        return cached[-1]
    # from a session, download the AGG holdings file from the last trading day/up to 7 days from last day.
    s = session or requests.Session()
    day = month.to_timestamp(how="end").normalize()
    for _ in range(MAX_STEP_BACK_DAYS):
        if day.weekday() < 5 and day <= pd.Timestamp.today().normalize():
            for i in range(4):
                try:
                    r = s.get(C.AGG_HOLDINGS_API.format(date=day), timeout=60)
                    r.raise_for_status()
                    break
                except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                    if i == 3: # timed out/couldnt find
                        raise
                    log.warning("holdings download failed (%s), retrying in %ds", type(e).__name__, 10 * (i + 1))
                    time.sleep(10 * (i + 1))
            if b"Holdings as of" in r.content[:300]:
                path = C.HOLDINGS_DIR / f"AGG_holdings_{day:%Y%m%d}.csv"
                path.write_bytes(r.content)
                log.info("downloaded AGG holdings as of %s", day.date())
                time.sleep(1)
                return path
        day -= pd.Timedelta(days=1)
    log.warning("no AGG holdings file found for %s", month)
    return None


def build_holdings_history():
    """Download the previous month's ending AGG portfolio (as of last trading day).
      Then we say these are the holdings we can invest in for the current period."""
    # Get the holdings for the beginning of each month, from our start date of our data pull to the end date.
    session = requests.Session()
    frames = []
    holdings_months = pd.period_range(pd.Period(C.START_DATE, "M"), pd.Period(C.END_DATE, "M"), freq="M")
    for month in holdings_months:
        path = download_month_end(month - 1, session)          # start-of-month portfolio = previous month-end file
        if path is None:
            continue
        df, asof = read_holdings(path)
        if asof.to_period("M") != month - 1:
            raise ValueError(f"{path.name} is dated {asof.date()}, expected a date in {month - 1}")
        frames.append(modelled_holdings(df, asof).assign(month=month))
    h = pd.concat(frames, ignore_index=True)

    membership = h[["month", "holdings_asof_date", "cusip", "sector", "asset_type", "is_tba",
                    "agg_market_value", "agg_par_value", "agg_weight_pct", "agg_weight"]].rename(
        columns={"holdings_asof_date": "holdings_date"})
    universe = h.sort_values("holdings_asof_date").drop_duplicates("cusip", keep="last").drop(columns="month")
    held = membership.groupby("cusip")["month"].agg(first_held_month="min", last_held_month="max", months_held="size")
    universe = universe.merge(held.reset_index(), on="cusip").reset_index(drop=True)

    by_month = membership.groupby("month").agg(bonds=("cusip", "size"), modelled_weight=("agg_weight", "sum"))
    log.info("AGG holdings: %d months %s to %s, %d-%d modelled bonds per month (%.1f%%-%.1f%% of bond value), "
             "%d distinct bonds", len(by_month), by_month.index.min(), by_month.index.max(), by_month["bonds"].min(),
             by_month["bonds"].max(), 100 * by_month["modelled_weight"].min(), 100 * by_month["modelled_weight"].max(),
             len(universe))
    return universe, membership


def restrict_to_output(universe, membership):
    """
    restrict to just the months we care about i.e start date to present
    """
    m = membership[membership["month"] >= pd.Period(C.OUTPUT_START, "M")].reset_index(drop=True)
    held = m.groupby("cusip")["month"].agg(first_held_month="min", last_held_month="max", months_held="size")
    u = (universe.drop(columns=["first_held_month", "last_held_month", "months_held"])
         .merge(held.reset_index(), on="cusip").reset_index(drop=True))
    log.info("output window: %d months from %s, %d bonds", m["month"].nunique(), m["month"].min(), len(u))
    return u, m
