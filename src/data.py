import logging
import os
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parent.parent
TREASURIES_DIR = ROOT / "data" / "raw" / "treasuries"

logger = logging.getLogger(__name__)
_HOLIDAYS = USFederalHolidayCalendar().holidays("2000-01-01", "2040-12-31")

TREASURIES_QUERY = """
    select i.tcusip         as cusip,
           d.caldt          as date,
           i.itype          as security_type,
           i.tcouprt        as coupon_rate,
           i.tdatdt         as dated_date,
           i.tmatdt         as maturity_date,
           d.tdbid          as bid_price,
           d.tdask          as ask_price,
           d.tdnomprc       as mid_price,
           d.tdnomprc_flg   as mid_price_flag,
           d.tdaccint       as accrued_interest,
           d.tdyld          as daily_yield,
           d.tdyld * 365    as annual_yield,
           d.tdduratn       as duration_days,
           d.tdduratn / 365 as duration_years,
           d.tdretnua       as daily_return,
           d.tdtotout       as total_outstanding,
           d.tdpubout       as public_outstanding
    from crsp.tfz_dly as d
    join crsp.tfz_iss as i
      on d.kycrspid = i.kycrspid
    where d.caldt between '{start}' and '{end}'
"""


def connect() -> wrds.Connection:
    load_dotenv(ROOT / ".env")
    return wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])


def treasury_path(date: pd.Timestamp) -> Path:
    return TREASURIES_DIR / f"crsp_treasuries_{date:%Y%m%d}.parquet"


def pull_treasuries(start: str, end: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return CRSP daily treasuries for [start, end], caching one file per day.

    CRSP only has rows on days the market traded, so the dates present in the
    result are the business days. Days already cached are read from disk. WRDS
    is only queried (for the span of the missing days) when a weekday that is
    not a US federal holiday has no cached file, or when force_refresh=True.
    Good Friday is not a federal holiday, so ranges containing one re-query.
    """
    TREASURIES_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start, end, freq="C", holidays=_HOLIDAYS)
    missing = [d for d in days if not treasury_path(d).exists()]

    if force_refresh or missing:
        lo, hi = (days.min(), days.max()) if force_refresh else (min(missing), max(missing))
        logger.info("querying WRDS %s to %s", lo.date(), hi.date())
        db = connect()
        try:
            df = db.raw_sql(
                TREASURIES_QUERY.format(start=f"{lo:%Y-%m-%d}", end=f"{hi:%Y-%m-%d}"),
                date_cols=["date", "dated_date", "maturity_date"],
            )
        finally:
            db.close()
        df["security_type"] = df["security_type"].astype("int")
        for date, day in df.groupby("date"):
            day.reset_index(drop=True).to_parquet(treasury_path(date))
            logger.info("%s: %d rows pulled", date.date(), len(day))

    frames = []
    for path in sorted(TREASURIES_DIR.glob("crsp_treasuries_*.parquet")):
        date = pd.Timestamp(path.stem.rsplit("_", 1)[1])
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            day = pd.read_parquet(path)
            logger.info("%s: %d rows loaded", date.date(), len(day))
            frames.append(day)
    return pd.concat(frames, ignore_index=True)
