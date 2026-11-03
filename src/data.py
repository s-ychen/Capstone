import os
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TREASURIES_DIR = ROOT / "data" / "raw" / "treasuries"

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


def pull_treasuries(start: str, end: str, force: bool = False) -> list[Path]:
    """Pull CRSP daily treasuries for [start, end] and cache one file per day.

    CRSP only has rows on days the market traded, so the dates present in the
    result are the business days. No holiday calendar needed. Days already
    cached are skipped unless force=True. Returns the paths for the range.
    """
    TREASURIES_DIR.mkdir(parents=True, exist_ok=True)

    db = connect()
    try:
        df = db.raw_sql(
            TREASURIES_QUERY.format(start=start, end=end),
            date_cols=["date", "dated_date", "maturity_date"],
        )
    finally:
        db.close()
    df["security_type"] = df["security_type"].astype("int")

    paths = []
    for date, day in df.groupby("date"):
        path = treasury_path(date)
        if force or not path.exists():
            day.reset_index(drop=True).to_parquet(path)
        paths.append(path)
    return paths


def load_treasuries(start: str, end: str) -> pd.DataFrame:
    """Load cached daily treasury files in [start, end]."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    files = [
        p
        for p in sorted(TREASURIES_DIR.glob("crsp_treasuries_*.parquet"))
        if lo <= pd.Timestamp(p.stem.rsplit("_", 1)[1]) <= hi
    ]
    return pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)
