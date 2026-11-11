"""Step 2: CRSP daily Treasury quotes -> clean daily prices for all Treasuries (curve input) and for AGG's Treasuries."""
import logging

import numpy as np
import pandas as pd

from . import config as C
from .finmath import to_cusip9
from .wrds_pull import cached, sql

log = logging.getLogger(__name__)

# CRSP itype: 1 noncallable bond, 2 noncallable note, 4 bill, 5/6 callable bond/note, 11/12 TIPS note/bond.
FIXED_COUPON_TYPES = (1, 2, 4)

QUERY = """
    select i.tcusip        as cusip8,
           d.caldt         as date,
           i.itype         as security_type,
           i.tcouprt       as coupon_rate,
           i.tdatdt        as dated_date,
           i.tmatdt        as maturity_date,
           d.tdbid         as bid_price,
           d.tdask         as ask_price,
           d.tdnomprc      as mid_price,
           d.tdnomprc_flg  as price_flag,
           d.tdaccint      as accrued_interest,
           d.tdyld         as daily_yield,
           d.tdduratn      as duration_days,
           d.tdretnua      as daily_return,
           d.tdtotout      as total_outstanding,
           d.tdpubout      as public_outstanding
    from crsp.tfz_dly as d
    join crsp.tfz_iss as i on d.kycrspid = i.kycrspid
    where d.caldt between %(start)s and %(end)s
"""


def pull_treasuries(refresh: bool = False) -> pd.DataFrame:
    """All CRSP daily Treasury quotes in [START_DATE, END_DATE] (cached as data/raw/crsp_treasuries.parquet)."""
    return cached("crsp_treasuries", lambda: sql(QUERY, {"start": C.START_DATE, "end": C.END_DATE},
                                                 date_cols=["date", "dated_date", "maturity_date"]), refresh)


def clean_treasuries(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per Treasury per quote date with the standard price columns.

    - TIPS and callables are dropped (the curve and the model use fixed-coupon nominals only)
    - mid = (bid + ask)/2 is the clean price; dirty = clean + CRSP accrued interest
    - bid_ask_spread is in price points per 100 face; yields are annualised from CRSP's daily yield
    """
    df = raw.copy()
    df["security_type"] = df["security_type"].astype(int)
    df = df[df["security_type"].isin(FIXED_COUPON_TYPES)]
    df = df[(df["bid_price"] > 0) & (df["ask_price"] >= df["bid_price"])]
    df = df[df["maturity_date"] > df["date"]]
    df["cusip"] = df["cusip8"].map(to_cusip9)
    df = df.dropna(subset=["cusip"])
    num = ["coupon_rate", "bid_price", "ask_price", "mid_price", "accrued_interest", "daily_yield",
           "duration_days", "daily_return", "total_outstanding", "public_outstanding"]
    df[num] = df[num].astype(float)
    df["clean_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["accrued_interest"] = df["accrued_interest"].fillna(0.0)
    df["dirty_price"] = df["clean_price"] + df["accrued_interest"]
    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    df["yield"] = df["daily_yield"] * 365                     # CRSP tdyld is a daily rate
    df["duration"] = df["duration_days"] / 365
    df["years_to_maturity"] = (df["maturity_date"] - df["date"]).dt.days / 365.25
    df["is_zero_coupon"] = df["security_type"] == 4
    cols = ["cusip", "date", "security_type", "is_zero_coupon", "coupon_rate", "dated_date", "maturity_date",
            "years_to_maturity", "bid_price", "ask_price", "clean_price", "accrued_interest", "dirty_price",
            "bid_ask_spread", "price_flag", "yield", "duration", "daily_return", "total_outstanding", "public_outstanding"]
    out = df[cols].sort_values(["cusip", "date"]).reset_index(drop=True)
    dup = out.duplicated(["cusip", "date"])
    if dup.any():
        log.warning("%d duplicate (cusip, date) Treasury rows dropped", dup.sum())
        out = out[~dup]
    return out
