"""
Everything that talks to WRDS: the connection, the parquet cache in data/raw, and the queries for each source.

Each pull is cached as data/raw/<name>_<START_DATE>_<END_DATE>[_<cusip hash>].parquet, so WRDS is only hit the first
time. The cache name includes the date window and a hash of the CUSIPs the query was filtered on, so a changed window
or universe never silently reuses an old pull. TRACE is pulled in C.TRACE_BUCKETS fixed hash buckets of CUSIPs, so
adding a month of holdings only re-pulls the buckets whose bonds changed.

  pull_treasuries   CRSP daily Treasury quotes (all Treasuries: the curve needs them all)
  pull_fisd_terms   FISD bond terms for the corporates
  pull_trace        raw trade-level TRACE (corporate, MBS pool or TBA table), one cached frame per bucket
"""
import hashlib
import logging
import os

import pandas as pd
from dotenv import load_dotenv

from . import config as C

log = logging.getLogger(__name__)
_db = None

CRSP_QUERY = f"""
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
    from {C.CRSP_DAILY} as d
    join {C.CRSP_ISSUES} as i on d.kycrspid = i.kycrspid
    where d.caldt between %(start)s and %(end)s
"""

FISD_QUERY = f"""
    select complete_cusip as cusip, issue_id, maturity as maturity_date, coupon as coupon_rate, coupon_type,
           interest_frequency, dated_date, first_interest_date, last_interest_date, day_count_basis,
           offering_date, offering_amt as offering_amount, amount_outstanding, principal_amt as principal_amount,
           denomination, bond_type, security_level, redeemable as is_callable, convertible as is_convertible,
           rule_144a as is_144a
    from {C.FISD_ISSUES}
    where complete_cusip in %(cusips)s
"""

# raw TRACE column -> our name (trace.py renames with this). Columns a table doesn't have (e.g. yld_pt in the MBS
# table) are left out of that table's query and created as NaN in the cleaning.
TRACE_COLS = {
    "cusip_id": "cusip", "trd_exctn_dt": "date", "trd_exctn_tm": "time", "rptd_pr": "price",
    "entrd_vol_qt": "quantity", "yld_pt": "yield_pct", "rpt_side_cd": "side", "cntra_mp_id": "contra",
    "trc_st": "status", "msg_seq_nb": "msg_seq", "orig_msg_seq_nb": "orig_msg_seq", "asof_cd": "asof",
    "wis_fl": "when_issued", "lckd_in_ind": "locked_in", "sale_cndtn_cd": "sale_cond", "sale_cndtn2_cd": "sale_cond2",
    "days_to_sttl_ct": "settle_days", "stlmnt_dt": "settle_date", "factor_on_file": "pool_factor", "factor": "pool_factor_alt",
}


def connect():
    """Open the WRDS connection once (only when something actually has to be pulled). 
    WRDS_USERNAME comes from the repo .env; the password from ~/.pgpass.
    sometimes it sends a Duo push still so have ur phone ready 
    TODO: add some log that senses if the code is stalled waiting for the push """
    global _db
    if _db is None:
        import wrds
        load_dotenv(C.ROOT / ".env")
        _db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    return _db


def sql(query, params=None, date_cols=None):
    db = connect()
    db.connection.rollback()                 # clears an aborted transaction left by a failed query
    return db.raw_sql(query, params=params, date_cols=date_cols)


def cached(name, pull, refresh=False, cusips=None):
    """Return data/raw/<name>_<window>[_<cusip hash>].parquet, or run pull() and save it. refresh=True re-pulls.
    Pass the CUSIPs a query is filtered on, so a changed universe never silently reuses an old pull."""
    suffix = f"_{hashlib.md5(','.join(sorted(cusips)).encode()).hexdigest()[:10]}" if cusips is not None else ""
    path = C.RAW_DIR / f"{name}_{C.WINDOW_TAG}{suffix}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        log.info("%s: %d rows from cache", name, len(df))
        return df
    df = pull()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("%s: %d rows pulled from WRDS", name, len(df))
    return df


def buckets(cusips, n):
    """Split CUSIPs into n buckets by a stable hash of each CUSIP. Adding bonds (e.g. another month of holdings)
    only changes the buckets they land in, so the other buckets' cached pulls are reused."""
    out = [[] for _ in range(n)]
    for c in sorted(set(cusips)):
        out[int(hashlib.md5(c.encode()).hexdigest()[:8], 16) % n].append(c)
    return out


def pull_treasuries(refresh=False):
    """All CRSP daily Treasury quotes in [START_DATE, END_DATE]."""
    return cached("crsp_treasuries", lambda: sql(CRSP_QUERY, {"start": C.START_DATE, "end": C.END_DATE},
                                                 date_cols=["date", "dated_date", "maturity_date"]), refresh)


def pull_fisd_terms(cusips, refresh=False):
    return cached("fisd_corporate_terms",
                  lambda: sql(FISD_QUERY, {"cusips": tuple(cusips)},
                              date_cols=["maturity_date", "dated_date", "first_interest_date", "last_interest_date", "offering_date"]),
                  refresh, cusips)


def pull_trace(cusips, table, name, refresh=False):
    """Yield raw trade-level TRACE frames, one per CUSIP bucket, cached as
    data/raw/<name>/bucket_NNN_<window>_<cusip hash>.parquet. A bucket is re-pulled only when its bonds change."""
    query = None

    def pull(batch):
        nonlocal query
        if query is None:            # look up the table's columns only when something actually has to be pulled
            have = set(connect().get_table(library=table.split(".")[0], table=table.split(".")[1], obs=1).columns)
            cols = ", ".join(c for c in TRACE_COLS if c in have)
            query = f"select {cols} from {table} where cusip_id in %(cusips)s and trd_exctn_dt between %(start)s and %(end)s"
        return sql(query, {"cusips": tuple(batch), "start": C.START_DATE, "end": C.END_DATE})

    for i, batch in enumerate(buckets(cusips, C.TRACE_BUCKETS)):
        if not batch:
            continue
        yield cached(f"{name}/bucket_{i:03d}", lambda: pull(batch), refresh, batch)
