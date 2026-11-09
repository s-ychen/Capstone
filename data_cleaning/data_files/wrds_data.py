"""WRDS pulls. Each pull selects only the needed columns/dates, and is cached to data/raw/*.pkl."""
import pandas as pd

from . import config as C


def connect():
    import wrds
    return wrds.Connection(wrds_username=C.WRDS_USERNAME)


def cached(name, fn, use_cache):
    """Run fn() or reload data/raw/<name>.pkl when use_cache and the file exists."""
    path = C.DATA_DIR / "raw" / f"{name}.pkl"
    if use_cache and path.exists():
        return pd.read_pickle(path)
    df = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    print(f"  pulled {name}: {len(df):,} rows")
    return df


def _sql(db, query, params):
    db.connection.rollback()
    return db.raw_sql(query, params=params)


def _end():
    return C.END_DATE or str(pd.Timestamp.today().date())


def trading_calendar(db):
    """Treasury quote dates (the trading calendar) between START_DATE and END_DATE."""
    d = _sql(db, f"SELECT DISTINCT caldt FROM {C.CRSP_DLY} WHERE caldt BETWEEN %(s)s AND %(e)s", {"s": C.START_DATE, "e": _end()})
    return pd.DatetimeIndex(sorted(pd.to_datetime(d["caldt"])))


BONDRET_COLS = ["cusip", "date", "dated_date", "offering_date", "maturity", "coupon", "ncoups", "price_l5m", "t_yld_pt", "t_spread", "t_dvolume"]


def pull_bondret(db, cusips):
    """BondRet rows for the corporate CUSIPs (only the columns we use; names that do not exist are skipped)."""
    have = set(db.raw_sql(f"SELECT * FROM {C.BONDRET_TABLE} LIMIT 1").columns)
    cols = ", ".join(c for c in BONDRET_COLS if c in have)
    return _sql(db, f"SELECT {cols} FROM {C.BONDRET_TABLE} WHERE cusip IN %(c)s AND date BETWEEN %(s)s AND %(e)s", {"c": tuple(cusips), "s": C.START_DATE, "e": _end()})


def pull_crsp_issues(db):
    return _sql(db, f"SELECT kytreasno, tcusip, tdatdt, tmatdt, tcouprt, itype FROM {C.CRSP_ISS} WHERE tmatdt >= %(s)s", {"s": C.START_DATE})


def pull_crsp_quotes(db, dates=None):
    """Daily bid/ask quotes on `dates` only (None = every date since START_DATE)."""
    where, p = ("caldt IN %(d)s", {"d": tuple(dates)}) if dates is not None else ("caldt BETWEEN %(s)s AND %(e)s", {"s": C.START_DATE, "e": _end()})
    return _sql(db, f"SELECT kytreasno, caldt, tdbid, tdask, tdaccint FROM {C.CRSP_DLY} WHERE {where}", p)


def pull_mbs_master(db, cusips):
    return _sql(db, f"SELECT * FROM {C.MBS_MASTER} WHERE cusip_id IN %(c)s", {"c": tuple(cusips)})


def trace_chunks(db, cusips, dates, table=C.TRACE_TABLE, chunk=300):
    """Yield raw TRACE trades for small batches of `cusips`, only on `dates` (the month-end windows). Missing columns are skipped."""
    have = set(db.raw_sql(f"SELECT * FROM {table} LIMIT 1").columns)
    cols = ", ".join(c for c in C.TRACE_COLS.values() if c in have)
    for i in range(0, len(cusips), chunk):
        print(f"  TRACE: {min(i + chunk, len(cusips)):,}/{len(cusips):,} CUSIPs", flush=True)
        yield _sql(db, f"SELECT {cols} FROM {table} WHERE {C.TRACE_COLS['cusip']} IN %(c)s AND {C.TRACE_COLS['trade_date']} IN %(d)s",
                   {"c": tuple(cusips[i:i + chunk]), "d": tuple(dates)})
