"""WRDS connection and a parquet cache so each raw pull hits WRDS once (data/raw/<name>.parquet)."""
import logging
import os
from collections.abc import Callable

import pandas as pd
from dotenv import load_dotenv

from . import config as C

log = logging.getLogger(__name__)
_db = None


def connect():
    """Lazy singleton WRDS connection; WRDS_USERNAME comes from the repo .env file."""
    global _db
    if _db is None:
        import wrds
        load_dotenv(C.ROOT / ".env")
        _db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    return _db


def sql(query: str, params: dict | None = None, date_cols: list[str] | None = None) -> pd.DataFrame:
    db = connect()
    db.connection.rollback()                 # clears an aborted transaction left by a failed query
    return db.raw_sql(query, params=params, date_cols=date_cols)


def cached(name: str, pull: Callable[[], pd.DataFrame], refresh: bool = False) -> pd.DataFrame:
    """Return data/raw/<name>_<window>.parquet, or run pull() and save it. refresh=True re-pulls."""
    path = C.RAW_DIR / f"{name}_{C.WINDOW_TAG}.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        log.info("%s: %d rows from cache", name, len(df))
        return df
    df = pull()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("%s: %d rows pulled from WRDS", name, len(df))
    return df


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]
