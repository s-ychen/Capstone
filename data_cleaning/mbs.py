"""Step 4 (MBS pass-throughs): enhanced TRACE SPDS pool trades (+ TBA trades) -> clean daily prices in current face."""
import logging

import numpy as np
import pandas as pd

from . import config as C
from . import trace_clean
from .corporates import pull_trace
from .finmath import mbs_accrued_interest

log = logging.getLogger(__name__)


def build_mbs_daily(universe: pd.DataFrame, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (mbs_daily, cleaning_log). Pools come from the SPDS MBS table (quantities scaled to current face by the
    pool factor on file); AGG's TBA positions come from the TBA table (face already current, no factor)."""
    u = universe[universe["asset_type"] == "mbs"]
    groups = [(u.loc[~u["is_tba"], "cusip"].tolist(), C.TRACE_MBS, "trace_mbs", "mbs_pool"),
              (u.loc[u["is_tba"], "cusip"].tolist(), C.TRACE_TBA, "trace_tba", "tba")]
    daily, logs = [], []
    for cusips, table, name, product in groups:
        if not cusips:
            continue
        for raw in pull_trace(cusips, table, name, refresh):
            trades, step_log = trace_clean.clean_trades(raw, product=product)
            logs.append(step_log.assign(group=name))
            if len(trades):
                daily.append(trace_clean.daily_prices(trades).assign(source=name))
    d = pd.concat(daily, ignore_index=True).sort_values(["cusip", "date"]).reset_index(drop=True)
    clean_log = pd.concat(logs).groupby(["group", "step"], sort=False)["trades_left"].sum().reset_index()

    # terms come from the AGG file (coupon, stated maturity); accrued is 30/360 from the 1st of the month
    terms = u.set_index("cusip")
    d["coupon_rate"] = d["cusip"].map(terms["agg_coupon_pct"])
    d["maturity_date"] = d["cusip"].map(terms["agg_maturity_date"])
    d["is_tba"] = d["cusip"].map(terms["is_tba"])
    d["accrued_interest"] = mbs_accrued_interest(d["date"], d["coupon_rate"].to_numpy())
    d["dirty_price"] = d["clean_price"] + d["accrued_interest"]
    d["bid_ask_spread_bps"] = d["bid_ask_spread"] / d["clean_price"] * 1e4
    if "pool_factor" not in d:
        d["pool_factor"] = np.nan


    # lalala manual part 
    
    log.info("mbs daily: %d rows, %d pools, %d TBAs, %s to %s", len(d), d.loc[~d["is_tba"], "cusip"].nunique(),
             d.loc[d["is_tba"], "cusip"].nunique(), d["date"].min().date(), d["date"].max().date())
    return d, clean_log
