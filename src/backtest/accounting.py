"""Realised returns and forced-sale costs between rebalances."""
import numpy as np
import pandas as pd

from src.params.frictions import COST_LOOKBACK_DAYS
from src.params.krd import cash_between


def realised_returns(P, d0, d1, data, table) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Realised one-month return of every bond in P from d0 to d1, the share that fell back to mu_hat, the mask of
    fallbacks and the d0 dirty price."""
    wide = data.panel[["dirty_price", "stale"]].unstack("month_end")
    p0 = wide["dirty_price"][d0].reindex(P.cusips).to_numpy(float)
    p1 = wide["dirty_price"][d1].reindex(P.cusips).to_numpy(float)
    st = (wide["stale"][d0].reindex(P.cusips) != False) | (wide["stale"][d1].reindex(P.cusips) != False)  # noqa: E712
    cash = pd.Series(cash_between(table, d0, d1), index=table.cusips).reindex(P.cusips).fillna(0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p1 + cash) / p0 - 1
    bad = st.to_numpy() | ~np.isfinite(r)
    r = np.where(bad, P.periods[0].mu_hat, r)
    return r, float(bad.mean()), bad, p0


def exit_cost(drop: dict, d, prices: pd.DataFrame, c_prev: dict) -> float:
    """Share of wealth lost selling `drop` (cusip -> weight) at d for half the bid-ask: each bond's last quote in the
    lookback window to d, else its cost at the previous rebalance."""
    if not drop:
        return 0.0
    d = pd.Timestamp(d)
    p = prices[(prices["date"] > d - pd.Timedelta(days=COST_LOOKBACK_DAYS)) & (prices["date"] <= d)
               & prices["cusip"].isin(list(drop))]
    half = (p.dropna(subset=["bid_ask_spread_bps"]).sort_values("date")
             .groupby("cusip")["bid_ask_spread_bps"].last() / 2 * 1e-4)
    return float(sum(w * half.get(c, c_prev.get(c, 0.0)) for c, w in drop.items()))
