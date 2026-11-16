"""Trading costs, minimum lots and the required return (the non-statistical inputs of the model)."""
import numpy as np
import pandas as pd

COST_LOOKBACK_DAYS = 90


def trading_cost(prices: pd.DataFrame, cusips: np.ndarray, sector: np.ndarray, as_of) -> np.ndarray:
    """Proportional one-way cost c_i = half the bid-ask spread, as a fraction of price.

    Uses each bond's last observed bid_ask_spread_bps in the 90 days to as_of. Bonds with none take their sector's
    median, then the overall median.
    """
    as_of = pd.Timestamp(as_of)
    p = prices[(prices["date"] > as_of - pd.Timedelta(days=COST_LOOKBACK_DAYS)) & (prices["date"] <= as_of)]
    last = (p.dropna(subset=["bid_ask_spread_bps"]).sort_values("date")
             .groupby("cusip")["bid_ask_spread_bps"].last())
    c = pd.Series(last.reindex(cusips).to_numpy(), index=pd.Index(sector)) / 2 * 1e-4
    sector_med = c.groupby(level=0).transform("median")
    c = c.fillna(sector_med).fillna(c.median())
    return np.nan_to_num(c.to_numpy())


def min_lot_weight(min_denomination: np.ndarray, dirty_price: np.ndarray, sector: np.ndarray, W: float) -> np.ndarray:
    """l_i = L_i / W: market value of the minimum face amount as a share of wealth W (same currency as face).

    Missing min_denomination takes the sector median, then the overall median. L_i = face * dirty / 100.
    """
    face = pd.Series(np.asarray(min_denomination, float), index=pd.Index(sector))
    face = face.fillna(face.groupby(level=0).transform("median")).fillna(face.median())
    return face.to_numpy() * np.asarray(dirty_price, float) / 100 / W


def terminal_goal(W0: float, g: float, T: float) -> float:
    """G = W0 (1 + g)^T."""
    return W0 * (1 + g) ** T


def required_return(G: float, W: float, t: int, N: int) -> float:
    """Per-period return needed to reach G from wealth W at rebalance t of N: (G / W)^(1 / (N - t)) - 1."""
    if t >= N:
        raise ValueError("t must be below N")
    return (G / W) ** (1 / (N - t)) - 1
