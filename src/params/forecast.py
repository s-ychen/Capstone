"""Forecasts for the multi-period model (Deliverable 3, Section 2.3), all under today's constant curve.

Index entries are not forecast here: they need Treasury auction announcements, which the data pipeline does not hold.
"""
import numpy as np
import pandas as pd

from .krd import CashflowTable, cash_between, price

MIN_YEARS_TO_MATURITY = 1.0     # index rule: bonds under one year to maturity leave the index


def step_date(as_of, h: int) -> pd.Timestamp:
    """Calendar month-end h months after as_of's month. as_of is often a business month-end (e.g. 2018-03-29), and
    adding MonthEnd(h) to it directly lands on the same month's calendar end, a step of a day or two."""
    t = pd.Timestamp(as_of)
    return t + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(h) if h else t


def expected_returns(table: CashflowTable, start, params: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """One-month gross-minus-one return from `start` to the next month-end, off the same curve.

    r = (price at the next month-end + cash paid in the month) / price now - 1, with each bond's spread held.
    Reprices at the shorter maturity rather than using carry + roll-down to first order (D3 Section 2.3.2).
    NaN where the bond has no value today. A bond that matures in the month returns its redemption.
    """
    end = step_date(start, 1)
    p0 = price(table, start, params, spread)
    p1 = price(table, end, params, spread)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (p1 + cash_between(table, start, end)) / p0 - 1
    return np.where(p0 > 0, r, np.nan)


def eligibility_exits(maturity: np.ndarray, as_of) -> np.ndarray:
    """E_t: True for bonds with under one year to maturity at as_of (not allowed to be held)."""
    ttm = (maturity - np.datetime64(pd.Timestamp(as_of), "D")).astype("timedelta64[D]").astype(float) / 365.25
    return ttm < MIN_YEARS_TO_MATURITY


def forecast_benchmark(b: np.ndarray, r: np.ndarray, exits_next: np.ndarray) -> np.ndarray:
    """b_hat for the next date: drift b by (1 + r), zero the bonds that exit, renormalize to sum 1."""
    drifted = np.where(exits_next, 0.0, b * (1 + np.nan_to_num(r)))
    return drifted / drifted.sum()


def benchmark_return(b: np.ndarray, r: np.ndarray) -> float:
    """r^b = b'r, the benchmark's forecast return over the period."""
    return float(b @ np.nan_to_num(r))


def drift_portfolio(x: np.ndarray, r: np.ndarray, r_b: float) -> np.ndarray:
    """D3 eq. 7: x_tilde = diag(1 + r) x / (1 + r^b)."""
    return (1 + np.nan_to_num(r)) * x / (1 + r_b)
