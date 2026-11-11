"""Bond repricing off the fitted Treasury curve: key-rate durations, duration, spread duration, convexity.

A bond is priced as the PV of its remaining cash flows at exp(-(z(t) + s + bump(t)) t), where z is the fitted NSS
zero curve and s the bond's constant spread over it (Deliverable 3, Section 2.1). Callables are treated as bullets.
All sensitivities are per unit (decimal) yield change, in years.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_cleaning import config as C
from data_cleaning.curve import NSS_PARAMS, nss
from data_cleaning.features import _bisect
from data_cleaning.finmath import DAYS_PER_YEAR, coupon_dates

KEY_TENORS = np.array(list(C.KEY_RATES.values()))
BP = 1e-4


@dataclass
class CashflowTable:
    """All scheduled cash flows of a set of bonds, flat. Row i belongs to bond idx[i] (position in cusips)."""
    cusips: np.ndarray
    idx: np.ndarray
    date: np.ndarray        # datetime64[D]
    coupon: np.ndarray      # per 100 face
    principal: np.ndarray   # per 100 face (100 on the final row of each bond)

    @property
    def amount(self) -> np.ndarray:
        return self.coupon + self.principal


@dataclass
class Analytics:
    price: np.ndarray       # model dirty price per 100 face (0 if no cash flows remain)
    K: np.ndarray           # n x 8 key-rate durations
    D: np.ndarray           # modified (effective) duration, parallel bump
    D_spr: np.ndarray       # spread duration, bump of the bond's spread
    C: np.ndarray           # convexity, parallel +/- convexity_bp
    dts: np.ndarray         # duration times spread (spread in decimal)
    spread: np.ndarray      # spread used, decimal
    spread_solved: np.ndarray   # True where the spread was solved from the price instead of read from prices_daily


def build_cashflows(terms: pd.DataFrame, earliest) -> CashflowTable:
    """Cash flows for terms (columns cusip, maturity_date, coupon_rate in %, coupon_frequency), from `earliest` on.

    Rows must have no missing maturity or coupon. earliest only needs to be before the first date we price at.
    """
    idx, dates, cpn, prin = [], [], [], []
    for i, r in enumerate(terms.itertuples(index=False)):
        freq = int(r.coupon_frequency) if r.coupon_frequency and r.coupon_frequency > 0 else 2
        sched = coupon_dates(r.maturity_date, freq, earliest)
        p = np.zeros(len(sched))
        p[-1] = 100.0
        idx.append(np.full(len(sched), i))
        dates.append(sched)
        cpn.append(np.full(len(sched), r.coupon_rate / freq))
        prin.append(p)
    return CashflowTable(terms["cusip"].to_numpy(), np.concatenate(idx), np.concatenate(dates),
                         np.concatenate(cpn), np.concatenate(prin))


def nss_params(curve_row: pd.Series) -> np.ndarray:
    return curve_row[NSS_PARAMS].to_numpy(float)


def _live(table: CashflowTable, as_of):
    """Rows after as_of and their times in years from as_of."""
    d0 = np.datetime64(pd.Timestamp(as_of), "D")
    m = table.date > d0
    return m, (table.date[m] - d0).astype(float) / DAYS_PER_YEAR


def price(table: CashflowTable, as_of, params: np.ndarray, spread: np.ndarray, shift=0.0, shape=None) -> np.ndarray:
    """Dirty price per 100 face at as_of for every bond. shift is added to the zero rate (parallel);
    shape (function of maturity in years) adds a maturity-dependent bump instead; both can be combined."""
    m, T = _live(table, as_of)
    idx = table.idx[m]
    rate = nss(T, params) + spread[idx] + shift
    if shape is not None:
        rate = rate + shape(T)
    return np.bincount(idx, table.amount[m] * np.exp(-rate * T), minlength=len(table.cusips))


def cash_between(table: CashflowTable, start, end) -> np.ndarray:
    """Coupons and principal paid in (start, end], per bond per 100 face."""
    d0, d1 = (np.datetime64(pd.Timestamp(x), "D") for x in (start, end))
    m = (table.date > d0) & (table.date <= d1)
    return np.bincount(table.idx[m], table.amount[m], minlength=len(table.cusips))


def solve_spread(table: CashflowTable, as_of, params: np.ndarray, dirty: np.ndarray) -> np.ndarray:
    """Constant spread (decimal) over the curve that reprices `dirty`; NaN where there is no root or no price."""
    n = len(table.cusips)
    m, T = _live(table, as_of)
    idx, A, z = table.idx[m], table.amount[m], nss(T, params)
    return _bisect(lambda x: np.bincount(idx, A * np.exp(-(z + x[idx]) * T), minlength=n) - dirty, n, -0.2, 0.5)


def bump_shape(j: int):
    """Triangular 1-unit bump at key rate j: linear to zero at the neighbouring key rates, flat beyond the first
    and last, so the eight bumps sum to a parallel shift (Fig. 1)."""
    e = np.eye(len(KEY_TENORS))[j]
    return lambda T: np.interp(T, KEY_TENORS, e)


def analytics(table: CashflowTable, as_of, params: np.ndarray, spread_bps=None, dirty=None,
              bump_bp: float = 1.0, convexity_bp: float = 10.0) -> Analytics:
    """Key-rate durations, duration, spread duration, convexity and DTS of every bond at as_of.

    spread_bps: per-bond spread from prices_daily (NaN allowed). Missing spreads are solved from `dirty`.
    Bonds still without a spread, or with no cash flows after as_of, get NaN analytics.
    """
    n = len(table.cusips)
    s = np.full(n, np.nan) if spread_bps is None else np.asarray(spread_bps, float) * BP
    solved = np.isnan(s)
    if solved.any() and dirty is not None:
        s = np.where(solved, solve_spread(table, as_of, params, np.asarray(dirty, float)), s)
    solved &= np.isfinite(s)
    s0 = np.nan_to_num(s)
    h, h2 = bump_bp * BP, convexity_bp * BP

    P = price(table, as_of, params, s0)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.column_stack([-(price(table, as_of, params, s0, shape=lambda T, f=bump_shape(j): h * f(T)) - P) / (h * P)
                             for j in range(len(KEY_TENORS))])
        D = -(price(table, as_of, params, s0, shift=h) - P) / (h * P)
        D_spr = -(price(table, as_of, params, s0 + h) - P) / (h * P)
        Cx = (price(table, as_of, params, s0, shift=h2) + price(table, as_of, params, s0, shift=-h2) - 2 * P) / (P * h2 ** 2)
    bad = ~np.isfinite(s) | (P <= 0)
    K[bad], D[bad], D_spr[bad], Cx[bad] = np.nan, np.nan, np.nan, np.nan
    return Analytics(P, K, D, D_spr, Cx, D_spr * s, np.where(np.isfinite(s), s, np.nan), solved)
