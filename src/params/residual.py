"""Residual variance psi_i of each bond not explained by the curve (Deliverable 3, Section 2.1).

For each month-end pair (t-1, t): total return r = (dirty_t + cash paid - dirty_{t-1}) / dirty_{t-1}, residual
e = r + k_{t-1}' dy_t, where k_{t-1} are the bond's key-rate durations at t-1. psi_i is the sample variance of e.
Returns with a stale endpoint are dropped. build_inputs then removes the credit factors (credit_factors) and shrinks
each bond's variance toward a duration-scaled sector prior (shrink_psi). estimate_psi is the older plain estimate:
sample variance, sector median for bonds with fewer than min_obs returns.
"""
import numpy as np
import pandas as pd

from data_cleaning.treasury_curve import KEY_COLS

from .krd import CashflowTable, analytics, build_cashflows, cash_between, nss_params


def residual_returns(table: CashflowTable, panel: pd.DataFrame, curve: pd.DataFrame, as_of,
                     window: int | None = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """n x (T-1) residual matrix (NaN where invalid) and the month-end dates of each return (period end).

    panel is the month-end panel; only month-ends up to as_of are used.
    """
    dates = _month_ends(panel)
    dates = dates[dates <= pd.Timestamp(as_of)]
    if window:
        dates = dates[-(window + 1):]
    dirty, spread, stale = _wide(panel, table.cusips, dates)
    cv = curve.set_index("date").loc[dates]
    E = np.full((len(table.cusips), len(dates) - 1), np.nan)
    for k in range(1, len(dates)):
        E[:, k - 1] = _residual(table, dates[k - 1], dates[k], cv.iloc[k - 1], cv.iloc[k], dirty[:, k - 1:k + 1],
                                spread[:, k - 1], stale[:, k - 1:k + 1])
    return E, dates[1:]


def _month_ends(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index.get_level_values("month_end").unique().sort_values()


def _wide(panel: pd.DataFrame, cusips, dates):
    """(dirty, spread_bps, stale) as cusips x dates arrays; a missing month counts as stale."""
    wide = {c: panel[c].unstack("month_end").reindex(index=cusips, columns=dates)
            for c in ("dirty_price", "spread_bps", "stale")}
    return (wide["dirty_price"].to_numpy(float), wide["spread_bps"].to_numpy(float),
            (wide["stale"] != False).to_numpy(bool))  # noqa: E712


def _residual(table, d0, d1, cv0, cv1, dirty, spread0, stale) -> np.ndarray:
    """Residual of every bond over (d0, d1], NaN where invalid. dirty and stale are n x 2 (d0, d1)."""
    a = analytics(table, d0, nss_params(cv0), spread0, dirty[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (dirty[:, 1] + cash_between(table, d0, d1) - dirty[:, 0]) / dirty[:, 0]
    e = r + a.K @ (cv1[KEY_COLS].to_numpy(float) - cv0[KEY_COLS].to_numpy(float))
    return np.where(~stale[:, 1] & ~stale[:, 0] & np.isfinite(e), e, np.nan)


def _subset(table: CashflowTable, rows: np.ndarray) -> CashflowTable:
    """The bonds where rows is True, cash flows kept in order."""
    m = rows[table.idx]
    remap = np.cumsum(rows) - 1
    return CashflowTable(table.cusips[rows], remap[table.idx[m]], table.date[m], table.coupon[m], table.principal[m])


class ResidualPanel:
    """Residuals of a fixed bond set, each month computed once and reused across as-of dates.

    A month's residual depends only on its two month-ends, so residual_returns at any as-of date is a row and column
    slice of this panel. Columns are filled lazily up to the latest as-of asked for. terms: cusip, maturity_date,
    coupon_rate, coupon_frequency (rows without maturity or coupon are dropped).
    """

    def __init__(self, terms: pd.DataFrame, panel: pd.DataFrame, curve: pd.DataFrame):
        terms = terms[terms["maturity_date"].notna() & terms["coupon_rate"].notna()].reset_index(drop=True)
        self.dates = _month_ends(panel)
        # same earliest as build_inputs, so every bond has its cash flows from the first month-end on
        self.table = build_cashflows(terms, self.dates.min() - pd.Timedelta(days=400))
        self.pos = pd.Index(self.table.cusips)
        self.dirty, self.spread, self.stale = _wide(panel, self.table.cusips, self.dates)
        self.curve = curve.set_index("date")
        self.cols: list[np.ndarray] = []

    def covers(self, cusips) -> bool:
        return bool(self.pos.get_indexer(cusips).min(initial=0) >= 0)

    def get(self, cusips, as_of, window: int | None = None) -> np.ndarray:
        """residual_returns(table of cusips, panel, curve, as_of, window)[0]."""
        n = int((self.dates <= pd.Timestamp(as_of)).sum())
        for k in range(len(self.cols) + 1, n):
            d0, d1 = self.dates[k - 1], self.dates[k]
            live = ~self.stale[:, k - 1] & ~self.stale[:, k]    # every other row is NaN anyway
            col = np.full(len(live), np.nan)
            col[live] = _residual(_subset(self.table, live), d0, d1, self.curve.loc[d0], self.curve.loc[d1],
                                  self.dirty[live, k - 1:k + 1], self.spread[live, k - 1],
                                  self.stale[live, k - 1:k + 1])
            self.cols.append(col)
        lo = max(0, n - 1 - window) if window else 0
        rows = self.pos.get_indexer(cusips)
        if (rows < 0).any():
            raise KeyError(f"{(rows < 0).sum()} cusips not in the residual panel")
        if n - 1 <= lo:
            return np.full((len(rows), 0), np.nan)
        return np.column_stack(self.cols[lo:n - 1])[rows]


def estimate_psi(E: np.ndarray, sector: np.ndarray, min_obs: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """(psi, n_obs) from the residual matrix. Bonds under min_obs take their sector median; a sector with none
    takes the overall median."""
    n_obs = np.isfinite(E).sum(axis=1)
    psi = np.full(len(E), np.nan)
    enough = n_obs >= min_obs
    psi[enough] = np.nanvar(E[enough], axis=1, ddof=1)
    if not enough.any():
        raise ValueError(f"no bond has {min_obs} residual returns; widen the window or lower min_obs")
    s = pd.Series(psi, index=pd.Index(sector))
    med = s.groupby(level=0).transform("median").to_numpy()
    psi = np.where(enough, psi, np.where(np.isfinite(med), med, np.nanmedian(psi)))
    return psi, n_obs


def credit_factors(E: np.ndarray, D_spr: np.ndarray, group: np.ndarray, min_bonds: int = 20, min_months: int = 3
                   ) -> tuple[list, np.ndarray, np.ndarray]:
    """Credit factor returns and the residuals left after removing them.

    Each credit group g (a corporate sector) gets one factor per month: the implied common spread change
    s_gk = -(D'e) / (D'D) over the bonds in g with a valid residual that month (least squares of e on -D_spr).
    Bonds outside every group (group None, e.g. Treasuries) keep their residual. D_spr is the spread duration at
    the as-of date, used for every past month (an approximation, spread durations move slowly).
    The factor covariance needs min_months months where every group has a value, so the sparsest groups are dropped
    until that holds. Bonds in a dropped group keep their full residual, so psi still carries their credit risk.
    Returns (groups kept, S g x T with NaN where under min_bonds, U = E + D_spr * s).
    """
    groups = sorted({g for g in group if g is not None})
    S = np.full((len(groups), E.shape[1]), np.nan)
    for j, g in enumerate(groups):
        m = (group == g) & (D_spr > 0)
        for k in range(E.shape[1]):
            ok = m & np.isfinite(E[:, k])
            if ok.sum() >= min_bonds:
                S[j, k] = -(D_spr[ok] @ E[ok, k]) / (D_spr[ok] @ D_spr[ok])
    keep = np.ones(len(groups), bool)
    while keep.any() and np.all(np.isfinite(S[keep]), axis=0).sum() < min_months:
        n_fin = np.where(keep, np.isfinite(S).sum(axis=1), np.inf)
        keep[np.argmin(n_fin)] = False
    groups, S = [g for g, k in zip(groups, keep) if k], S[keep]
    U = E.copy()
    for j, g in enumerate(groups):
        m = (group == g) & (D_spr > 0)
        for k in np.flatnonzero(np.isfinite(S[j])):
            ok = m & np.isfinite(E[:, k])
            U[ok, k] = E[ok, k] + D_spr[ok] * S[j, k]
    return groups, S, U


def shrink_psi(U: np.ndarray, D: np.ndarray, sector: np.ndarray, min_obs: int = 6, n0: float = 12,
               d_floor: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """(psi, n_obs) with each bond's sample variance shrunk toward a duration-scaled sector prior.

    prior_i = D_i^2 * kappa_s, kappa_s = median of s2_i / D_i^2 over the sector's bonds with at least min_obs returns
    (D floored at d_floor). psi_i = (n_i s2_i + n0 prior_i) / (n_i + n0), and the prior alone under min_obs.
    n0 is the prior's weight in months of data.
    """
    n_obs = np.isfinite(U).sum(axis=1)
    enough = n_obs >= min_obs
    if not enough.any():
        raise ValueError(f"no bond has {min_obs} residual returns; widen the window or lower min_obs")
    s2 = np.zeros(len(U))
    s2[enough] = np.nanvar(U[enough], axis=1, ddof=1)
    D2 = np.maximum(np.nan_to_num(D), d_floor) ** 2
    k = pd.Series(np.where(enough, s2 / D2, np.nan)).groupby(np.asarray(sector)).transform("median").to_numpy()
    k = np.where(np.isfinite(k), k, np.nanmedian(np.where(enough, s2 / D2, np.nan)))
    prior = k * D2
    psi = np.where(enough, (n_obs * s2 + n0 * prior) / (n_obs + n0), prior)
    return psi, n_obs
