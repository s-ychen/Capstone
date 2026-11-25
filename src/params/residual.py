"""Residual variance psi_i of each bond not explained by the curve (Deliverable 3, Section 2.1).

For each month-end pair (t-1, t): total return r = (dirty_t + cash paid - dirty_{t-1}) / dirty_{t-1}, residual
e = r + k_{t-1}' dy_t, where k_{t-1} are the bond's key-rate durations at t-1. psi_i is the sample variance of e.
Returns with a stale endpoint are dropped. build_inputs then removes the credit factors (credit_factors) and shrinks
each bond's variance toward a duration-scaled sector prior (shrink_psi). estimate_psi is the older plain estimate:
sample variance, sector median for bonds with fewer than min_obs returns.
"""
import numpy as np
import pandas as pd

from data_cleaning.curve import KEY_COLS

from .krd import CashflowTable, analytics, cash_between, nss_params


def residual_returns(table: CashflowTable, panel: pd.DataFrame, curve: pd.DataFrame, as_of,
                     window: int | None = None) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """n x (T-1) residual matrix (NaN where invalid) and the month-end dates of each return (period end).

    panel is the month-end panel; only month-ends up to as_of are used.
    """
    cusips = table.cusips
    dates = panel.index.get_level_values("month_end").unique().sort_values()
    dates = dates[dates <= pd.Timestamp(as_of)]
    if window:
        dates = dates[-(window + 1):]
    wide = {c: panel[c].unstack("month_end").reindex(index=cusips, columns=dates)
            for c in ("dirty_price", "spread_bps", "stale")}
    dirty = wide["dirty_price"].to_numpy(float)
    spread = wide["spread_bps"].to_numpy(float)
    stale = (wide["stale"] != False).to_numpy(bool)  # noqa: E712  (missing counts as stale)
    cv = curve.set_index("date").loc[dates]
    y = cv[KEY_COLS].to_numpy(float)

    E = np.full((len(cusips), len(dates) - 1), np.nan)
    for k in range(1, len(dates)):
        a = analytics(table, dates[k - 1], nss_params(cv.iloc[k - 1]), spread[:, k - 1], dirty[:, k - 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (dirty[:, k] + cash_between(table, dates[k - 1], dates[k]) - dirty[:, k - 1]) / dirty[:, k - 1]
        e = r + a.K @ (y[k] - y[k - 1])
        ok = ~stale[:, k] & ~stale[:, k - 1] & np.isfinite(e)
        E[ok, k - 1] = e[ok]
    return E, dates[1:]


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


def credit_factors(E: np.ndarray, D_spr: np.ndarray, group: np.ndarray, min_bonds: int = 20
                   ) -> tuple[list, np.ndarray, np.ndarray]:
    """Credit factor returns and the residuals left after removing them.

    Each credit group g (a corporate sector) gets one factor per month: the implied common spread change
    s_gk = -(D'e) / (D'D) over the bonds in g with a valid residual that month (least squares of e on -D_spr).
    Bonds outside every group (group None, e.g. Treasuries) keep their residual. D_spr is the spread duration at
    the as-of date, used for every past month (an approximation, spread durations move slowly).
    Returns (groups, S g x T with NaN where under min_bonds, U = E + D_spr * s).
    """
    groups = sorted({g for g in group if g is not None})
    S = np.full((len(groups), E.shape[1]), np.nan)
    U = E.copy()
    for j, g in enumerate(groups):
        m = (group == g) & (D_spr > 0)
        for k in range(E.shape[1]):
            ok = m & np.isfinite(E[:, k])
            if ok.sum() >= min_bonds:
                S[j, k] = -(D_spr[ok] @ E[ok, k]) / (D_spr[ok] @ D_spr[ok])
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
