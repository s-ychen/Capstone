"""Residual variance psi_i of each bond not explained by the curve (Deliverable 3, Section 2.1).

For each month-end pair (t-1, t): total return r = (dirty_t + cash paid - dirty_{t-1}) / dirty_{t-1}, residual
e = r + k_{t-1}' dy_t, where k_{t-1} are the bond's key-rate durations at t-1. psi_i is the sample variance of e.
Returns with a stale endpoint are dropped. Bonds with fewer than min_obs returns get their sector's median psi,
so new issues still get a value.
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
