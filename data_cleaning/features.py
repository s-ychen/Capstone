"""Step 5: features that need the bond's cash flows and the fitted curve: yield to maturity and the constant
spread over the Treasury zero curve (both solved from the dirty price), and years to maturity."""
import logging

import numpy as np
import pandas as pd

from .curve import NSS_PARAMS, nss
from .finmath import DAYS_PER_YEAR, coupon_dates

log = logging.getLogger(__name__)
YIELD_FREQ = 2          # yields are quoted semiannual bond-equivalent


def _bisect(f, n, lo, hi, iters=50):
    """Vectorised bisection over n problems for a decreasing function f (price error) on [lo, hi]. NaN where no root."""
    lo, hi = np.full(n, lo, float), np.full(n, hi, float)
    valid = (f(lo) > 0) & (f(hi) < 0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        up = f(mid) > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.where(valid, (lo + hi) / 2, np.nan)


def solve_yield_and_spread(d: pd.DataFrame, terms: pd.DataFrame, curve: pd.DataFrame | None) -> pd.DataFrame:
    """Add ytm_solved (decimal, semiannual) and spread_bps (constant spread over the same-day NSS zero curve) to a
    daily price frame with columns cusip, date, dirty_price. terms: one row per cusip with maturity_date, coupon_rate,
    coupon_frequency. Callable bonds are priced to maturity. Rows whose date has no fitted curve get spread NaN."""
    d = d.copy()
    ytm, spread = np.full(len(d), np.nan), np.full(len(d), np.nan)
    params = None
    if curve is not None:
        params = curve.set_index("date")[NSS_PARAMS].reindex(d["date"]).to_numpy(float)   # NaN rows = no curve that day
    tm = terms.set_index("cusip")
    for cusip, idx in d.groupby("cusip").indices.items():
        if cusip not in tm.index:
            continue
        b = tm.loc[cusip]
        if pd.isna(b["maturity_date"]) or pd.isna(b["coupon_rate"]):
            continue
        freq = int(b["coupon_frequency"])
        dates = d["date"].values[idx].astype("datetime64[D]")
        sched = coupon_dates(b["maturity_date"], freq, dates.min())
        T = (sched[None, :] - dates[:, None]).astype(float) / DAYS_PER_YEAR          # rows x cash flows
        live = T > 0
        A = np.where(live, b["coupon_rate"] / freq, 0.0)
        A[:, -1] += np.where(live[:, -1], 100.0, 0.0)
        T = np.where(live, T, 0.0)
        P = d["dirty_price"].values[idx]
        ok = live.any(axis=1) & np.isfinite(P)
        if not ok.any():
            continue
        Tk, Ak, Pk = T[ok], A[ok], P[ok]
        m = YIELD_FREQ
        ytm[idx[ok]] = _bisect(lambda y: (Ak * (1 + y[:, None] / m) ** (-m * Tk)).sum(1) - Pk, len(Pk), -0.2, 1.0)
        if params is not None:
            pk = params[idx[ok]]
            has = np.isfinite(pk).all(axis=1)
            if has.any():
                Z = np.stack([nss(Tk[i], pk[i]) for i in np.flatnonzero(has)])
                Th, Ah, Ph = Tk[has], Ak[has], Pk[has]
                s = _bisect(lambda x: (Ah * np.exp(-(Z + x[:, None]) * Th)).sum(1) - Ph, len(Ph), -0.2, 0.5)
                spread[idx[ok][has]] = s * 1e4
    d["ytm_solved"] = ytm
    d["spread_bps"] = spread
    d["years_to_maturity"] = (d["cusip"].map(tm["maturity_date"]) - d["date"]).dt.days / DAYS_PER_YEAR
    log.info("solved yields for %.1f%% of rows, spreads for %.1f%%", 100 * np.isfinite(ytm).mean(), 100 * np.isfinite(spread).mean())
    return d
