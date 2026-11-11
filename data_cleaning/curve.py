"""Step 3: fit a Treasury zero curve per date (Nelson-Siegel-Svensson) and read off the key-rate zero yields."""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from .finmath import cashflows

log = logging.getLogger(__name__)

NSS_PARAMS = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
NSS_START = np.array([0.04, -0.02, 0.0, 0.0, 2.0, 10.0])
NSS_LOW = np.array([0.0, -0.20, -0.30, -0.30, 0.20, 1.0])
NSS_HIGH = np.array([0.20, 0.20, 0.30, 0.30, 10.0, 40.0])
MIN_BONDS = 20          # skip a date with fewer usable quotes
OUTLIER_SIGMAS = 5.0    # bonds whose price error exceeds this many robust sigmas are refit with zero weight
KEY_COLS = [f"zero_{k.lower()}" for k in C.KEY_RATES]   # zero_6m, zero_1y, ... (decimal, continuously compounded)


def nss(t, p):
    """NSS continuously compounded zero rate at maturities t (years). p = [b0, b1, b2, b3, tau1, tau2]."""
    t = np.maximum(np.asarray(t, float), 1e-6)
    b0, b1, b2, b3, t1, t2 = p
    x1, x2 = t / t1, t / t2
    s1, s2 = (1 - np.exp(-x1)) / x1, (1 - np.exp(-x2)) / x2
    return b0 + b1 * s1 + b2 * (s1 - np.exp(-x1)) + b3 * (s2 - np.exp(-x2))


def _fit_day(T, A, idx, P, dur, x0):
    """Duration-weighted price least squares with one robust refit. Returns (params, price rmse, bonds used)."""
    w0 = 1 / np.maximum(dur, 0.1)                       # 1/duration ~ price errors -> yield errors

    def resid(p, w):
        model = np.bincount(idx, A * np.exp(-nss(T, p) * T), minlength=len(P))
        return (model - P) * w

    w = w0.copy()
    p = np.clip(x0, NSS_LOW, NSS_HIGH)
    for _ in range(2):
        p = least_squares(resid, p, args=(w,), bounds=(NSS_LOW, NSS_HIGH)).x
        err = resid(p, np.ones(len(P)))
        mad = np.median(np.abs(err[w > 0] - np.median(err[w > 0]))) * 1.4826
        if mad == 0:
            break
        w = np.where(np.abs(err) > OUTLIER_SIGMAS * mad, 0.0, w0)
    used = w > 0
    rmse = float(np.sqrt(np.mean(resid(p, np.ones(len(P)))[used] ** 2)))
    return p, rmse, int(used.sum())


def fit_curves(tsy: pd.DataFrame) -> pd.DataFrame:
    """Fit one NSS curve per quote date from clean Treasury quotes (output of treasuries.clean_treasuries).

    Uses fixed-coupon notes/bonds and bills (no TIPS). Returns one row per date: NSS parameters,
    fit diagnostics and the zero yields at the key-rate tenors (decimal, continuously compounded).
    """
    tsy = tsy[tsy["years_to_maturity"] > 0.02]        # drop quotes within a week of maturity: price ~ 100, no information
    rows, x0 = [], NSS_START.copy()
    for date, day in tsy.groupby("date", sort=True):
        flows = [cashflows(date, r.maturity_date, r.coupon_rate, 2) for r in day.itertuples()]
        keep = [i for i, (t, _) in enumerate(flows) if len(t)]
        if len(keep) < MIN_BONDS:
            log.warning("%s: only %d usable Treasuries, curve skipped", date.date(), len(keep))
            continue
        P = day["dirty_price"].to_numpy(float)[keep]
        T = np.concatenate([flows[i][0] for i in keep])
        A = np.concatenate([flows[i][1] for i in keep])
        idx = np.repeat(np.arange(len(keep)), [len(flows[i][0]) for i in keep])
        dur = np.bincount(idx, T * A) / np.bincount(idx, A)   # cash-flow-weighted time: cheap duration proxy for weights
        x0, rmse, n_used = _fit_day(T, A, idx, P, dur, x0)
        zeros = nss(np.array(list(C.KEY_RATES.values())), x0)
        rows.append({"date": date, **dict(zip(NSS_PARAMS, x0)), "fit_rmse_price": rmse,
                     "n_bonds_used": n_used, "n_bonds_quoted": len(keep), **dict(zip(KEY_COLS, zeros))})
    out = pd.DataFrame(rows)
    log.info("fitted %d daily curves, median price rmse %.3f", len(out), out["fit_rmse_price"].median())
    return out[["date"] + KEY_COLS + NSS_PARAMS + ["fit_rmse_price", "n_bonds_used", "n_bonds_quoted"]]
