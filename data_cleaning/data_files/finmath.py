"""Numerics: CUSIPs, coupon schedules, accrued interest, cash flows, NSS zero curve, batched yield/spread solvers."""
import numpy as np
import pandas as pd
from . import config as C


def cusip_check_digit(stem):
    """9-character CUSIP from an 8-character stem (None if invalid)."""
    if not isinstance(stem, str) or len(stem) != 8:
        return None
    vals = [int(c) if c.isdigit() else ord(c) - 55 if c.isalpha() else {"*": 36, "@": 37, "#": 38}.get(c) for c in stem.upper()]
    if None in vals:
        return None
    total = sum((v * (2 if i % 2 else 1)) // 10 + (v * (2 if i % 2 else 1)) % 10 for i, v in enumerate(vals))
    return stem.upper() + str((10 - total % 10) % 10)


def coupon_dates(maturity, freq, earliest):
    """Coupon dates counted back from maturity in 12/freq-month steps, down to the first one <= earliest. Ascending datetime64[D]."""
    mat, earliest, step = pd.Timestamp(maturity), pd.Timestamp(earliest), max(round(12 / max(freq, 1)), 1)
    span = (mat.year - earliest.year) * 12 + mat.month - earliest.month
    back = step * np.arange(max(span // step + 2, 2))
    m = (mat.year - 1970) * 12 + mat.month - 1 - back                                  # months since 1970
    first = m.astype("datetime64[M]").astype("datetime64[D]")
    days_in_month = ((m + 1).astype("datetime64[M]").astype("datetime64[D]") - first).astype(int)
    return (first + (np.minimum(mat.day, days_in_month) - 1).astype("timedelta64[D]"))[::-1]   # day clipped to month length


def _ymd(d):
    mo = d.astype("datetime64[M]")
    return d.astype("datetime64[Y]").astype(int) + 1970, mo.astype(int) % 12 + 1, (d - mo.astype("datetime64[D]")).astype(int) + 1


def _days360(a, b):
    (ya, ma, da), (yb, mb, db) = _ymd(a), _ymd(b)
    da = np.minimum(da, 30)
    return 360 * (yb - ya) + 30 * (mb - ma) + (np.where((db == 31) & (da == 30), 30, db) - da)


def accrued(val_dates, sched, coupon, freq, day_count):
    """Accrued interest per 100 for one bond on each valuation date (30/360 or ACT/ACT)."""
    v = np.asarray(val_dates).astype("datetime64[D]")
    i = np.searchsorted(sched, v, side="right")
    ok = (i > 0) & (i < len(sched)) & (coupon > 0)
    prev, nxt = sched[np.clip(i - 1, 0, len(sched) - 1)], sched[np.clip(i, 0, len(sched) - 1)]
    with np.errstate(all="ignore"):
        frac = (v - prev) / (nxt - prev) if day_count == "ACT/ACT" else _days360(prev, v) / _days360(prev, nxt)
    return np.where(ok, coupon / freq * frac, 0.0)


def cashflows(settle, sched, coupon, freq, amortising=False):
    """Remaining (times in years, amounts) per 100 face. Bullet bonds pay coupons + 100; pass-throughs amortise level-pay with C.MBS_CPR."""
    s = pd.Timestamp(settle).to_datetime64().astype("datetime64[D]")
    rem = sched[sched > s]
    if not rem.size:
        return np.array([]), np.array([])
    t = (rem - s).astype(float) / C.DAYS_PER_YEAR
    if not amortising:
        a = np.full(rem.size, coupon / freq)
        a[-1] += 100.0
        return t, a
    n, r, k = min(rem.size, C.MBS_MAX_MONTHS), coupon / 1200.0, np.arange(min(rem.size, C.MBS_MAX_MONTHS) + 1)
    sched_bal = 100 * ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1) if r > 0 else 100 * (1 - k / n)
    bal = sched_bal * (1 - (1 - (1 - C.MBS_CPR) ** (1 / 12))) ** k                     # survival after prepayments
    return t[:n], bal[:-1] * r + (bal[:-1] - bal[1:])                                   # interest + principal


def nss(t, p):
    """Nelson-Siegel-Svensson continuously compounded zero rate (decimal). p = [b0,b1,b2,b3,tau1,tau2], 1-D or one row per t-row."""
    p, t = np.asarray(p, float), np.maximum(np.asarray(t, float), 1e-6)
    b0, b1, b2, b3, t1, t2 = (p[..., i, None] if p.ndim == 2 else p[i] for i in range(6))
    x1, x2 = t / t1, t / t2
    s1 = (1 - np.exp(-x1)) / x1
    return b0 + b1 * s1 + b2 * (s1 - np.exp(-x1)) + b3 * ((1 - np.exp(-x2)) / x2 - np.exp(-x2))


def solve_batch(dirty, times, amounts, kind, params=None, chunk=20000):
    """
    Vectorised bisection over many bonds. kind='spread': constant spread (bps) over the NSS curve (params, one row per bond);
    kind='ytm': semiannual yield (%). Both reprice the cash flows to the dirty price. NaN where no root.
    """
    out, dirty = np.full(len(dirty), np.nan), np.asarray(dirty, float)
    rows = [i for i in range(len(times)) if len(times[i]) and np.isfinite(dirty[i])]
    rows.sort(key=lambda i: len(times[i]))
    for s in range(0, len(rows), chunk):
        idx = rows[s:s + chunk]
        L = len(times[idx[-1]])
        T, A = np.zeros((len(idx), L)), np.zeros((len(idx), L))
        for r, i in enumerate(idx):
            T[r, :len(times[i])], A[r, :len(times[i])] = times[i], amounts[i]
        P = dirty[idx]
        if kind == "spread":
            Z = nss(T, np.asarray(params)[idx])
            f, lo, hi, scale = (lambda x: (A * np.exp(-(Z + x[:, None]) * T)).sum(1) - P), -0.2, 0.5, 1e4
        else:
            m = C.YIELD_FREQ
            f, lo, hi, scale = (lambda x: (A * (1 + x[:, None] / m) ** (-m * T)).sum(1) - P), -0.2, 1.0, 100.0
        lo, hi = np.full(len(idx), lo), np.full(len(idx), hi)
        valid = (f(lo) > 0) & (f(hi) < 0)
        for _ in range(60):                                       # price is decreasing in the rate
            mid = (lo + hi) / 2
            up = f(mid) > 0
            lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
        out[idx] = np.where(valid, (lo + hi) / 2 * scale, np.nan)
    return out
