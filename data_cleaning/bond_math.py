"""
All the helper functions for bond stuff (cusip8 to 9 conversions, accrued interest, cash flow calcs)
Bond maths used across the pipeline: CUSIP check digits, coupon schedules, accrued interest, cash flows, the
Nelson-Siegel-Svensson zero curve formula, and solving yield to maturity / spread over the curve from a dirty price.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DAYS_PER_YEAR = 365.25
YIELD_FREQ = 2          # yields are quoted semiannual bond-equivalent
NSS_PARAMS = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]


def cusip_check_digit(cusip8):
    """9th CUSIP character (modulus 10 double-add-double) for an 8-character stem."""
    total = 0
    for i, ch in enumerate(cusip8.upper()):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch) - ord("A") + 10
        else:
            v = {"*": 36, "@": 37, "#": 38}[ch]
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - total % 10) % 10)


def to_cusip9(cusip8):
    """CRSP Treasury CUSIPs are 8 characters; append the check digit so they join on the AGG 9-character CUSIP."""
    if not isinstance(cusip8, str) or len(cusip8.strip()) != 8:
        return None
    stem = cusip8.strip().upper()
    return stem + cusip_check_digit(stem)


def coupon_dates(maturity, freq, earliest):
    """Coupon dates stepped back from maturity every 12/freq months until one falls on/before `earliest`.
    Returns ascending datetime64[D]. Day-of-month is clipped to the month length (e.g. 31 -> 30 in June)."""
    mat, earliest = pd.Timestamp(maturity), pd.Timestamp(earliest)
    step = max(round(12 / freq), 1)
    n_months = (mat.year - earliest.year) * 12 + mat.month - earliest.month
    back = step * np.arange(max(n_months // step + 2, 2))
    m = (mat.year - 1970) * 12 + mat.month - 1 - back                       # months since 1970, descending
    first_of_month = m.astype("datetime64[M]").astype("datetime64[D]")
    days_in_month = ((m + 1).astype("datetime64[M]").astype("datetime64[D]") - first_of_month).astype(int)
    return (first_of_month + (np.minimum(mat.day, days_in_month) - 1).astype("timedelta64[D]"))[::-1]


def days_30_360(start, end):
    """30/360 (US) day count between two datetime64[D] arrays."""
    def ymd(d):
        month = d.astype("datetime64[M]")
        return (d.astype("datetime64[Y]").astype(int) + 1970,
                month.astype(int) % 12 + 1,
                (d - month.astype("datetime64[D]")).astype(int) + 1)
    (y1, m1, d1), (y2, m2, d2) = ymd(start), ymd(end)
    d1 = np.minimum(d1, 30)
    d2 = np.where((d2 == 31) & (d1 == 30), 30, d2)
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


def accrued_interest(dates, maturity, coupon_pct, freq, day_count):
    """Accrued interest per 100 face on each date for a bullet bond with a regular coupon schedule.
    day_count: 'ACT/ACT' (Treasuries) or '30/360' (corporates)."""
    v = pd.DatetimeIndex(dates).values.astype("datetime64[D]")
    out = np.zeros(len(v))
    if not (coupon_pct > 0) or pd.isna(maturity) or len(v) == 0:
        return out
    sched = coupon_dates(maturity, freq, v.min() - np.timedelta64(400, "D"))
    i = np.searchsorted(sched, v, side="right")
    ok = (i > 0) & (i < len(sched))
    prev = sched[np.clip(i - 1, 0, len(sched) - 1)]
    nxt = sched[np.clip(i, 0, len(sched) - 1)]
    with np.errstate(all="ignore"):
        if day_count == "ACT/ACT":
            frac = (v - prev).astype(float) / (nxt - prev).astype(float)
        else:
            frac = days_30_360(prev, v) / days_30_360(prev, nxt)
    return np.where(ok, coupon_pct / freq * frac, 0.0)


def cashflows(settle, maturity, coupon_pct, freq):
    """Remaining (times in years, amounts per 100 face) of a bullet bond after `settle`."""
    s = pd.Timestamp(settle).to_datetime64().astype("datetime64[D]")
    sched = coupon_dates(maturity, freq, s)
    rem = sched[sched > s]
    if rem.size == 0:
        return np.array([]), np.array([])
    t = (rem - s).astype(float) / DAYS_PER_YEAR
    a = np.full(rem.size, coupon_pct / freq)
    a[-1] += 100.0
    return t, a


def nss(t, p):
    """NSS continuously compounded zero rate at maturities t (years). p = [b0, b1, b2, b3, tau1, tau2]."""
    t = np.maximum(np.asarray(t, float), 1e-6)
    b0, b1, b2, b3, t1, t2 = p
    x1, x2 = t / t1, t / t2
    s1, s2 = (1 - np.exp(-x1)) / x1, (1 - np.exp(-x2)) / x2
    return b0 + b1 * s1 + b2 * (s1 - np.exp(-x1)) + b3 * (s2 - np.exp(-x2))


def _bisect(f, n, lo, hi, iters=50):
    """Vectorised bisection over n problems for a decreasing function f (price error) on [lo, hi]. nan where no root."""
    lo, hi = np.full(n, lo, float), np.full(n, hi, float)
    valid = (f(lo) > 0) & (f(hi) < 0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        up = f(mid) > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.where(valid, (lo + hi) / 2, np.nan)


def solve_yield_and_spread(d, terms, curve):
    """Add ytm_solved (decimal, semiannual) and spread_bps (constant spread over the same-day NSS zero curve) to a
    price frame with columns cusip, date, dirty_price. 
    We use bisection to find the yield and spreads, so search for the number that makes the DCFs = obs dirty price numerically work
    
    terms: one row per cusip with maturity_date, coupon_rate,coupon_frequency
    Callable bonds are priced to maturity. 
    Rows whose date has no fitted curve get spread nan."""
    d = d.copy()
    # initialize the arrays for ytm and spread
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
        # cash flow times (years) and amounts for every row of this bond: rows x cash flows
        freq = int(b["coupon_frequency"])
        dates = d["date"].values[idx].astype("datetime64[D]")
        sched = coupon_dates(b["maturity_date"], freq, dates.min())
        T = (sched[None, :] - dates[:, None]).astype(float) / DAYS_PER_YEAR
        live = T > 0
        A = np.where(live, b["coupon_rate"] / freq, 0.0)
        A[:, -1] += np.where(live[:, -1], 100.0, 0.0)
        T = np.where(live, T, 0.0)
        P = d["dirty_price"].values[idx]
        ok = live.any(axis=1) & np.isfinite(P)
        if not ok.any():
            continue
        # yield: the semiannual rate that discounts the cash flows back to the dirty price
        Tk, Ak, Pk = T[ok], A[ok], P[ok]
        m = YIELD_FREQ
        ytm[idx[ok]] = _bisect(lambda y: (Ak * (1 + y[:, None] / m) ** (-m * Tk)).sum(1) - Pk, len(Pk), -0.2, 1.0)
        # spread: the constant shift of that day's zero curve that reprices the bond
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
