"""Bond maths shared by the cleaning steps: CUSIP check digits, coupon schedules, accrued interest, cash flows."""
import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def cusip_check_digit(cusip8: str) -> str:
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


def to_cusip9(cusip8) -> str | None:
    """CRSP Treasury CUSIPs are 8 characters; append the check digit so they join on the AGG 9-character CUSIP."""
    if not isinstance(cusip8, str) or len(cusip8.strip()) != 8:
        return None
    stem = cusip8.strip().upper()
    return stem + cusip_check_digit(stem)


def coupon_dates(maturity, freq: int, earliest) -> np.ndarray:
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


def _ymd(d: np.ndarray):
    month = d.astype("datetime64[M]")
    return (d.astype("datetime64[Y]").astype(int) + 1970,
            month.astype(int) % 12 + 1,
            (d - month.astype("datetime64[D]")).astype(int) + 1)


def days_30_360(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """30/360 (US) day count between two datetime64[D] arrays."""
    (y1, m1, d1), (y2, m2, d2) = _ymd(start), _ymd(end)
    d1 = np.minimum(d1, 30)
    d2 = np.where((d2 == 31) & (d1 == 30), 30, d2)
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


def accrued_interest(dates, maturity, coupon_pct: float, freq: int, day_count: str) -> np.ndarray:
    """Accrued interest per 100 face on each date for a bullet bond with a regular coupon schedule.
    day_count: 'ACT/ACT' (Treasuries) or '30/360' (corporates). Zero-coupon bonds accrue nothing."""
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


def mbs_accrued_interest(dates, coupon_pct: float) -> np.ndarray:
    """Agency pass-throughs accrue 30/360 from the 1st of the month: coupon/12 * (day - 1)/30 per 100 face."""
    day = pd.DatetimeIndex(dates).day.values
    return coupon_pct / 12 * (np.minimum(day, 30) - 1) / 30


def cashflows(settle, maturity, coupon_pct: float, freq: int):
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
