"""Steps 2-3: clean CRSP Treasury quotes, fit a zero curve per date (NSS), read off the 8 key-rate yields."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from .finmath import cashflows, coupon_dates, nss

NSS_NAMES = ["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]
KEY_COLS = {k: f"zero_yield_{k.lower()}_pct" for k in C.KEY_RATES}


def clean_crsp(issues, daily):
    """
    One row per Treasury per quote date: mid_clean_price = (bid+ask)/2, dirty_price = mid + CRSP accrued interest,
    half_spread_frac = (ask-bid)/2/mid. Columns: treasury_id, cusip8, issue_date, maturity_date, coupon_pct, quote_date, ...
    """
    iss = issues.rename(columns={"kytreasno": "treasury_id", "tcusip": "cusip8", "tdatdt": "issue_date",
                                 "tmatdt": "maturity_date", "tcouprt": "coupon_pct"})
    iss = iss[~iss["itype"].isin(C.CRSP_EXCLUDE_ITYPES)].drop_duplicates("treasury_id")
    q = daily.rename(columns={"kytreasno": "treasury_id", "caldt": "quote_date", "tdbid": "bid_price", "tdask": "ask_price",
                              "tdaccint": "accrued_interest"})
    q = q[(q["bid_price"] > 0) & (q["ask_price"] >= q["bid_price"])]
    out = q.merge(iss[["treasury_id", "cusip8", "issue_date", "maturity_date", "coupon_pct"]], on="treasury_id")
    for c in ["quote_date", "issue_date", "maturity_date"]:
        out[c] = pd.to_datetime(out[c])
    out["mid_clean_price"] = (out["bid_price"] + out["ask_price"]) / 2
    out["dirty_price"] = out["mid_clean_price"] + out["accrued_interest"].fillna(0.0)
    out["half_spread_frac"] = (out["ask_price"] - out["bid_price"]) / 2 / out["mid_clean_price"]
    return out[out["maturity_date"] > out["quote_date"]].reset_index(drop=True)


def _fit_day(T, A, I, P, dur, x0):
    """Duration-weighted NSS least squares with robust refits that zero-weight outlier bonds. Returns (params, rmse, n_used)."""
    w0 = 1 / np.maximum(dur, 0.1)
    res = lambda p, w: (np.bincount(I, A * np.exp(-nss(T, p) * T), len(P)) - P) * w
    w, p = w0.copy(), np.clip(x0, C.NSS_LOW, C.NSS_HIGH)
    for _ in range(1 + C.CURVE_PASSES):
        p = least_squares(res, p, args=(w,), bounds=(C.NSS_LOW, C.NSS_HIGH)).x
        z = res(p, w0)
        mad = np.median(np.abs(z[w > 0] - np.median(z[w > 0]))) * 1.4826
        if not mad:
            break
        w = np.where(np.abs(z) > C.CURVE_SIGMAS * mad, 0.0, w0)
    used = w > 0
    return p, float(np.sqrt(np.mean((res(p, np.ones(len(P))) [used]) ** 2))), int(used.sum())


def fit_curves(quotes):
    """NSS curve per quote date (all dates, or month-end dates if C.CURVE_FREQ == 'month_end'). Returns params df indexed by date."""
    dates = pd.Series(sorted(quotes["quote_date"].unique()))
    if C.CURVE_FREQ == "month_end":
        dates = dates.groupby(dates.dt.to_period("M")).max()
    first = quotes["quote_date"].min() - pd.DateOffset(years=1)
    sched = {r.treasury_id: coupon_dates(r.maturity_date, 2, first)
             for r in quotes.drop_duplicates("treasury_id").itertuples()}
    rows, x0 = [], np.array(C.NSS_START)
    for d in dates:
        day = quotes[quotes["quote_date"] == d]
        flows = [(r.dirty_price, *cashflows(d, sched[r.treasury_id], r.coupon_pct, 2)) for r in day.itertuples()]
        flows = [f for f in flows if len(f[1])]
        if len(flows) < C.CURVE_MIN_BONDS:
            continue
        P = np.array([f[0] for f in flows])
        T, A = np.concatenate([f[1] for f in flows]), np.concatenate([f[2] for f in flows])
        I = np.repeat(np.arange(len(flows)), [len(f[1]) for f in flows])
        dur = np.bincount(I, T * A) / np.bincount(I, A)                       # cash-flow-weighted time, a duration proxy
        x0, rmse, n_used = _fit_day(T, A, I, P, dur, x0)
        rows.append({"date": pd.Timestamp(d), **dict(zip(NSS_NAMES, x0)), "rmse_price": rmse, "n_used": n_used, "n_quoted": len(P)})
    final_df = pd.DataFrame(rows).set_index("date")
    return final_df


def key_rate_table(params):
    """Zero yields (continuously compounded, %) at the 8 key-rate tenors for every fitted date (the Yield Curve table)."""
    z = np.vstack([100 * nss(np.array(list(C.KEY_RATES.values())), p) for p in params[NSS_NAMES].to_numpy()])
    kr_df = pd.DataFrame(z, index=params.index, columns=list(KEY_COLS.values())).join(params[["rmse_price", "n_used", "n_quoted"]])
    return kr_df


def params_asof(params, dates):
    """NSS parameters of the latest fitted curve on or before each date (NaN if none)."""
    i = params.index.searchsorted(pd.DatetimeIndex(dates), side="right") - 1
    out = params[NSS_NAMES].to_numpy()[np.clip(i, 0, None)].astype(float)
    out[i < 0] = np.nan
    return out
