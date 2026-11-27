"""
Clean the CRSP daily Treasury quotes, and fit a Treasury zero curve per day from them (NSS methodology),
reading off the zero yields at the key-rate tenors. The curve is the only output saved daily (besides prices_daily if you want it).

"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import config as C
from .bond_math import NSS_PARAMS, cashflows, nss, to_cusip9

log = logging.getLogger(__name__)
# crsp notation: 1 is noncallable bonds, 2 is noncallable notes, 4 are bills, 5/6 are callable bonds/notes, 11/12 are Treasury Inflation-Protected Securities notes/bonds
FIXED_COUPON_TYPES = (1, 2, 4)

NSS_START = np.array([0.04, -0.02, 0.0, 0.0, 2.0, 10.0])
NSS_LOW = np.array([0.0, -0.20, -0.30, -0.30, 0.20, 1.0])
NSS_HIGH = np.array([0.20, 0.20, 0.30, 0.30, 10.0, 40.0])
MIN_BONDS = 20          # skip date with few usable quotes
OUTLIER_SIGMAS = 5.0    # bonds whose price error exceeds this many robust sigmas are refit with zero weight
KEY_COLS = [f"zero_{k.lower()}" for k in C.KEY_RATES]   # zero_6m, zero_1y, ... (decimal, continuously compounded)


def clean_treasuries(raw):
    """One row per Treasury per quote date with the standard price columns.

    - drop callables and tips, such that we only have the fixed-coupon bonds
    - mid = (bid + ask)/2 is the clean price
    - dirty = clean + CRSP accrued interest
    - bid_ask_spread is in price points per 100 face; yields are annualised from crsp's daily yield
    """
    df = raw.copy()
    # Filter out variable coupon bonds, bad data (i.e bid price pos & ask > bid & mat date > bond date)
    df["security_type"] = df["security_type"].astype(int)
    df = df[df["security_type"].isin(FIXED_COUPON_TYPES)]
    df = df[(df["bid_price"] > 0) & (df["ask_price"] >= df["bid_price"])]
    df = df[df["maturity_date"] > df["date"]]
    # since AGG uses cusip9 we convert all the data to cusip9 from cusip8 
    df["cusip"] = df["cusip8"].map(to_cusip9) 
    df = df.dropna(subset=["cusip"])
    # calculate clean price: bid + ask / 2, dirty price: clean + acc int, bid-ask, 
    num = ["coupon_rate", "bid_price", "ask_price", "mid_price", "accrued_interest", "daily_yield",
           "duration_days", "daily_return", "total_outstanding", "public_outstanding"]
    df[num] = df[num].astype(float)
    df["clean_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["accrued_interest"] = df["accrued_interest"].fillna(0.0)
    df["dirty_price"] = df["clean_price"] + df["accrued_interest"]
    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    # clean up other cols
    df["yield"] = df["daily_yield"] * 365                     # CRSP tdyld is a daily rate so * 365
    df["duration"] = df["duration_days"] / 365
    df["years_to_maturity"] = (df["maturity_date"] - df["date"]).dt.days / 365.25 # use this
    df["is_zero_coupon"] = df["security_type"] == 4

    # construct the final cols for the treasury data that we need
    cols = ["cusip", "date", "security_type", "is_zero_coupon", "coupon_rate", "dated_date", "maturity_date",
            "years_to_maturity", "bid_price", "ask_price", "clean_price", "accrued_interest", "dirty_price",
            "bid_ask_spread", "price_flag", "yield", "duration", "daily_return", "total_outstanding", "public_outstanding"]
    out = df[cols].sort_values(["cusip", "date"]).reset_index(drop=True)

    # drop any duplicate rows, by cusip x date
    dup = out.duplicated(["cusip", "date"])
    if dup.any():
        log.warning("%d duplicate (cusip, date) Treasury rows dropped", dup.sum())
        out = out[~dup]
    return out


def _fit_day(T, A, idx, P, dur, x0):
    """Duration-weighted price least squares with one robust refit. Returns (params, price rmse, bonds used)."""
    def resid(p, w):
        '''helper func to calc the residuals between our fit'''
        model = np.bincount(idx, A * np.exp(-nss(T, p) * T), minlength=len(P))
        return (model - P) * w
    # we do this weight scaling because price error scaled with duration*yield error
    # price err ~ yield error * duration
    # we add a floor of .1 so short time bonds dont get huge weights
    w0 = 1 / np.maximum(dur, 0.1)
    w = w0.copy() 
    # clip x0 to be betewen our configged values
    p = np.clip(x0, NSS_LOW, NSS_HIGH)
    # we do a two pass robust fit using mean abs dev instead of std dev for mesasuring error spread
    #   1. fit least squares with duration weights
    #   2. compute the median absolute deviation of the unweighted price errors, scaled by 1.4826 = 1/Phi^-1(0.75)
    #      so it estimates the standard deviation under normality. Using the median instead of the std keeps
    #      mispriced bonds from inflating the threshold
    #   3. for outlier bonds (error > 5 sigma out), weight them as 0, and refit
    for _ in range(2):
        p = least_squares(resid, p, args=(w,), bounds=(NSS_LOW, NSS_HIGH)).x
        err = resid(p, np.ones(len(P)))
        # we wanna use a diff measure of variance then standard deviation, because outliers will
        # stretch that. so instead we use the median of the absolute deviations
        # calculate median(|error - median(error)|) (where 50% of data lies within one MAD of median)
        # assuming normalish data, 50% of the data falls within +- .6745 of the mean, 
        # mean absolute dev = sigma * .6745
        # sigma = mean absolute dev * 1/.6745 
        sigma = np.median(np.abs(err[w > 0] - np.median(err[w > 0]))) * 1.4826
        if sigma == 0:
            break
        w = np.where(np.abs(err) > OUTLIER_SIGMAS * sigma, 0.0, w0)
    used = w > 0
    rmse = float(np.sqrt(np.mean(resid(p, np.ones(len(P)))[used] ** 2)))
    return p, rmse, int(used.sum())


def fit_curves(tsy):
    """Fit one NSS curve per quote date from the clean Treasury quotes (clean_treasuries), using fixed-coupon
    notes/bonds and bills (no TIPS). Returns one row per date: the zero yields at the key-rate tenors (decimal,
    continuously compounded), the NSS parameters and fit rmse."""
    # drop quotes within a week of maturity: price ~ 100, no information
    tsy = tsy[tsy["years_to_maturity"] > 0.02]
    rows, x0 = [], NSS_START.copy()
    for date, day in tsy.groupby("date", sort=True):
        # get the owed cashflows still (t=schedule times, a=amounts per 100 face), and make an index list of the bonds that 
        # still have casfhlows
        flows = [cashflows(date, r.maturity_date, r.coupon_rate, 2) for r in day.itertuples()]
        keep = [i for i, (t, _) in enumerate(flows) if len(t)]
        if len(keep) < MIN_BONDS: # we want a usable (>20) of bonds to fit the curve on the day
            log.warning("%s: only %d usable Treasuries, curve skipped", date.date(), len(keep))
            continue
        # get the dirty prices, scheduled times, amounts for just our kept bonds
        P = day["dirty_price"].to_numpy(float)[keep]
        T = np.concatenate([flows[i][0] for i in keep]) # the scheduled times
        A = np.concatenate([flows[i][1] for i in keep]) # the amounts
        idx = np.repeat(np.arange(len(keep)), [len(flows[i][0]) for i in keep])
        # cash-flow-weighted time: cheap duration proxy for weights
        dur = np.bincount(idx, T * A) / np.bincount(idx, A) 
        # each day starts from the previous day's fit
        x0, rmse, n_used = _fit_day(T, A, idx, P, dur, x0)  
        zeros = nss(np.array(list(C.KEY_RATES.values())), x0)
        # make a row for the daily info + fit on the day
        rows.append({"date": date, **dict(zip(NSS_PARAMS, x0)), "fit_rmse_price": rmse,
                     "n_bonds_used": n_used, "n_bonds_quoted": len(keep), **dict(zip(KEY_COLS, zeros))})
    out = pd.DataFrame(rows)
    log.info("fitted %d daily curves, median price rmse %.3f", len(out), out["fit_rmse_price"].median())

    # just to order the cols cleanly 
    return out[["date"] + KEY_COLS + NSS_PARAMS + ["fit_rmse_price", "n_bonds_used", "n_bonds_quoted"]]
