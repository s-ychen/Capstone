"""Sanity checks on the outputted parquet files:  uv run python -m data_cleaning.validate"""
import logging

import numpy as np
import pandas as pd

from . import config as C
from .treasury_curve import KEY_COLS

log = logging.getLogger("data_cleaning.validate")


def main():
    u, p, c, h, d = (pd.read_parquet(C.PROCESSED_DIR / f"{name}.parquet") for name in
                     ["bond_universe", "prices_monthly", "yield_curve_daily", "agg_holdings_monthly", "prices_daily"])
    start = pd.Timestamp(C.OUTPUT_START)
    months = pd.DatetimeIndex(h["month"].unique()).sort_values()
    last_traded = (p[p["price_source"].isin(["traded_window", "traded_month"])]
                   .merge(d[["cusip", "date", "clean_price"]], left_on=["cusip", "price_date"], right_on=["cusip", "date"],
                          how="left", suffixes=("", "_daily")))
    held = pd.MultiIndex.from_arrays([h["cusip"], h["month"]])
    w = h.groupby("month")["model_weight"].sum()
    tsy = p[p["asset_type"] == "treasury"]
    checks = {
        "unique CUSIPs in universe": u["cusip"].is_unique,
        "holdings: month m uses the file from the end of month m-1 (no look-ahead)":
            (h["holdings_date"].dt.to_period("M") == h["month"].dt.to_period("M") - 1).all(),
        "holdings: one row per (month, cusip)": not h.duplicated(["month", "cusip"]).any(),
        "model weights sum to 1 in every month with prices": np.allclose(w[w > 0], 1.0),
        "prices_monthly: one row per (cusip, month)": not p.duplicated(["cusip", "month"]).any(),
        "prices_monthly: only months the bond was held": pd.MultiIndex.from_arrays([p["cusip"], p["month"]]).isin(held).all(),
        "prices_monthly: price date not after month-end": (p["price_date"] <= p["month_end_date"]).all(),
        "all prices positive": (p["clean_price"] > 0).all(),
        "dirty >= clean": (p["dirty_price"] >= p["clean_price"] - 1e-9).all(),
        "curve: all 8 key rates on every date": c[KEY_COLS].notna().all().all(),
        "curve: one row per trading day": c["date"].is_unique,
        "output starts at OUTPUT_START (holdings, prices, curve)":
            min(h["month"].min(), p["month"].min(), d["date"].min(), c["date"].min()) >= start,
        "holdings: every month from OUTPUT_START present (no gaps)":
            len(months) == len(pd.date_range(months.min(), months.max(), freq="MS")),
        "prices_daily: one row per (cusip, date)": not d.duplicated(["cusip", "date"]).any(),
        "prices_daily: only months the bond was held":
            pd.MultiIndex.from_arrays([d["cusip"], d["month"]]).isin(held).all(),
        "prices_daily: month-end traded prices match the daily row": np.allclose(
            last_traded["clean_price"], last_traded["clean_price_daily"], equal_nan=False),
        "treasury: CRSP yield vs solved ytm within 5 bp (median)": (tsy["yield"] - tsy["ytm_solved"]).abs().median() < 5e-4,
    }
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)

    print("\nShare of AGG benchmark weight with a month-end price, by year (observed or filled):")
    h["year"] = h["month"].dt.year
    priced_w = h["agg_weight"].where(h["price_status"] != "none", 0.0)
    print((priced_w.groupby([h["year"], h["asset_type"]]).sum() / h.groupby(["year", "asset_type"])["agg_weight"].sum())
          .unstack().round(3))
    print("\nprices_monthly by asset type:")
    print(p.groupby("asset_type").agg(rows=("month", "size"), bonds=("cusip", "nunique"), first=("month", "min"),
                                      last=("month", "max")))
    print("\nprices_daily by asset type:")
    print(d.groupby("asset_type").agg(rows=("date", "size"), bonds=("cusip", "nunique"), first=("date", "min"),
                                      last=("date", "max"), ytm_solved=("ytm_solved", lambda x: x.notna().mean())))
    print("\nprice_source share by asset type:")
    print(pd.crosstab(p["asset_type"], p["price_source"], normalize="index").round(3))
    print("\nTreasury spread over the fitted curve (bps), should be centred near 0:")
    print(tsy["spread_bps"].describe()[["count", "mean", "50%", "std"]].round(1))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
