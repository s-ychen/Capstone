"""Sanity checks on data/processed/*.parquet:  uv run python -m data_cleaning.validate"""
import logging

import numpy as np
import pandas as pd

from . import config as C
from .curve import KEY_COLS

log = logging.getLogger("data_cleaning.validate")


def load(name):
    return pd.read_parquet(C.PROCESSED_DIR / f"{name}.parquet")


def main():
    u, p, c = load("bond_universe"), load("prices_daily"), load("yield_curve_daily")
    corp, tsy = load("corporate_daily"), load("treasury_daily")
    checks = {
        "unique CUSIPs in universe": u["cusip"].is_unique,
        "model weights sum to 1": np.isclose(u["model_weight"].sum(), 1.0),
        "one row per (cusip, date)": not p.duplicated(["cusip", "date"]).any(),
        "all prices positive": (p["clean_price"] > 0).all(),
        "dirty >= clean": (p["dirty_price"] >= p["clean_price"] - 1e-9).all(),
        "every price CUSIP is in the universe": p["cusip"].isin(u["cusip"]).all(),
        "curve: all 8 key rates on every date": c[KEY_COLS].notna().all().all(),
        "curve: one row per trading day": c["date"].is_unique,
        "treasury: CRSP yield vs solved ytm within 5 bp (median)": (tsy["yield"] - tsy["ytm_solved"]).abs().median() < 5e-4,
    }
    for k, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + k)

    print("\nCoverage by asset type (share of AGG weight with at least one price):")
    print(u.groupby("asset_type").apply(lambda g: pd.Series({"bonds": len(g), "with_prices": g["has_prices"].sum(),
          "weight_covered": g.loc[g["has_prices"], "agg_weight"].sum() / g["agg_weight"].sum()}), include_groups=False).round(3))
    print("\nDate range and rows per asset type:")
    print(p.groupby("asset_type").agg(rows=("date", "size"), first=("date", "min"), last=("date", "max"),
                                      days_per_bond=("date", lambda s: s.size / p.loc[s.index, "cusip"].nunique())).round(1))
    if corp["dk_price_vw"].notna().any():
        both = corp.dropna(subset=["dk_price_vw"])
        diff = (both["clean_price"] - both["dk_price_vw"]).abs()
        print(f"\nTRACE cleaning vs Dickerson volume-weighted price on {len(both):,} overlapping bond-days: "
              f"median |diff| {diff.median():.3f} pts, 95th pct {diff.quantile(.95):.3f}, within 0.25 pt: {(diff < .25).mean():.1%}")
        acc = (both["accrued_interest"] - both["dk_accrued_interest"]).abs()
        print(f"accrued interest vs Dickerson: median |diff| {acc.median():.4f}, 95th pct {acc.quantile(.95):.4f}")
    print("\nTreasury spread over the fitted curve (bps), should be centred near 0:")
    print(tsy["spread_bps"].describe()[["count", "mean", "50%", "std"]].round(1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
