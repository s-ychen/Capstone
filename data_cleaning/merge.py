"""Step 6: merge the per-source tables into the master parquet tables under data/processed/."""
import logging

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

# columns every asset type provides, in this order, for the combined prices_daily table
COMMON = ["cusip", "date", "asset_type", "sector", "source", "clean_price", "accrued_interest", "dirty_price",
          "yield", "ytm_solved", "spread_bps", "bid_ask_spread", "bid_ask_spread_bps", "par_volume", "dollar_volume",
          "trade_count", "years_to_maturity"]


def build_bond_universe(universe: pd.DataFrame, corp_terms: pd.DataFrame, tsy: pd.DataFrame,
                        prices: pd.DataFrame) -> pd.DataFrame:
    """One row per universe bond: AGG snapshot, bond terms (CRSP / FISD / AGG file), data coverage, and weights.

    agg_weight      = AGG market value share of the whole holdings file (the file's own Weight column is rounded to
                      2 dp, which zeroes most corporates, so it is recomputed from market value)
    model_weight    = agg_weight renormalised over bonds that have at least one price (sums to 1), NaN otherwise
    """
    u = universe.copy()
    u["agg_weight"] = u["agg_market_value"] / u["agg_market_value"].sum()

    t = tsy.drop_duplicates("cusip").set_index("cusip")
    ct = corp_terms.set_index("cusip")
    is_t, is_c = u["asset_type"] == "treasury", u["asset_type"] == "corporate"
    u["maturity_date"] = u["agg_maturity_date"]
    u["coupon_rate"] = u["agg_coupon_pct"]
    u.loc[is_t, "maturity_date"] = u.loc[is_t, "cusip"].map(t["maturity_date"]).fillna(u.loc[is_t, "maturity_date"])
    u.loc[is_t, "coupon_rate"] = u.loc[is_t, "cusip"].map(t["coupon_rate"]).fillna(u.loc[is_t, "coupon_rate"])
    u.loc[is_c, "maturity_date"] = u.loc[is_c, "cusip"].map(ct["maturity_date"])
    u.loc[is_c, "coupon_rate"] = u.loc[is_c, "cusip"].map(ct["coupon_rate"])
    u["coupon_frequency"] = u["asset_type"].map(C.COUPON_FREQ)
    u.loc[is_c, "coupon_frequency"] = u.loc[is_c, "cusip"].map(ct["coupon_frequency"])
    u["day_count"] = u["asset_type"].map(C.DAY_COUNT)
    u.loc[is_c, "day_count"] = u.loc[is_c, "cusip"].map(ct["day_count"])
    u["terms_source"] = np.select([is_t & u["cusip"].isin(t.index), is_c], ["crsp", u["cusip"].map(ct["terms_source"])], "agg_file")
    u["dated_date"] = pd.NaT
    u.loc[is_t, "dated_date"] = u.loc[is_t, "cusip"].map(t["dated_date"])
    u.loc[is_c, "dated_date"] = u.loc[is_c, "cusip"].map(ct["dated_date"])
    for c in ["coupon_type", "min_denomination", "amount_outstanding", "bond_type", "security_level", "is_callable", "is_144a"]:
        u[c] = u["cusip"].map(ct[c])
    u.loc[is_t, "min_denomination"] = 100.0

    cov = prices.groupby("cusip").agg(n_price_days=("date", "size"), first_price_date=("date", "min"), last_price_date=("date", "max"))
    u = u.merge(cov, on="cusip", how="left")
    u["n_price_days"] = u["n_price_days"].fillna(0).astype(int)
    u["has_prices"] = u["n_price_days"] > 0
    u["model_weight"] = np.where(u["has_prices"], u["agg_weight"] / u.loc[u["has_prices"], "agg_weight"].sum(), np.nan)
    cols = ["cusip", "name", "sector", "asset_type", "agg_sector", "is_tba", "holdings_asof_date",
            "agg_market_value", "agg_par_value", "agg_weight", "model_weight", "has_prices", "n_price_days",
            "first_price_date", "last_price_date", "agg_duration", "agg_mod_duration", "agg_ytm_pct",
            "maturity_date", "coupon_rate", "coupon_frequency", "coupon_type", "day_count", "dated_date", "terms_source",
            "min_denomination", "amount_outstanding", "bond_type", "security_level", "is_callable", "is_144a"]
    out = u[cols].sort_values(["asset_type", "cusip"]).reset_index(drop=True)
    cov_w = out.groupby("asset_type").apply(lambda g: g.loc[g["has_prices"], "agg_weight"].sum() / g["agg_weight"].sum(), include_groups=False)
    log.info("universe coverage by AGG weight: %s", cov_w.round(3).to_dict())
    return out


def build_prices_daily(tsy: pd.DataFrame, corp: pd.DataFrame, mbs: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Stack the three daily tables on the common columns. Only observed days are present (no forward filling):
    Treasuries are quoted every trading day, corporates and MBS only on days they traded."""
    sec = universe.set_index("cusip")["sector"]
    frames = []
    for d, asset_type in [(tsy, "treasury"), (corp, "corporate"), (mbs, "mbs")]:
        d = d.copy()
        d["asset_type"] = asset_type
        d["sector"] = d["cusip"].map(sec)
        for c in COMMON:
            if c not in d:
                d[c] = np.nan
        frames.append(d[COMMON])
    p = pd.concat(frames, ignore_index=True).sort_values(["asset_type", "cusip", "date"]).reset_index(drop=True)
    assert not p.duplicated(["cusip", "date"]).any(), "duplicate (cusip, date) rows in prices_daily"
    return p


def save(tables: dict[str, pd.DataFrame]) -> None:
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = C.PROCESSED_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("wrote %s: %d rows x %d cols", path.name, len(df), df.shape[1])
