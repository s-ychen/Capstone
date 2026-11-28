"""Monthly benchmark: the AGG holdings of the chosen asset types, market value weighted to sum to 1.

    b_i = agg_market_value_i / sum over selected bonds of agg_market_value

build_inputs takes agg_weight from the same month's holdings file and renormalizes over the bonds of the universe
priced at the as-of date, so data_for only cuts the universe to the month's benchmark (agg_weight is proportional to
market value within a file, so the result is these weights).
"""
import pandas as pd

from data_cleaning import config as C


class Benchmark:
    def __init__(self, asset_types, path=C.PROCESSED_DIR / "agg_holdings_monthly.parquet"):
        h = pd.read_parquet(path, columns=["holdings_date", "cusip", "asset_type", "agg_market_value"])
        h = h[h.asset_type.isin(asset_types)]
        h["ym"] = h.holdings_date.dt.to_period("M")
        self.asset_types = tuple(asset_types)
        self._mv = h.groupby(["ym", "cusip"]).agg_market_value.sum()

    def weights_at(self, asof) -> pd.Series:
        """Weights of asof's month, indexed by cusip, summing to 1."""
        mv = self._mv.loc[pd.Timestamp(asof).to_period("M")]
        return mv / mv.sum()

    def data_for(self, data, asof):
        """data with the universe cut to the bonds of the benchmark at asof."""
        b = self.weights_at(asof)
        return data._replace(universe=data.universe[data.universe.cusip.isin(b.index)])
