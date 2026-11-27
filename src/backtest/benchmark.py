"""Monthly benchmark: the AGG holdings of the chosen asset types, market value weighted to sum to 1.

    b_i = agg_market_value_i / sum over selected bonds of agg_market_value

build_inputs reads agg_weight from the universe and renormalizes over the bonds priced at the as-of date, so data_for
hands it a universe cut to the month's benchmark with agg_weight overwritten.
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
        """data with the universe cut to the benchmark at asof and agg_weight set to its weights."""
        b = self.weights_at(asof)
        u = data.universe[data.universe.cusip.isin(b.index)].copy()
        u["agg_weight"] = u.cusip.map(b)
        return data._replace(universe=u)
