"""Load the processed tables and build the month-end price panel (Deliverable 3, data step 5).

Month-end price rule: for each month take the last daily price within the final `window` trading days. A bond with
no price in that window carries its previous month-end price forward and is marked stale. Stale prices are only
used to value holdings; returns that touch one are excluded from the residual-variance estimate.
"""
import numpy as np
import pandas as pd

from data_cleaning import config as C

PANEL_COLS = ["asset_type", "sector", "clean_price", "accrued_interest", "dirty_price", "ytm_solved",
              "spread_bps", "bid_ask_spread_bps"]


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(bond_universe, prices_daily, yield_curve_daily) from data/processed."""
    d = C.PROCESSED_DIR
    return (pd.read_parquet(d / "bond_universe.parquet"),
            pd.read_parquet(d / "prices_daily.parquet"),
            pd.read_parquet(d / "yield_curve_daily.parquet").sort_values("date").reset_index(drop=True))


def month_ends(calendar, window: int = 5) -> pd.DataFrame:
    """Last trading day of each month in `calendar`, and the first day of its final-`window`-days window."""
    cal = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(calendar))))
    grp = pd.Series(cal, index=cal).groupby([cal.year, cal.month])
    return pd.DataFrame({"month_end": grp.max().values,
                         "window_start": grp.apply(lambda s: s.iloc[-window:].iloc[0]).values})


def month_end_prices(prices: pd.DataFrame, calendar, window: int = 5,
                     max_stale_months: int | None = None) -> pd.DataFrame:
    """Month-end price panel indexed by (cusip, month_end).

    Columns: PANEL_COLS plus obs_date (date of the underlying price) and stale (carried forward from an earlier
    month-end). Rows before a bond's first observation do not exist. max_stale_months=None carries forward without
    limit, as Deliverable 3 describes.
    """
    me = month_ends(calendar, window)
    month_end, win_start = me["month_end"].to_numpy(), me["window_start"].to_numpy()

    p = prices[(prices["date"] >= win_start.min()) & (prices["date"] <= month_end.max())]
    pos = np.searchsorted(month_end, p["date"].to_numpy())            # first month-end on/after the price date
    p = p.assign(month_end=month_end[pos], obs_date=p["date"])
    p = p[p["date"].to_numpy() >= win_start[pos]]
    obs = (p.sort_values("date").drop_duplicates(["cusip", "month_end"], keep="last")
             .set_index(["cusip", "month_end"])[PANEL_COLS + ["obs_date"]])

    grid = pd.MultiIndex.from_product([obs.index.get_level_values(0).unique(), me["month_end"]],
                                      names=["cusip", "month_end"])
    panel = obs.reindex(grid)
    stale = panel["obs_date"].isna()
    panel = panel.groupby(level=0).ffill()
    panel["stale"] = stale
    if max_stale_months is not None:
        run = (~stale).groupby(level=0).cumsum()
        age = stale.groupby([panel.index.get_level_values(0), run]).cumsum()
        panel = panel[age <= max_stale_months]
    return panel[panel["dirty_price"].notna()]


def curve_row(curve: pd.DataFrame, date) -> pd.Series:
    """Fitted curve (NSS parameters and key-rate zeros) on `date`."""
    row = curve[curve["date"] == pd.Timestamp(date)]
    if row.empty:
        raise ValueError(f"no fitted curve on {pd.Timestamp(date).date()}; curve ends {curve['date'].max().date()}")
    return row.iloc[0]
