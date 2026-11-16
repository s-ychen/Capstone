"""Assemble every optimizer input of the D3 multi-period model for one rebalance date.

    uv run python -m src.params --asof 2025-06-30 --H 2

Conventions: dates are month-ends, covariances and returns are monthly, durations are in years per decimal yield
change, weights b are the AGG holdings (Oct 2026 file) renormalised over the bonds priced at the as-of date.
"""
import argparse
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import NamedTuple

import numpy as np
import pandas as pd

from .covariance import TRADING_DAYS_PER_MONTH, key_rate_changes
from .factor_model import FactorModel, fit_factor_model, loadings, theta_factor
from .forecast import (benchmark_return, eligibility_exits, expected_returns, forecast_benchmark, step_date)
from .frictions import min_lot_weight, required_return, trading_cost
from .krd import BP, analytics, build_cashflows, nss_params
from .prices import curve_row, load_processed, month_end_prices
from .residual import estimate_psi, residual_returns

log = logging.getLogger(__name__)


class Data(NamedTuple):
    universe: pd.DataFrame
    prices: pd.DataFrame
    curve: pd.DataFrame
    panel: pd.DataFrame     # month-end price panel


@lru_cache(maxsize=1)
def load_data() -> Data:
    universe, prices, curve = load_processed()
    return Data(universe, prices, curve, month_end_prices(prices, curve["date"]))


@dataclass
class PeriodInputs:
    """Inputs for planning date t+h. Arrays are over the bonds in Params.cusips."""
    h: int
    date: pd.Timestamp
    b_hat: np.ndarray           # forecast benchmark weights
    mu_hat: np.ndarray          # forecast one-month returns from this date (return-goal constraint)
    K: np.ndarray               # n x 8 key-rate durations at this date (aged bonds, constant curve)
    D: np.ndarray               # modified duration
    D_spr: np.ndarray           # spread duration (equal to D when spread enters additively, see README note in main)
    dts: np.ndarray             # duration times spread, the alternative credit-risk measure from D2 eq. 6
    C: np.ndarray               # convexity
    F: np.ndarray               # n x p, Q_hat = F F' + diag(psi)
    exits: np.ndarray           # E_{t+h}: bool, bonds that must have z = 0 (under one year to maturity)
    ell: np.ndarray             # minimum lot as a weight, at forecast wealth
    r_req: float                # required per-period return at forecast wealth (NaN if no goal given)
    wealth: float               # forecast wealth W_{t+h}
    drift_gross: np.ndarray | None  # 1 + r_hat over the previous period, diag(R_hat) of eq. 7 (None at h=0)
    r_b: float                  # forecast benchmark return over the previous period (NaN at h=0)


@dataclass
class Params:
    asof: pd.Timestamp
    cusips: np.ndarray
    sector: np.ndarray
    asset_type: np.ndarray
    b: np.ndarray               # b_t
    model: FactorModel          # Sigma, Gamma_p, Lambda_p, explained variance, psi, T_s
    c: np.ndarray               # proportional trading costs
    theta_F: np.ndarray         # Theta = Q / T_s in factor form
    theta_psi: np.ndarray
    stale: np.ndarray           # month-end price is carried forward
    spread_solved: np.ndarray   # spread solved from price (e.g. MBS) rather than read from prices_daily
    n_resid_obs: np.ndarray     # returns behind each psi
    periods: list[PeriodInputs]


def _snap(curve: pd.DataFrame, panel: pd.DataFrame, asof) -> pd.Timestamp:
    ends = panel.index.get_level_values("month_end").unique().sort_values()
    ok = ends[ends <= pd.Timestamp(asof)]
    if ok.empty:
        raise ValueError(f"no month-end on or before {asof}; data starts {ends.min().date()}")
    if ok[-1] != pd.Timestamp(asof):
        log.warning("asof %s is not a month-end; using %s", pd.Timestamp(asof).date(), ok[-1].date())
    if ok[-1] > curve["date"].max():
        raise ValueError(f"curve ends {curve['date'].max().date()}")
    return ok[-1]


def build_inputs(asof, W: float, H: int = 2, G: float | None = None, t_index: int = 0, N: int | None = None,
                 sigma_method: str = "ledoit_wolf", p: int = 3, freq: str = "M", window: int | None = None,
                 halflife: float = 12, min_obs: int = 6, data: Data | None = None) -> Params:
    """Estimate Q_t (factor form), K, D, D_spr, C, mu_hat, b_hat, exits, c, lots, r_req and Theta at `asof`.

    W: current wealth W_t in dollars (sets the minimum-lot weights). G: terminal wealth goal (see frictions.terminal_goal); with t_index (months elapsed) and
    N (total rebalances) it gives r_req, otherwise r_req is NaN. window limits the months behind Sigma and psi.
    """
    data = data or load_data()
    universe, prices, curve, panel = data
    asof = _snap(curve, panel, asof)
    cv = curve_row(curve, asof)
    params = nss_params(cv)

    now = panel.xs(asof, level="month_end")
    terms = universe.set_index("cusip").join(now[["dirty_price", "spread_bps", "stale"]], how="inner")
    terms = terms[(terms["maturity_date"] > asof) & terms["coupon_rate"].notna()].reset_index()
    first_end = panel.index.get_level_values("month_end").min()
    earliest = pd.Timestamp(first_end) - pd.Timedelta(days=400)

    table = build_cashflows(terms, earliest)
    a0 = analytics(table, asof, params, terms["spread_bps"].to_numpy(), terms["dirty_price"].to_numpy())
    keep = np.isfinite(a0.D)
    if not keep.all():
        log.warning("dropping %d bonds without a valid price or spread at %s", (~keep).sum(), asof.date())
        terms = terms[keep].reset_index(drop=True)
        table = build_cashflows(terms, earliest)
        a0 = analytics(table, asof, params, terms["spread_bps"].to_numpy(), terms["dirty_price"].to_numpy())

    cusips, sector = terms["cusip"].to_numpy(), terms["sector"].to_numpy()
    maturity = terms["maturity_date"].to_numpy().astype("datetime64[D]")
    b = terms["agg_weight"].to_numpy(float)
    b = b / b.sum()

    # Q_t: key-rate covariance from curve changes up to asof, residual variance from month-end returns
    sub = panel[panel.index.get_level_values("cusip").isin(cusips)]
    E, _ = residual_returns(table, sub, curve, asof, window)
    psi, n_obs = estimate_psi(E, sector, min_obs)
    dY = key_rate_changes(curve, end=asof, freq=freq, window=window)
    scale = TRADING_DAYS_PER_MONTH if freq == "D" else 1.0
    fm = fit_factor_model(dY, psi, sigma_method, p, halflife, scale)

    c = trading_cost(prices, cusips, sector, asof)
    spread_bps = a0.spread / BP
    min_denom = terms["min_denomination"].to_numpy(float)

    periods, b_h, W_h, prev = [], b, W, None
    for h in range(H):
        date = step_date(asof, h)
        a = a0 if h == 0 else analytics(table, date, params, spread_bps)
        r = expected_returns(table, date, params, a0.spread)
        exits = eligibility_exits(maturity, date)
        if prev is not None:
            r_prev, b_prev = prev
            b_h = forecast_benchmark(b_prev, r_prev, exits)
            r_b, drift = benchmark_return(b_prev, r_prev), 1 + np.nan_to_num(r_prev)
        else:
            r_b, drift = np.nan, None
        dirty = terms["dirty_price"].to_numpy(float) if h == 0 else np.where(a.price > 0, a.price, terms["dirty_price"])
        K = np.nan_to_num(a.K)
        periods.append(PeriodInputs(
            h=h, date=date, b_hat=b_h, mu_hat=np.nan_to_num(r), K=K, D=np.nan_to_num(a.D),
            D_spr=np.nan_to_num(a.D_spr), dts=np.nan_to_num(a.dts), C=np.nan_to_num(a.C), F=loadings(K, fm),
            exits=exits, ell=min_lot_weight(min_denom, dirty, sector, W_h),
            r_req=required_return(G, W_h, t_index + h, N) if G is not None and N is not None else np.nan,
            wealth=W_h, drift_gross=drift, r_b=r_b))
        prev = (r, b_h)
        W_h *= 1 + benchmark_return(b_h, r)

    th_F, th_psi = theta_factor(loadings(np.nan_to_num(a0.K), fm), psi, fm.n_obs)
    return Params(asof, cusips, sector, terms["asset_type"].to_numpy(), b, fm, c, th_F, th_psi,
                  terms["stale"].to_numpy(bool), a0.spread_solved, n_obs, periods)
