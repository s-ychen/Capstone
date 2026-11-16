import numpy as np
import pandas as pd
import pytest

from data_cleaning import config as C
from src.params import (build_cashflows, build_inputs, dense_Q, estimate_sigma, expected_returns,
                        forecast_benchmark, drift_portfolio, load_data, pca, required_return, terminal_goal,
                        tracking_variance, analytics)
from src.params.forecast import benchmark_return, eligibility_exits
from src.params.krd import BP

ASOF = pd.Timestamp("2025-06-30")
FLAT = np.array([0.04, 0, 0, 0, 2.0, 10.0])      # NSS parameters of a flat 4% continuously compounded curve


def bonds(rows):
    """rows: (years to maturity, coupon %) -> cashflow table at ASOF."""
    terms = pd.DataFrame({
        "cusip": [f"B{i}" for i in range(len(rows))],
        "maturity_date": [ASOF + pd.Timedelta(days=round(y * 365.25)) for y, _ in rows],
        "coupon_rate": [c for _, c in rows], "coupon_frequency": 2})
    return build_cashflows(terms, ASOF - pd.Timedelta(days=400)), terms


# synthetic: repricing

def test_zero_coupon_krd_sits_at_its_key_rate():
    table, _ = bonds([(5, 0.0)])
    a = analytics(table, ASOF, FLAT, spread_bps=np.array([0.0]))
    j = list(C.KEY_RATES).index("5Y")
    assert a.K[0, j] == pytest.approx(5.0, abs=0.01)
    assert np.delete(a.K[0], j) == pytest.approx(0.0, abs=0.01)
    assert a.C[0] == pytest.approx(25.0, rel=0.01)          # continuous compounding: C = T^2


def test_krds_sum_to_duration_and_spread_duration_equals_it():
    table, _ = bonds([(2, 3.0), (7, 4.5), (20, 5.0)])
    a = analytics(table, ASOF, FLAT, spread_bps=np.array([50.0, 80.0, 120.0]))
    np.testing.assert_allclose(a.K.sum(axis=1), a.D, rtol=2e-3)
    np.testing.assert_allclose(a.D_spr, a.D, rtol=1e-9)     # additive spread: spread duration is duration
    np.testing.assert_allclose(a.dts, a.D * np.array([50, 80, 120]) * BP)


def test_missing_spread_is_solved_from_price():
    table, _ = bonds([(5, 4.0)])
    price = analytics(table, ASOF, FLAT, spread_bps=np.array([75.0])).price
    a = analytics(table, ASOF, FLAT, spread_bps=np.array([np.nan]), dirty=price)
    assert a.spread[0] == pytest.approx(75 * BP, abs=1e-8) and a.spread_solved[0]


def test_flat_curve_expected_return_is_carry_only():
    table, _ = bonds([(5, 0.0)])
    r = expected_returns(table, ASOF, FLAT, np.array([0.0]))
    assert r[0] == pytest.approx(np.exp(0.04 * 31 / 365.25) - 1, rel=2e-3)


# synthetic: covariance

def rate_changes(n=60, seed=0):
    rng = np.random.default_rng(seed)
    level, slope = rng.normal(0, 0.003, n), rng.normal(0, 0.001, n)
    tenors = np.array(list(C.KEY_RATES.values()))
    return level[:, None] + slope[:, None] * np.log(tenors) + rng.normal(0, 2e-4, (n, 8))


@pytest.mark.parametrize("method", ["sample", "ewma", "ledoit_wolf"])
def test_sigma_is_symmetric_psd(method):
    S = estimate_sigma(rate_changes(), method)
    assert S.shape == (8, 8)
    np.testing.assert_allclose(S, S.T)
    assert np.linalg.eigvalsh(S).min() > 0


def test_pca_orthonormal_and_level_first():
    S = estimate_sigma(rate_changes(), "sample")
    pc = pca(S, 3)
    np.testing.assert_allclose(pc.Gamma.T @ pc.Gamma, np.eye(3), atol=1e-10)
    assert np.all(np.diff(pc.Lambda) <= 0) and pc.explained.sum() <= 1
    assert np.all(pc.Gamma[:, 0] > 0)                       # PC1 is a parallel move


def test_factor_form_matches_dense_Q():
    rng = np.random.default_rng(1)
    F, psi, a = rng.normal(size=(50, 3)), rng.uniform(0.1, 1, 50), rng.normal(size=50)
    assert tracking_variance(F, psi, a) == pytest.approx(a @ dense_Q(F, psi) @ a)


# synthetic: forecasts

def test_benchmark_forecast_and_drift():
    b = np.array([0.5, 0.3, 0.2])
    r = np.array([0.004, 0.003, 0.002])
    exits = np.array([False, True, False])
    bh = forecast_benchmark(b, r, exits)
    assert bh.sum() == pytest.approx(1) and bh[1] == 0
    x = drift_portfolio(b, r, benchmark_return(b, r))      # holding the index drifts like the index
    assert x.sum() == pytest.approx(1)


def test_eligibility_and_required_return():
    mat = np.array(["2026-01-15", "2028-01-15"], dtype="datetime64[D]")
    assert eligibility_exits(mat, ASOF).tolist() == [True, False]
    G = terminal_goal(100.0, 0.05, 2)
    assert G == pytest.approx(110.25)
    assert (1 + required_return(G, 100.0, 0, 24)) ** 24 == pytest.approx(1.1025)


# real data (skipped when data/processed is absent)

@pytest.fixture(scope="module")
def real():
    if not (C.PROCESSED_DIR / "yield_curve_daily.parquet").exists():
        pytest.skip("no processed data")
    data = load_data()
    return data, build_inputs(ASOF, 1e6, H=2, data=data)


def test_treasury_duration_matches_crsp(real):
    data, P = real
    t = pd.read_parquet(C.PROCESSED_DIR / "treasury_daily.parquet")
    crsp = t[t["date"] == ASOF].set_index("cusip")["duration"]
    m = P.asset_type == "treasury"
    ours = pd.Series(P.periods[0].D[m], index=P.cusips[m])
    assert ours.corr(crsp.reindex(ours.index)) > 0.999
    assert (ours / crsp.reindex(ours.index) - 1).abs().median() < 0.01


def test_covariance_and_curve_factors(real):
    _, P = real
    m = P.model
    assert m.explained.sum() > 0.9 and m.explained[0] > 0.6
    assert np.all(m.Gamma[:, 0] > 0)
    assert (m.psi > 0).all() and np.isfinite(P.c).all()


def test_period_inputs_are_consistent(real):
    _, P = real
    n = len(P.cusips)
    for q in P.periods:
        assert q.b_hat.sum() == pytest.approx(1) and (q.b_hat >= 0).all()
        assert q.F.shape == (n, 3) and np.isfinite(q.F).all()
        assert np.isfinite([q.D, q.D_spr, q.C, q.mu_hat, q.ell]).all()
    assert P.periods[0].drift_gross is None and P.periods[1].drift_gross.shape == (n,)
    # aging shortens duration by roughly a month for the benchmark
    d0, d1 = (q.b_hat @ q.D for q in P.periods)
    assert 0 < d0 - d1 < 0.2
