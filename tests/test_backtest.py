import numpy as np
import pandas as pd
import pytest

from src.backtest import RunConfig, summarize
from src.backtest.loop import goal_rate


def test_goal_rate_on_track_asks_for_the_original_rate():
    g_m = 0.0004
    assert goal_rate(g_m, 12, 3, (1 + g_m) ** 3) == pytest.approx(g_m)


def test_goal_rate_clips_to_floor_and_cap():
    assert goal_rate(0.0004, 12, 1, 1.05) == 0.0          # far ahead: ease off to the floor
    assert goal_rate(0.0004, 12, 11, 0.95) == pytest.approx(0.0008)   # far behind: at most cap x g_m


def test_config_validates_and_labels():
    c = RunConfig(asset_types=("treasury", "corporate"), overrides={"H": 1})
    assert c.asset_types == ("corporate", "treasury")
    assert c.label() == "tracking-medium-k150-g50"
    assert c.settings().H == 1
    with pytest.raises(ValueError):
        RunConfig(asset_types=("equity",))
    with pytest.raises(ValueError):
        RunConfig(overrides={"not_a_field": 1})
    with pytest.raises(ValueError):           # above the information ratio limit
        RunConfig(g_annual=0.02, risk_tol="low").settings()


def test_config_from_toml(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[inputs]\nasset_types = ["treasury"]\nT = 2\n[overrides]\nH = 1\n[catch_up]\non = false\n'
                 'cap = 2.0\nfloor = 0.0\n')
    c = RunConfig.from_toml(f)
    assert c.asset_types == ("treasury",) and c.T == 2 and c.overrides == {"H": 1} and not c.catch_up["on"]


def test_summarize_annualizes_active_growth():
    n, a = 12, 0.0005
    m = pd.DataFrame(dict(ret_port=np.full(n, 0.01 + a), ret_bench=np.full(n, 0.01), turnover=0.1, cost=1e-4,
                          n_hold=150, te_fcst_ann=0.003, te_cap_ann=0.005, robust_active=a, goal_req=a,
                          shortfall=0.0, delta_scale=1.0, mip_skipped=False, share_fallback=0.0, seconds=0.1,
                          active_share=0.8))
    m["active"] = m.ret_port - m.ret_bench
    m["growth_active"] = ((1 + m.ret_port) / (1 + m.ret_bench)).cumprod()
    s = summarize(m, g=0.005)
    assert s["active_ann"] == pytest.approx(((1.0105 / 1.01) ** 12 - 1), rel=1e-9)
    assert s["goal_met"] and s["max_active_drawdown"] == 0 and s["hit_rate"] == 1
