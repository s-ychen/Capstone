"""User inputs to Settings, and checks on the horizon.

The user picks a return goal g (annualized active), a horizon T (years) from a start month, a risk tolerance
(low | medium | high), k_max and the wealth W, plus a mode:
  * "tracking": track the benchmark as closely as possible while aiming for g (soft goal, shortfall penalized)
  * "return": earn as much robust active return as the risk tolerance allows, g is only reported against
Risk tolerance sets three things together: the ex-ante tracking error cap, eps2 (how much confidence the return goal
needs, larger = more cautious) and tol_scale (multiplier on the duration, key rate, DTS and convexity tolerances).
Low tolerance is tight on all three. Every value here is a placeholder until calibration.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .model import Settings

RISK_TOL = {
    "low": dict(te_cap=0.0025, eps2=0.50, tol_scale=0.5),
    "medium": dict(te_cap=0.0050, eps2=0.25, tol_scale=1.0),
    "high": dict(te_cap=0.0100, eps2=0.10, tol_scale=2.0),
}
MODES = {
    # gamma: weight on tracking variance vs trading cost. In "return" mode it is only a tie-breaker.
    # goal_penalty: cost per unit of monthly goal shortfall, large so the goal gives way only when it must.
    # cost_weight: "return" amortizes trading cost over a year, else one month of return never pays for a trade
    "tracking": dict(gamma=3e3, goal_penalty=100.0, cost_weight=1.0),
    "return": dict(gamma=10.0, goal_penalty=None, cost_weight=1 / 12),
}
DELTAS = ("delta_D", "delta_K", "delta_DTS", "delta_C")
WARMUP = 7      # month-ends of history needed before the start (6 monthly curve changes for Sigma and psi)


def make_settings(g: float, risk_tol: str, k_max: int, mode: str = "tracking", **overrides) -> Settings:
    """Settings from the user inputs. overrides set any Settings field after the presets (advanced use), e.g.
    te_cap=0.004, delta_D=0.3 (taken as given, not scaled), H=1, method="heuristic"."""
    if risk_tol not in RISK_TOL:
        raise ValueError(f"risk_tol must be one of {list(RISK_TOL)}, got {risk_tol!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
    if k_max < 1:
        raise ValueError("k_max must be at least 1")
    r, base = RISK_TOL[risk_tol], Settings()
    kw = dict(
        MODES[mode],
        objective=mode,
        k_max=int(k_max),
        te_cap=r["te_cap"],
        eps2=r["eps2"],
        # the model's goal is robust active return >= -ret_slack per month, so the slack is minus the monthly goal
        ret_slack=-((1 + g) ** (1 / 12) - 1),
        **{k: getattr(base, k) * r["tol_scale"] for k in DELTAS},
    )
    kw.update(overrides)
    H = kw.get("H", base.H)
    if "omega" not in overrides:
        kw["omega"] = (1 / H,) * H
    return replace(base, **kw)


def horizon_dates(ends, start, T: float) -> tuple[list, pd.Timestamp]:
    """Rebalance month-ends (N = 12 T) and the month-end the last return is measured to. start can be any date,
    it snaps to the month-end of its month."""
    ends = pd.DatetimeIndex(ends).sort_values()
    N = round(12 * T)
    if N < 1:
        raise ValueError("T must be at least one month")
    hit = np.flatnonzero(ends.to_period("M") == pd.Timestamp(start).to_period("M"))
    if not len(hit):
        raise ValueError(f"no month-end in {pd.Timestamp(start):%Y-%m}, data covers {ends[0].date()} .. {ends[-1].date()}")
    i0 = int(hit[0])
    if i0 < WARMUP:
        raise ValueError(f"start needs {WARMUP} month-ends of history, earliest start is {ends[WARMUP].date()}")
    if i0 + N >= len(ends):
        raise ValueError(f"horizon runs past the data, latest start for T = {T} is {ends[len(ends) - 1 - N].date()}")
    return list(ends[i0 : i0 + N]), ends[i0 + N]
