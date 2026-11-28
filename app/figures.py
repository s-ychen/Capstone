"""Interactive (plotly) versions of the src.analytics plots, one per Plot in PLOTS, same series and colors."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.analytics.metrics import TE_WINDOW_D, TE_WINDOW_M, Metrics
from src.analytics.plots import BENCH, OTHER, PC, PORT, SECTOR, SERIES, YLABEL

REF = "#9a9a9a"     # reference lines (goal, cap, bounds), readable on light and dark surfaces
LW = 2


def _fig(name: str, hover: str = "x unified") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(height=340, margin=dict(l=8, r=8, t=8, b=8), hovermode=hover, bargap=0.18, barcornerradius=3,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      yaxis_title=YLABEL[name], xaxis=dict(showgrid=False))
    return fig


def _line(fig, x, y, name, color, fmt, dash=None, shape="linear", markers=False):
    fig.add_scatter(x=x, y=y, name=name, mode="lines+markers" if markers else "lines", line_shape=shape,
                    line=dict(color=color, width=LW if dash is None else 1.5, dash=dash), marker=dict(size=8),
                    hovertemplate=f"%{{y:{fmt}}}<extra>{name}</extra>")


def _bar(fig, x, y, name, color, fmt):
    fig.add_bar(x=x, y=y, name=name, marker=dict(color=color, line=dict(width=0)),
                hovertemplate=f"%{{y:{fmt}}}<extra>{name}</extra>")


def _label(k: str) -> str:
    return k.replace("_", " ")


def _sectors(mx: Metrics) -> list:
    present = mx.sectors.sector.unique()
    return [k for k in SECTOR if k in present] + sorted(set(present) - set(SECTOR))


def growth(mx):
    d, f = mx.daily, _fig("growth")
    _line(f, d.index, d.wealth_port, "Portfolio", PORT, ".4f")
    _line(f, d.index, d.wealth_bench, "Benchmark", BENCH, ".4f")
    return f


def relative(mx):
    d, f = mx.daily, _fig("relative")
    _line(f, d.index, 1e4 * d.rel, "Portfolio vs benchmark", PORT, ".1f")
    _line(f, d.index, 1e4 * d.goal, "Goal path", REF, ".1f", dash="dash")
    return f


def drawdown(mx):
    d, f = mx.daily, _fig("drawdown")
    f.add_scatter(x=d.index, y=1e4 * d.drawdown, name="Active drawdown", fill="tozeroy", showlegend=False,
                  line=dict(color=PORT, width=1.5), hovertemplate="%{y:.1f} bp<extra></extra>")
    return f


def monthly_active(mx):
    m, f = mx.monthly, _fig("monthly_active")
    _bar(f, m.index, 1e4 * m.active, "Realised active return (net)", PORT, ".1f")
    _line(f, m.index, 1e4 * m.goal_req, "Required (catch-up goal)", REF, ".2f", dash="dash", shape="hvh")
    return f


def te(mx):
    m, d, f = mx.monthly, mx.daily, _fig("te")
    _line(f, m.index, 1e4 * m.te_fcst_ann, "Ex-ante (optimizer)", SERIES[1], ".1f", shape="hv")
    _line(f, d.index, 1e4 * d.te_roll, f"Realised, daily, rolling {TE_WINDOW_D} days", PORT, ".1f")
    _line(f, mx.te_roll_m.index, 1e4 * mx.te_roll_m, f"Realised, monthly, rolling {TE_WINDOW_M} months",
          SERIES[2], ".1f")
    if m.te_cap_ann.notna().any():
        _line(f, m.index, 1e4 * m.te_cap_ann, "Cap", REF, ".0f", dash="dash", shape="hv")
    f.update_yaxes(rangemode="tozero")
    return f


def te_ex_ante_source(mx):
    r, m, f = mx.risk, mx.monthly, _fig("te_ex_ante_source")
    groups = [("Level (PC1)", ["pc1"], PC["pc1"]), ("Slope (PC2)", ["pc2"], PC["pc2"]),
              ("Curvature (PC3)", ["pc3"], PC["pc3"]),
              ("Credit factors", [c for c in r if c.startswith("credit_")], SERIES[1]),
              ("Residual (issuer)", [c for c in r if c.startswith("resid_")], SERIES[6])]
    te_ann = 1e4 * m.te_fcst_ann.reindex(r.index)
    tot = r.sum(axis=1)
    for name, cols, color in groups:
        _bar(f, r.index, te_ann * r[cols].sum(axis=1) / tot, name, color, ".1f")
    f.update_layout(barmode="stack")
    return f


def te_realised_source(mx):
    s, f = mx.real_src, _fig("te_realised_source")
    for c in s:
        _bar(f, s.index, 1e4 * s[c], _label(c), SECTOR.get(c, OTHER), ".1f")
    _line(f, mx.te_roll_m.index, 1e4 * mx.te_roll_m, "Realised TE (monthly)", REF, ".1f", dash="dash")
    f.update_layout(barmode="relative")
    return f


def sector_weights(mx):
    w, f = mx.sectors.pivot(index="date", columns="sector", values="w_port"), _fig("sector_weights")
    for k in _sectors(mx):
        f.add_scatter(x=w.index, y=100 * w[k], name=_label(k), stackgroup="w", line_shape="hv",
                      line=dict(color=SECTOR.get(k, OTHER), width=0.5), fillcolor=SECTOR.get(k, OTHER),
                      hovertemplate=f"%{{y:.1f}}%<extra>{_label(k)}</extra>")
    f.update_yaxes(range=[0, 100])
    return f


def active_sector(mx):
    w, f = mx.sectors.pivot(index="date", columns="sector", values="w_active"), _fig("active_sector")
    for k in _sectors(mx):
        _line(f, w.index, 100 * w[k], _label(k), SECTOR.get(k, OTHER), "+.2f")
    f.add_hline(0, line=dict(color=REF, width=1))
    return f


def active_duration(mx):
    m, f = mx.monthly, _fig("active_duration")
    dD = mx.run.meta["settings"]["delta_D"]
    _line(f, m.index, m.act_D, "Active duration", PORT, "+.3f", markers=True)
    for v in (-dD, dD):
        _line(f, [m.index[0], m.index[-1]], [v, v], "Bound ±δ_D", REF, "+.2f", dash="dash")
    f.data[-1].showlegend = False
    f.add_hline(0, line=dict(color=REF, width=1))
    f.update_yaxes(range=[-1.5 * dD, 1.5 * dD])
    return f


def active_krd(mx):
    m, f = mx.monthly, _fig("active_krd", hover="closest")
    cols = [c for c in m if c.startswith("act_krd_")]
    z = m[cols].to_numpy().T
    lim = float(np.nanmax(np.abs(z))) or 1.0
    f.add_heatmap(x=m.index, y=[c[8:] for c in cols], z=z, zmid=0, zmin=-lim, zmax=lim, xgap=1, ygap=1,
                  colorscale=[[0, "#ec4647"], [0.5, "#c8c8c8"], [1, "#257de0"]],
                  colorbar=dict(title=dict(text="years", side="right"), thickness=12, outlinewidth=0),
                  hovertemplate="%{x|%Y-%m}, %{y}: %{z:+.3f} yrs<extra></extra>")
    return f


def holdings(mx):
    c, m, f = mx.conc, mx.monthly, _fig("holdings")
    _line(f, c.index, c.eff_n, "Effective holdings (1 / HHI)", PORT, ".0f", markers=True)
    _line(f, m.index, m.n_hold, "Names held", BENCH, ".0f", dash="dash")
    f.update_yaxes(rangemode="tozero")
    return f


def concentration(mx):
    c, f = mx.conc, _fig("concentration")
    for (col, name), color in zip([("top1", "Largest weight"), ("top10", "Top 10 weights"),
                                   ("coverage", "Benchmark weight in held names")], SERIES):
        _line(f, c.index, 100 * c[col], name, color, ".1f")
    _line(f, mx.monthly.index, 100 * mx.monthly.active_share, "Active share", REF, ".1f", dash="dash")
    f.update_yaxes(range=[0, 100])
    return f


def turnover(mx):
    m, f = mx.monthly.iloc[1:], _fig("turnover")
    _bar(f, m.index, 100 * m.turnover, "Turnover", PORT, ".1f")
    f.update_layout(showlegend=False)
    return f


def cost(mx):
    m, f = mx.monthly, _fig("cost")
    trade = 1 - (1 - m.cost) / (1 - m.exit_cost)
    _bar(f, m.index, 1e4 * trade, "Rebalance trades", PORT, ".2f")
    _bar(f, m.index, 1e4 * m.exit_cost, "Forced sales (index exits)", SERIES[1], ".2f")
    f.update_layout(barmode="stack")
    return f


FIGURES = {f.__name__: f for f in (growth, relative, drawdown, monthly_active, te, te_ex_ante_source,
                                   te_realised_source, sector_weights, active_sector, active_duration, active_krd,
                                   holdings, concentration, turnover, cost)}
