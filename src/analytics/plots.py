"""One figure per metric. Each Plot carries its explanation and formula for the report."""
from dataclasses import dataclass
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .metrics import TE_WINDOW_D, TE_WINDOW_M, Metrics

# theme (light), one place
SURFACE, INK, MUTED, GRID, AXIS = "#ffffff", "#191919", "#666666", "#e7e7e7", "#c2c2c2"
PORT, BENCH = "#257de0", "#8a8a8a"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
SECTOR = dict(zip(["treasury", "industrial", "financial", "utility", "mbs_passthrough"], SERIES))
OTHER = "#8a8a8a"
PC = {"pc1": "#1b64be", "pc2": "#519ced", "pc3": "#9dc8f7"}
DIVERGING = LinearSegmentedColormap.from_list("div", ["#ec4647", "#efefef", "#257de0"])
LW = 1.6

plt.rcParams.update({"font.size": 9,
                     "axes.titleweight": "normal", "axes.labelcolor": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False,
                     "legend.fontsize": 8.5})


@dataclass
class Plot:
    name: str
    title: str
    explain: str
    formula: str            # LaTeX, empty if none
    draw: Callable[[plt.Axes, Metrics], None]


def _axes(title, ylabel):
    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.tick_params(axis="x", which="major", length=4, color=AXIS)
    ax.tick_params(axis="x", which="minor", length=2, color=AXIS)
    loc = mdates.MonthLocator(bymonth=[1, 4, 7, 10])      # labelled quarterly, year shown at January
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, show_offset=False))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())     # tick every month
    for k, sp in ax.spines.items():
        sp.set_visible(k == "bottom")
        sp.set_color(AXIS)
    return fig, ax


def _zero(ax):
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)


def _bars(ax, x, y, bottom=None, **kw):
    ax.bar(x, y, width=22, bottom=bottom, edgecolor=SURFACE, linewidth=0.8, **kw)


# ---- wealth (daily) ----

def growth(ax, mx):
    d = mx.daily
    ax.plot(d.index, d.wealth_port, color=PORT, lw=LW, label="Portfolio")
    ax.plot(d.index, d.wealth_bench, color=BENCH, lw=LW, label="Benchmark")
    ax.legend(loc="upper right")


def relative(ax, mx):
    d = mx.daily
    _zero(ax)
    ax.plot(d.index, 1e4 * d.goal, color=INK, lw=1.1, ls="--", label="Goal path")
    ax.plot(d.index, 1e4 * d.rel, color=PORT, lw=LW, label="Portfolio vs benchmark")
    ax.legend(loc="upper left")


def drawdown(ax, mx):
    d = mx.daily
    ax.fill_between(d.index, 1e4 * d.drawdown, 0, color=PORT, alpha=0.18, lw=0)
    ax.plot(d.index, 1e4 * d.drawdown, color=PORT, lw=1.2)
    _zero(ax)


# ---- returns ----

def monthly_active(ax, mx):
    m = mx.monthly
    _zero(ax)
    _bars(ax, m.index, 1e4 * m.active, color=PORT, label="Realised active return (net)")
    ax.step(m.index, 1e4 * m.goal_req, where="mid", color=INK, lw=1.1, ls="--", label="Required (catch-up goal)")
    ax.legend(loc="upper left")


# ---- tracking error ----

def te(ax, mx):
    m, d = mx.monthly, mx.daily
    ax.step(m.index, 1e4 * m.te_fcst_ann, where="post", color=SERIES[1], lw=LW, label="Ex-ante (optimizer)")
    ax.plot(d.index, 1e4 * d.te_roll, color=PORT, lw=LW, label=f"Realised, daily, rolling {TE_WINDOW_D} days")
    ax.plot(mx.te_roll_m.index, 1e4 * mx.te_roll_m, color=SERIES[2], lw=LW,
            label=f"Realised, monthly, rolling {TE_WINDOW_M} months")
    if m.te_cap_ann.notna().any():
        ax.step(m.index, 1e4 * m.te_cap_ann, where="post", color=INK, lw=1.1, ls="--", label="Cap")
    ax.set_ylim(0, 1.25 * ax.get_ylim()[1])
    ax.legend(loc="upper left", ncol=2)


def te_ex_ante_src(ax, mx):
    r, m = mx.risk, mx.monthly
    groups = [("Level (PC1)", ["pc1"], PC["pc1"]), ("Slope (PC2)", ["pc2"], PC["pc2"]),
              ("Curvature (PC3)", ["pc3"], PC["pc3"]),
              ("Credit factors", [c for c in r if c.startswith("credit_")], SERIES[1]),
              ("Residual (issuer)", [c for c in r if c.startswith("resid_")], SERIES[6])]
    tot = r.sum(axis=1)
    te_ann = 1e4 * m.te_fcst_ann.reindex(r.index)
    bottom = np.zeros(len(r))
    for name, cols, col in groups:
        h = (te_ann * r[cols].sum(axis=1) / tot).to_numpy()
        _bars(ax, r.index, h, bottom=bottom, color=col, label=name)
        bottom += h
    ax.set_ylim(0, 1.3 * bottom.max())
    ax.legend(loc="upper left", ncol=5, handlelength=1.0, columnspacing=1.2)


def te_real_src(ax, mx):
    s = mx.real_src
    pos, neg = np.zeros(len(s)), np.zeros(len(s))
    for c in s:
        v = s[c].fillna(0).to_numpy()
        _bars(ax, s.index, 1e4 * v, bottom=1e4 * np.where(v >= 0, pos, neg), color=SECTOR.get(c, OTHER),
              label=c.replace("_", " "))
        pos += np.where(v >= 0, v, 0)
        neg += np.where(v < 0, v, 0)
    ax.plot(mx.te_roll_m.index, 1e4 * mx.te_roll_m, color=INK, lw=1.1, ls="--", label="Realised TE (monthly)")
    _zero(ax)
    ax.legend(loc="upper left", ncol=3, handlelength=1.0, columnspacing=1.2)


# ---- positioning ----

def _sectors(mx):
    present = mx.sectors.sector.unique()
    return [k for k in SECTOR if k in present] + sorted(set(present) - set(SECTOR))


def sector_weights(ax, mx):
    w = mx.sectors.pivot(index="date", columns="sector", values="w_port")
    secs = _sectors(mx)
    ax.stackplot(w.index, *[100 * w[k] for k in secs], colors=[SECTOR.get(k, OTHER) for k in secs], step="post",
                 labels=[k.replace("_", " ") for k in secs], edgecolor=SURFACE, linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(secs))


def active_sector(ax, mx):
    w = mx.sectors.pivot(index="date", columns="sector", values="w_active")
    _zero(ax)
    for k in _sectors(mx):
        ax.plot(w.index, 100 * w[k], color=SECTOR.get(k, OTHER), lw=LW, label=k.replace("_", " "))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)


def active_duration(ax, mx):
    m = mx.monthly
    _zero(ax)
    dD = mx.run.meta["settings"]["delta_D"]
    for i, v in enumerate((-dD, dD)):
        ax.axhline(v, color=INK, lw=1.1, ls="--", label=r"Bound $\pm\delta_D$" if i else None)
    ax.plot(m.index, m.act_D, color=PORT, lw=LW, marker="o", ms=4, mec=SURFACE, label="Active duration")
    ax.set_ylim(-1.5 * dD, 1.5 * dD)
    ax.legend(loc="upper left", ncol=2)


def active_krd(ax, mx):
    m = mx.monthly
    cols = [c for c in m if c.startswith("act_krd_")]
    z = m[cols].to_numpy().T
    lim = np.nanmax(np.abs(z)) or 1.0
    x = mdates.date2num(m.index)
    edges = np.r_[x[0] - (x[1] - x[0]) / 2, (x[1:] + x[:-1]) / 2, x[-1] + (x[-1] - x[-2]) / 2]
    im = ax.pcolormesh(edges, np.arange(len(cols) + 1), z, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim),
                       edgecolors=SURFACE, linewidth=0.5)
    ax.set_yticks(np.arange(len(cols)) + 0.5, [c[8:] for c in cols])
    ax.grid(False)
    cb = ax.figure.colorbar(im, ax=ax, pad=0.01, aspect=25)
    cb.set_label("Active KRD (years)", color=MUTED)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)


# ---- holdings ----

def holdings(ax, mx):
    c, m = mx.conc, mx.monthly
    ax.plot(m.index, m.n_hold, color=BENCH, lw=1.1, ls="--", label="Names held")
    ax.plot(c.index, c.eff_n, color=PORT, lw=LW, marker="o", ms=4, mec=SURFACE, label="Effective holdings (1 / HHI)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower right")


def concentration(ax, mx):
    c = mx.conc
    for (col, name), colr in zip([("top1", "Largest weight"), ("top10", "Top 10 weights"),
                                  ("coverage", "Benchmark weight in held names")], SERIES):
        ax.plot(c.index, 100 * c[col], color=colr, lw=LW, label=name)
    ax.plot(mx.monthly.index, 100 * mx.monthly.active_share, color=INK, lw=1.1, ls="--", label="Active share")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)


# ---- trading ----

def turnover(ax, mx):
    m = mx.monthly.iloc[1:]
    _bars(ax, m.index, 100 * m.turnover, color=PORT)


def cost(ax, mx):
    m = mx.monthly
    trade = 1 - (1 - m.cost) / (1 - m.exit_cost)
    _bars(ax, m.index, 1e4 * trade, color=PORT, label="Rebalance trades")
    _bars(ax, m.index, 1e4 * m.exit_cost, bottom=1e4 * trade, color=SERIES[1], label="Forced sales (index exits)")
    ax.legend(loc="upper right")


PLOTS = [
    Plot("growth", "Growth of 1, daily", "Wealth of the portfolio (net of trading costs) and the benchmark from the "
         "first rebalance, marked to market every trading day. Weights are held between monthly rebalances.",
         r"W_t = W_{d_0}\,\sum_i w_i\,\frac{P_{i,t} + \mathrm{cash}_i(d_0,t)}{P_{i,d_0}},\qquad "
         r"w = x \text{ (portfolio, times } 1-c_{d_0}) \text{ or } \hat b \text{ (benchmark)}", growth),
    Plot("relative", "Cumulative active return vs goal (bp)", "Portfolio wealth relative to the benchmark, daily, "
         "against the goal path that compounds the annual goal g. Above the dashed line the run is ahead of its goal.",
         r"\mathrm{rel}_t = \frac{W^p_t}{W^b_t} - 1,\qquad \mathrm{goal}_t = (1+g)^{\tau_t} - 1,\ "
         r"\tau_t = \text{years since start}", relative),
    Plot("drawdown", "Active drawdown, daily (bp)", "Fall of relative wealth from its running peak (the peak starts "
         "at 1, so an early loss counts as drawdown). Measures how far the run fell behind the benchmark at worst.",
         r"A_t = \frac{W^p_t}{W^b_t},\qquad \mathrm{DD}_t = \frac{A_t}{\max(1, \max_{s \le t} A_s)} - 1", drawdown),
    Plot("monthly_active", "Monthly active return vs required (bp)", "Net active return of each month against the "
         "monthly goal the optimizer aimed for. With catch-up on, the requirement rises after a miss and falls when "
         "ahead, within [floor, cap times g_m].",
         r"a_t = r^p_t - r^b_t,\qquad g^{req}_k = \mathrm{clip}\!\left(\left(\frac{(1+g_m)^N}{A_k}\right)^{1/(N-k)}"
         r" - 1,\ \mathrm{floor},\ \mathrm{cap}\cdot g_m\right)", monthly_active),
    Plot("te", "Tracking error, annualised (bp)", "Ex-ante TE is the optimizer's forecast at each rebalance from the "
         f"factor model. Realised TE is the standard deviation of active returns over a rolling window, from daily "
         f"marks ({TE_WINDOW_D} days) and from monthly returns ({TE_WINDOW_M} months). Daily TE runs higher because "
         "illiquid corporates are marked at their last quote, so the portfolio's daily active return carries "
         "pricing noise that cancels over the month; the monthly figure is the one comparable to the forecast.",
         r"\mathrm{TE}^{ex}_t = \sqrt{12\,a_t^\top \hat Q a_t},\quad \hat Q = F F^\top + \mathrm{diag}(\psi);"
         r"\qquad \mathrm{TE}^{real} = \sqrt{252}\,\mathrm{sd}(a^{daily}) \text{ or } \sqrt{12}\,\mathrm{sd}(a^{monthly})", te),
    Plot("te_ex_ante_source", "Ex-ante tracking error by source (bp)", "Forecast TE split in proportion to each "
         "source's share of active variance. The factors are orthogonal so the variances add exactly. Curve PCs are "
         "level, slope and curvature of the key-rate changes, credit factors are sector spread factors, residual is "
         "issuer-specific variance.",
         r"a^\top \hat Q a = \sum_j (F_j^\top a)^2 + \sum_i \psi_i a_i^2,\qquad "
         r"\mathrm{TE}_s = \mathrm{TE}^{ex}\cdot \frac{\mathrm{var}_s}{a^\top \hat Q a}", te_ex_ante_src),
    Plot("te_realised_source", f"Realised tracking error by source, rolling {TE_WINDOW_M} months (bp)",
         "Monthly active return split by sector (active weight times realised return, gross) plus the remainder "
         "(costs and the gross/net cross term). Each source's share of realised TE is its covariance with the total "
         "over the variance of the total, so shares sum to the realised TE and a negative bar is a hedge.",
         r"a_t = \sum_s \underbrace{\sum_{i \in s} (x_i - b_i) r_i}_{a_{s,t}} + e_t,\qquad "
         r"\mathrm{TE}_s = \mathrm{TE}\cdot \frac{\mathrm{cov}(a_s, a)}{\mathrm{var}(a)}", te_real_src),
    Plot("sector_weights", "Portfolio weight by sector (%)", "Post-trade portfolio weight in each sector at every "
         "rebalance.", r"w_s = \sum_{i \in s} x_i", sector_weights),
    Plot("active_sector", "Active sector weight (pct points)", "Portfolio minus benchmark weight per sector. "
         "Positive is overweight.", r"w_s - b_s = \sum_{i \in s} (x_i - b_i)", active_sector),
    Plot("active_duration", "Active duration (years)", "Portfolio minus benchmark modified duration at each "
         "rebalance. The optimizer bounds it by delta_D.", r"D^{act} = D^\top (x - b),\qquad |D^{act}| \le \delta_D",
         active_duration),
    Plot("active_krd", "Active key-rate durations by tenor", "Active key-rate duration at each tenor and rebalance. "
         "Blue is long duration at that tenor relative to the benchmark, red is short. Rows sum to the active "
         "duration.", r"K^{act}_k = K_{\cdot k}^\top (x - b),\qquad \sum_k K^{act}_k = D^{act}", active_krd),
    Plot("holdings", "Number of holdings", "Names held against the effective number of holdings, the inverse "
         "Herfindahl index of portfolio weights. Equal weights give eff. N = names held.",
         r"N_{eff} = \frac{1}{\sum_i x_i^2}", holdings),
    Plot("concentration", "Concentration (%)", "Largest single weight and the top 10 combined, the benchmark weight "
         "of the bonds held (how much of the index the portfolio touches), and active share.",
         r"\mathrm{coverage} = \sum_{i: x_i > 0} b_i,\qquad \mathrm{active\ share} = \tfrac12 \sum_i |x_i - b_i|",
         concentration),
    Plot("turnover", "Turnover after the initial purchase (% of portfolio bought)", "Share of the portfolio bought "
         "at each rebalance, from the drifted holdings. The first month (100%, building the portfolio from cash) is "
         "left out.", r"\mathrm{turnover}_t = \sum_i (x_{i,t} - x^{init}_{i,t})^+", turnover),
    Plot("cost", "Trading cost (bp of wealth)", "Wealth lost to the bid-ask spread each month: rebalance trades at "
         "the proportional cost c, and forced sales of bonds that left the benchmark at half the bid-ask.",
         r"\mathrm{cost}_t = 1 - (1 - c^{exit}_t)(1 - c^\top |x_t - x^{init}_t|)", cost),
]

YLABEL = {"growth": "Wealth (start = 1)", "relative": "Active return (bp)", "drawdown": "Drawdown (bp)",
          "monthly_active": "Active return (bp / month)", "te": "TE (bp / year)", "te_ex_ante_source": "TE (bp / year)",
          "te_realised_source": "TE (bp / year)", "sector_weights": "Weight (%)",
          "active_sector": "Active weight (pct points)", "active_duration": "Active duration (years)",
          "active_krd": "Key-rate tenor", "holdings": "Number of bonds", "concentration": "Share (%)",
          "turnover": "Bought (% of portfolio)", "cost": "Cost (bp)"}


def render(p: Plot, mx: Metrics, path):
    fig, ax = _axes(p.title, YLABEL[p.name])
    p.draw(ax, mx)
    fig.savefig(path, dpi=150)
    plt.close(fig)
