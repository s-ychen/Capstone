"""Series behind each plot, from a run and its daily paths."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.results import Run

TE_WINDOW_D = 252       # rolling realised TE window, trading days
TE_WINDOW_M = 12        # rolling window for the realised TE split by source, months


@dataclass
class Metrics:
    run: Run
    daily: pd.DataFrame         # indexed by date: wealth, returns, relative wealth, goal path, drawdown
    monthly: pd.DataFrame       # run.monthly with a DatetimeIndex
    conc: pd.DataFrame          # per rebalance: effective holdings, top 1 and top 10 weight, coverage
    risk: pd.DataFrame          # date x factor, ex-ante active variance (monthly)
    sectors: pd.DataFrame       # long, run.sectors with active weight and active contribution
    real_src: pd.DataFrame      # date x source, rolling realised TE share (annualised)
    te_roll_m: pd.Series        # rolling 12 month realised TE from monthly active returns (annualised)


def build(run: Run, daily: pd.DataFrame) -> Metrics:
    g = run.meta["inputs"]["g_annual"]
    d = daily.set_index(pd.to_datetime(daily.date)).drop(columns="date")
    d["ret_port"] = d.wealth_port.pct_change()
    d["ret_bench"] = d.wealth_bench.pct_change()
    d["active"] = d.ret_port - d.ret_bench
    d["rel"] = d.wealth_port / d.wealth_bench - 1
    years = (d.index - d.index[0]).days / 365.25
    d["goal"] = (1 + g) ** years - 1
    A = 1 + d.rel
    d["drawdown"] = A / A.cummax().clip(lower=1) - 1
    d["te_roll"] = d.active.rolling(TE_WINDOW_D, min_periods=TE_WINDOW_D // 2).std() * np.sqrt(252)

    m = run.monthly.copy()
    m.index = pd.to_datetime(m.index)

    h = run.holdings[run.holdings.x > 1e-9]
    conc = []
    for dt, gr in h.groupby("date"):
        w = np.sort(gr.x.to_numpy() / gr.x.sum())[::-1]
        conc.append(dict(date=dt, eff_n=1 / np.sum(w ** 2), top1=w[0], top10=w[:10].sum(), coverage=gr.b.sum()))
    conc = pd.DataFrame(conc).set_index("date")
    conc.index = pd.to_datetime(conc.index)

    risk = run.risk.pivot(index="date", columns="factor", values="var")
    risk.index = pd.to_datetime(risk.index)

    s = run.sectors.copy()
    s["date"] = pd.to_datetime(s.date)
    s["w_active"] = s.w_port - s.w_bench
    s["contrib_active"] = s.contrib_port - s.contrib_bench

    src = s.pivot(index="date", columns="sector", values="contrib_active").reindex(m.index).fillna(0)
    src["cost and other"] = m.active - src.sum(axis=1)
    tot = m.active
    roll = dict(window=TE_WINDOW_M, min_periods=TE_WINDOW_M // 2)
    te_m = tot.rolling(**roll).std() * np.sqrt(12)
    share = pd.DataFrame({c: src[c].rolling(**roll).cov(tot) / tot.rolling(**roll).var() for c in src})
    return Metrics(run, d, m, conc, risk, s, share.mul(te_m, axis=0), te_m)


def headline(mx: Metrics) -> dict:
    """Numbers quoted in the report: run summary plus daily-path statistics."""
    s = dict(mx.run.summary)
    d = mx.daily
    s["te_ann_daily"] = float(d.active.std() * np.sqrt(252))
    s["max_active_drawdown_daily"] = float(d.drawdown.min())
    s["eff_n_avg"] = float(mx.conc.eff_n.mean())
    return s
