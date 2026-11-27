"""One goal-based run end to end, and sweeps over the user inputs.

    from src.backtest import RunConfig, run, sweep
    r = run(RunConfig(g_annual=0.005, start="2021-01", T=4))      # Run with frames and meta, saved under results/runs
    table = sweep(RunConfig(), dict(risk_tol=["low", "medium", "high"], mode=["tracking", "return"]))
"""
import logging
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from src.params import build_cashflows, load_data
from src.params.residual import ResidualPanel
from src.optimize.presets import horizon_dates

from . import loop
from .benchmark import Benchmark
from .cache import ParamsCache
from .config import RunConfig
from .results import RESULTS, Run, base_meta, make_run, summarize

log = logging.getLogger(__name__)


class Context:
    """Data shared by every run with the same asset types and horizon: loaded data, benchmark, cash flow table and
    the params cache (one per H and W). Pass one to run() to reuse it across runs."""

    def __init__(self, cfg: RunConfig, cache_root=None):
        self.data = load_data()
        self.ends = self.data.panel.index.get_level_values("month_end").unique().sort_values()
        self.dates, self.end = horizon_dates(self.ends, cfg.start, cfg.T)
        self.asset_types, self.start, self.T = cfg.asset_types, cfg.start, cfg.T
        self.bench = Benchmark(cfg.asset_types)
        u = self.data.universe
        self.table = build_cashflows(
            u[u.asset_type.isin(cfg.asset_types) & u.maturity_date.notna() & u.coupon_rate.notna()]
            .reset_index(drop=True), self.dates[0] - pd.Timedelta(days=400))
        self.cache_root = cache_root
        self._caches: dict = {}
        self._resid = None

    @property
    def resid(self) -> ResidualPanel:
        """Residuals of every bond of these asset types, built on first use and shared by all caches."""
        if self._resid is None:
            u = self.data.universe
            self._resid = ResidualPanel(u[u.asset_type.isin(self.asset_types)], self.data.panel, self.data.curve)
        return self._resid

    def fits(self, cfg: RunConfig) -> bool:
        return (cfg.asset_types, cfg.start, cfg.T) == (self.asset_types, self.start, self.T)

    def cache(self, H: int, W: float) -> ParamsCache:
        if (H, W) not in self._caches:
            kw = {} if self.cache_root is None else {"root": self.cache_root}
            self._caches[(H, W)] = ParamsCache(self.asset_types, H, W, resid=self.resid, **kw)
        return self._caches[(H, W)]

    def data_for(self, d):
        return self.bench.data_for(self.data, d)


def run(cfg: RunConfig, ctx: Context | None = None, save: bool = True, root: Path = RESULTS / "runs") -> Run:
    """Roll cfg over its horizon. Raises ValueError for an infeasible goal or horizon before any estimation."""
    settings = cfg.settings()
    if ctx is None or not ctx.fits(cfg):
        ctx = Context(cfg)
    cache = ctx.cache(settings.H, cfg.W)
    built0 = cache.built
    log.info("%s: %d rebalances %s .. %s, %s %s", cfg.label(), len(ctx.dates), ctx.dates[0].date(),
             ctx.dates[-1].date(), settings.backend, settings.method)
    started = datetime.now()
    frames = loop.run(settings, ctx.dates, ctx.ends, ctx.data, ctx.data_for, ctx.table, cache, cfg.n_cand,
                      cfg.catch_up)
    meta = base_meta(cfg, settings, ctx.dates, ctx.end, cache, started, datetime.now(), cache.built - built0)
    meta["summary"] = summarize(frames.monthly, cfg.g_annual)
    r = make_run(frames, meta)
    if save:
        log.info("saved %s", r.save(root))
    return r


def sweep(cfg: RunConfig, grid: dict, save: bool = True) -> pd.DataFrame:
    """Run cfg with every combination of grid (RunConfig field -> values). Infeasible combinations are skipped and
    reported. Saved under results/sweeps/<time>/<label>/, with sweep.csv holding one summary row per run."""
    root = RESULTS / "sweeps" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ctx, rows = None, []
    for vals in product(*grid.values()):
        c = cfg.with_(**dict(zip(grid, vals)))
        try:
            c.settings()
        except ValueError as e:
            log.warning("skip %s: %s", c.label(), e)
            rows.append({**dict(zip(grid, vals)), "label": c.label(), "error": str(e)})
            continue
        if ctx is None or not ctx.fits(c):
            ctx = Context(c)
        r = run(c, ctx, save=save, root=root)
        rows.append({**dict(zip(grid, vals)), "label": c.label(), "path": str(r.path) if r.path else "",
                     **r.summary})
    table = pd.DataFrame(rows)
    if save:
        root.mkdir(parents=True, exist_ok=True)
        table.to_csv(root / "sweep.csv", index=False)
        log.info("sweep saved to %s", root)
    return table
