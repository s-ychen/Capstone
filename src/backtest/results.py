"""Run outputs: the summary against the goal, and the run folder on disk.

results/runs/<yyyy-mm-dd_HHMMSS>_<label>/
    run.json            meta (time, git, versions), inputs, resolved settings, horizon, cache key, summary
    monthly.parquet     one row per rebalance (loop.run columns)
    holdings.parquet    (date, cusip): weights before and after the trade, benchmark weight, realised return, cost
    sectors.parquet     (date, sector): weights and gross return contributions, portfolio and benchmark
    risk.parquet        (date, factor): ex-ante active variance per curve PC, credit factor and residual sector
"""
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .loop import Frames

RESULTS = Path(__file__).resolve().parents[2] / "results"
FRAMES = ("monthly", "holdings", "sectors", "risk")


def summarize(m: pd.DataFrame, g: float) -> dict:
    """Headline numbers of one run (monthly frame) against the annual goal g."""
    n = len(m)
    if not n:
        return {"months": 0}
    cp, cb = float((1 + m.ret_port).prod()), float((1 + m.ret_bench).prod())
    act_ann = (cp / cb) ** (12 / n) - 1
    te = float(m.active.std(ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    return {
        "months": n,
        "cum_port": cp - 1,
        "cum_bench": cb - 1,
        "active_ann": act_ann,
        "goal_ann": g,
        "goal_met": bool(act_ann >= g),
        "te_ann": te,
        "info_ratio": act_ann / te if te and np.isfinite(te) and te > 0 else float("nan"),
        "hit_rate": float((m.active > 0).mean()),
        "max_active_drawdown": float((m.growth_active / m.growth_active.cummax().clip(lower=1)).min() - 1),
        "te_fcst_ann_avg": float(m.te_fcst_ann.mean()),
        "te_cap_ann": float(m.te_cap_ann.iloc[0]),
        "robust_active_ann_avg": float(12 * m.robust_active.mean()),
        "goal_req_ann_avg": float((1 + m.goal_req.mean()) ** 12 - 1),
        "turnover_avg": float(m.turnover.mean()),
        "turnover_avg_ex_first": float(m.turnover.iloc[1:].mean()) if n > 1 else float("nan"),
        "cost_bp_avg": float(1e4 * m.cost.mean()),
        "cost_bp_total": float(1e4 * (1 - (1 - m.cost).prod())),
        "holdings_avg": float(m.n_hold.mean()),
        "active_share_avg": float(m.active_share.mean()),
        "shortfall_months": int((m.shortfall > 1e-6).sum()),
        "shortfall_bp_avg": float(1e4 * m.shortfall.mean()),
        "widened_months": int((m.delta_scale > 1).sum()),
        "mip_skipped_months": int(m.mip_skipped.sum()),
        "fallback_share_avg": float(m.share_fallback.mean()),
        "solve_s_avg": float(m.seconds.mean()),
    }


def _git() -> dict:
    root = RESULTS.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root)
        dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True, cwd=root)
        return {"commit": commit.stdout.strip(), "src_dirty": bool(dirty.stdout.strip())}
    except OSError:
        return {"commit": "", "src_dirty": None}


def _versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import gurobipy
        out["gurobi"] = ".".join(map(str, gurobipy.gurobi.version()))
    except ImportError:
        pass
    return out


def _jsonable(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return str(o)


def _clean(o):
    """NaN and inf to null, so run.json is valid JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


@dataclass
class Run:
    meta: dict
    monthly: pd.DataFrame
    holdings: pd.DataFrame
    sectors: pd.DataFrame
    risk: pd.DataFrame
    path: Path | None = None

    @property
    def summary(self) -> dict:
        return self.meta["summary"]

    def save(self, root: Path = RESULTS / "runs", name: str | None = None) -> Path:
        out = Path(root) / (name or f"{self.meta['run']['started'][:19].replace('T', '_').replace(':', '')}"
                                    f"_{self.meta['label']}")
        out.mkdir(parents=True, exist_ok=False)
        for k in FRAMES:
            getattr(self, k).to_parquet(out / f"{k}.parquet")
        (out / "run.json").write_text(json.dumps(_clean(self.meta), indent=2, default=_jsonable))
        self.path = out
        return out


def make_run(frames: Frames, meta: dict) -> Run:
    return Run(meta=meta, **frames._asdict())


def base_meta(cfg, settings, dates, end, cache, started, finished, built: int) -> dict:
    return {
        "label": cfg.label(),
        "run": {"started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
                "seconds": round((finished - started).total_seconds(), 1), "params_built": built,
                "git": _git(), "versions": _versions()},
        "inputs": cfg.to_dict(),
        "settings": asdict(settings),
        "horizon": {"n_rebalances": len(dates), "first": str(dates[0].date()), "last": str(dates[-1].date()),
                    "returns_to": str(end.date())},
        "params_cache": cache.key,
    }


def load_run(path) -> Run:
    path = Path(path)
    meta = json.loads((path / "run.json").read_text())
    fr = {k: pd.read_parquet(path / f"{k}.parquet") for k in FRAMES}
    return Run(meta=meta, path=path, **fr)


def list_runs(root: Path = RESULTS / "runs") -> pd.DataFrame:
    """One row per saved run: folder, label and summary."""
    rows = []
    for f in sorted(Path(root).glob("*/run.json")):
        m = json.loads(f.read_text())
        if "summary" not in m:      # legacy src.optimize.backtest runs
            continue
        rows.append({"path": str(f.parent), "label": m["label"], "started": m["run"]["started"], **m["summary"]})
    return pd.DataFrame(rows)
