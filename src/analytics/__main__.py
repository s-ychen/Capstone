"""uv run python -m src.analytics [run]  ->  results/runs/<run>/analytics/{*.png, README.md, daily.parquet}

run is a folder name under results/runs or a path (default: latest run with run.json from src.backtest).
"""
import argparse
from pathlib import Path

from src.backtest.results import RESULTS, load_run

from . import daily as D
from .metrics import build, headline
from .plots import PLOTS, render


def _resolve(arg) -> Path:
    if arg:
        p = Path(arg)
        return p if p.exists() else RESULTS / "runs" / arg
    runs = [f.parent for f in sorted((RESULTS / "runs").glob("*/run.json")) if '"summary"' in f.read_text()]
    if not runs:
        raise SystemExit("no src.backtest runs under results/runs")
    return runs[-1]


def _report(run, h, gap_bp) -> str:
    i = run.meta["inputs"]
    lines = [
        f"# Analytics: {run.meta['label']}", "",
        f"{run.meta['horizon']['first']} to {run.meta['horizon']['returns_to']}, "
        f"{h['months']} monthly rebalances, {', '.join(i['asset_types'])}, goal {1e4 * i['g_annual']:g} bp / year, "
        f"risk tolerance {i['risk_tol']}, k_max {i['k_max']}, {i['mode']} mode.", "",
        "| metric | value |", "| --- | --- |",
        f"| cumulative return, portfolio / benchmark | {h['cum_port']:.2%} / {h['cum_bench']:.2%} |",
        f"| active return, annualised | {1e4 * h['active_ann']:+.1f} bp (goal {'met' if h['goal_met'] else 'missed'}) |",
        f"| tracking error, realised (daily / monthly) | {1e4 * h['te_ann_daily']:.0f} / {1e4 * h['te_ann']:.0f} bp |",
        f"| tracking error, ex-ante average | {1e4 * h['te_fcst_ann_avg']:.0f} bp |",
        f"| information ratio (monthly) | {h['info_ratio']:.2f} |",
        f"| hit rate (months beating benchmark) | {h['hit_rate']:.0%} |",
        f"| max active drawdown (daily) | {1e4 * h['max_active_drawdown_daily']:.0f} bp |",
        f"| turnover, average after first month | {h['turnover_avg_ex_first']:.1%} |",
        f"| trading cost, average / total | {h['cost_bp_avg']:.2f} / {h['cost_bp_total']:.1f} bp |",
        f"| holdings, names / effective | {h['holdings_avg']:.0f} / {h['eff_n_avg']:.0f} |",
        f"| active share, average | {h['active_share_avg']:.0%} |", "",
        f"Daily paths reproduce the backtest's monthly returns to within {gap_bp:.3f} bp.", "",
    ]
    for p in PLOTS:
        lines += [f"## {p.title}", "", f"![{p.name}]({p.name}.png)", "", p.explain, ""]
        if p.formula:
            lines += ["$$", p.formula, "$$", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m src.analytics", description=__doc__.splitlines()[0])
    ap.add_argument("run", nargs="?", help="run folder name or path (default: latest)")
    ap.add_argument("--fresh", action="store_true", help="recompute the cached daily paths")
    args = ap.parse_args()
    path = _resolve(args.run)
    run = load_run(path)
    out = path / "analytics"
    if args.fresh and (out / "daily.parquet").exists():
        (out / "daily.parquet").unlink()
    daily = D.daily_paths(run, out)
    gap = D.check(daily, run.monthly)
    mx = build(run, daily)
    for p in PLOTS:
        render(p, mx, out / f"{p.name}.png")
    (out / "README.md").write_text(_report(run, headline(mx), gap))
    print(f"saved {len(PLOTS)} plots and README.md to {out}")
    print(f"max |daily-compounded - monthly return|: {gap:.3f} bp")


if __name__ == "__main__":
    main()
