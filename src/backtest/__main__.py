"""uv run python -m src.backtest configs/goal.toml [--sweep risk_tol=low,medium,high estimation.psi_n0=none,12]

One run of the config, or a sweep over the listed RunConfig fields or dotted section keys (overrides.*, estimation.*,
catch_up.*). Values are parsed as TOML (numbers, strings, booleans); none gives None."""
import argparse
import logging
import tomllib

from .config import RunConfig
from .driver import run, sweep


def _value(v: str):
    if v.lower() == "none":
        return None
    try:
        return tomllib.loads(f"v = {v}")["v"]
    except tomllib.TOMLDecodeError:
        return v


def main():
    ap = argparse.ArgumentParser(prog="python -m src.backtest", description=__doc__.splitlines()[0])
    ap.add_argument("config", help="TOML file, see configs/goal.toml")
    ap.add_argument("--sweep", nargs="+", metavar="FIELD=V1,V2", help="grid over RunConfig fields or section.key")
    ap.add_argument("--name", default="", help="appended to the sweep folder name")
    ap.add_argument("--tag", help="appended to run folder names")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("gurobipy", "cvxpy"):
        logging.getLogger(name).setLevel(logging.WARNING)
    cfg = RunConfig.from_toml(args.config)
    if args.tag:
        cfg = cfg.with_(tag=args.tag)
    if args.sweep:
        grid = {}
        for item in args.sweep:
            k, _, vs = item.partition("=")
            grid[k] = [_value(v) for v in vs.split(",")]
        print(sweep(cfg, grid, save=not args.no_save, name=args.name).to_string())
        return
    r = run(cfg, save=not args.no_save)
    for k, v in r.summary.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")
    if r.path:
        print("saved to", r.path)


if __name__ == "__main__":
    main()
