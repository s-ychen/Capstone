"""Goal-based driver: roll the optimizer monthly over a user horizon and save the run (port of
workbook/goal_optimizer.ipynb). Entry point: python -m src.backtest configs/goal.toml"""
from .config import RunConfig
from .driver import Context, run, sweep
from .results import Run, list_runs, load_run, summarize

__all__ = ["RunConfig", "Context", "run", "sweep", "Run", "load_run", "list_runs", "summarize"]
