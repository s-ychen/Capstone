"""Run the pipeline from a terminal: `python run_pipeline.py [--cache] [--no-download] [--no-auctions]`.
Prints each step, and a stack trace even if the interpreter crashes."""
import faulthandler
import sys

faulthandler.enable()
from data_files.pipeline import checks, run

if __name__ == "__main__":
    a = sys.argv
    checks(run(use_cache="--cache" in a, download="--no-download" not in a, auctions="--no-auctions" not in a))
