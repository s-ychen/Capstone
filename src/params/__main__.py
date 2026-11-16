"""Command line summary of build_inputs: uv run python -m src.params --asof 2025-06-30 --H 2"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from .inputs import build_inputs, load_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asof", default="2025-06-30")
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--W", type=float, default=1e6, help="current wealth in dollars")
    ap.add_argument("--method", default="ledoit_wolf", choices=["sample", "ewma", "ledoit_wolf"])
    ap.add_argument("--freq", default="M", choices=["M", "D"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    data = load_data()
    t1 = time.time()
    P = build_inputs(args.asof, args.W, args.H, sigma_method=args.method, freq=args.freq, data=data)
    t2 = time.time()
    m, p0 = P.model, P.periods[0]
    print(f"as of {P.asof.date()}: {len(P.cusips)} bonds, load {t1 - t0:.1f}s, build {t2 - t1:.1f}s")
    print("explained variance of PCs:", np.round(m.explained, 3), "T_s =", m.n_obs)
    print("sum b:", round(P.b.sum(), 6), "| bonds by sector:", dict(pd.Series(P.sector).value_counts()))
    print(f"b-weighted D {p0.b_hat @ p0.D:.2f}, D_spr {p0.b_hat @ p0.D_spr:.2f}, C {p0.b_hat @ p0.C:.1f}, "
          f"mu {p0.b_hat @ p0.mu_hat:.4%}/month")
    print(f"psi median {np.median(m.psi):.2e}, stale {P.stale.mean():.1%}, spread solved {P.spread_solved.mean():.1%}")
    for q in P.periods:
        print(f"h={q.h} {q.date.date()}: exits {q.exits.sum()}, sum b_hat {q.b_hat.sum():.6f}")


if __name__ == "__main__":
    main()
