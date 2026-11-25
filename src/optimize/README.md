# src/optimize

Preliminary D3 multi-period tracking model on top of `src.params`, with a rolling backtest and dashboard.

    uv run python -m src.optimize            # params (cached) -> grid backtest -> results/runs/<timestamp>/ (run.json, backtest.parquet, summary.csv)
    uv run python -m src.optimize.dashboard  # results/runs/<timestamp>/dashboard.png (--run to pick one)

Data grain: monthly rebalances on month-end prices (24 month-ends, 2024-01 to 2025-12, set by where the fitted Treasury curve ends). The first 6 are Sigma/psi warm-up, so 17 rebalances: train 2024-07..12, val 2025-01..06, test 2025-07..11. Config (gamma, H) chosen on validation RMS active return, quoted on test.

Cardinality and minimum lots: `Settings.method = "heuristic"` (default) is a two-stage heuristic (relax, keep top k_max, re-solve) on CLARABEL. `"misocp"` solves the written MISOCP once with binary z on Gurobi (`uv add gurobipy` is in the env, needs a license), stopping at `time_limit` seconds and using the best incumbent. `Settings.backend = "gurobipy"` (`grb.py`) runs relax, truncate, re-solve and the MIP in one Gurobi model, with the heuristic solution as the MIP start, and is far faster than the cvxpy route for `"misocp"` (about 0.5 s vs 22 s on the 2024-07-31 Treasuries test). Default `mip_gap` 1e-2, `MIPFocus` 2. On the cvxpy backend the heuristic runs first as the MIP start, binaries exist only for the `mip_cand` (150) largest stage-1 weights, and the model is scaled (weights in percent, objective x1e3) because Gurobi's tolerances were too loose for raw weights of about 4e-3.

Simplifications: Candidate set is the 400 largest benchmark weights plus current holdings. Tolerances (delta), the return goal (benchmark expected return minus 25 bp) and eps2 are placeholders. Where the return goal is infeasible it is dropped and flagged in `return_goal_dropped`.
