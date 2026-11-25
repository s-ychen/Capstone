# Solve pathways

`solve_rebalance(P, S, x_init, s)` in `model.py`. `Settings.method` picks what is solved, `Settings.backend` picks the solver.

```mermaid
flowchart TD
    IN["Inputs<br/>P, S, x_init, Settings"] --> M{method}

    M -->|heuristic| B1{backend}
    M -->|misocp| B2{backend}

    B1 -->|cvxpy| H1["Stage 1: relaxation (CLARABEL)<br/>Stage 2: top k_max, re-solve"]
    B1 -->|gurobipy| H2["Stage 1: relaxation (Gurobi)<br/>Stage 2: top k_max, re-solve"]

    B2 -->|cvxpy| C1["Heuristic (CLARABEL)"]
    B2 -->|gurobipy| G1["Stage 1 + 2 in one Gurobi model"]

    C1 -->|"incumbent = MIP start<br/>(if mip_warm)"| C2["Rebuild in cvxpy, add binaries z<br/>solve with Gurobi"]
    G1 -->|"incumbent = MIP start<br/>(always)"| G2["Add binaries z to same model<br/>solve with Gurobi"]

    H1 --> OUT[Solution]
    H2 --> OUT
    C2 --> OUT
    G2 --> OUT

    FB["Fallback ladder if infeasible:<br/>drop return goal, then widen deltas x2, x4, x8"] -.-> H1
    FB -.-> H2
    FB -.-> C2
    FB -.-> G2
```

Heuristic outputs are the stage-2 solution. MIP outputs are the best Gurobi incumbent, stopping at `mip_gap` or `time_limit`.

## Inputs

- `P`: `src.params` Params for one date. Per bond: `c`, `model.psi`, `theta_F`, `theta_psi`. Per planning date `periods[h]`: `b_hat`, `F`, `ell`, `exits`, `D`, `D_spr`, `C`, `K`, `mu_hat`, `drift_gross`, `r_b`.
- `S`: candidate index set (400 largest benchmark weights plus current holdings in the backtest).
- `x_init`: current drifted holdings on `S`.
- `Settings`:
  - Objective: `H`, `omega`, `gamma`
  - Constraints: `k_max`, `max_active_w`, `delta_D`, `delta_K`, `delta_S`, `delta_C`, `eps2`, `ret_slack`
  - Estimation: `psi_floor` (applied to `P` by the backtest, off by default since `src.params` shrinks psi)
  - Path: `method`, `backend`
  - Solver: `solver`, `mip_solver`
  - MIP: `mip_gap`, `mip_focus`, `time_limit`, `mip_cand`, `mip_warm`
