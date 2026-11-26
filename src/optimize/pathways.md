# Solve pathways

## Full process

Follows `workbook/optimizer.ipynb`. The solve node expands into the diagram under Solve step.

```mermaid
flowchart TD
    subgraph N1["1. Inputs"]
        U["ASSET_TYPES, G_ANNUAL, START_DATE, T_YEARS,<br/>RISK, K_MAX, H, W, METHOD, TIME_LIMIT"] --> MS["make_settings<br/>RISK sets gamma and scales deltas<br/>ret_slack = minus monthly g"]
    end

    subgraph N2["2. Data and dates"]
        LD["load_data<br/>universe, prices, curve, month-end panel"] --> RB["REB = first N = 12 T month-ends from START_DATE"]
    end

    MS --> LOOP
    RB --> LOOP
    LOOP{"next rebalance date d"}

    subgraph N3["3. Benchmark"]
        BM["benchmark_at(d)<br/>b = agg MV / sleeve MV"] --> DF["data_for(d)<br/>universe cut to benchmark, agg_weight = b"]
    end

    LOOP -->|more dates| CH{"PARAMS cached<br/>for (d, H)?"}
    CH -->|no| BM

    subgraph N4["4. Estimation: build_inputs(d, W, H)"]
        E1["fit NSS curve, reprice bonds<br/>K, D, D_spr, C"] --> E2["Sigma via Ledoit-Wolf, keep 3 PCs"]
        E2 --> E3["credit factor per corporate sector"]
        E3 --> E4["psi shrunk to sector prior"]
        E4 --> E5["c, ell, mu_hat, b_hat"]
    end

    DF --> E1
    E5 --> P["Params P"]
    CH -->|yes| P

    subgraph N5["5. Solve"]
        HD["drop holdings not in P, renormalize<br/>(empty at first date)"] --> CA["candidates<br/>S = top 400 eligible by b_hat + holdings<br/>x_init = holdings on S"]
        CA --> SR["solve_rebalance(P, S, x_init, settings)<br/>see Solve step"]
    end

    P --> HD

    subgraph N6["6. Execute and measure"]
        EX["x0 = x[0] normalized<br/>cost = c' abs(x0 - x_init)"] --> RR["realised_returns d to next month-end"]
        RR --> AR["net, bench = b_hat' r, active = net - bench"]
        AR --> DR["drift x0 by r, renormalize"]
    end

    SR --> EX
    DR -->|new holdings| LOOP

    LOOP -->|done| SUM["7. summary<br/>cum returns, annualized active vs g,<br/>tracking error, turnover, cost, holdings"]
    SUM --> SEN["8. sensitivity<br/>re-run loop over RISK x k_max, reusing PARAMS"]
```

## Solve step

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

    FB["Fallback ladder if infeasible:<br/>hard goal: drop it, then widen deltas and te_cap x2, x4, x8<br/>soft goal or return mode: only widen"] -.-> H1
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
  - Objective: `objective` ("tracking" or "return"), `H`, `omega`, `gamma`, `cost_weight`, `goal_penalty` (None = hard goal)
  - Constraints: `k_max`, `max_active_w`, `delta_D`, `delta_K`, `delta_DTS`, `delta_C`, `te_cap`, `eps2`, `ret_slack`
  - User inputs map to these through `presets.make_settings` (driven by `workbook/goal_optimizer.ipynb`)
  - Estimation: `psi_floor` (applied to `P` by the backtest, off by default since `src.params` shrinks psi)
  - Path: `method`, `backend`
  - Solver: `solver`, `mip_solver`
  - MIP: `mip_gap`, `mip_focus`, `time_limit`, `mip_cand`, `mip_warm`, `mip_skip_shortfall` (gurobipy: keep the heuristic when it misses a soft goal)
