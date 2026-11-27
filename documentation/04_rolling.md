# Rolling loop

`loop.run(settings, dates, ends, data, data_for, table, params, n_cand, catch_up)` (`src/backtest/loop.py`) runs one solve per rebalance month-end over the horizon and returns four frames (below). `driver.run(cfg)` builds its inputs from a `RunConfig` and saves the run.

## One month

```mermaid
%%{init: {"flowchart": {"wrappingWidth": 1000}, "markdownAutoWrap": false, "themeCSS": ".node .label div, .node .nodeLabel, .node p { text-align: left !important; }"}}%%
flowchart TD
    D["rebalance d (k of N)"] --> CH{"ParamsCache has d?"}
    CH -->|no| EST["build_inputs(d, W, H,<br/>data_for(d))"]
    CH -->|yes| P["Params P"]
    EST --> P
    P --> GR["goal for this month<br/>g_req (catch-up) or g_m"]
    P --> EX["sell holdings not in P<br/>(left benchmark<br/>or unpriced)<br/>at half bid-ask"]
    EX --> CA["candidates: S, x_init"]
    GR --> SO["solve_rebalance with<br/>ret_slack = -g_req"]
    CA --> SO
    SO --> TR["trade x0 = x[0]<br/>cost c' abs(x0 - x_init)"]
    TR --> RR["realised returns d to<br/>next month-end"]
    RR --> AC["net, benchmark and<br/>active return,<br/>update active growth A"]
    AC --> DR["drift x0 by returns<br/>= new holdings"]
    DR --> ROW["append results row"]
    DR -->|"k + 1"| D
```

## Steps

1. Estimate. `build_inputs` at $d$ with the month's benchmark, through `ParamsCache` (memory, then `results/cache/params/<key>/<date>.pkl`). The key covers asset types, `H`, `W`, the `src/params` source and the processed data files, so re-running with another risk tolerance, mode or `k_max` reuses it, and any change that would alter `Params` builds afresh.
2. Goal. Tracking mode with catch-up on: the monthly goal is reset (below). Otherwise $g_m$.
3. Forced sales. Holdings whose cusip is not in `P.cusips` are sold at half their last bid-ask in the 90 days to $d$ (else their cost at the previous rebalance). This is `exit_cost`. The remaining holdings are rescaled to post-sale wealth and sum below 1, so the solve reinvests the cash and pays the buy cost on it.
4. Solve from the drifted holdings (`candidates`, then `solve_rebalance`). Trade cost $\text{cost} = c_S^\top \lvert x_0 - x^{init} \rvert$.
5. Measure. `realised_returns`: month-end dirty prices plus coupons paid between $d$ and the next month-end. A bond with a stale price at either end, or no valid return, takes its $\hat\mu$.

$$r^{port} = (1 - \text{exit\_cost})(1 - \text{cost})(1 + x_0^\top r_S) - 1, \qquad r^{bench} = \hat b_0^\top r, \qquad \text{active} = r^{port} - r^{bench}$$

6. Drift. New holdings $x_{0,i}(1 + r_i) / (1 + x_0^\top r_S)$ for bonds above $10^{-9}$, and $A \leftarrow A\,(1 + r^{port}) / (1 + r^{bench})$.

## Goal catch-up

$g$ is a goal over the whole horizon, so in tracking mode (`CATCH_UP["on"]`) each month asks for the monthly rate that still reaches it from the cumulative active growth $A$ so far:

$$g_{req} = \operatorname{clip}\!\left( \left( \frac{(1 + g_m)^N}{A} \right)^{1/(N - k)} - 1,\ \text{floor},\ \text{cap} \cdot g_m \right)$$

with `cap = 2`, `floor = 0`. Behind the goal it asks for at most twice the original rate; ahead of it, no less than the benchmark. Return mode ignores the goal in the objective, so catch-up is off.

Example, 3 months from 2022-01, $g$ = 50 bp/yr ($g_m$ = 4.16 bp): January earned +12.8 bp active, already ahead of the 3-month target, so February and March aimed for 0.

## Results

`driver.run(cfg)` saves each run to `results/runs/<yyyy-mm-dd_HHMMSS>_<label>/` (label e.g. `tracking-medium-k150-g50`). `load_run(path)` reads it back, `list_runs()` tabulates every saved run.

| file | grain | contents |
|---|---|---|
| `run.json` | run | time, git commit and whether `src` was dirty, package versions, inputs, resolved `Settings`, horizon, params cache key, summary |
| `monthly.parquet` | rebalance | columns below |
| `holdings.parquet` | (date, cusip) | every bond held before or after the trade: `x_init`, `x`, `b`, realised `r`, `fallback`, `c`, `mu_hat`, `D`, sector, asset type |
| `sectors.parquet` | (date, sector) | `w_port`, `w_bench`, gross return contributions `contrib_port` ($x_i r_i$), `contrib_bench` ($\hat b_i r_i$) |
| `risk.parquet` | (date, factor) | ex-ante active variance per curve PC ($(F_{pc}^\top a)^2$), credit factor and residual sector ($\sum \psi_i a_i^2$), monthly |

`monthly.parquet` columns:

| column | meaning |
|---|---|
| `ret_port`, `ret_gross`, `ret_bench`, `active` | monthly net and gross portfolio, benchmark and active return |
| `growth_active` | cumulative $A$ after the month |
| `cost`, `exit_cost`, `n_exits` | total cost share of wealth, the forced-sale part, bonds force-sold |
| `turnover` | sum of buys, $\sum \max(0, x_0 - x^{init})$ (1.0 the first month) |
| `n_hold`, `n_cand`, `n_bonds` | bonds held, $\lvert S \rvert$, benchmark bonds in `P` |
| `share_fallback` | share of benchmark bonds whose realised return fell back to $\hat\mu$ |
| `active_share` | $\tfrac12 \sum \lvert a_i \rvert$ |
| `te_fcst_ann`, `te_cap_ann` | ex-ante annualized TE, and the cap after widening |
| `port_D`, `bench_D`, `act_D`, `act_DTS`, `act_C`, `act_krd_<tenor>` | portfolio, benchmark and active exposures |
| `goal_req`, `robust_active`, `shortfall` | goal this solve aimed for, $\rho_0$, shortfall against `goal_req` (monthly) |
| `goal_dropped`, `mip_skipped`, `delta_scale`, `status`, `seconds` | solve diagnostics |

`summary` in `run.json` (`results.summarize`): cumulative portfolio and benchmark returns, annualized active return $\left(\prod(1+r^{port}) / \prod(1+r^{bench})\right)^{12/N} - 1$ against $g$ and whether it was met, realised TE (annualized standard deviation of monthly active returns), information ratio, hit rate, maximum drawdown of $A$, average forecast TE against the cap, turnover (with and without the first month), cost, holdings, active share, shortfall, widened and MIP-skipped months, fallback share and solve time.

`driver.sweep(cfg, grid)` (or `--sweep risk_tol=low,medium,high mode=tracking,return`) runs every combination, sharing the loaded data and params cache. A combination whose goal fails the information ratio check (see [05_settings.md](05_settings.md)) is skipped with its message. Runs go to `results/sweeps/<time>/<label>/`, with `sweep.csv` holding one summary row per run.
