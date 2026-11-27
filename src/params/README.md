# src/params

Computes the optimizer inputs of the D3 multi-period tracking model for one rebalance date. It does not optimize, backtest or measure performance. Proof of concept: MBS inputs are rough (sparse data, no prepayment model).

## Call

```python
from src.params import build_inputs
P = build_inputs(asof, W, H=2, G=None, t_index=0, N=None,
                 sigma_method="ledoit_wolf", p=3, freq="M", window=None)
```

| arg | meaning |
|---|---|
| `asof` | rebalance date (month-end, must have a fitted curve; data ends 2025-12-31) |
| `W` | current wealth in dollars, sets the minimum-lot weights |
| `H` | planning horizon in rebalances |
| `G`, `t_index`, `N` | terminal wealth goal, months elapsed, total rebalances. Give all three to get `r_req`, else it is NaN |
| `sigma_method` | `sample`, `ewma`, `ledoit_wolf` for the key-rate covariance |
| `p` | principal components kept (3) |
| `freq`, `window` | monthly (`M`) or daily (`D`, scaled to monthly) curve changes, optional lookback in periods |

Reads `data/processed/` only (bond_universe, agg_holdings_monthly, prices_daily, yield_curve_daily). CLI summary: `uv run python -m src.params --asof 2025-06-30`.

## Output

`Params` (fixed across the horizon, arrays over the `n` bonds in `P.cusips`):

| field | what | how |
|---|---|---|
| `b` | benchmark weights b_t | AGG weights from the last holdings file on or before `asof` (at a month-end, the file dated `asof`), renormalised over bonds priced at `asof` |
| `model.Sigma`, `Gamma`, `Lambda` | 8x8 key-rate covariance, top-p eigenvectors and eigenvalues | month-end changes of the 8 fitted zero yields up to `asof`, then PCA |
| `model.psi` | residual variance per bond | variance of (month-end return + K'dy); stale prices excluded; under 6 observations uses sector median |
| `c` | proportional trading cost | half the last bid-ask spread (90 days), sector median fallback |
| `theta_F`, `theta_psi` | estimation-error covariance in factor form | Q / T_s (assumption, D3 does not define it) |
| `stale`, `spread_solved`, `n_resid_obs` | data quality flags | |

`P.periods[h]` is a `PeriodInputs` for planning date t+h, h = 0..H-1:

| field | model symbol | how |
|---|---|---|
| `b_hat` | b_hat | drift previous b_hat by (1 + r), zero the exits, renormalise |
| `mu_hat` | mu_hat | one-month return: reprice at next month-end off the same curve, plus coupons paid |
| `K` (n x 8), `D`, `D_spr`, `C`, `dts` | K, D, D_spr, C | reprice off the NSS curve plus the bond's spread; 1 bp triangular bump per key rate for K, parallel bump for D, spread bump for D_spr, +/-10 bp for C; `dts` = D_spr x spread |
| `F` (n x p) | Q_hat = F F' + diag(psi) | K Gamma Lambda^(1/2). Q is never formed |
| `exits` | E | under one year to maturity at that date, forces z = 0 |
| `ell` | l_i | min denomination x dirty price / 100 / wealth |
| `wealth`, `r_req` | W_{t+h}, r_req | wealth grown at forecast benchmark return; (G / W)^(1/(N - t)) - 1 |
| `drift_gross`, `r_b` | R_hat, r^b | (1 + r) and b'r from the previous period; None / NaN at h = 0 |

Units: monthly, decimal returns, durations in years per decimal yield change. Tracking variance of active weights a: `||F'a||^2 + sum(psi a^2)` (`tracking_variance`).

## Multi-period

Yes. Each `PeriodInputs` is built for its own date by aging every bond under today's curve held constant (D3 Section 2.3). Aging changes b_hat, mu_hat, K/D/D_spr/C, F, exits, ell and r_req. Sigma and psi do not change with h.

The drifted portfolio x_tilde depends on the decision x_{t+h-1}, so the optimizer builds it: `x_tilde[h] = drift_gross[h] * x[h-1] / (1 + r_b[h])` (`drift_portfolio` does the same for fixed x). At h = 0, x_tilde is the actual current holdings.

Not forecast: yield curve moves, Treasury entries (needs auction data), defaults and downgrades, prepayments.

## Where this sits

| piece | status |
|---|---|
| Parameter estimation (this module) | done |
| MISOCP (Gurobi) objective and constraints | not started |
| Inputs the solver still needs: delta tolerances from risk aversion, k_max, omega_h, eps_2, transaction cost scaling | not started |
| Rolling backtest, tracking error and goal feasibility, estimator grid search | not started |

Next step is a function that takes `Params`, the current holdings and the user inputs, builds the model from `periods`, and returns trades for t.

## Known gaps

- Spread duration equals duration (constant additive spread), so those two D3 constraints coincide. Use `dts` if a distinct spread-risk measure is wanted.
- MBS: bullet approximation, duration far above AGG's, psi mostly the sector median.
- Sigma comes from about 17 to 24 monthly observations, so it is noisy.
- AGG drops bonds under one year to maturity at each month-end, so `exits` is usually empty at h = 0.
