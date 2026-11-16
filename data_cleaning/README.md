# data_cleaning

Builds the daily bond dataset for the GoalPath-Bond model from WRDS, starting from the ideal universe in
`data/AGG_holdings_clean.csv` (iShares AGG holdings). Everything is keyed on the 9-character CUSIP and the
trading date.

```sh
uv run python -m data_cleaning.run_pipeline          # pulls from WRDS once, caches to data/raw, writes data/processed
uv run python -m data_cleaning.run_pipeline --refresh # ignore the caches and re-pull
uv run python -m data_cleaning.validate               # sanity checks + comparison against Dickerson's cleaned prices
```

WRDS login: `WRDS_USERNAME` in the repo `.env`, with the password in `~/.pgpass` (the `wrds` package offers to
create it on the first interactive connection). The full run pulls ~55M TRACE trade rows and takes 15-30 minutes;
later runs use the parquet caches in `data/raw/` (both `data/raw` and `data/processed` are gitignored).

## Pipeline

| step | module | what it does |
|---|---|---|
| 1 | `universe.py` | Parse the AGG file; keep Treasury, Industrial, Financial Institutions, Utility and MBS Pass-Through (sectors in `config.SECTOR_MAP`, ~94% of the fund). |
| 2 | `treasuries.py` | CRSP daily Treasury quotes (`crsp.tfz_dly` x `tfz_iss`). Drops TIPS/callables, mid = (bid+ask)/2, dirty = mid + CRSP accrued, 8-char CRSP CUSIPs get their check digit. |
| 3 | `curve.py` | Nelson-Siegel-Svensson zero curve fitted to **all** nominal Treasury dirty prices each day (duration-weighted, outliers down-weighted); zero yields read off at the 8 key-rate tenors. |
| 4 | `corporates.py`, `trace_clean.py` | Enhanced TRACE trades for the AGG corporates, cleaned trade-by-trade (Dick-Nielsen cancels/corrections/reversals, inter-dealer duplicates, non-standard and long-settlement trades, Edwards-Harris-Piwowar median + reversal filters), collapsed to one row per bond-day. Terms from FISD (`fisd.fisd_mergedissue`), accrued interest from the coupon schedule (30/360). Dickerson's daily analytics (`contrib_bond_dickerson`) are merged in as `dk_*` columns where available. |
| 4 | `mbs.py` | Same cleaning on TRACE SPDS enhanced pool trades (`trace_enhanced.trace_spds_mbs_enhanced`) and TBA trades (`..._tba_enhanced`). Pool quantities are converted to current face with the pool factor on file. |
| 5 | `features.py` | Yield to maturity (semiannual) and the constant spread over the same-day zero curve, both solved from the dirty price and the bond's cash flows. |
| 6 | `merge.py` | Bond universe with terms, coverage flags and renormalised weights; the stacked `prices_daily` table. |

`config.py` holds the date window, sector map, conventions and table names. `wrds_pull.py` is the connection and
cache layer; `finmath.py` the coupon-schedule / accrued / cash-flow maths.

## Output tables (`data/processed/*.parquet`)

**`bond_universe`** - one row per AGG bond in the modelled sectors (12,069 bonds, holdings as of 2026-10-12).
AGG snapshot (`agg_market_value`, `agg_par_value`, `agg_duration`, `agg_ytm_pct`, ...), terms (`maturity_date`,
`coupon_rate`, `coupon_frequency`, `day_count`, `dated_date`, `min_denomination`, `is_callable`, ...; `terms_source`
says whether they came from CRSP, FISD or the AGG file), coverage (`has_prices`, `n_price_days`, `first/last_price_date`)
and weights:
- `agg_weight` = market-value share of the whole fund (recomputed; the file's own Weight column is rounded to 2 dp)
- `model_weight` = `agg_weight` renormalised over bonds with at least one price, so it sums to 1 (this is b_t)

**`prices_daily`** - the master price table, one row per (cusip, date) **on days with an observation**: Treasuries
have a quote every trading day, corporates/MBS only on days they traded. Nothing is forward-filled; month-end
selection and stale-price carry-forward are left to the model code (see the Deliverable 3 step 5 rules).

| column | meaning |
|---|---|
| `asset_type`, `sector`, `source` | treasury / corporate / mbs; AGG sector; crsp / trace_enhanced / trace_mbs / trace_tba |
| `clean_price` | per 100 face: CRSP bid-ask mid, or face-weighted average TRACE trade price |
| `accrued_interest`, `dirty_price` | accrued **at the quote/trade date** (CRSP convention; Dickerson's `dk_accrued_interest` is at T+2 settlement). dirty = clean + accrued |
| `yield` | reported: CRSP yield (annualised) or face-weighted TRACE `yld_pt`; NaN for MBS |
| `ytm_solved` | semiannual yield solved from the dirty price and bullet cash flows (NaN for MBS) |
| `spread_bps` | constant spread over the same-day fitted zero curve that reprices the dirty price (NaN for MBS and after the curve ends) |
| `bid_ask_spread`, `bid_ask_spread_bps` | CRSP ask-bid; for TRACE, dealer-sell-to-customer VWAP minus dealer-buy-from-customer VWAP (NaN unless both sides traded that day) |
| `par_volume`, `dollar_volume`, `trade_count` | face traded (current face for MBS pools), its dollar value, number of clean trades. NaN for CRSP quotes |
| `years_to_maturity` | from the terms |

**`treasury_daily`**, **`corporate_daily`**, **`mbs_daily`** - the same rows with every source-specific column
(CRSP duration/return/outstanding; TRACE low/high/bid/ask prices and the `dk_*` Dickerson analytics such as
`dk_modified_duration`, `dk_convexity`, `dk_credit_spread`; MBS `pool_factor`, `is_tba`).

**`yield_curve_daily`** - one row per CRSP trading day: `zero_6m ... zero_30y` (decimal, continuously compounded),
the NSS parameters and fit diagnostics. Its `date` column is also the trading calendar.

**`trace_cleaning_log`** - trades remaining after each cleaning step, per product.

**`mbs_monthly`** (from `uv run python -m data_cleaning.mbs_monthly`, built on `mbs_daily`) - a gap-free month-end
panel for the MBS pools: last trade of the month as the price (`price_source` = `traded_window` if within the final
5 trading days, else `traded_month`), months with no trade filled by moving the pool's last traded price with the
median price change of its coupon cohort (`cohort_proxy`: same GNMA/conventional program, term and coupon; flat
`carried` when no cohort traded). `is_stale` is True unless `traded_window`. Bid-ask spread is the month's observed
mean, else the pool's last observed, else the cohort median (`bid_ask_source`). The pool factor is only known on
trade days and is carried forward (`pool_factor_date`), so current face is overstated for pools that stop trading.
An incomplete final month is dropped. On traded months held out as a test, the cohort proxy has a median error
of 0.6 price points against 1.3 for a flat carry.

## Coverage and caveats

- WRDS end dates differ by source (Nov 2026): CRSP Treasuries 2025-12-31, TRACE corporates 2026-03-17, TRACE MBS
  pools 2025-03-17, TBAs 2025-01-29, Dickerson 2025-03-31. The curve (and `spread_bps`) therefore stops at 2025-12-31.
- The universe is the Oct 2026 holdings file, so bonds issued after a source's end date have no history
  (about 16% of AGG's Treasury market value is 2026 issues).
- Coverage of the modelled sectors by AGG weight is printed by `validate.py`.
- Not covered: CMBS, ABS, sovereign, supranational, agency and local-authority holdings (~5% of AGG). Agency debentures
  are in `trace.trace_agency` with a different schema; the same cleaning could be pointed at it if ever needed.
- Callable and fixed-to-float corporates (`coupon_type == "V"`) are priced to maturity with their current coupon.
- MBS pass-through yields/spreads need a prepayment model and are left NaN; AGG's stated coupon/maturity are used for accrued.
- `wrdsapps_bondret` (monthly) is not used: it is derived from the same TRACE data at month-end frequency.
