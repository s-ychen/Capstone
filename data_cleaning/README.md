Process

This folder has all the scripts that a) pull data b) process it c) make the necessary features for the model.

```run_pipeline.py``` is the main script ran, with params --refresh, --skip-mbs, --skip-corporates

```holdings.py``` is the script that gets the historical AGG portfolios from download url

```treasury_curve.py``` is the script that takes the treasury data, cleans it, and fits a yield curve to it using NSS least squares (as we found this was a good way to fit the curve from literature)

```merge.py```  builds the output tables (bond universe, holdings monthly, prices daily + monthly)

Helpers:

```pull.py``` is the actual WRDS calls

```panels.py``` is more data engineering steps, to construct panels of data where each bond has columns for monthly/daily freq (so bond x date)

```trace.py``` has the TRACE trade-level data cleaning. Note we do not use Bond-Dickerson for a multitude of reasons: it cuts off at January 2025, it does not process MBS trades and all the corporates in sources such as FISD. Since we are going to have to do the TRACE data processing anyways, we just take the data processing/trade filter techniques from the Dickerson website and do it ourselves on all the data.

```bond_math.py``` is  helpers for feature engineering e.g if a data source doesn't report yield or accrued interest, we calculate it.

```config.py``` is where all the common used params are stored. Make edits  here for stuff such as start date of the data pull and directory names.

# data_cleaning

Builds the monthly bond dataset from WRDS, plus the daily Treasury curve. The ideal bond universe is AGG's actual monthly holdings, which we query in ```holdings.py```. We get the latest end of previous month portfolio holdings (last trading day), for our current month's model such that there is no forward leakage. All the bonds are keyed using the CUSIP9 code and month.

Prices are saved twice: `prices_monthly` (****one row per bond per held month, the modelling table) and `prices_daily`
(every observed bond-day in a held month, no filling). The yield curve is modeled daily.

Raw data is pulled for `config.START_DATE`-`END_DATE` (2016-01 to 2026-03, cached in `data/raw`), but output starts at
`config.OUTPUT_START` (2017-08). AGG's download link has no AGG files from 2017-01-03 to 2017-07-05, so
Feb-Jul 2017 have no start-of-month portfolio. The earlier history still feeds carried prices, the MBS cohort index and
the trade filters.

```sh
uv run python -m data_cleaning.run_pipeline          # pulls from WRDS once, caches to data/raw, writes data/processed
uv run python -m data_cleaning.run_pipeline --refresh # ignore the caches and re-pull
uv run python -m data_cleaning.validate               # sanity checks
```


WRDS login: in .env, put your WRDS_USERNAME and password. Requires 2FA with Duo set up. This takes a while to run (maybe few hours). This is all cached though so only needed to be run once.


The rest of the data cleaning takes maybe an hour itself. All the data is cached in fixed hash buckets so if we want to extend the window or add a month it will only re-pull the cumulative, not the entire thing all over again. 

## Pipeline

```run_pipeline.py```: run this script, with `config.py` holds the date window, sector map, conventions and table names.


| step | module              | what it does                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ---- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `holdings.py`       | For every month m, download AGG's holdings on the last trading day of m-1 from BlackRock's fund-document API (`config.AGG_HOLDINGS_API`, cached in `data/raw/agg_holdings/`); keep Treasury, Industrial, Financial Institutions, Utility and MBS Pass-Through (`config.SECTOR_MAP`, ~95% of bond value). The union of all months' bonds is what gets pulled from WRDS.                                                                                                             |
| 2    | `pull.py`           | Using the union of all months' bonds is what gets pulled from WRDS. Called from different data sources that are part of WRDS: TRACE/CRSP/FISD.                                                                                                                                                                                                                                                                                                                                     |
| 3    | `treasury_curve.py` | Drop TIPS/callables, mid = (bid+ask)/2, dirty = mid + CRSP accrued, 8-char CRSP CUSIPs get their check digit. Do NSS formula to fit to all observed Treasury dirty prices each day (duration-weighted, outliers down-weighted to zero); zero yields read off at the 8 key-rate tenors.**Saved daily.**                                                                                                                                                                          |
| 4    | `trace.py`          | Enhanced TRACE trades for the corporates, cleaned trade-by-trade (Dick-Nielsen cancels/corrections/reversals, inter-dealer duplicates, non-standard and long-settlement trades, Edwards-Harris-Piwowar median + reversal filters), collapsed to one row per bond-day in memory. Terms from FISD (`fisd.fisd_mergedissue`).                                                                                                                                                         |
| 4    | `trace.py`          | Same cleaning on TRACE SPDS enhanced pool (MBS in our case) trades (`trace_enhanced.trace_spds_mbs_enhanced`). Pool quantities are converted to current face with the pool factor on file.                                                                                                                                                                                                                                                                                         |
| 5    | `panels.py`         | Treasuries and corporates: basically we want to construct the end dataframes to have all the same shape so we process the data more by doing things. Month-end price per held month (last observation, flagged by whether it fell in the final 5 trading days; carried forward if the bond did not trade), monthly bid-ask and volume, then accrued interest, dirty price, yield to maturity and spread over the zero curve (`bond_math.py`), solved only on these monthly rows.  |
| 5    | `panels.py`         | The same for MBS, except months without a trade are filled from the pool's coupon cohort.                                                                                                                                                                                                                                                                                                                                                                                          |
| 6    | `merge.py`          | Stacks the three panels into`prices_monthly` and the daily rows into `prices_daily`; bond terms and coverage per bond; monthly benchmark weights. Everything is cut to months from `OUTPUT_START`.                                                                                                                                                                                                                                                                                 |

`config.py` holds the date window, sector map, conventions and table names. `pull.py` is the WRDS connection, cache
layer and every WRDS query; `bond_math.py` the coupon-schedule / accrued / cash-flow / yield-solving maths.

## Output tables (`data/processed/*.parquet`)

**`agg_holdings_monthly`** - the benchmark, one row per (month, cusip) in AGG's start-of-month portfolio.
`month` is the first day of the month the holdings apply to; `holdings_date` is the file's date (last trading day
of the previous month). AGG `agg_market_value`, `agg_par_value` and:

- `agg_weight` = market value / the fund's total bond market value on `holdings_date` (recomputed; the file's
  Weight column is rounded to 2 dp). Sums to ~0.95 per month, the excluded sectors are the rest.
- `price_status` = `observed` (quoted or traded that month), `filled` (no trade that month: an earlier price carried
  forward, or an MBS cohort proxy) or `none`
- `model_weight` = `agg_weight` renormalised over bonds with a month-end price (observed or filled), so it sums to 1
  each month (b_t)

**`bond_universe`** - one row per bond held in any month of the window (~16,000 for 2022-2026). When it was held
(`first_held_month`, `last_held_month`, `months_held`), the AGG fields from the latest file it appears in
(`snapshot_date`, `agg_market_value`, `agg_duration`, `agg_ytm_pct`, ...), terms (`maturity_date`, `coupon_rate`,
`coupon_frequency`, `day_count`, `dated_date`, `min_denomination`, `is_callable`, ...; `terms_source` says whether
they came from CRSP, FISD or the AGG file) and coverage while held (`has_prices`, `n_price_months`,
`first/last_price_month`, counting observed months only).

**`prices_monthly`** - the master price table, one row per (cusip, month) for every month the bond was in AGG's
start-of-month portfolio and has a price, from its first observation up to the last complete month of its source.


| column                                                                | meaning                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `month`, `month_end_date`                                             | first day of the month; its last trading day                                                                                                                                                                                                                                                                                                   |
| `asset_type`, `sector`, `source`                                      | treasury / corporate / mbs; AGG sector; crsp / trace_enhanced / trace_mbs / trace_tba                                                                                                                                                                                                                                                          |
| `price_source`, `is_stale`                                            | `traded_window` (last observation in the final 5 trading days: the month-end price), `traded_month` (earlier in the month), `carried` (no observation this month: last price carried forward) or `cohort_proxy` (MBS: moved with its coupon cohort). `is_stale` unless `traded_window`; exclude stale months when estimating residual variance |
| `price_date`, `valuation_date`, `months_since_trade`                  | day the price was observed; day it is valued at (`price_date` if observed this month, else `month_end_date`); months since it last traded                                                                                                                                                                                                      |
| `clean_price`                                                         | per 100 face: CRSP bid-ask mid, or face-weighted average TRACE trade price on`price_date`                                                                                                                                                                                                                                                      |
| `accrued_interest`, `dirty_price`                                     | accrued at`valuation_date` (CRSP's own for quoted Treasuries; coupon schedule otherwise; 30/360 from the 1st for MBS). dirty = clean + accrued                                                                                                                                                                                                 |
| `yield`                                                               | reported on`price_date`: CRSP yield (annualised) or face-weighted TRACE `yld_pt`; NaN when carried and for MBS                                                                                                                                                                                                                                 |
| `ytm_solved`                                                          | semiannual yield solved from the dirty price and bullet cash flows (NaN for MBS)                                                                                                                                                                                                                                                               |
| `spread_bps`                                                          | constant spread over the`valuation_date` zero curve that reprices the dirty price (NaN for MBS and after the curve ends)                                                                                                                                                                                                                       |
| `years_to_maturity`                                                   | at`month_end_date`                                                                                                                                                                                                                                                                                                                             |
| `bid_ask_spread`, `bid_ask_spread_bps`, `bid_ask_source`              | mean of the month's daily spreads: CRSP ask-bid, or for TRACE dealer-sell-to-customer VWAP minus dealer-buy-from-customer VWAP on days with both.`bid_ask_source` = observed / carried (bond's last observed month) / cohort (MBS) / missing                                                                                                   |
| `par_volume`, `dollar_volume`, `trade_count`, `trade_days`            | month totals of face traded (current face for MBS pools), its dollar value, clean trades and days with a trade. 0 for CRSP quotes                                                                                                                                                                                                              |
| `pool_factor`, `pool_factor_date`, `is_tba`, `cohort`, `cohort_index` | MBS only. The pool factor is only known on trade days and is carried forward, so current face is overstated for pools that stop trading                                                                                                                                                                                                        |

**`prices_daily`** - one row per (cusip, date) with a CRSP quote or a clean TRACE trade, on trading days in months
the bond was in AGG's start-of-month portfolio (~11.3M rows, ~680 MB). Nothing is filled: Treasuries appear every
trading day, corporates and MBS only on days they traded. Columns as in `prices_monthly` (valued on `date`): `month`,
`asset_type`, `sector`, `source`, `clean_price`, `accrued_interest`, `dirty_price`, `yield`, `ytm_solved`, `spread_bps`,
`years_to_maturity`, `bid_ask_spread(_bps)`, `par_volume`, `dollar_volume`, `trade_count` (volumes 0 for CRSP
quotes), `pool_factor`, `is_tba`. Unlike `prices_monthly` it runs to each source's last date, including the incomplete
final month. The month-end traded rows of `prices_monthly` are the last daily row of the month.

**`yield_curve_daily`** - one row per CRSP trading day: `zero_6m ... zero_30y` (decimal, continuously compounded),
the NSS parameters and fit diagnostics. Its `date` column is also the trading calendar.

## Coverage and caveats

- WRDS end dates differ by source (Nov 2026): CRSP Treasuries 2025-12-31, TRACE corporates 2026-03-17, TRACE MBS
  pools 2025-03-17, TBAs 2025-01-29. A source's last month is only built if its data reaches that month's final
  5 trading days, so corporates end Feb 2026, MBS Feb 2025 and Treasuries Dec 2025. The curve (and `spread_bps`)
  stops at 2025-12-31.
- Months after a source's end still have holdings, so those bonds get `price_status` = `none` and `model_weight`
  is spread over what is priced. Restrict a backtest to months where every asset type is priced (Aug 2017 to Feb
  2025 with the current sources).
- A bond added to AGG at the end of month m (new issue) only enters the panel in month m+1.
- On MBS months held out as a test, the cohort proxy has a median error of 0.6 price points against 1.3 for a flat
  carry.
- Not covered: CMBS, ABS, sovereign, supranational, agency and local-authority holdings (~5% of AGG)
- Callable and fixed-to-float corporates (`coupon_type == "V"`) are priced to maturity with their current coupon.
- MBS pass-through yields/spreads need a prepayment model and are left NaN; AGG's stated coupon/maturity are used for accrued.
- `wrdsapps_bondret` (monthly) is not used: it is derived from the same TRACE data at month-end frequency.
