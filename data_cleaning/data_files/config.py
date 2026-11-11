import re
from pathlib import Path

WRDS_USERNAME = "rosanagao"
START_DATE, END_DATE = "2025-01-01", "2025-12-31"   # price history. Fixed end: WRDS CRSP Treasury quotes stop at 2025-12-31 (None = today)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"                         # data/raw = downloads + cached pulls, data/processed = outputs

# --- iShares AGG universe
AGG_URL = "https://www.ishares.com/us/products/239458/ishares-core-total-us-bond-market-etf"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"}
LINK_TEXT = re.compile(r"detailed holdings and analytics|download latest fund holdings", re.I)
SECTOR_RULES = [("treasury", r"treasur"), ("industrial", r"industrial"), ("financial", r"financial"),
                ("utility", r"utilit"), ("mbs_passthrough", r"mbs pass")]      # regex on the iShares "Sector" label; VERIFY vs printed table
MODEL_SECTORS = ["treasury", "industrial", "financial", "utility", "mbs_passthrough"]
CORP_SECTORS = ["industrial", "financial", "utility"]

# --- WRDS tables
BONDRET_TABLE, CRSP_ISS, CRSP_DLY = "wrdsapps_bondret.bondret", "crsp.tfz_iss", "crsp.tfz_dly"
MBS_MASTER, TRACE_TABLE = "trace_enhanced.mbsmasterfile", "trace.trace_enhanced"            # enhanced corporate TRACE (ends ~18 months before today)
TRACE_MBS_TABLE = "trace.trace_spds_mbs_enhanced"   # agency pass-through pools; the standard SPDS table has no CUSIP, so enhanced is the only option
CRSP_EXCLUDE_ITYPES = (11, 12)                                                               # TIPS; VERIFY
TRACE_COLS = {"cusip": "cusip_id", "trade_date": "trd_exctn_dt", "trade_time": "trd_exctn_tm", "report_date": "trd_rpt_dt",
              "price": "rptd_pr", "quantity": "entrd_vol_qt", "yield_pct": "yld_pt", "side": "rpt_side_cd",
              "contra": "cntra_mp_id", "status": "trc_st", "msg_seq": "msg_seq_nb", "orig_msg_seq": "orig_msg_seq_nb",
              "wi": "wis_fl", "locked": "lckd_in_ind", "cond1": "sale_cndtn_cd", "cond2": "sale_cndtn2_cd",
              "settle_days": "days_to_sttl_ct", "settle_date": "stlmnt_dt",
              "pool_factor": "factor_on_file"}                                              # columns missing from a table are filled with NaN

# --- conventions and assumptions
KEY_RATES = {"6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "30Y": 30.0}
DAYS_PER_YEAR = 365.25
DAY_COUNT = {"treasury": "ACT/ACT", "industrial": "30/360", "financial": "30/360", "utility": "30/360", "mbs_passthrough": "30/360"}
COUPON_FREQ = {"treasury": 2, "industrial": 2, "financial": 2, "utility": 2, "mbs_passthrough": 12}
MIN_DENOM = {"treasury": 100.0, "industrial": 1000.0, "financial": 1000.0, "utility": 1000.0, "mbs_passthrough": 1.0}   # ASSUMED
HALF_SPREAD_BPS = {"treasury": 2.0, "industrial": 25.0, "financial": 25.0, "utility": 25.0, "mbs_passthrough": 5.0}   # PLACEHOLDER
YIELD_FREQ = 2                     # yields are semiannual bond-equivalent
MBS_CPR, MBS_MAX_MONTHS = 0.0, 360   # pass-through cash flows: level-pay amortisation, constant prepayment

# --- month-end prices
WINDOW_DAYS = 5                    # last price within the final 5 trading days
MAX_STALE_MONTHS = None            # None = carry stale prices forward indefinitely
BONDRET_SPREAD_TO_BPS = 100.0      # VERIFY: BondRet t_spread assumed in percentage points

# --- zero curve
CURVE_FREQ = "month_end"           # "month_end" (fast, enough for monthly rebalancing) or "daily"
CURVE_MIN_BONDS, CURVE_SIGMAS, CURVE_PASSES = 20, 5.0, 2
NSS_START = [0.04, -0.02, 0.0, 0.0, 2.0, 10.0]
NSS_LOW = [0.0, -0.20, -0.30, -0.30, 0.20, 1.0]
NSS_HIGH = [0.20, 0.20, 0.30, 0.30, 10.0, 40.0]

# --- TRACE cleaning
TRACE_MARGIN_DAYS = 4              # extra trading days pulled before each month-end window (for the 9-day median)
TRACE_PRICE_DEV = 0.10
TRACE_MAX_SETTLE_DAYS = 2          # set to None to skip (MBS often settle later)
TRACE_OK_CONDITIONS = ("", "@")
