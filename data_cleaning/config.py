"""Settings for the data pipeline. Edit here, not in the step modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # cached WRDS pulls (gitignored)
PROCESSED_DIR = DATA_DIR / "processed"  # pipeline outputs (gitignored)
AGG_HOLDINGS_CSV = DATA_DIR / "AGG_holdings_clean.csv"

# Price history window (inclusive). Each source simply stops where WRDS stops; as of Nov 2026:
#   CRSP Treasury quotes        -> 2025-12-31   (so the zero curve, and spreads over it, end there)
#   TRACE enhanced (corporates) -> 2026-03-17
#   TRACE SPDS enhanced (MBS)   -> 2025-03-17, TBA table -> 2025-01-29
#   Dickerson daily corporates  -> 2025-03-31
START_DATE = "2024-01-01"
END_DATE = "2026-03-31"
WINDOW_TAG = f"{START_DATE}_{END_DATE}"   # suffix on data/raw caches, so changing the window re-pulls

# iShares "Sector" label -> model sector. Sectors not listed are outside the universe
# (CMBS, ABS, sovereign, supranational, agency, local authority: ~5% of AGG, no daily source we process).
SECTOR_MAP = {
    "Treasury": "treasury",
    "Industrial": "industrial",
    "Financial Institutions": "financial",
    "Utility": "utility",
    "MBS Pass-Through": "mbs_passthrough",
}
ASSET_TYPE = {"treasury": "treasury", "industrial": "corporate", "financial": "corporate",
              "utility": "corporate", "mbs_passthrough": "mbs"}

# Conventions used when a source does not report accrued interest itself (FISD overrides these for corporates).
DAY_COUNT = {"treasury": "ACT/ACT", "corporate": "30/360", "mbs": "30/360"}
COUPON_FREQ = {"treasury": 2, "corporate": 2, "mbs": 12}

# Key-rate tenors (years) for the fitted Treasury zero curve.
KEY_RATES = {"6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "30Y": 30.0}

# WRDS tables
CRSP_ISSUES, CRSP_DAILY = "crsp.tfz_iss", "crsp.tfz_dly"
FISD_ISSUES = "fisd.fisd_mergedissue"
TRACE_CORP = "trace.trace_enhanced"
TRACE_MBS = "trace_enhanced.trace_spds_mbs_enhanced"
TRACE_TBA = "trace_enhanced.trace_spds_tba_enhanced"
DICKERSON_DAILY = "contrib_bond_dickerson.dickerson_bonds_daily"
TRACE_CHUNK_SIZE = 250          # CUSIPs per TRACE query (about 5M raw rows per chunk for liquid corporates)
