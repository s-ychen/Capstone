"""Step 1: iShares AGG holdings -> bond universe (one row per CUSIP, descriptive column names)."""
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from . import config as C

COLS = {"CUSIP": "cusip", "Name": "name", "Sector": "ishares_sector", "Asset Class": "asset_class",
        "Market Value": "agg_market_value", "Weight (%)": "agg_weight_pct", "Par Value": "agg_par_value",
        "Maturity": "agg_maturity_date", "Coupon (%)": "agg_coupon_pct", "Duration": "agg_duration",
        "Mod. Duration": "agg_mod_duration", "YTM (%)": "agg_ytm_pct"}


def download_holdings(path):
    """Download the latest holdings CSV from the iShares fund page."""
    import requests                                  # imported only when downloading
    from bs4 import BeautifulSoup
    with requests.Session() as s:
        s.headers.update(C.HEADERS)
        page = s.get(C.AGG_URL, timeout=30)
        page.raise_for_status()
        links = BeautifulSoup(page.text, "html.parser").find_all("a", href=True)
        href = next((a["href"] for a in links if C.LINK_TEXT.search(a.get_text(" ", strip=True))), None) \
            or next(a["href"] for a in links if "holdings" in a["href"].lower() and "csv" in a["href"].lower())
        resp = s.get(urljoin(page.url, href), headers={"Referer": page.url}, timeout=60)
        resp.raise_for_status()
        if "html" in resp.headers.get("Content-Type", "").lower():
            raise ValueError("Got HTML instead of a CSV (possibly blocked).")
    Path(path).write_bytes(resp.content)


def read_holdings(path):
    """Parse the CSV (header row found automatically). Returns (df, holdings_as_of_date)."""
    lines = Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines()
    head = next(i for i, l in enumerate(lines) if "CUSIP" in l and "Weight" in l)
    asof = next((pd.to_datetime(l.split(",", 1)[1].strip().strip('"'), errors="coerce")
                 for l in lines[:head] if "as of" in l.lower()), pd.NaT)
    df = pd.read_csv(path, skiprows=head, thousands=",", encoding="utf-8-sig", dtype={"CUSIP": "string"}, on_bad_lines="skip")
    df = df.rename(columns=COLS)[[c for c in COLS.values() if c in df.rename(columns=COLS).columns]]
    df["cusip"] = df["cusip"].str.strip().str.upper().str.zfill(9)
    total = df["agg_weight_pct"].sum()
    df = df[df["cusip"].str.fullmatch(r"[A-Z0-9]{9}", na=False)].drop_duplicates("cusip").copy()   # drops cash rows and footer
    print(f"  holdings weight in file: {total:.1f}%, kept after CUSIP filter: {df['agg_weight_pct'].sum():.1f}%")
    df["agg_maturity_date"] = pd.to_datetime(df["agg_maturity_date"], errors="coerce", format="mixed")  # "mixed": May/Sep parse per value
    df["agg_weight_raw"] = df["agg_weight_pct"] / 100.0
    df["cusip8"] = df["cusip"].str[:8]
    return df.reset_index(drop=True), asof


def classify(df):
    """Map each iShares row to one of C.MODEL_SECTORS from its Sector label (NaN = outside the model: agencies, CMBS, ABS, ...)."""
    label = df["ishares_sector"].fillna("").str.lower()
    sector = pd.Series(pd.NA, index=df.index, dtype="object")
    for model_sector, pattern in C.SECTOR_RULES:
        sector[sector.isna() & label.str.contains(pattern)] = model_sector
    return sector


def build_universe(download=True):
    """Returns (bond_universe_df, holdings_as_of_date)."""
    path = C.DATA_DIR / "raw" / "latest_AGG_holdings.CSV"
    path.parent.mkdir(parents=True, exist_ok=True)
    if download:
        download_holdings(path)
    raw, asof = read_holdings(path)
    raw["model_sector"] = classify(raw)
    print(raw.assign(model_sector=raw["model_sector"].fillna("(excluded)")).groupby(["ishares_sector", "model_sector"])
          .agg(n=("cusip", "size"), weight_pct=("agg_weight_pct", "sum")).round(2))      # check the sector mapping
    return raw[raw["model_sector"].isin(C.MODEL_SECTORS)].reset_index(drop=True), asof
