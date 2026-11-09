"""Step 6: bond terms, Bond Features table (terms + AGG weights), and the optional Treasury auction announcements."""
import numpy as np
import pandas as pd

from . import config as C


def clean_mbs_master(raw, valuation_date):
    """One MBS reference row per CUSIP: prefer the record active on valuation_date, then the latest start."""
    d = raw.assign(cusip=raw["cusip_id"], start=pd.to_datetime(raw["stdt"]), end=pd.to_datetime(raw.get("enddt")),
                   mbs_maturity_date=pd.to_datetime(raw["mtrty_dt"]), mbs_coupon_pct=pd.to_numeric(raw["cpn_rt"], errors="coerce"),
                   mbs_issue_date=pd.to_datetime(raw.get("issue_dt")))
    d = d[d["start"].isna() | (d["start"] <= valuation_date)]
    d["active"] = (d["end"].isna() | (d["end"] >= valuation_date)).astype(int)
    d = d.sort_values(["cusip", "active", "start"], na_position="first").drop_duplicates("cusip", keep="last")
    return d[["cusip", "mbs_issue_date", "mbs_maturity_date", "mbs_coupon_pct"]]


def build_terms(universe, bondret, tsy_quotes, mbs_terms):
    """
    One row per universe bond: issue_date, maturity_date, coupon_pct, coupon_frequency, day_count, is_amortising, min_denomination.
    Sources: CRSP (Treasuries), BondRet (corporates), TRACE MBS master (MBS); the AGG file is the fallback (terms_source).
    """
    c = bondret.assign(date=pd.to_datetime(bondret["date"])).sort_values("date").drop_duplicates("cusip", keep="last")
    corp = pd.DataFrame({"cusip": c["cusip"], "c_issue": pd.to_datetime(c["dated_date"]).fillna(pd.to_datetime(c["offering_date"])),
                         "c_mat": pd.to_datetime(c["maturity"]), "c_cpn": pd.to_numeric(c["coupon"], errors="coerce"),
                         "c_freq": pd.to_numeric(c["ncoups"], errors="coerce")})
    t = tsy_quotes.dropna(subset=["cusip"]).drop_duplicates("cusip")[["cusip", "issue_date", "maturity_date", "coupon_pct"]]
    t.columns = ["cusip", "t_issue", "t_mat", "t_cpn"]
    m = mbs_terms.rename(columns={"mbs_issue_date": "m_issue", "mbs_maturity_date": "m_mat", "mbs_coupon_pct": "m_cpn"})
    d = universe[["cusip", "model_sector", "agg_coupon_pct", "agg_maturity_date"]].merge(corp, how="left").merge(t, how="left").merge(m, how="left")
    first = lambda *cols: d[list(cols)].bfill(axis=1).iloc[:, 0]
    d["issue_date"], d["maturity_date"], d["coupon_pct"] = first("t_issue", "c_issue", "m_issue"), first("t_mat", "c_mat", "m_mat"), first("t_cpn", "c_cpn", "m_cpn")
    d["terms_source"] = np.where(d["maturity_date"].notna() & d["coupon_pct"].notna(), "wrds", "agg_fallback")
    d["maturity_date"], d["coupon_pct"] = d["maturity_date"].fillna(d["agg_maturity_date"]), d["coupon_pct"].fillna(d["agg_coupon_pct"])
    d["coupon_frequency"] = d["c_freq"].where(d["c_freq"] > 0).fillna(d["model_sector"].map(C.COUPON_FREQ)).astype(int)
    d["day_count"], d["min_denomination"] = d["model_sector"].map(C.DAY_COUNT), d["model_sector"].map(C.MIN_DENOM)
    d["is_amortising"] = d["model_sector"].eq("mbs_passthrough")
    return d[["cusip", "model_sector", "issue_date", "maturity_date", "coupon_pct", "coupon_frequency", "day_count",
              "is_amortising", "min_denomination", "terms_source"]]


def build_features(universe, terms, prices):
    """
    Bond Features table = AGG snapshot + terms, restricted to bonds with terms and at least one price; weights renormalised to sum to 1.
    Returns (bond_features_df, dropped_df).
    """
    agg = [c for c in universe.columns if c.startswith("agg_") and c not in ("agg_coupon_pct", "agg_maturity_date", "agg_weight_pct")]
    f = universe[["cusip", "name", "ishares_sector", "asset_class"] + agg].merge(terms, on="cusip")
    has_terms, has_price = f["maturity_date"].notna() & f["coupon_pct"].notna(), f["cusip"].isin(prices["cusip"])
    f["drop_reason"] = np.select([~has_terms, ~has_price], ["missing maturity/coupon", "no price in any month"], "")
    dropped = f[f["drop_reason"] != ""][["cusip", "name", "model_sector", "agg_weight_raw", "drop_reason"]]
    f = f[f["drop_reason"] == ""].drop(columns="drop_reason")
    f["agg_weight"] = f["agg_weight_raw"] / f["agg_weight_raw"].sum()                        # so that 1'b = 1
    return f.reset_index(drop=True), dropped.reset_index(drop=True)


def fetch_treasury_auctions():
    """Upcoming/announced Treasury auctions (TreasuryDirect JSON API; VERIFY field names). Empty frame on failure."""
    import requests                                  # imported only when fetching auctions
    cols = {"cusip": "cusip", "securityType": "security_type", "securityTerm": "security_term", "announcementDate": "announcement_date",
            "auctionDate": "auction_date", "issueDate": "issue_date", "maturityDate": "maturity_date", "offeringAmount": "offering_amount"}
    frames = []
    for kind in ("upcoming", "announced"):
        try:
            r = requests.get(f"https://www.treasurydirect.gov/TA_WS/securities/{kind}?format=json", headers=C.HEADERS, timeout=30)
            r.raise_for_status()
            frames.append(pd.DataFrame(r.json()))
        except Exception as e:
            print(f"  auctions ({kind}) unavailable: {e}")
    if not frames:
        return pd.DataFrame(columns=list(cols.values()))
    d = pd.concat(frames, ignore_index=True).rename(columns=cols)
    d = d[[c for c in cols.values() if c in d]].drop_duplicates()
    for c in [c for c in d if c.endswith("_date")]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    return d.reset_index(drop=True)
