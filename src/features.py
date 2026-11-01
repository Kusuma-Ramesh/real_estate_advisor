"""Shared cleaning, feature engineering and *proxy-target* construction.

IMPORTANT - what is learned and what is assumed
-----------------------------------------------
* The CSV contains one observed price per listing (``Price_in_Lakhs``). It contains NO
  future prices and NO "good investment" label.
* ``Good_Investment`` is a RULE-BASED PROXY label (see ``investment_score``). A model trained on it
  measures how well it recovers that rule, not real-world investment performance.
* ``Projected_Price_5Y_Lakhs`` is an ASSUMPTION-BASED projection (price x (1+r)^5, with r from
  ``growth_rate``). It is not learned from data and has no measured forecasting accuracy.
* The only genuinely learned, observed-target task is estimating the *current* price
  (``Price_in_Lakhs``) from non-price attributes - see ``src/train.py``.

Every assumption lives in the constants below so it is easy to audit and change.
"""
import sys

import numpy as np
import pandas as pd

# Snapshot year of the dataset: Age_of_Property + Year_Built == 2025 for every row.
REFERENCE_YEAR = 2025
HORIZON_YEARS = 5
BASE_GROWTH = 0.08            # assumed annual appreciation before adjustments
GROWTH_MIN, GROWTH_MAX = 0.04, 0.12
GOOD_SCORE_THRESHOLD = 5      # out of 8, AND price per sq ft <= city median
TRANSPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}

# Locality is kept in the frame (app filters, EDA) but is not a model input: it has 500 levels
# and every locality id appears in many cities, so it is not a geographic hierarchy.
CATEGORICAL = ["State", "City", "Locality", "Property_Type", "Furnished_Status",
               "Public_Transport_Accessibility", "Facing", "Owner_Type", "Availability_Status"]
MODEL_CATS = [c for c in CATEGORICAL if c != "Locality"]
NUMERIC = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs", "Age_of_Property",
           "Floor_No", "Total_Floors", "Floor_Ratio", "Floor_Inconsistent", "Nearby_Schools",
           "Nearby_Hospitals", "Parking_Space", "Security", "Amenity_Count", "Infra_Score",
           "School_Density_Score"]
PRICE_COLUMNS = ["Price_in_Lakhs", "Price_per_SqFt_Rs"]
# Classifier sees every rule input (the label is a function of them -> rule recovery).
CLS_FEATURES = MODEL_CATS + NUMERIC
# Price regressor must NOT see its own target or anything derived from it.
REG_FEATURES = MODEL_CATS + [c for c in NUMERIC if c not in PRICE_COLUMNS]
ESSENTIAL_COLS = ["Price_in_Lakhs", "Size_in_SqFt", "Year_Built", "City", "State"]


def load_raw(path):
    return pd.read_csv(path)


REQUIRED_COLS = ["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
                 "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
                 "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type",
                 "Availability_Status"]


def check_schema(df):
    """Fail early, with the column names, if the CSV does not have the columns the pipeline needs."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def clean(df):
    """Drop exact duplicates (ignoring ID), rows missing essentials and rows with non-positive size/price
    (they would give infinite price per sq ft); tidy strings. Every drop is reported, never silent."""
    check_schema(df)
    n0 = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "ID"]).copy()
    n_dup = n0 - len(df)
    df = df.dropna(subset=ESSENTIAL_COLS)
    n_na = n0 - n_dup - len(df)
    bad = (df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0)
    if n_dup or n_na or bad.any():
        print(f"[clean] dropped {n_dup} duplicate, {n_na} missing-essential and {int(bad.sum())} non-positive size/price rows", file=sys.stderr)
    df = df[~bad]
    for c in df.select_dtypes(exclude="number").columns:   # works on pandas 2 and 3
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    if "Amenities" in df:
        df["Amenities"] = df["Amenities"].replace("", "None")
    return df.reset_index(drop=True)


def _yes(series):
    """Idempotent Yes/No -> 1/0 (also accepts values that are already 0/1)."""
    return series.astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)


def engineer(df):
    """Create model features. Does NOT create targets. Safe to call on a single new row."""
    df = df.copy()
    # Dates: derive age from Year_Built against ONE reference year so train/app/EDA agree.
    df["Age_of_Property"] = REFERENCE_YEAR - df["Year_Built"]
    # Source Price_per_SqFt is in lakhs rounded to 2dp (0.05-0.1) -> recompute in rupees/sq ft.
    df["Price_per_SqFt_Rs"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]
    # Source has Floor_No > Total_Floors in ~46% of rows. Keep raw values, flag it, bound the ratio.
    df["Floor_Inconsistent"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)
    df["Floor_Ratio"] = (df["Floor_No"] / df["Total_Floors"].clip(lower=1)).clip(upper=1)
    df["Parking_Space"] = _yes(df["Parking_Space"])
    df["Security"] = _yes(df["Security"])
    df["Amenity_Count"] = df["Amenities"].apply(
        lambda s: 0 if s in ("None", "Unknown", "") else len([a for a in s.split(",") if a.strip()]))
    transport = df["Public_Transport_Accessibility"].map(TRANSPORT_MAP).fillna(0)
    # Infrastructure score (about 0.7-10): schools, hospitals, transport.
    df["Infra_Score"] = (0.35 * df["Nearby_Schools"] + 0.35 * df["Nearby_Hospitals"]
                         + 0.30 * transport * 5)
    df["School_Density_Score"] = df["Nearby_Schools"] / (df["Size_in_SqFt"] / 1000)
    return df


def city_stats(df):
    """City price/sqft medians. Compute from TRAINING rows only when training models."""
    ppsf_city = df.groupby("City")["Price_per_SqFt_Rs"].median()
    return {"city_median_ppsf": ppsf_city.to_dict(),
            "state_of_city": df.drop_duplicates("City").set_index("City")["State"].to_dict()}


def growth_rate(df, stats):
    """ASSUMPTION-BASED annual appreciation rate (no price history exists in the data).

    8% base; city +/-1.5% by city price/sqft percentile; Apartment +0.3%, Villa +0.5%;
    infrastructure +/-1%; ready-to-move +0.5%; -0.05% per year of age. Clipped to 4-12%.
    """
    pct = pd.Series(stats["city_median_ppsf"]).rank(pct=True)
    city_adj = df["City"].map(pct).fillna(0.5).sub(0.5).mul(0.03)
    type_adj = df["Property_Type"].map({"Apartment": 0.003, "Villa": 0.005}).fillna(0.0)
    infra_adj = (df["Infra_Score"] - 5.0).div(5.0).clip(-1, 1).mul(0.01)
    ready_adj = (df["Availability_Status"] == "Ready_to_Move").astype(float).mul(0.005)
    age_adj = -0.0005 * df["Age_of_Property"]
    return (BASE_GROWTH + city_adj + type_adj + infra_adj + ready_adj + age_adj).clip(GROWTH_MIN, GROWTH_MAX)


def investment_score(df, stats):
    """Multi-factor score (0-8) used by the proxy Good-Investment rule."""
    city_med = df["City"].map(stats["city_median_ppsf"])
    pts = pd.DataFrame({
        "cheaper_than_city_median": df["Price_per_SqFt_Rs"] <= city_med,
        "bhk_3plus": df["BHK"] >= 3,
        "ready_to_move": df["Availability_Status"] == "Ready_to_Move",
        "good_transport": df["Public_Transport_Accessibility"].isin(["Medium", "High"]),
        "security": df["Security"] == 1,
        "parking": df["Parking_Space"] == 1,
        "social_infra": (df["Nearby_Schools"] + df["Nearby_Hospitals"]) >= 11,
        "young_building": df["Age_of_Property"] <= 18,
    })
    return pts.sum(axis=1), pts


def project_price(price, rate, years=HORIZON_YEARS):
    """Assumption-based compound projection."""
    return price * (1 + rate) ** years


def add_targets(df, stats):
    """Adds the proxy label and the assumption-based projection (neither is observed)."""
    df = df.copy()
    df["Growth_Rate"] = growth_rate(df, stats)
    df["Projected_Price_5Y_Lakhs"] = project_price(df["Price_in_Lakhs"], df["Growth_Rate"])
    score, _ = investment_score(df, stats)
    df["Investment_Score"] = score
    cheaper = df["Price_per_SqFt_Rs"] <= df["City"].map(stats["city_median_ppsf"])
    df["Good_Investment"] = (cheaper & (score >= GOOD_SCORE_THRESHOLD)).astype(int)
    return df


def prepare(path, stats=None):
    """Load -> clean -> engineer -> targets. Pass train-only ``stats`` to stay leak-free;
    if omitted, stats come from the whole file (fine for descriptive EDA, NOT for modelling)."""
    df = engineer(clean(load_raw(path)))
    stats = stats or city_stats(df)
    return add_targets(df, stats), stats
