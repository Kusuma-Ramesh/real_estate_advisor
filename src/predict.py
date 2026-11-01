"""Prediction helpers shared by the Streamlit app and the tests (no Streamlit dependency)."""
import json

import joblib
import pandas as pd

from src.features import (BASE_GROWTH, GROWTH_MAX, GROWTH_MIN, HORIZON_YEARS, REFERENCE_YEAR, add_targets,
                          engineer, investment_score, project_price)

AMENITIES = ["Playground", "Gym", "Garden", "Pool", "Clubhouse"]
# Ranges observed in the training data; outside them a model/rule is extrapolating.
TRAIN_RANGES = {"Size_in_SqFt": (500, 5000), "Price_in_Lakhs": (10, 500), "Year_Built": (1990, 2023),
                "Nearby_Schools": (1, 10), "Nearby_Hospitals": (1, 10), "Total_Floors": (1, 30), "Floor_No": (0, 30)}


def _load(model_dir):
    meta = json.load(open(f"{model_dir}/metadata.json", encoding="utf-8"))
    return joblib.load(f"{model_dir}/classifier.joblib"), joblib.load(f"{model_dir}/regressor.joblib"), meta


def load_artifacts(model_dir="models", retrain_on_failure=True):
    """Load saved models. Pickles are tied to the scikit-learn version that wrote them, so if loading
    fails (e.g. "No module named '_loss'" after a version change) retrain once with the installed
    version (~1 minute) and load the fresh files."""
    try:
        return _load(model_dir)
    except (ImportError, AttributeError, ValueError, KeyError, EOFError, OSError, TypeError) as exc:
        if not retrain_on_failure or model_dir != "models":
            raise
        print(f"[predict] could not load saved models ({exc!r}); retraining with the installed scikit-learn...", flush=True)
        from src.train import main as train_main
        train_main()
        return _load(model_dir)


def validate(inputs):
    """Return (errors, warnings). Errors block prediction; warnings flag extrapolation/odd data."""
    errors, warns = [], []
    if inputs["Floor_No"] > inputs["Total_Floors"]:
        errors.append("Floor number cannot exceed total floors in the building.")
    if inputs["Year_Built"] > REFERENCE_YEAR:
        errors.append(f"Year built cannot be after {REFERENCE_YEAR} (the dataset's reference year).")
    for k, (lo, hi) in TRAIN_RANGES.items():
        if not lo <= inputs[k] <= hi:
            warns.append(f"{k} = {inputs[k]} is outside the training range {lo}-{hi}; treat results with extra caution.")
    age = REFERENCE_YEAR - inputs["Year_Built"]
    if inputs["Availability_Status"] == "Under_Construction" and age > 5:
        warns.append(f"Under-construction listing with a build year {age} years before {REFERENCE_YEAR} is internally inconsistent.")
    per_bhk = inputs["Size_in_SqFt"] / max(inputs["BHK"], 1)
    if per_bhk < 250 or per_bhk > 2000:
        warns.append(f"{per_bhk:,.0f} sq ft per BHK is unusual (heuristic range 250-2,000); check size and BHK.")
    if inputs["Property_Type"] not in ("Apartment", "Independent House", "Villa"):
        warns.append("Unrecognised property type.")
    return errors, warns


def build_row(i):
    row = pd.DataFrame([{
        "State": i["State"], "City": i["City"], "Locality": i["Locality"], "Property_Type": i["Property_Type"],
        "BHK": i["BHK"], "Size_in_SqFt": i["Size_in_SqFt"], "Price_in_Lakhs": i["Price_in_Lakhs"],
        "Year_Built": i["Year_Built"], "Furnished_Status": i["Furnished_Status"], "Floor_No": i["Floor_No"],
        "Total_Floors": i["Total_Floors"], "Age_of_Property": REFERENCE_YEAR - i["Year_Built"],
        "Nearby_Schools": i["Nearby_Schools"], "Nearby_Hospitals": i["Nearby_Hospitals"],
        "Public_Transport_Accessibility": i["Public_Transport_Accessibility"],
        "Parking_Space": "Yes" if i["Parking_Space"] else "No", "Security": "Yes" if i["Security"] else "No",
        "Amenities": ", ".join(i["Amenities"]) or "None", "Facing": i["Facing"], "Owner_Type": i["Owner_Type"],
        "Availability_Status": i["Availability_Status"]}])
    return engineer(row)


def assess(inputs, clf, reg, meta):
    """Full assessment of one property. Keeps learned outputs and assumption-based outputs separate."""
    errors, warns = validate(inputs)
    if errors:
        return {"errors": errors, "warnings": warns}
    stats = meta["stats"]
    row = build_row(inputs)
    if row["City"].iloc[0] not in stats["city_median_ppsf"]:
        return {"errors": [f"City '{inputs['City']}' not seen in training data."], "warnings": warns}
    scored = add_targets(row, stats)
    score, pts = investment_score(scored, stats)
    price = float(inputs["Price_in_Lakhs"])
    rate = float(scored["Growth_Rate"].iloc[0])
    out = {
        "errors": [], "warnings": warns,
        # --- learned (classifier trained on the rule-based proxy label) ---
        "good_probability": float(clf.predict_proba(row[meta["classifier_features"]])[0, 1]),
        # --- rule (exact, deterministic) ---
        "rule_label": int(scored["Good_Investment"].iloc[0]), "rule_score": int(score.iloc[0]),
        "checklist": pts.iloc[0].to_dict(),
        "ppsf": float(row["Price_per_SqFt_Rs"].iloc[0]),
        "city_median_ppsf": float(stats["city_median_ppsf"][inputs["City"]]),
        # --- learned, observed target (current price); weak by construction on this data ---
        "market_estimate_lakhs": float(reg.predict(row[meta["regressor_features"]])[0]),
        "regressor_has_skill": meta["regressor_skill_vs_mean_r2"] >= 0.01,
        # --- assumption-based (NOT learned, NOT validated) ---
        "assumed_growth_rate": rate,
        "projected_price_lakhs": float(project_price(price, rate)),
        "scenarios": {f"{r:.0%}/yr": float(project_price(price, r)) for r in (GROWTH_MIN, BASE_GROWTH, GROWTH_MAX)},
        "horizon_years": HORIZON_YEARS, "target_year": REFERENCE_YEAR + HORIZON_YEARS,
    }
    return out
