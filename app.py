"""Real Estate Investment Advisor - Streamlit app.   Run:  streamlit run app.py

Three kinds of output are kept visibly separate (badges in the UI):
  LEARNED     - classifier probability (trained on a rule-based proxy label) and a current-price estimate
  RULE        - the exact deterministic investment checklist
  ASSUMPTION  - 5-year projection = price x (1 + assumed growth)^5; never learned, never validated
"""
import io
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from src import ui
from src.features import (BASE_GROWTH, GOOD_SCORE_THRESHOLD, GROWTH_MAX, GROWTH_MIN, HORIZON_YEARS,
                          REFERENCE_YEAR, prepare)
from src.predict import AMENITIES, assess, load_artifacts

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
DATA = "data/india_housing_prices.csv"
HTML = dict(unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading saved models...")
def load_models():
    # Deployment must use the committed artifacts; it cannot retrain without the private CSV.
    return load_artifacts("models", retrain_on_failure=False)


def validate_prepared_dataset(frame):
    if frame.empty:
        raise ValueError("The CSV contains no usable property rows after cleaning.")
    known_cities = set(meta["stats"]["city_median_ppsf"])
    unknown_cities = sorted(set(frame["City"]) - known_cities)
    if unknown_cities:
        preview = ", ".join(unknown_cities[:8])
        suffix = " ..." if len(unknown_cities) > 8 else ""
        raise ValueError(
            "The CSV contains cities not supported by the saved models: "
            f"{preview}{suffix}. Use listings from the model's supported cities."
        )
    return frame


@st.cache_data(show_spinner="Preparing property listings...")
def load_local_data(_stats_key):
    frame = prepare(DATA, stats=meta["stats"])[0]
    return validate_prepared_dataset(frame)


@st.cache_data(show_spinner="Validating and preparing uploaded listings...")
def load_uploaded_data(file_bytes, _stats_key):
    frame = prepare(io.BytesIO(file_bytes), stats=meta["stats"])[0]
    return validate_prepared_dataset(frame)


clf, reg, meta = load_models()

with st.sidebar:
    st.markdown(ui.brand(), **HTML)
    uploaded_file = st.file_uploader(
        "Upload property dataset (CSV)",
        type=["csv"],
        help="Required for Explore and Market insights when the bundled local dataset is unavailable. "
             "Use a CSV with the columns listed in data/DATA_DICTIONARY.md."
    )
    page = st.radio("Navigate", ["Predict", "Explore properties", "Market insights", "Model details",
                                 "Methodology & limitations"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="note">Screening aid only - not financial advice. Targets are proxies or assumptions; '
                'see Methodology.</div>', **HTML)

try:
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file.getvalue(), meta["n_train"])
        data_source = "uploaded CSV"
    elif os.path.exists(DATA):
        df = load_local_data(meta["n_train"])
        data_source = "local dataset"
    else:
        df = None
        data_source = None
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    df = None
    data_source = None
    st.sidebar.error(f"Could not load the dataset: {exc}")

if df is not None:
    st.sidebar.markdown(
        f'<div class="note" style="margin-top:.8rem">Dataset: {len(df):,} listings ({data_source})'
        f'<br>Reference year: {REFERENCE_YEAR}</div>', **HTML
    )
else:
    st.sidebar.markdown(