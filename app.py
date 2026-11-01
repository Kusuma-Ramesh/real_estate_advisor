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
        f'<div class="note" style="margin-top:.8rem">No listing dataset loaded.<br>'
        f'Prediction and model information remain available.</div>', **HTML
    )


def model_category_values(model):
    """Read the fitted categorical vocabularies from the saved sklearn pipeline."""
    preprocessor = model.named_steps.get("pre")
    if preprocessor is None:
        return {}
    encoder = preprocessor.named_transformers_.get("cat")
    if encoder is None:
        return {}
    categorical_columns = next(
        columns for name, _, columns in preprocessor.transformers_ if name == "cat"
    )
    return {
        column: sorted(str(value) for value in categories)
        for column, categories in zip(categorical_columns, encoder.categories_)
    }


MODEL_OPTIONS = model_category_values(clf)
CITY_TO_STATE = meta["stats"]["state_of_city"]


def show(fig, height=None, container=st):
    container.plotly_chart(ui.style_fig(fig, height), width="stretch", config={"displayModeBar": False})


def show_scroll(fig, height, window=560):
    """Tall charts (one row per city) scroll inside a fixed-height window instead of lengthening the page."""
    if height <= window:
        show(fig, height)
    else:
        with st.container(height=window, border=False):
            show(fig, height)
        st.caption("Scroll inside the chart to see every city.")


def page_header(eyebrow, title, text):
    st.markdown(ui.hero(eyebrow, title, text), **HTML)


# ---------------------------------------------------------------- Predict
if page == "Predict":
    page_header("Property analytics", "Real Estate Investment Advisor",
                "Evaluate a listing against a transparent investment checklist, a classifier trained on it, and an "
                "assumption-based five-year projection. Each result is tagged <b>Learned</b>, <b>Rule</b> or <b>Assumption</b>.")

    # Location sits OUTSIDE the form: widgets inside st.form only update on submit, which left the City
    # list showing the previous State's cities.
    with st.container(border=True):
        st.markdown(ui.section("Location"), **HTML)
        c1, c2, c3 = st.columns(3)
        available_states = sorted(df.State.unique()) if df is not None else sorted(set(CITY_TO_STATE.values()))
        state = c1.selectbox("State", available_states, key="state")
        available_cities = (sorted(df[df.State == state].City.unique()) if df is not None
                            else sorted(city_name for city_name, state_name in CITY_TO_STATE.items() if state_name == state))
        city = c2.selectbox("City", available_cities, key=f"city_{state}")
        if df is not None:
            localities = sorted(df.Locality.unique(), key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)
        else:
            localities = ["Locality_1"]
        locality = c3.selectbox("Locality", localities,
                                help="Locality is not used as a model input; the default placeholder is used without a dataset.")

    with st.form("prop"):
        st.markdown(ui.section("Property"), **HTML)
        c1, c2, c3, c4 = st.columns(4)
        ptype = c1.selectbox("Property type", sorted(df.Property_Type.unique()) if df is not None else MODEL_OPTIONS["Property_Type"])
        bhk = c2.slider("BHK", 1, 5, 3)
        size = c3.number_input("Size (sq ft)", 100, 10000, 1500, step=50)
        price = c4.number_input("Asking price (₹ lakhs)", 1.0, 5000.0, 120.0, step=5.0)

        c1, c2, c3, c4 = st.columns(4)
        year = c1.number_input("Year built", 1900, REFERENCE_YEAR, 2012)
        furn = c2.selectbox("Furnishing", sorted(df.Furnished_Status.unique()) if df is not None else MODEL_OPTIONS["Furnished_Status"])
        facing = c3.selectbox("Facing", sorted(df.Facing.unique()) if df is not None else MODEL_OPTIONS["Facing"])
        avail = c4.selectbox("Availability", sorted(df.Availability_Status.unique()) if df is not None else MODEL_OPTIONS["Availability_Status"])

        st.markdown(ui.section("Building & neighbourhood"), **HTML)
        c1, c2, c3, c4 = st.columns(4)
        total_floors = c1.number_input("Total floors", 1, 60, 10)
        floor_no = c2.number_input("Floor number", 0, 60, 3)
        schools = c3.slider("Nearby schools (1-10)", 1, 10, 5)
        hospitals = c4.slider("Nearby hospitals (1-10)", 1, 10, 5)

        st.markdown(ui.section("Features & ownership"), **HTML)
        c1, c2, c3, c4 = st.columns(4)
        transport = c1.selectbox("Public transport", ["Low", "Medium", "High"], index=1)
        owner = c2.selectbox("Owner type", sorted(df.Owner_Type.unique()) if df is not None else MODEL_OPTIONS["Owner_Type"])
        parking = c3.checkbox("Parking space", True)
        security = c4.checkbox("Security", True)
        amen = st.multiselect("Amenities", AMENITIES, default=["Gym", "Garden"])
        go = st.form_submit_button("Analyse property", type="primary")

    if go:
        res = assess({
            "State": state, "City": city, "Locality": locality, "Property_Type": ptype, "BHK": bhk,
            "Size_in_SqFt": size, "Price_in_Lakhs": price, "Year_Built": int(year), "Furnished_Status": furn,
            "Floor_No": int(floor_no), "Total_Floors": int(total_floors), "Nearby_Schools": schools,
            "Nearby_Hospitals": hospitals, "Public_Transport_Accessibility": transport, "Parking_Space": parking,
            "Security": security, "Amenities": amen, "Facing": facing, "Owner_Type": owner,
            "Availability_Status": avail}, clf, reg, meta)
        for e in res["errors"]:
            st.error(e)
        for w in res["warnings"]:
            st.warning(w)
        if not res["errors"]:
            good = bool(res["rule_label"])
            st.markdown(ui.verdict(good, "Good investment" if good else "Not a strong investment",
                                   f"Rule score {res['rule_score']}/8 - needs ≥ {GOOD_SCORE_THRESHOLD} and price per sq ft at or "
                                   f"below the city median"), **HTML)
            st.write("")
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(ui.kpi("Price per sq ft", f"₹{res['ppsf']:,.0f}", f"City median ₹{res['city_median_ppsf']:,.0f}"), **HTML)
            k2.markdown(ui.kpi("Rule score", f"{res['rule_score']} / 8", ui.badge("rule", "Rule")), **HTML)
            k3.markdown(ui.kpi(f"Projected {res['target_year']}", f"₹{res['projected_price_lakhs']:,.1f}L",
                               ui.badge("assumption", "Assumption") + f" &nbsp;{res['assumed_growth_rate']:.2%}/yr"), **HTML)
            k4.markdown(ui.kpi("Model price estimate", f"₹{res['market_estimate_lakhs']:,.1f}L", ui.badge("learned", "Learned")), **HTML)
            st.write("")

            left, right = st.columns([1, 1.15])
            with left:
                st.markdown("### Investment assessment")
                st.markdown(ui.badge("learned", "Learned") + " &nbsp;Classifier probability of 'good'", **HTML)
                show(ui.gauge(res["good_probability"]))
                st.caption("The classifier is trained on a rule-based proxy label: it reproduces the rule and is not a market prediction.")
                names = {"cheaper_than_city_median": "Price per sq ft ≤ city median", "bhk_3plus": "3+ BHK",
                         "ready_to_move": "Ready to move", "good_transport": "Medium / High transport",
                         "security": "Security", "parking": "Parking", "social_infra": "Schools + hospitals ≥ 11",
                         "young_building": "Age ≤ 18 years"}
                st.markdown(ui.badge("rule", "Rule") + " &nbsp;Checklist", **HTML)
                st.markdown(ui.checklist([(names[k], v) for k, v in res["checklist"].items()]), **HTML)
            with right:
                st.markdown(f"### Five-year outlook ({res['target_year']})")
                st.markdown(ui.badge("assumption", "Assumption") + " &nbsp;Not a learned forecast - accuracy is unmeasured", **HTML)
                sc = pd.DataFrame({"Scenario": list(res["scenarios"]), "Price": list(res["scenarios"].values())})
                fig = px.bar(sc, x="Scenario", y="Price", text=sc.Price.round(1), title="Illustrative growth scenarios (₹ lakhs)",
                             labels={"Price": "Price (₹ lakhs)", "Scenario": ""})
                fig.add_hline(y=price, line_dash="dot", line_color="#E6EAF2", annotation_text="Asking price",
                              annotation_position="top left", annotation_font=dict(color="#F1F4FA", size=11),
                              annotation_bgcolor="rgba(10,16,28,.9)")
                fig.update_traces(marker_color=ui.GOLD, textposition="outside", cliponaxis=False,
                                  hovertemplate="%{x}<br>₹%{y:,.1f} lakhs<extra></extra>")
                show(fig, 300)
                st.caption(f"Growth is an assumed rule: {BASE_GROWTH:.0%} base ± city, type, infrastructure, availability and age "
                           f"(clipped {GROWTH_MIN:.0%}-{GROWTH_MAX:.0%}). Assumed rate for this property: {res['assumed_growth_rate']:.2%}/yr. "
                           f"The horizon is counted from the dataset's {REFERENCE_YEAR} reference year, so it ends in {res['target_year']}, not five years from today.")
                if not res["regressor_has_skill"]:
                    st.info("The model price estimate is **not better than predicting the average price** (see Model details): "
                            "in this dataset price is unrelated to the listed attributes. Do not use it to judge fair value.")

            st.markdown("### Feature importance")
            c1, c2 = st.columns(2)
            for col, key, title in [(c1, "classification", "Classifier"), (c2, "regression", "Current-price regressor")]:
                imp = pd.Series(meta["importance"][key]).head(8).sort_values()
                imp.index = [ui.pretty(i) for i in imp.index]
                fig = px.bar(imp, orientation="h", title=title, labels={"value": "Importance (permutation, held-out)", "index": ""})
                fig.update_traces(marker_color=ui.BLUE if key == "classification" else ui.GOLD,
                                  hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>")
                fig.update_yaxes(automargin=True)
                fig.update_xaxes(tickformat=".3f" if key == "classification" else ".4f", nticks=5)
                fig.update_layout(showlegend=False)
                show(fig, 330, col)

# ---------------------------------------------------------------- Explore
elif page == "Explore properties":
    if df is None:
        page_header("Dataset required", "Explore properties", "Upload a property dataset to explore listings.")
        st.info("Upload a compatible CSV using the sidebar to enable this page. Your CSV is processed in the app and is not added to GitHub.")
    else:
        page_header("Explore", "Property explorer", "Filter the 250,000 listings and inspect rule-based labels and assumption-based projections.")
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            states = c1.multiselect("State", sorted(df.State.unique()))
            cities = c2.multiselect("City", sorted(df[df.State.isin(states)].City.unique() if states else df.City.unique()))
            types = c3.multiselect("Property type", sorted(df.Property_Type.unique()))
            c1, c2, c3 = st.columns(3)
            bhks = c1.multiselect("BHK", [1, 2, 3, 4, 5])
            pr = c2.slider("Price (₹ lakhs)", 10, 500, (10, 500))
            sz = c3.slider("Size (sq ft)", 500, 5000, (500, 5000))
            c1, c2 = st.columns(2)
            avail = c1.multiselect("Availability", sorted(df.Availability_Status.unique()))
            only_good = c2.checkbox("Only show rule-based 'Good Investment' label")

        f = df[df.Price_in_Lakhs.between(*pr) & df.Size_in_SqFt.between(*sz)]
        for col, vals in [("State", states), ("City", cities), ("Property_Type", types), ("BHK", bhks), ("Availability_Status", avail)]:
            if vals:
                f = f[f[col].isin(vals)]
        if only_good:
            f = f[f.Good_Investment == 1]
        if len(f):
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(ui.kpi("Matching listings", f"{len(f):,}", f"of {len(df):,}"), **HTML)
            k2.markdown(ui.kpi("Avg price", f"₹{f.Price_in_Lakhs.mean():,.0f}L", "lakhs"), **HTML)
            k3.markdown(ui.kpi("Avg price / sq ft", f"₹{f.Price_per_SqFt_Rs.mean():,.0f}", ""), **HTML)
            k4.markdown(ui.kpi("Rule-label rate", f"{f.Good_Investment.mean():.0%}", ui.badge("rule", "Rule")), **HTML)
            st.write("")
            show_cols = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
                         "Price_per_SqFt_Rs", "Age_of_Property", "Availability_Status", "Investment_Score",
                         "Good_Investment", "Projected_Price_5Y_Lakhs"]
            T = st.column_config
            st.dataframe(f[show_cols].head(500).round(1), width="stretch", hide_index=True, column_config={
                "ID": T.NumberColumn("ID", format="%d", width="small"), "State": "State", "City": "City", "Locality": "Locality",
                "Property_Type": "Property type", "BHK": T.NumberColumn("BHK", format="%d", width="small"),
                "Size_in_SqFt": T.NumberColumn("Size (sq ft)", format="%d", width="small"),
                "Price_in_Lakhs": T.NumberColumn("Price (₹ L)", format="%.1f", width="small"),
                "Price_per_SqFt_Rs": T.NumberColumn("₹ / sq ft", format="%.1f", width="small"),
                "Age_of_Property": T.NumberColumn("Age (yrs)", format="%d", width="small"), "Availability_Status": "Availability",
                "Investment_Score": T.NumberColumn("Rule score", format="%d", width="small"),
                "Good_Investment": T.NumberColumn("Rule label", help="1 = rule-based Good Investment", format="%d", width="small"),
                "Projected_Price_5Y_Lakhs": T.NumberColumn("Projected 5Y (₹ L)", help="Assumption-based, not learned", format="%.1f", width="small")})
            st.caption("Showing first 500 matches. 'Projected_Price_5Y_Lakhs' is assumption-based, not learned.")
            g = f.groupby(["City", "Property_Type"])["Price_in_Lakhs"].mean().reset_index()
            if g.City.nunique() > 1:
                fig = px.density_heatmap(g, x="Property_Type", y="City", z="Price_in_Lakhs", histfunc="avg",
                                         color_continuous_scale="YlOrBr", title="Average price (₹L) - city × type",
                                         labels={"Property_Type": "Property type"})
                fig.update_yaxes(dtick=1)
                fig.update_coloraxes(colorbar_title_text="₹ lakhs")
                fig.update_traces(hovertemplate="City: %{y}<br>Property type: %{x}<br>Average price: ₹%{z:,.1f} lakhs<extra></extra>")
                show_scroll(fig, max(380, 20 * g.City.nunique() + 130), 520)
        else:
            st.info("No matches for these filters.")

# ---------------------------------------------------------------- Insights
elif page == "Market insights":
    if df is None:
        page_header("Dataset required", "Market insights", "Upload a property dataset to view listing-based market analytics.")
        st.info("Upload a compatible CSV using the sidebar to enable this page. Your CSV is processed in the app and is not added to GitHub.")
    else:
        page_header("Analytics", "Market insights",
                    "This dataset looks synthetic: price barely varies by city, size, BHK, schools or amenities. "
                    "The charts are accurate; they mostly show that those factors are flat.")
        t1, t2, t3, t4 = st.tabs(["Location heatmaps", "City prices", "Cohorts & projections", "Investment rate"])
        NAMES = {"Price_per_SqFt_Rs": "price per sq ft (₹)", "Price_in_Lakhs": "asking price (₹ lakhs)",
                 "Projected_Price_5Y_Lakhs": "projected 5-year price (₹ lakhs, ASSUMPTION-based)"}
        UNIT = {"Price_per_SqFt_Rs": "₹ per sq ft", "Price_in_Lakhs": "₹ lakhs", "Projected_Price_5Y_Lakhs": "₹ lakhs"}
        with t1:
            m = st.radio("Metric", list(NAMES), format_func=NAMES.get, horizontal=True)
            pv = df.pivot_table(index="State", columns="Property_Type", values=m, aggfunc="mean")
            fig = px.imshow(pv, text_auto=",.0f", color_continuous_scale="YlOrBr", aspect="auto", title=f"Average {NAMES[m]} - state × property type",
                            labels=dict(x="Property type", y="State", color=UNIT[m]))
            fig.update_traces(hovertemplate="State: %{y}<br>Property type: %{x}<br>" + UNIT[m] + ": %{z:,.0f}<extra></extra>")
            show(fig, 480)
            pv2 = df.pivot_table(index="City", columns="BHK", values=m, aggfunc="mean")
            fig = px.imshow(pv2, text_auto=",.0f", color_continuous_scale="Blues", aspect="auto", title=f"Average {NAMES[m]} - city × BHK",
                            labels=dict(x="BHK", y="City", color=UNIT[m]))
            fig.update_xaxes(type="category")
            fig.update_yaxes(dtick=1)
            fig.update_traces(hovertemplate="City: %{y}<br>BHK: %{x}<br>" + UNIT[m] + ": %{z:,.0f}<extra></extra>")
            show_scroll(fig, 780, 600)
        with t2:
            c = df.groupby("City")[["Price_in_Lakhs", "Price_per_SqFt_Rs"]].mean().reset_index().sort_values("Price_in_Lakhs")
            fig = px.bar(c, x="Price_in_Lakhs", y="City", orientation="h", text="Price_in_Lakhs",
                         title="Average asking price by city (₹ lakhs), highest first",
                         labels={"Price_in_Lakhs": "Average asking price (₹ lakhs)", "City": ""})
            fig.update_traces(marker_color=ui.GOLD, texttemplate="₹%{x:,.1f}L", textposition="outside", cliponaxis=False,
                              hovertemplate="%{y}<br>Average asking price: ₹%{x:,.1f} lakhs<extra></extra>")
            fig.update_yaxes(categoryorder="total ascending", dtick=1)
            fig.update_xaxes(range=[0, float(c["Price_in_Lakhs"].max()) * 1.15], tickformat=",.0f")
            fig.update_layout(bargap=0.25)
            show_scroll(fig, 780, 600)
        with t3:
            available_cities = sorted(df.City.unique())
            default_cities = [c for c in ["Mumbai", "Bangalore", "Chennai"] if c in available_cities]
            sel = st.multiselect("Cities", available_cities, default=default_cities)
            d = df[df.City.isin(sel)].groupby(["Year_Built", "City"])["Price_per_SqFt_Rs"].mean().reset_index()
            fig = px.line(d, x="Year_Built", y="Price_per_SqFt_Rs", color="City",
                          title="Average price per sq ft by year built<br><sup>Construction cohorts, not a price trend</sup>",
                          labels={"Year_Built": "Year built", "Price_per_SqFt_Rs": "Average price per sq ft (₹)", "City": "City"})
            fig.update_yaxes(tickformat=",.0f", tickprefix="₹")
            show(fig, 360)
            st.markdown(ui.badge("assumption", "Assumption") + " &nbsp;5-year projection vs today - not a learned forecast", **HTML)
            if sel:
                p = df.groupby("City")[["Price_in_Lakhs", "Projected_Price_5Y_Lakhs"]].mean().loc[sel].reset_index().melt("City")
                p["variable"] = p["variable"].map({"Price_in_Lakhs": "Average asking price today",
                                                   "Projected_Price_5Y_Lakhs": "Assumption-based projection"})
                fig = px.bar(p, x="City", y="value", color="variable", barmode="group", labels={"value": "₹ lakhs", "variable": "", "City": ""},
                             title=f"Mean asking price vs assumption-based projection<br><sup>Projection = price × (1 + assumed rate)^{HORIZON_YEARS}, in ₹ lakhs</sup>")
                fig.update_yaxes(tickformat=",.0f")
                show(fig, 360)
        with t4:
            t4a, t4b = st.columns([3, 2], gap="large")
            g = df.groupby("City")["Good_Investment"].mean().sort_values().reset_index()
            fig = px.bar(g, x="Good_Investment", y="City", orientation="h", text="Good_Investment",
                         title="Share of listings with the rule-based 'Good Investment' label, by city",
                         labels={"Good_Investment": "Share labelled good (rule-based)", "City": ""})
            fig.update_traces(marker_color=ui.GREEN, texttemplate="%{x:.1%}", textposition="outside", cliponaxis=False,
                              hovertemplate="%{y}<br>Share labelled good: %{x:.1%}<extra></extra>")
            fig.update_yaxes(categoryorder="total ascending", dtick=1)
            fig.update_xaxes(range=[0, float(g["Good_Investment"].max()) * 1.15], tickformat=".0%")
            fig.update_layout(bargap=0.25)
            with t4a:
                show_scroll(fig, 780, 560)
            g2 = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reset_index()
            fig = px.bar(g2, x="Public_Transport_Accessibility", y="Good_Investment", text="Good_Investment",
                         category_orders={"Public_Transport_Accessibility": ["Low", "Medium", "High"]},
                         title="Rule-based label rate by transport access<br><sup>Higher by construction: the rule awards points for transport</sup>",
                         labels={"Public_Transport_Accessibility": "Public transport access", "Good_Investment": "Share labelled good (rule-based)"})
            fig.update_traces(marker_color=ui.GREEN, texttemplate="%{y:.1%}", textposition="outside",
                              hovertemplate="Transport access: %{x}<br>Share labelled good: %{y:.1%}<extra></extra>")
            fig.update_yaxes(tickformat=".0%", range=[0, float(g2["Good_Investment"].max()) * 1.2])
            show(fig, 360, t4b)

# ---------------------------------------------------------------- Model details
elif page == "Model details":
    page_header("Evaluation", "Model details",
                "Classifier accuracy measures recovery of a hand-written rule, not real investment outcomes. "
                "The 5-year price is an assumption, so it has no accuracy figure.")
    res = meta["results"]
    best_c, best_r = res["classification"][meta["best_classifier"]], res["regression"][meta["best_regressor"]]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(ui.kpi("Classifier accuracy", f"{best_c['accuracy']:.1%}", "reproduces the rule, not investment success"), **HTML)
    k2.markdown(ui.kpi("Classifier ROC-AUC", f"{best_c['roc_auc']:.3f}", f"Selected: {ui.pretty(meta['best_classifier'])}"), **HTML)
    k3.markdown(ui.kpi("Price model R² (held-out)", f"{best_r['r2']:.5f}", f"Gain vs predicting the mean: {meta['regressor_skill_vs_mean_r2']:+.5f}"), **HTML)
    k4.markdown(ui.kpi("Price model RMSE (held-out)", f"₹{best_r['rmse']:.1f} L", "Error in ₹ lakhs; lower is better"), **HTML)
    st.write("")
    base_best = min(res["regression"]["baseline_mean"]["rmse"], res["regression"]["baseline_city_mean"]["rmse"])
    st.warning(f"**Price model: no usable skill.** R² = {best_r['r2']:.5f} (≈ 0); MAE is ₹{best_r['mae']:.2f} lakhs. "
               f"Test RMSE {best_r['rmse']:.2f} vs {base_best:.2f} for the better of the mean / per-city-mean baselines - "
               f"{'it does not beat them' if best_r['rmse'] >= base_best else 'it beats them only marginally'}. "
               f"The {best_c['accuracy']:.1%} classifier accuracy means it matches a hand-written rule, not that listings succeed as investments.")
    def metric_table(results, order, selected):
        t = pd.DataFrame(results).T
        t = t[[c for c in order if c in t.columns]]
        t.index = [ui.pretty(i) + ("  ✓ selected" if i == selected else "") for i in t.index]
        t.index.name = "Model"
        return t

    cls_cols = {"accuracy": ("Accuracy", "%.4f"), "precision": ("Precision", "%.4f"), "recall": ("Recall", "%.4f"), "f1": ("F1", "%.4f"),
                "roc_auc": ("ROC-AUC", "%.4f"), "pr_auc": ("PR-AUC", "%.4f"), "cv_roc_auc": ("CV ROC-AUC", "%.4f")}
    reg_cols = {"rmse": ("RMSE (₹ lakhs)", "%.3f"), "mae": ("MAE (₹ lakhs)", "%.3f"), "r2": ("R²", "%.5f"), "cv_rmse": ("CV RMSE (₹ lakhs)", "%.3f")}
    cfg = lambda spec: {k: st.column_config.NumberColumn(lbl, format=f) for k, (lbl, f) in spec.items()}
    st.markdown("### Classification - rule-based proxy label")
    st.dataframe(metric_table(res["classification"], list(cls_cols), meta["best_classifier"]), width="stretch", column_config=cfg(cls_cols))
    st.markdown("### Regression - current price (₹ lakhs, observed target)")
    st.dataframe(metric_table(res["regression"], list(reg_cols), meta["best_regressor"]), width="stretch", column_config=cfg(reg_cols))
    st.caption(f"Models chosen by cross-validation on the training split. Tracking backend for the shipped run: {meta['tracking_backend']}.")
    st.markdown("### Feature importance")
    c1, c2 = st.columns(2, gap="large")
    for col, key, title, xt in [(c1, "classification", "Classifier", "Importance (drop in ROC-AUC)"),
                                (c2, "regression", "Price regressor", "Importance (drop in R²)")]:
        imp = pd.Series(meta["importance"][key]).head(10).sort_values()
        imp.index = [ui.pretty(i) for i in imp.index]
        fig = px.bar(imp, orientation="h", title=title, labels={"value": xt, "index": ""})
        fig.update_traces(marker_color=ui.BLUE if key == "classification" else ui.GOLD,
                          hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>")
        fig.update_yaxes(automargin=True)
        fig.update_xaxes(tickformat=".3f" if key == "classification" else ".4f", nticks=5)
        fig.update_layout(showlegend=False)
        show(fig, 380, col)
    if os.path.exists("reports/evaluation_report.md"):
        with st.expander("Full evaluation report"):
            st.markdown(open("reports/evaluation_report.md", encoding="utf-8").read())

# ---------------------------------------------------------------- Methodology
else:
    page_header("Transparency", "Methodology & limitations", "What is learned, what is a rule, and what is assumed.")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card">' + ui.badge("learned", "Learned") +
                "<p style='margin-top:.6rem'>A classifier trained on a rule-based proxy label, and a current-price estimate trained on observed "
                "prices. Price is statistically unrelated to the other columns, so price skill is ≈ none.</p></div>", **HTML)
    c2.markdown('<div class="card">' + ui.badge("rule", "Rule") +
                f"<p style='margin-top:.6rem'>An 8-point checklist: price per sq ft ≤ city median <b>and</b> score ≥ {GOOD_SCORE_THRESHOLD}. "
                "Deterministic and fully transparent.</p></div>", **HTML)
    c3.markdown('<div class="card">' + ui.badge("assumption", "Assumption") +
                f"<p style='margin-top:.6rem'>{HORIZON_YEARS}-year price = price × (1 + r)^{HORIZON_YEARS}; r = {BASE_GROWTH:.0%} base with fixed "
                f"adjustments, clipped to {GROWTH_MIN:.0%}-{GROWTH_MAX:.0%}. No historical future prices exist, so accuracy is not measured or claimed.</p></div>", **HTML)
    st.write("")
    with st.container(key="method"):
        st.markdown(f"""
**Data.** 250,000 listings across India. One observed price per listing; no sale dates, price history, investment outcome or crime data.
Reference year **{REFERENCE_YEAR}** (`Age_of_Property + Year_Built = {REFERENCE_YEAR}` in every row).

**Leakage controls.** Train/test split happens before any statistic is computed; city medians come from training rows only; the price
regressor never sees price or price per sq ft; model choice uses cross-validation on training data only.

**What the accuracy figures mean.** The classifier reaches about 99.9% because the label is a deterministic rule of its own inputs; that is
reproduction of the rule. No investment outcome (resale, rental yield, returns) exists in the data, so real investment success is not measured.
The price regressor does not beat predicting the mean (R² ≈ 0), with typical errors near half the mean listing price.

**Projection horizon.** Five years is counted from the dataset's {REFERENCE_YEAR} reference year, so the projection ends in {REFERENCE_YEAR + HORIZON_YEARS}.
The 4% / 8% / 12% scenarios are arithmetic on assumed rates, not forecasts.

**Known data issues.** About 46% of rows have Floor_No > Total_Floors (flagged, ratio capped, rows kept); 500 locality ids each appear in many
cities, so Locality is not a geographic hierarchy (not used by models). The data are near-uniform and balanced, BHK is unrelated to size, price is
unrelated to every attribute, and many under-construction listings have old build years: the dataset appears synthetic, so nothing here describes
a real market. Details: `reports/data_audit.md`.
""")

st.markdown('<div class="footer">Real Estate Investment Advisor · Screening aid, not financial advice · '
            'Learned, rule-based and assumption-based outputs are labelled separately</div>', **HTML)