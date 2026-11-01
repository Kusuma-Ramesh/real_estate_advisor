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