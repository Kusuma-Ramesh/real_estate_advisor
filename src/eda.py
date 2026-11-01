"""EDA for the 20 questions in the brief. Run: python3 -m src.eda
Writes PNG charts to eda_outputs/ and numeric answers to eda_outputs/findings.md
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import json, os
from src.features import prepare

sns.set_theme(style="whitegrid", context="notebook")
OUT = "eda_outputs"
# Descriptive EDA: use the training-split city medians saved by train.py when available so the
# rule-based label matches the app; otherwise fall back to whole-file medians.
_saved = json.load(open("models/metadata.json", encoding="utf-8"))["stats"] if os.path.exists("models/metadata.json") else None
df, stats = prepare("data/india_housing_prices.csv", stats=_saved)
os.makedirs(OUT, exist_ok=True)
df["PPSF"] = df["Price_per_SqFt_Rs"]
lines = []


def save(name):
    import textwrap
    ax = plt.gca()
    if ax.get_title():                       # wrap long titles so they are never clipped
        ax.set_title("\n".join(textwrap.wrap(ax.get_title(), 62)), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=110)
    plt.close()


def note(q, text):
    lines.append(f"**Q{q}.** {text}\n")


# ---- 1-5 Price & size -------------------------------------------------
plt.figure(figsize=(7, 4)); sns.histplot(df.Price_in_Lakhs, bins=50)
plt.title("Q1. Distribution of property prices (₹ lakhs)"); save("q01_price_dist")
p = df.Price_in_Lakhs
note(1, f"Prices are roughly uniform between ₹{p.min():.0f}L and ₹{p.max():.0f}L (mean {p.mean():.0f}L, "
        f"median {p.median():.0f}L, skew {p.skew():.2f}). There is no long right tail as in real markets.")

plt.figure(figsize=(7, 4)); sns.histplot(df.Size_in_SqFt, bins=50)
plt.title("Q2. Distribution of property sizes (sq ft)"); save("q02_size_dist")
s = df.Size_in_SqFt
note(2, f"Sizes are roughly uniform from {s.min():.0f} to {s.max():.0f} sq ft (median {s.median():.0f}).")

plt.figure(figsize=(7, 4)); sns.boxplot(data=df, x="Property_Type", y="PPSF", showfliers=False)
plt.title("Q3. Price per sq ft by property type (₹)"); save("q03_ppsf_by_type")
g = df.groupby("Property_Type")["PPSF"].median().round(0).to_dict()
note(3, f"Median price per sq ft by type: {g}. Differences are small, so type is a weak price driver here.")

smp = df.sample(8000, random_state=1)
plt.figure(figsize=(7, 4.5)); sns.scatterplot(data=smp, x="Size_in_SqFt", y="Price_in_Lakhs", alpha=.25, s=10)
plt.title("Q4. Size vs price"); save("q04_size_vs_price")
r = df[["Size_in_SqFt", "Price_in_Lakhs"]].corr().iloc[0, 1]
note(4, f"Correlation between size and price is {r:.3f}, i.e. effectively none. Larger homes are NOT priced higher "
        f"in this dataset; instead price per sq ft falls as size rises (corr {df[['Size_in_SqFt','PPSF']].corr().iloc[0,1]:.2f}), "
        f"which is a mathematical by-product of price being independent of size.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.boxplot(y=df.PPSF, ax=ax[0]); ax[0].set_title("Price per sq ft (₹)")
sns.boxplot(y=df.Size_in_SqFt, ax=ax[1]); ax[1].set_title("Size (sq ft)")
fig.suptitle("Q5. Outliers"); save("q05_outliers")
q1, q3 = df.PPSF.quantile([.25, .75]); iqr = q3 - q1
out_ppsf = (df.PPSF > q3 + 1.5 * iqr).mean(); out_low = (df.PPSF < q1 - 1.5 * iqr).mean()
note(5, f"{out_ppsf:.1%} of properties are IQR-outliers on price per sq ft (high side; {out_low:.1%} on the low side), "
        f"driven by small homes with high prices; max ₹{df.PPSF.max():,.0f}/sq ft. Size has no outliers (bounded 500-5000). "
        f"Outliers were kept for modelling because they are valid small-size/high-price combinations, and the tree models are robust to them.")

# ---- 6-10 Location ----------------------------------------------------
st = df.groupby("State")["PPSF"].mean().sort_values()
plt.figure(figsize=(7, 6)); st.plot.barh(); plt.title("Q6. Avg price per sq ft by state (₹)"); save("q06_ppsf_by_state")
note(6, f"Highest: {st.index[-1]} (₹{st.iloc[-1]:,.0f}); lowest: {st.index[0]} (₹{st.iloc[0]:,.0f}). "
        f"Spread between states is only {(st.max()/st.min()-1):.0%}.")

ct = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
plt.figure(figsize=(7, 9)); ct.plot.barh(); plt.title("Q7. Avg price by city (₹ lakhs)"); save("q07_price_by_city")
note(7, f"Highest: {ct.index[-1]} ({ct.iloc[-1]:.0f}L); lowest: {ct.index[0]} ({ct.iloc[0]:.0f}L). "
        f"Cities are all within {ct.min():.0f}-{ct.max():.0f}L, i.e. very flat.")

loc_age = df.groupby("Locality")["Age_of_Property"].median()
plt.figure(figsize=(7, 4)); sns.histplot(loc_age, bins=25)
plt.title("Q8. Median property age across 500 localities (years) - locality ids are not real neighbourhoods"); save("q08_age_by_locality")
note(8, f"Median age per locality ranges {loc_age.min():.0f}-{loc_age.max():.0f} years (overall median "
        f"{loc_age.median():.0f}). Oldest: {', '.join(loc_age.nlargest(3).index)}; newest: {', '.join(loc_age.nsmallest(3).index)}. "
        f"Caveat: every locality id appears in {df.groupby('Locality').City.nunique().median():.0f} cities on median, so locality ids are not real neighbourhoods and these differences are sampling noise.")

bhk = pd.crosstab(df.City, df.BHK, normalize="index")
plt.figure(figsize=(7, 11)); sns.heatmap(bhk, annot=True, fmt=".0%", cmap="Blues", cbar=False)
plt.title("Q9. BHK mix by city"); save("q09_bhk_by_city")
note(9, f"BHK is near-uniform in every city (each of 1-5 BHK is roughly 20%; max city deviation "
        f"{(bhk-0.2).abs().max().max():.1%}).")

loc_n = df.groupby("Locality").size()
top5 = df[df.Locality.isin(loc_n[loc_n >= 200].index)].groupby("Locality")["PPSF"].median().nlargest(5).index
d5 = df[df.Locality.isin(top5)].copy()
d5["Built"] = (d5.Year_Built // 5 * 5).astype(int)
tr = d5.groupby(["Built", "Locality"])["PPSF"].mean().reset_index()
plt.figure(figsize=(8, 4.5)); sns.lineplot(data=tr, x="Built", y="PPSF", hue="Locality", marker="o")
plt.title("Q10. Price per sq ft by year-built cohort, top-5 locality ids (not a time trend; ids are not real places)"); save("q10_top5_locality_trend")
note(10, f"Top-5 localities by median ₹/sq ft (min 200 listings): {', '.join(top5)}. The data has no sale-date field, so "
         f"'trend' is shown by year-built cohort; lines are noisy with no consistent upward trend, so no appreciation signal can be read from the data itself "
         f"(this is why the 5-year price is an assumption-based projection, not a learned one). Locality ids are not nested in cities, so this ranking is noise.")

# ---- 11-15 Correlation ------------------------------------------------
num = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "PPSF", "Age_of_Property", "Nearby_Schools",
       "Nearby_Hospitals", "Floor_No", "Total_Floors", "Amenity_Count", "Infra_Score"]
cm = df[num].corr()
_raw = ['BHK','Size_in_SqFt','Price_in_Lakhs','Age_of_Property','Nearby_Schools','Nearby_Hospitals','Floor_No','Total_Floors']
_c = df[_raw].corr().abs().where(~np.eye(len(_raw), dtype=bool)); raw_max = float(np.nanmax(_c.values))
plt.figure(figsize=(9, 7)); sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7})
plt.title("Q11. Correlation matrix (PPSF and Infra_Score are derived columns: their correlations are mechanical)", fontsize=9); save("q11_correlation")
off = cm.where(~np.eye(len(cm), dtype=bool)).abs().stack().sort_values(ascending=False)
pairs = [f"{a} vs {b} ({cm.loc[a, b]:.2f})" for (a, b) in off.index[::2][:3]]
note(11, f"Only mechanical relationships appear: {'; '.join(pairs)}. Largest |r| among raw, non-derived attributes (BHK, size, price, age, schools, hospitals, floors): {raw_max:.3f}.")

for q, col, nm in [(12, "Nearby_Schools", "schools"), (13, "Nearby_Hospitals", "hospitals")]:
    plt.figure(figsize=(7, 4)); sns.barplot(data=df, x=col, y="PPSF", estimator="mean", errorbar=None)
    plt.title(f"Q{q}. Nearby {nm} vs avg price per sq ft (₹)"); save(f"q{q}_{nm}_vs_ppsf")
    r = df[[col, "PPSF"]].corr().iloc[0, 1]
    note(q, f"Correlation between {nm} count and price per sq ft is {r:.3f}: no relationship. Locations with more {nm} are not priced higher.")

plt.figure(figsize=(7, 4)); sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", showfliers=False)
plt.title("Q14. Price by furnished status (₹ lakhs)"); save("q14_price_by_furnished")
g = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean().round(1).to_dict()
note(14, f"Mean price (lakhs): {g}. Differences under 1%, so furnishing does not drive price.")

plt.figure(figsize=(7, 4)); sns.barplot(data=df, x="Facing", y="PPSF", errorbar=None)
plt.title("Q15. Price per sq ft by facing (₹)"); save("q15_ppsf_by_facing")
g = df.groupby("Facing")["PPSF"].mean().round(0).to_dict()
note(15, f"Mean ₹/sq ft by facing: {g}. Effectively identical.")

# ---- 16-20 Investment / amenities / ownership -------------------------
plt.figure(figsize=(6, 4)); df.Owner_Type.value_counts().plot.bar(); plt.title("Q16. Properties by owner type"); save("q16_owner_type")
note(16, f"Owner types are evenly split: {df.Owner_Type.value_counts().to_dict()}.")

plt.figure(figsize=(6, 4)); df.Availability_Status.value_counts().plot.bar(); plt.title("Q17. Properties by availability"); save("q17_availability")
note(17, f"{df.Availability_Status.value_counts().to_dict()}. Only these two statuses exist in the data (no 'Sold').")

plt.figure(figsize=(6, 4)); sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", errorbar=None)
plt.title("Q18. Price by parking availability (₹ lakhs)"); save("q18_parking_price")
g = df.groupby("Parking_Space")["Price_in_Lakhs"].mean().round(1).to_dict()
note(18, f"Mean price with/without parking (lakhs): {g}. No meaningful effect.")

am = df.Amenities.str.split(",").explode().str.strip()
exploded = df.assign(Amenity=df.Amenities.str.split(",")).explode("Amenity")
exploded["Amenity"] = exploded["Amenity"].str.strip()
a = exploded.groupby("Amenity")["PPSF"].mean().sort_values()
plt.figure(figsize=(7, 4)); a.plot.barh(); plt.title("Q19. Avg price per sq ft by amenity present (₹)"); save("q19_amenities")
cnt = df.groupby("Amenity_Count")["PPSF"].mean()
note(19, f"Listings with each amenity average ₹{a.min():,.0f}-{a.max():,.0f}/sq ft (spread {a.max()/a.min()-1:.1%}). "
         f"Price per sq ft by number of amenities ranges only {cnt.min():,.0f}-{cnt.max():,.0f}; amenities do not drive price.")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
order = ["Low", "Medium", "High"]
sns.barplot(data=df, x="Public_Transport_Accessibility", y="PPSF", order=order, errorbar=None, ax=ax[0])
ax[0].set_title("Avg ₹/sq ft")
sns.barplot(data=df, x="Public_Transport_Accessibility", y="Good_Investment", order=order, errorbar=None, ax=ax[1])
ax[1].set_title("Rule-based good-investment rate (proxy label)"); fig.suptitle("Q20. Public transport accessibility"); save("q20_transport")
g1 = df.groupby("Public_Transport_Accessibility")["PPSF"].mean().round(0).to_dict()
g2 = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().round(3).to_dict()
note(20, f"₹/sq ft by transport level: {g1} (flat). Good-investment rate: {g2}. The rate rises with transport access "
         f"because the investment rule awards points for it (rule effect, not market evidence).")

# ---- Extra brief EDA: crime proxy / infrastructure vs resale ----------
df["Infra_Bin"] = pd.cut(df.Infra_Score, 5)
plt.figure(figsize=(7, 4)); sns.barplot(data=df, x="Infra_Bin", y="Projected_Price_5Y_Lakhs", errorbar=None)
plt.xticks(rotation=20); plt.title("Extra: infrastructure score vs ASSUMPTION-based projected price (₹ lakhs) - rises by construction, not evidence"); save("extra_infra_vs_future")
lines.append("**Extra (brief step 2).** The dataset has no crime-rate column, so that analysis cannot be done. "
             "Infrastructure score vs assumption-based projected value is shown in `extra_infra_vs_future.png`; it is mildly positive only because the "
             "growth-rate assumption rewards infrastructure.\n")

open(f"{OUT}/findings.md", "w", encoding="utf-8").write("# EDA findings (20 questions)\n\n" + "\n".join(lines))
print("EDA done:", len(lines), "findings")
