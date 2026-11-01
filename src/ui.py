"""Presentation helpers for the Streamlit app (pure strings/figures; no business logic)."""
import plotly.graph_objects as go

GOLD, BLUE, GREEN, RED, MUTED = "#D4AF37", "#4C8DFF", "#2ECC9A", "#FF6B6B", "#9AA6BC"
COLORWAY = [GOLD, BLUE, GREEN, "#B07CFF", RED, "#5CC8FF"]

FEATURE_LABELS = {
    "Price_per_SqFt_Rs": "Price per sq ft (₹)", "Price_in_Lakhs": "Asking price (₹ lakhs)", "Size_in_SqFt": "Size (sq ft)",
    "BHK": "BHK", "Age_of_Property": "Property age (years)", "Floor_No": "Floor number", "Total_Floors": "Total floors",
    "Floor_Ratio": "Floor ÷ total floors", "Floor_Inconsistent": "Floor above total (flag)",
    "Nearby_Schools": "Nearby schools", "Nearby_Hospitals": "Nearby hospitals", "Parking_Space": "Parking space",
    "Security": "Security", "Amenity_Count": "Amenity count", "Infra_Score": "Infrastructure score",
    "School_Density_Score": "School density score", "Public_Transport_Accessibility": "Public transport access",
    "Property_Type": "Property type", "Furnished_Status": "Furnishing", "Owner_Type": "Owner type",
    "Availability_Status": "Availability", "State": "State", "City": "City", "Facing": "Facing"}
MODEL_LABELS = {
    "baseline_majority_class": "Baseline: majority class", "baseline_cheaper_than_city_median": "Baseline: cheaper than city median",
    "baseline_mean": "Baseline: predict the mean", "baseline_city_mean": "Baseline: per-city mean",
    "logreg": "Logistic regression", "ridge": "Ridge regression", "hist_gb": "Hist. gradient boosting",
    "hist_gb_no_price_features": "Hist. GB without price features"}


def pretty(name):
    """Readable label for a column / feature / model key (display only)."""
    return FEATURE_LABELS.get(name) or MODEL_LABELS.get(name) or str(name).replace("_", " ")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 85% -10%, rgba(212,175,55,.08), transparent 60%),
                     radial-gradient(900px 500px at -10% 10%, rgba(76,141,255,.07), transparent 55%), #0A101C; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, #MainMenu, footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -.01em; font-weight: 700; }
h3 { color: #F1F4FA; }
[data-testid="stSidebar"] { background: #0D1424; border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .45rem .7rem; border-radius: 8px; margin-bottom: 2px; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(212,175,55,.08); }
.brand { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 1.1rem; }
.brand .logo { width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, #D4AF37, #8C6D12); color:#0A101C; font-weight:800; font-size:1.05rem; }
.brand .name { font-weight:700; line-height:1.1; } .brand .tag { color:#8B97AD; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
.hero { border:1px solid rgba(212,175,55,.25); border-radius:18px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;
  background: linear-gradient(120deg, rgba(212,175,55,.10), rgba(76,141,255,.07) 60%, rgba(16,24,40,.6)); }
.hero .eyebrow { color:#D4AF37; font-size:.75rem; font-weight:600; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { margin:.2rem 0 .35rem; font-size:2.1rem; }
.hero p { margin:0; color:#B8C1D3; max-width:860px; }
.card { background:#101828; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:1rem 1.2rem; height:100%; }
.kpi .label { color:#9AA6BC; line-height:1.3; font-size:.72rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase; }
.kpi .value { font-size:clamp(1.4rem, 1.9vw, 1.75rem); font-weight:700; margin:.15rem 0; color:#F4F6FB; font-variant-numeric:tabular-nums; }
.kpi .sub { color:#9AA6BC; font-size:.82rem; line-height:1.4; }
.badge { display:inline-block; padding:.18rem .6rem; border-radius:999px; font-size:.72rem; white-space:nowrap; vertical-align:middle; font-weight:700; letter-spacing:.08em;
  text-transform:uppercase; border:1px solid; }
.badge.learned { color:#4C8DFF; border-color:rgba(76,141,255,.5); background:rgba(76,141,255,.10); }
.badge.rule { color:#2ECC9A; border-color:rgba(46,204,154,.5); background:rgba(46,204,154,.10); }
.badge.assumption { color:#D4AF37; border-color:rgba(212,175,55,.5); background:rgba(212,175,55,.10); }
.verdict { border-radius:14px; padding:1rem 1.3rem; line-height:1.3; font-weight:700; font-size:1.15rem; display:flex; align-items:center; gap:.7rem; }
.verdict > span { flex:none; }
.verdict.good { background:rgba(46,204,154,.10); border:1px solid rgba(46,204,154,.45); color:#2ECC9A; }
.verdict.bad  { background:rgba(255,107,107,.10); border:1px solid rgba(255,107,107,.45); color:#FF6B6B; }
.verdict small { display:block; font-weight:400; color:#C3CBDB; font-size:.86rem; margin-top:.15rem; }
.check { display:flex; align-items:center; gap:.6rem; padding:.32rem 0; border-bottom:1px solid rgba(255,255,255,.05); font-size:.92rem; }
.check:last-child { border-bottom:none; } .check > span:first-child { flex:none; width:1rem; text-align:center; } .check .ok { color:#2ECC9A; } .check .no { color:#8794AB; }
.section-title { color:#D4AF37; font-size:.75rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:.4rem 0 .6rem; }
.note { color:#9AA6BC; font-size:.82rem; line-height:1.5; }
[data-testid="stFormSubmitButton"] button { background:linear-gradient(135deg,#D4AF37,#A9871F); color:#0A101C; font-weight:700;
  border:none; border-radius:10px; padding:.6rem 1.4rem; }
[data-testid="stFormSubmitButton"] button:hover { filter:brightness(1.08); color:#0A101C; }
[data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }
[data-testid="stMetric"] { background:#101828; border:1px solid rgba(255,255,255,.07); border-radius:14px; padding:.8rem 1rem; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { padding:.5rem 1rem; border-radius:8px 8px 0 0; }
.footer { color:#8794AB; font-size:.78rem; text-align:center; margin-top:2.5rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,.06); }

/* Model details: evaluation report inside the expander - tighter hierarchy */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] h1 { font-size:1.3rem; margin:.2rem 0 .6rem; color:#F1F4FA; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] h2 { font-size:1.05rem; color:#D4AF37; margin:1.6rem 0 .6rem;
  padding-bottom:.35rem; border-bottom:1px solid rgba(212,175,55,.25); letter-spacing:.01em; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] h3 { font-size:.95rem; margin:1.1rem 0 .4rem; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { font-size:.9rem; line-height:1.65; color:#C9D1E0; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li { margin-bottom:.35rem; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p { margin-bottom:.8rem; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] table { font-size:.82rem; width:100%; border-collapse:collapse; margin:.4rem 0 1rem; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th { background:rgba(212,175,55,.10); color:#E6EAF2; text-align:right; padding:.45rem .6rem; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th:first-child, 
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] td:first-child { text-align:left; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] td { padding:.4rem .6rem; text-align:right; border-bottom:1px solid rgba(255,255,255,.06); }
/* Methodology: readable line length and paragraph rhythm */
.st-key-method { max-width: 920px; }
.st-key-method p { line-height:1.75; margin-bottom:1.05rem; color:#C9D1E0; }
.st-key-method strong { color:#F1F4FA; }

/* ---- UI polish: equal-height cards, readable captions/alerts, inline code, tables ---- */
/* A column whose only content is a card stretches that card to the tallest card in the row */
[data-testid="stHorizontalBlock"]:has(.card) { align-items:stretch; }
.card { box-sizing:border-box; }
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) > [data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stElementContainer"],
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stMarkdown"],
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stMarkdown"] > div,
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stMarkdownContainer"] { height:100%; }
[data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stMarkdown"], [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stMarkdown"] > div, [data-testid="stColumn"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:only-child .card) [data-testid="stMarkdownContainer"] { margin-bottom:0 !important; }
.card p { margin:.6rem 0 0; line-height:1.6; font-size:.92rem; color:#C9D1E0; }
.kpi .sub:empty { display:none; }

h3, [data-testid="stHeading"] h3 { font-size:1.3rem; margin-top:.4rem; padding-bottom:.25rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#9AA6BC !important; opacity:1 !important; font-size:.84rem; line-height:1.55; }
[data-testid="stWidgetLabel"] p { font-size:.86rem; font-weight:500; color:#C9D1E0; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.08); margin:1rem 0; }
[data-testid="stAlert"] { border-radius:12px; line-height:1.6; }
[data-testid="stAlert"] p { font-size:.92rem; }
[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,.08); }
[data-testid="stExpander"] details { border-radius:12px; }
[data-testid="stExpander"] summary { align-items:center; padding:.7rem 1rem; }
[data-testid="stExpander"] summary p { margin:0; font-weight:600; }
.stTabs [data-baseweb="tab"] { font-weight:500; }
.stTabs [aria-selected="true"] { font-weight:700; }

/* Inline code (Methodology, captions, evaluation report) */
.stMarkdown code, [data-testid="stCaptionContainer"] code, [data-testid="stAlert"] code {
  font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:.84em; color:#F2D675;
  background:rgba(212,175,55,.12); border:1px solid rgba(212,175,55,.25); border-radius:6px; padding:.08rem .4rem;
  white-space:break-spaces; word-break:break-word; }
.stMarkdown pre code { background:none; border:none; padding:0; color:inherit; white-space:pre; }
.st-key-method { max-width:920px; }

/* Evaluation report tables: bolder headers, aligned digits */
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] th { font-weight:600; white-space:nowrap; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] td { font-variant-numeric:tabular-nums; }
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] tr:hover td { background:rgba(255,255,255,.03); }

/* Navigation: active page + visible keyboard focus */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:rgba(212,175,55,.12); font-weight:600; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:focus-visible),
button:focus-visible, [role="tab"]:focus-visible, summary:focus-visible, a:focus-visible,
[data-testid="stCheckbox"] label:has(input:focus-visible) { outline:2px solid #F2D675; outline-offset:2px; border-radius:8px; }
[role="slider"]:focus-visible { box-shadow:0 0 0 4px rgba(242,214,117,.35); outline:none; }
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within, [data-baseweb="base-input"]:focus-within {
  border-color:#D4AF37; box-shadow:0 0 0 2px rgba(212,175,55,.35); }
[data-testid="stFormSubmitButton"] button:focus-visible { outline-offset:3px; }

/* Responsive */
@media (max-width: 1200px) { .kpi .label { min-height:2.6em; } }
@media (max-width: 900px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
  .hero { padding:1.1rem 1.2rem; border-radius:14px; } .hero h1 { font-size:1.6rem; }
  .verdict { font-size:1.02rem; padding:.85rem 1rem; }
  [data-testid="stExpander"] [data-testid="stMarkdownContainer"] table { display:block; overflow-x:auto; }
  .st-key-method p { line-height:1.65; }
}

/* Checkboxes sit level with neighbouring inputs (side-by-side columns only) */
@media (min-width: 641px) { [data-testid="stHorizontalBlock"] [data-testid="stCheckbox"] { padding-top:1.7rem; } }
/* Scroll windows for tall charts */
[data-testid="stVerticalBlockBorderWrapper"] { scrollbar-width:thin; scrollbar-color:rgba(212,175,55,.45) transparent; }

/* Methodology: paragraphs as readable panels; accuracy panel highlighted */
.st-key-method p { padding:.85rem 1.1rem; background:#0E1626; border:1px solid rgba(255,255,255,.06);
  border-left:3px solid rgba(212,175,55,.55); border-radius:10px; margin-bottom:.8rem; }
.st-key-method p:nth-of-type(3) { border-left-color:#4C8DFF; background:rgba(76,141,255,.07); }
.st-key-method p:nth-of-type(5) { border-left-color:#FF6B6B; background:rgba(255,107,107,.05); }

/* ---- final pass: tablet wrapping, nav accent, tabs ---- */
@media (min-width: 641px) and (max-width: 900px) {
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: .6rem; }
  [data-testid="stColumn"] { flex: 1 1 calc(50% - 1rem) !important; min-width: calc(50% - 1rem) !important; }
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { box-shadow: inset 3px 0 0 #D4AF37; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#F4F6FB; }
.stTabs [data-baseweb="tab-list"] { overflow-x:auto; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"] { white-space:nowrap; }
.stTabs [aria-selected="true"] { color:#D4AF37; }
.stTabs [data-baseweb="tab-panel"] { padding-top:1.1rem; }
</style>
"""


def brand():
    return ('<div class="brand"><div class="logo">RE</div><div><div class="name">Investment Advisor</div>'
            '<div class="tag">Property analytics</div></div></div>')


def hero(eyebrow, title, text):
    return f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{text}</p></div>'


def kpi(label, value, sub=""):
    return (f'<div class="card kpi"><div class="label">{label}</div><div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def badge(kind, text):
    return f'<span class="badge {kind}">{text}</span>'


def verdict(good, title, sub):
    cls, icon = ("good", "✔") if good else ("bad", "✖")
    return f'<div class="verdict {cls}"><span>{icon}</span><div>{title}<small>{sub}</small></div></div>'


def checklist(items):
    rows = "".join(f'<div class="check"><span class="{"ok" if ok else "no"}">{"●" if ok else "○"}</span><span>{name}</span></div>'
                   for name, ok in items)
    return f'<div class="card">{rows}</div>'


def section(text):
    return f'<div class="section-title">{text}</div>'


def style_fig(fig, height=None):
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else ""
    top = 78 if "<br>" in title else 56          # two-line titles (title + <sup> subtitle) need more headroom
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=12, color="#C9D1E0"), colorway=COLORWAY,
                      margin=dict(l=10, r=10, t=top, b=10), title_font=dict(size=15, color="#F1F4FA"),
                      title_x=0, title_xanchor="left",
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
                      hoverlabel=dict(bgcolor="#101828", bordercolor="rgba(212,175,55,.5)", font=dict(color="#F1F4FA", size=12)))
    axis = dict(gridcolor="rgba(255,255,255,.06)", automargin=True, tickfont=dict(size=11, color="#C9D1E0"),
                title_font=dict(size=12, color="#9AA6BC"), zerolinecolor="rgba(255,255,255,.12)")
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    fig.update_coloraxes(colorbar=dict(thickness=14, outlinewidth=0, tickfont=dict(size=11, color="#C9D1E0"),
                                       title_font=dict(size=12, color="#9AA6BC")))
    if height:
        fig.update_layout(height=height)
    return fig


def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 38, "color": "#F4F6FB"}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 11, "color": "#C9D1E0"}}, "bar": {"color": GOLD, "thickness": .28},
               "bgcolor": "rgba(255,255,255,.04)", "borderwidth": 0,
               "steps": [{"range": [0, 50], "color": "rgba(255,107,107,.18)"},
                         {"range": [50, 100], "color": "rgba(46,204,154,.18)"}],
               "threshold": {"line": {"color": "#fff", "width": 2}, "value": 50}}))
    style_fig(fig, 230)
    fig.update_layout(margin=dict(l=28, r=28, t=14, b=6))
    return fig