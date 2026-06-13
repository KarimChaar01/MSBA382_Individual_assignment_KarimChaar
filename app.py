"""
SleepWatch: Sleep Disorders in the Context of Stress, Anxiety and Conflict
MSBA 382 - Healthcare Analytics | Individual Project
AUB Olayan School of Business, Summer 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import shap

st.set_page_config(
    page_title="SleepWatch",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display&display=swap');

html, body, [class*="css"], h1, h2, h3, h4, p, span, div, label, button,
.stMarkdown, .stText, section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', 'Segoe UI', sans-serif !important;
}

.block-container { padding-top: 1rem !important; }

[data-testid="metric-container"] {
    background: #FAFAF8;
    border: 1px solid #E8E3DA;
    border-radius: 10px;
    padding: 0.6rem 0.9rem 0.5rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: #1B3A5C !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    color: #8C8680 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] > div {
    font-size: 0.70rem !important;
    color: #8C8680 !important;
}

section[data-testid="stSidebar"] {
    background: #F5F3EF;
    border-right: 1px solid #E8E3DA;
}

.stTabs [data-baseweb="tab"] { font-size: 0.85rem; }

.callout {
    background: #FEF9EC;
    border-left: 3px solid #D97706;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.84rem;
    color: #44403C;
    margin-bottom: 1rem;
    line-height: 1.5;
}
.callout-blue {
    background: #EFF6FF;
    border-left: 3px solid #1D4ED8;
}

.risk-card {
    border-radius: 12px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# â"€â"€ Password gate â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def check_password():
    if st.session_state.get("authenticated"):
        return True
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; padding:2.2rem 2rem 1.8rem;
                    background:#FAFAF8; border-radius:16px; border:1px solid #E8E3DA;'>
            <div style='font-size:2.8rem;'>🌙</div>
            <div style='font-size:1.5rem; font-weight:600; color:#1B3A5C; margin:0.4rem 0 0;'>SleepWatch</div>
            <div style='color:#8C8680; font-size:0.84rem; margin:0.1rem 0 1.4rem;'>
                Healthcare Analytics &nbsp;·&nbsp; MSBA 382 &nbsp;·&nbsp; AUB
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        pwd = st.text_input("", type="password", placeholder="Enter password",
                            label_visibility="collapsed")
        _, b, _ = st.columns([1, 2, 1])
        with b:
            if st.button("Enter", use_container_width=True, type="primary"):
                if pwd == "karim2001":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Wrong password.")
    return False

if not check_password():
    st.stop()


# â"€â"€ Data â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
@st.cache_data(show_spinner="Loading data…")
def load_data():
    p = pd.read_csv("data/patient_data.csv")
    c = pd.read_csv("data/global_prevalence.csv")
    t = pd.read_csv("data/trend_data.csv")
    return p, c, t

@st.cache_resource(show_spinner="Loading risk model…")
def load_model():
    bundle = joblib.load("model/xgb_model.pkl")
    return bundle["model"], bundle["feats"], bundle["X_test"], bundle["y_test"]

@st.cache_resource(show_spinner=False)
def get_explainer(_model):
    return shap.TreeExplainer(_model)

patients, country, trend = load_data()

DISORDERS = ["Insomnia", "Sleep Apnea", "Hypersomnia", "Narcolepsy", "Restless Leg Syndrome"]
ALL_CATS  = DISORDERS + ["No Disorder"]

PALETTE = {
    "Insomnia":              "#2563EB",
    "Sleep Apnea":           "#DC2626",
    "Hypersomnia":           "#7C3AED",
    "Narcolepsy":            "#0369A1",
    "Restless Leg Syndrome": "#0891B2",
    "No Disorder":           "#9CA3AF",
}

BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="#FAFAF8",
    font=dict(family="DM Sans, Segoe UI, sans-serif", color="#1C1917"),
    margin=dict(t=36, b=28, l=8, r=8)
)

LB_EVENTS = {2019: "Economic Collapse", 2020: "Port Explosion",
             2024: "War 1", 2026: "War 2"}

FEAT_LABELS = ["Age", "Gender", "BMI", "Stress", "Anxiety (GAD-7)",
               "Sleep Duration", "Caffeine", "Screen Time",
               "Physical Activity", "Conflict Exposed",
               "Job Stress", "Smoking", "Alcohol", "Chronic Pain"]


# â"€â"€ Sidebar â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
with st.sidebar:
    st.markdown("""
    <div style='padding:0.6rem 0 0.2rem; text-align:center;'>
        <span style='font-size:1.8rem;'>🌙</span>
        <div style='font-size:1rem; font-weight:600; color:#1B3A5C; margin-top:0.2rem;'>SleepWatch</div>
        <div style='font-size:0.68rem; color:#A8A29E; margin-top:0.1rem;'>
            Sleep &amp; Conflict · MSBA 382
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("", ["Patient Overview", "Global Burden", "Risk Predictor"],
                    label_visibility="collapsed")

    st.divider()
    st.markdown("<span style='font-size:0.78rem; font-weight:600; color:#57534E;'>FILTERS</span>",
                unsafe_allow_html=True)
    st.caption("Apply to Patient Overview only")

    gender_filter   = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_range       = st.slider("Age", 18, 80, (18, 80))
    disorder_filter = st.multiselect("Disorder", ALL_CATS, default=ALL_CATS)
    conflict_filter = st.selectbox("Conflict Exposure",
                                   ["All", "Exposed only", "Not exposed only"])
    st.divider()
    st.caption("WHO GHO · IHME GBD · DASS-42 · Kaggle · Hallit 2020 · BMC 2025")


# â"€â"€ Filter helper â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def apply_filters(df):
    m = (df["Gender"].isin(gender_filter) &
         df["Age"].between(*age_range) &
         df["Sleep_Disorder"].isin(disorder_filter))
    if conflict_filter == "Exposed only":
        m &= df["Conflict_Exposed"] == 1
    elif conflict_filter == "Not exposed only":
        m &= df["Conflict_Exposed"] == 0
    return df[m].copy()

df = apply_filters(patients)

def polish(fig):
    fig.update_layout(**BASE)
    fig.update_xaxes(showgrid=False, linecolor="#E5E7EB", tickfont=dict(color="#6B7280", size=11))
    fig.update_yaxes(gridcolor="#F3F4F6", linecolor="#E5E7EB", tickfont=dict(color="#6B7280", size=11))
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 1 "" PATIENT OVERVIEW
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "Patient Overview":

    st.markdown("## Patient Overview")

    total    = len(df)
    affected = (df["Sleep_Disorder"] != "No Disorder").sum()

    st.markdown(f"""
    <div class='callout callout-blue'>
    Analysis covers <strong>{total:,} patient records</strong> from the DASS-42 OpenPsychometrics
    global survey, the Kaggle Sleep Health clinical dataset, and clinical data from Lebanon and
    the MENA region. Sidebar filters apply to all charts below.
    </div>
    """, unsafe_allow_html=True)

    # â"€â"€ KPIs row 1 â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Disorder Prevalence",
                  f"{affected/total*100:.1f}%" if total else "N/A",
                  f"{affected:,} of {total:,} patients")
    with c2:
        st.metric("Avg Stress Score",
                  f"{df['Stress_Score'].mean():.1f} / 10",
                  "self-reported, 1""10 scale")
    with c3:
        st.metric("Avg Anxiety (GAD-7)",
                  f"{df['Anxiety_Score'].mean():.1f} / 21",
                  "generalised anxiety scale")

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

    # â"€â"€ KPIs row 2 â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Avg Sleep Duration",
                  f"{df['Sleep_Duration_Hrs'].mean():.1f} hrs",
                  "average per night")
    with c5:
        st.metric("Conflict-Exposed",
                  f"{df['Conflict_Exposed'].mean()*100:.1f}%",
                  "war or crisis exposure")
    with c6:
        top = (df[df["Sleep_Disorder"] != "No Disorder"]["Sleep_Disorder"]
               .value_counts().idxmax() if affected else "N/A")
        st.metric("Most Common Disorder", top, "by patient count")

    st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)

    # â"€â"€ Chart row 1: donut | stress bar | conflict lollipop â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    col1, col2, col3 = st.columns([1, 1.1, 1.1])

    with col1:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Disorder breakdown</span>", unsafe_allow_html=True)
        counts = (df[df["Sleep_Disorder"] != "No Disorder"]["Sleep_Disorder"]
                  .value_counts().reset_index())
        counts.columns = ["Disorder", "Count"]
        fig1 = px.pie(counts, names="Disorder", values="Count",
                      color="Disorder", color_discrete_map=PALETTE, hole=0.52)
        fig1.update_layout(**{**BASE, "margin": dict(t=4, b=4, l=4, r=4)},
                           height=230,
                           legend=dict(orientation="h", y=-0.18, font=dict(size=9)))
        fig1.update_traces(textinfo="percent", textfont_size=10,
                           marker=dict(line=dict(color="white", width=2)),
                           hovertemplate="<b>%{label}</b><br>%{value:,} patients · %{percent}<extra></extra>")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Stress levels by disorder</span>", unsafe_allow_html=True)
        fig2 = px.box(df, x="Sleep_Disorder", y="Stress_Score",
                      color="Sleep_Disorder", color_discrete_map=PALETTE,
                      labels={"Sleep_Disorder": "", "Stress_Score": "Stress (1""10)"})
        polish(fig2)
        fig2.update_layout(showlegend=False, xaxis_tickangle=-20, height=230,
                           margin=dict(t=4, b=36, l=8, r=8))
        fig2.update_traces(marker=dict(size=2, opacity=0.3))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Conflict exposure rate</span>", unsafe_allow_html=True)
        conf_rate = (df[df["Sleep_Disorder"] != "No Disorder"]
                     .groupby("Sleep_Disorder")["Conflict_Exposed"]
                     .mean().reset_index())
        conf_rate.columns = ["Disorder", "Rate"]
        conf_rate["Pct"] = conf_rate["Rate"] * 100
        conf_rate = conf_rate.sort_values("Pct", ascending=True)
        fig3 = px.bar(conf_rate, x="Pct", y="Disorder", orientation="h",
                      color="Disorder", color_discrete_map=PALETTE,
                      labels={"Pct": "% conflict-exposed", "Disorder": ""},
                      text="Pct")
        fig3.update_traces(texttemplate="%{x:.0f}%", textposition="auto",
                           showlegend=False)
        fig3.update_layout(**{**BASE, "margin": dict(t=4, b=28, l=130, r=20)},
                           height=230,
                           xaxis=dict(title="% conflict-exposed", range=[0, 100],
                                      showgrid=True, gridcolor="#F3F4F6",
                                      tickfont=dict(size=10)),
                           yaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig3, use_container_width=True)

    # â"€â"€ Scatter (stress vs sleep) + Heatmap â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    sc_col, hm_col = st.columns(2)

    with sc_col:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Stress level vs sleep duration by disorder</span>", unsafe_allow_html=True)
        samp = df.sample(min(1500, len(df)), random_state=42)
        fig_sc2 = px.scatter(
            samp, x="Stress_Score", y="Sleep_Duration_Hrs",
            color="Sleep_Disorder", color_discrete_map=PALETTE,
            opacity=0.45,
            labels={"Stress_Score": "Stress (1-10)",
                    "Sleep_Duration_Hrs": "Sleep hrs / night",
                    "Sleep_Disorder": ""}
        )
        fig_sc2.update_traces(marker=dict(size=5))
        polish(fig_sc2)
        fig_sc2.update_layout(
            height=240,
            margin=dict(t=4, b=50, l=8, r=8),
            legend=dict(orientation="h", y=-0.30, font=dict(size=9))
        )
        st.plotly_chart(fig_sc2, use_container_width=True)

    with hm_col:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Disorder profile heatmap (relative severity)</span>", unsafe_allow_html=True)
        heat_raw = df.groupby("Sleep_Disorder").agg(
            Stress    =("Stress_Score",       "mean"),
            Anxiety   =("Anxiety_Score",      "mean"),
            BMI       =("BMI",                "mean"),
            Conflict  =("Conflict_Exposed",   lambda x: x.mean() * 100),
            Caffeine  =("Caffeine_Daily_Cups","mean"),
        ).round(1)
        heat_norm = (heat_raw - heat_raw.min()) / (heat_raw.max() - heat_raw.min() + 1e-9)
        text_grid = []
        for metric in heat_raw.columns:
            row = []
            for dis in heat_raw.index:
                v = heat_raw.loc[dis, metric]
                row.append(f"{v:.0f}%" if metric == "Conflict" else f"{v:.1f}")
            text_grid.append(row)
        fig_hm = px.imshow(
            heat_norm.T.values,
            x=heat_raw.index.tolist(),
            y=heat_raw.columns.tolist(),
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            labels={"color": "Relative level"}
        )
        fig_hm.update_traces(
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(size=10, color="white")
        )
        polish(fig_hm)
        fig_hm.update_layout(
            height=240,
            margin=dict(t=4, b=4, l=70, r=10),
            coloraxis_showscale=False,
            xaxis=dict(tickfont=dict(size=9), tickangle=-18),
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig_hm, use_container_width=True)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 2 "" GLOBAL BURDEN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "Global Burden":

    st.markdown("## Global Burden of Sleep Disorders")
    st.markdown("""
    <div class='callout'>
    Conflict-affected countries consistently rank at the top. Lebanon's sleep disorder prevalence
    has climbed from roughly 38% in 2018 to over 55% in 2026 "" a direct reflection of compounding
    crises: economic collapse, the Beirut port explosion, and two successive wars.
    </div>
    """, unsafe_allow_html=True)

    # â"€â"€ KPIs â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    lbn_now  = trend[trend["Year"] == 2026]["Lebanon_Prevalence"].values[0]
    lbn_base = trend[trend["Year"] == 2018]["Lebanon_Prevalence"].values[0]
    pal_prev = country[country["Country"] == "Palestine"]["Any_Sleep_Disorder_Pct"].values[0]
    mena_now = trend[trend["Year"] == 2026]["MENA_Prevalence"].values[0]
    glob_now = trend[trend["Year"] == 2026]["Global_Prevalence"].values[0]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Lebanon (2026)",    f"{lbn_now:.1f}%",  "up from 38.2% in 2018")
    with c2: st.metric("Palestine (2025)",  f"{pal_prev:.1f}%", "highest in dataset")
    with c3: st.metric("MENA Average",      f"{mena_now:.1f}%", "regional 2026 estimate")

    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4: st.metric("Global Average",     f"{glob_now:.1f}%",            "worldwide 2026")
    with c5: st.metric("Lebanon War Surge",  f"+{lbn_now - lbn_base:.1f} pp", "since 2018 baseline")
    with c6: st.metric("Countries Covered",  str(len(country)),             "across 6 regions")

    st.markdown("---")

    # â"€â"€ Trend chart â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    st.markdown("**How Lebanon's sleep disorder rate compares over time (2000""2026)**")

    TREND_COLORS = {
        "Global_Prevalence":      "#CBD5E1",
        "MENA_Prevalence":        "#3B82F6",
        "MENA_Conflict_Subgroup": "#F59E0B",
        "Lebanon_Prevalence":     "#DC2626",
    }
    TREND_LABELS = {
        "Global_Prevalence":      "Global",
        "MENA_Prevalence":        "MENA",
        "MENA_Conflict_Subgroup": "MENA Conflict",
        "Lebanon_Prevalence":     "Lebanon",
    }
    TREND_WIDTH = {"Global_Prevalence": 1.5, "MENA_Prevalence": 1.8,
                   "MENA_Conflict_Subgroup": 2.0, "Lebanon_Prevalence": 3.0}

    fig_t = go.Figure()
    for col, color in TREND_COLORS.items():
        fig_t.add_trace(go.Scatter(
            x=trend["Year"], y=trend[col], name=TREND_LABELS[col],
            line=dict(color=color, width=TREND_WIDTH[col]),
            mode="lines+markers",
            marker=dict(size=4 if col != "Lebanon_Prevalence" else 6),
            hovertemplate=f"<b>{TREND_LABELS[col]}</b>: %{{y:.1f}}%<extra></extra>"
        ))
    for yr, lbl in LB_EVENTS.items():
        y_val = trend[trend["Year"] == yr]["Lebanon_Prevalence"].values[0]
        fig_t.add_annotation(x=yr, y=y_val + 3.2, text=lbl,
                             showarrow=True, arrowhead=2, arrowcolor="#DC2626",
                             arrowsize=0.8, ax=0, ay=-28,
                             font=dict(size=9, color="#DC2626"),
                             bgcolor="rgba(255,255,255,0.92)",
                             bordercolor="#DC2626", borderwidth=1, borderpad=3)
        fig_t.add_vline(x=yr, line_dash="dot", line_color="#DC2626", opacity=0.22)
    fig_t.update_layout(
        **{**BASE, "margin": dict(t=55, b=36, l=8, r=8)},
        height=280,
        xaxis_title="Year", yaxis_title="Estimated prevalence (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        hovermode="x unified", yaxis=dict(range=[8, 72])
    )
    st.plotly_chart(fig_t, use_container_width=True)

    # â"€â"€ Map + Top 15 side by side â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    map_col, bar_col = st.columns([1.6, 1.4])

    with map_col:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Global prevalence map</span>", unsafe_allow_html=True)
        fig_map = px.choropleth(
            country, locations="ISO3", color="Any_Sleep_Disorder_Pct",
            hover_name="Country",
            hover_data={"ISO3": False, "Region": True,
                        "Any_Sleep_Disorder_Pct": ":.1f", "Conflict_Index": ":.2f"},
            color_continuous_scale="Blues",
            projection="natural earth",
            labels={"Any_Sleep_Disorder_Pct": "Prevalence (%)"}
        )
        fig_map.update_layout(
            paper_bgcolor="white",
            geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#D1D5DB",
                     showland=True, landcolor="#F9FAFB"),
            margin=dict(t=0, b=0, l=0, r=0),
            height=270,
            coloraxis_colorbar=dict(title="Prev. (%)", tickfont=dict(size=9),
                                    thickness=10, len=0.7)
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with bar_col:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Top 15 countries by prevalence</span>", unsafe_allow_html=True)
        top15 = country.nlargest(15, "Any_Sleep_Disorder_Pct").sort_values("Any_Sleep_Disorder_Pct").copy()
        top15["Conflict_Tier"] = top15["Conflict_Index"].apply(
            lambda x: "High conflict" if x >= 0.7 else ("Moderate" if x >= 0.4 else "Low conflict")
        )
        fig_b = px.bar(
            top15, x="Any_Sleep_Disorder_Pct", y="Country", orientation="h",
            color="Conflict_Tier",
            color_discrete_map={"High conflict": "#DC2626", "Moderate": "#F59E0B", "Low conflict": "#60A5FA"},
            text="Any_Sleep_Disorder_Pct",
            labels={"Any_Sleep_Disorder_Pct": "Prevalence (%)", "Conflict_Tier": "Conflict level"},
            category_orders={"Conflict_Tier": ["High conflict", "Moderate", "Low conflict"]}
        )
        fig_b.update_traces(texttemplate="%{x:.1f}%", textposition="inside",
                            textfont=dict(color="white", size=9))
        fig_b.update_layout(
            **{**BASE, "margin": dict(t=4, b=28, l=100, r=10)},
            height=270,
            xaxis=dict(range=[0, top15["Any_Sleep_Disorder_Pct"].max() * 1.05],
                       showgrid=True, gridcolor="#F3F4F6", tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
            legend=dict(orientation="h", y=-0.18, font=dict(size=9)),
            bargap=0.2,
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # â"€â"€ Scatter (conflict vs prevalence) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    left, right = st.columns([1.5, 1.5])

    with left:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Does conflict drive prevalence? (OLS)</span>", unsafe_allow_html=True)
        fig_sc = px.scatter(
            country, x="Conflict_Index", y="Any_Sleep_Disorder_Pct",
            color="Region", size="Any_Sleep_Disorder_Pct",
            hover_name="Country", trendline="ols",
            labels={"Any_Sleep_Disorder_Pct": "Prevalence (%)",
                    "Conflict_Index": "Conflict Index"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        polish(fig_sc)
        fig_sc.update_layout(height=250,
                             legend=dict(orientation="h", y=-0.28, font=dict(size=9)),
                             margin=dict(t=4, b=36, l=8, r=8))
        st.plotly_chart(fig_sc, use_container_width=True)

    with right:
        st.markdown("<span style='font-size:0.82rem;font-weight:600;color:#374151;'>Country data</span>", unsafe_allow_html=True)
        disp = ["Country", "Region", "Any_Sleep_Disorder_Pct", "Insomnia_Prev_Pct",
                "Sleep_Apnea_Prev_Pct", "Conflict_Index"]
        st.dataframe(
            country[disp].sort_values("Any_Sleep_Disorder_Pct", ascending=False)
            .rename(columns={"Any_Sleep_Disorder_Pct": "Disorder Prev. (%)",
                             "Insomnia_Prev_Pct": "Insomnia (%)",
                             "Sleep_Apnea_Prev_Pct": "Apnea (%)",
                             "Conflict_Index": "Conflict Index"}),
            use_container_width=True, hide_index=True, height=250
        )

    st.caption("Sources: WHO GHO · IHME GBD · Hallit et al. (2020) · BMC Public Health (2025) · "
               "Research Square (2025) · Tandfonline (2025) | MSBA 382, AUB, Summer 2026")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PAGE 3 "" RISK PREDICTOR
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "Risk Predictor":

    st.markdown("## Predictive Risk Model")
    st.markdown("""
    <div class='callout'>
    <strong>Bonus component.</strong> An XGBoost classifier estimates individual sleep disorder risk.
    SHAP values show exactly which inputs drive each prediction "" making the model transparent
    and clinically interpretable.
    </div>
    """, unsafe_allow_html=True)

    model, feats, X_test, y_test = load_model()

    tab1, tab2 = st.tabs(["Personal Risk Assessment", "Model Performance"])

    with tab1:
        st.markdown("**Fill in a patient profile to get the estimated disorder risk**")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_in     = st.slider("Age", 18, 80, 32)
            gender_in  = st.selectbox("Gender", ["Male", "Female"])
            bmi_in     = st.slider("BMI", 16.0, 50.0, 25.0, 0.5)
            stress_in  = st.slider("Stress (1""10)", 1, 10, 5)
            anxiety_in = st.slider("Anxiety / GAD-7 (0""21)", 0, 21, 7)
        with c2:
            sleep_in    = st.slider("Avg sleep (hrs)", 2.0, 10.0, 7.0, 0.5)
            caffeine_in = st.slider("Caffeine (cups/day)", 0, 6, 2)
            screen_in   = st.slider("Screen time (hrs/day)", 0.0, 12.0, 4.0, 0.5)
            pa_in       = st.selectbox("Physical activity", ["None", "Low", "Moderate", "High"])
            occ_in      = st.selectbox("Job stress level", ["Low", "Medium", "High"])
        with c3:
            conflict_in = st.selectbox("War / conflict exposure", ["No", "Yes"])
            smoking_in  = st.selectbox("Smoking", ["No", "Yes"])
            alcohol_in  = st.selectbox("Alcohol use", ["No", "Yes"])
            pain_in     = st.selectbox("Chronic pain", ["No", "Yes"])

        inp = pd.DataFrame([{
            "Age":               age_in,
            "Gender_enc":        int(gender_in == "Male"),
            "BMI":               bmi_in,
            "Stress_Score":      stress_in,
            "Anxiety_Score":     anxiety_in,
            "Sleep_Duration_Hrs": sleep_in,
            "Caffeine_Daily_Cups": caffeine_in,
            "Screen_Time_Hrs":   screen_in,
            "PA_enc":            {"None": 0, "Low": 1, "Moderate": 2, "High": 3}[pa_in],
            "Conflict_Exposed":  int(conflict_in == "Yes"),
            "Occ_enc":           {"Low": 0, "Medium": 1, "High": 2}[occ_in],
            "Smoking":           int(smoking_in == "Yes"),
            "Alcohol_Use":       int(alcohol_in == "Yes"),
            "Chronic_Pain":      int(pain_in    == "Yes"),
        }])

        inp_ordered  = inp[feats]           # enforce exact column order from training
        prob         = float(model.predict_proba(inp_ordered)[0][1])
        risk_pct     = prob * 100
        risk_display = f"{risk_pct:.1f}"
        color        = "#16A34A" if risk_pct < 30 else ("#D97706" if risk_pct < 60 else "#DC2626")
        label        = "Low Risk" if risk_pct < 30 else ("Moderate Risk" if risk_pct < 60 else "High Risk")

        st.divider()
        res_col, shap_col = st.columns([1, 2.2])

        with res_col:
            st.markdown(f"""
            <div class='risk-card' style='border: 2px solid {color}; background:#FAFAF8;'>
                <div style='font-size:2.8rem; font-weight:700; color:{color};'>{risk_display}%</div>
                <div style='font-size:1rem; font-weight:600; color:{color}; margin-top:0.2rem;'>{label}</div>
                <div style='font-size:0.78rem; color:#78716C; margin-top:0.6rem; line-height:1.5;'>
                    Estimated probability of a sleep disorder being present based on this profile.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with shap_col:
            st.markdown("**Which factors matter most for this prediction?**")
            try:
                explainer = get_explainer(model)
                sv        = explainer.shap_values(inp_ordered)

                # Handle SHAP API differences across versions
                if hasattr(sv, "values"):
                    raw = sv.values
                    sv_arr = raw[0] if raw.ndim == 2 else raw[0, :, 1]
                elif isinstance(sv, list):
                    sv_arr = sv[1][0] if len(sv) == 2 else sv[0]
                else:
                    sv_arr = sv[0]

                sv_arr  = np.array(sv_arr, dtype=float).flatten()[:len(FEAT_LABELS)]
                shap_df = pd.DataFrame({"Feature": FEAT_LABELS, "SHAP": sv_arr})
                shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values().index)

                max_abs = float(np.abs(sv_arr).max()) * 1.15 or 0.1
                fig_sh = px.bar(shap_df, x="SHAP", y="Feature", orientation="h",
                                color="SHAP", color_continuous_scale="RdBu_r",
                                color_continuous_midpoint=0,
                                labels={"SHAP": "Impact on risk score"})
                fig_sh.add_vline(x=0, line_width=1, line_color="#9CA3AF")
                polish(fig_sh)
                fig_sh.update_layout(
                    coloraxis_showscale=False,
                    margin=dict(t=10, b=10, l=8, r=8),
                    height=340,
                    xaxis=dict(range=[-max_abs, max_abs], zeroline=False)
                )
                st.caption("Red = increases risk · Blue = reduces risk")
                st.plotly_chart(fig_sh, use_container_width=True)
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

    with tab2:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = round(roc_auc_score(y_test, y_prob), 3)
        acc    = round((y_pred == y_test).mean() * 100, 1)

        st.markdown(f"Trained on a stratified sample of **4,000 records** (80/20 split) · "
                    f"Accuracy **{acc}%** · ROC-AUC **{auc}**")
        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual"),
                               x=["No Disorder", "Has Disorder"],
                               y=["No Disorder", "Has Disorder"])
            polish(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        with m2:
            st.markdown("**ROC curve**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"AUC = {auc}",
                                         line=dict(color="#2563EB", width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dot", color="#CBD5E1"),
                                         showlegend=False))
            fig_roc.update_layout(**BASE, xaxis_title="False positive rate",
                                  yaxis_title="True positive rate",
                                  legend=dict(x=0.55, y=0.08))
            st.plotly_chart(fig_roc, use_container_width=True)

        with m3:
            st.markdown("**Global feature importance**")
            imp = pd.DataFrame({"Feature": FEAT_LABELS,
                                "Importance": model.feature_importances_})
            imp = imp.sort_values("Importance", ascending=True)
            fig_i = px.bar(imp, x="Importance", y="Feature", orientation="h",
                           color="Importance", color_continuous_scale="Blues")
            polish(fig_i)
            fig_i.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_i, use_container_width=True)

