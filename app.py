"""
SleepWatch: Sleep Disorders in the Context of Stress, Anxiety and Conflict
MSBA 382 - Healthcare Analytics | Individual Project
AUB - Olayan School of Business, Summer 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SleepWatch | Healthcare Analytics",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"],
h1, h2, h3, h4, h5, h6, p, span, div, label, button,
.stMarkdown, .stText, .stSelectbox, .stMultiSelect,
.stSlider, .stRadio, .stDataFrame, .stMetric,
section[data-testid="stSidebar"] * {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

.block-container { padding-top: 1.2rem !important; }

.kpi-card {
    background: #f8f9fc;
    border: 1px solid #e0e4ef;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4361EE;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.3rem;
}
.kpi-sub {
    font-size: 0.75rem;
    color: #aaa;
    margin-top: 0.1rem;
}
.section-note {
    background: #f0f4ff;
    border-left: 3px solid #4361EE;
    padding: 0.65rem 1rem;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #333;
    margin-bottom: 1.2rem;
}
.kpi-accent { color: #F72585 !important; }
.kpi-warn   { color: #f39c12 !important; }
</style>
""", unsafe_allow_html=True)


# ── Password protection ────────────────────────────────────────────────────────
def check_password() -> bool:
    if st.session_state.get("authenticated"):
        return True
    col1, col2, col3 = st.columns([1, 1.3, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; padding:2rem 2rem 1.5rem 2rem;
                    background:#f8f9fc; border-radius:14px; border:1px solid #e0e4ef;'>
            <div style='font-size:3rem; line-height:1;'>🌙</div>
            <h2 style='margin:0.5rem 0 0.1rem; color:#1a1a2e; font-size:1.6rem;'>SleepWatch</h2>
            <p style='color:#555; font-size:0.88rem; margin:0;'>Healthcare Analytics Dashboard</p>
            <p style='color:#aaa; font-size:0.78rem; margin:0.2rem 0 1.2rem;'>
                MSBA 382 &nbsp;|&nbsp; AUB Olayan School of Business
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        pwd = st.text_input("Password", type="password", placeholder="Enter dashboard password",
                            label_visibility="collapsed")
        b1, b2, b3 = st.columns([1, 2, 1])
        with b2:
            if st.button("Enter Dashboard", use_container_width=True, type="primary"):
                if pwd == "karim2001":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data():
    patients = pd.read_csv("data/patient_data.csv")
    country  = pd.read_csv("data/global_prevalence.csv")
    trend    = pd.read_csv("data/trend_data.csv")
    return patients, country, trend


patients, country, trend = load_data()

DISORDERS = ["Insomnia", "Sleep Apnea", "Hypersomnia", "Narcolepsy", "Restless Leg Syndrome"]
ALL_CATEGORIES = DISORDERS + ["No Disorder"]

PALETTE = {
    "Insomnia":              "#4361EE",
    "Sleep Apnea":           "#F72585",
    "Hypersomnia":           "#7209B7",
    "Narcolepsy":            "#3A0CA3",
    "Restless Leg Syndrome": "#4CC9F0",
    "No Disorder":           "#ADB5BD",
}

CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#f8f9fc",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#1a1a2e"),
    margin=dict(t=30, b=30, l=10, r=10)
)

LB_EVENTS = {
    2019: "Economic Collapse",
    2020: "Port Explosion",
    2024: "War 1",
    2026: "War 2",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.8rem 0 0.4rem;'>
        <div style='font-size:2rem;'>🌙</div>
        <div style='font-size:1.1rem; font-weight:700; color:#1a1a2e;'>SleepWatch</div>
        <div style='font-size:0.7rem; color:#888; margin-top:0.2rem;'>
            Sleep Disorders &amp; Conflict<br>MSBA 382 | Summer 2026
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "nav",
        ["📊 Patient Overview", "🌍 Global Burden", "🤖 Risk Predictor"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Filters** — Patient Overview only")

    gender_filter = st.multiselect("Gender", ["Male", "Female"], default=["Male", "Female"])
    age_range     = st.slider("Age Range", 18, 80, (18, 80))
    disorder_filter = st.multiselect("Disorder Type", ALL_CATEGORIES, default=ALL_CATEGORIES)
    conflict_filter = st.selectbox("Conflict Exposure", ["All", "Exposed only", "Not exposed only"])

    st.divider()
    st.caption("Sources: WHO GHE · IHME GBD · OpenPsychometrics DASS-42 · Kaggle · Hallit et al. 2020 · BMC Public Health 2025")


# ── Filters ────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["Gender"].isin(gender_filter) &
        df["Age"].between(age_range[0], age_range[1]) &
        df["Sleep_Disorder"].isin(disorder_filter)
    )
    if conflict_filter == "Exposed only":
        mask &= df["Conflict_Exposed"] == 1
    elif conflict_filter == "Not exposed only":
        mask &= df["Conflict_Exposed"] == 0
    return df[mask].copy()


df = apply_filters(patients)


# ── Helpers ────────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "", accent: bool = False):
    val_class = "kpi-value kpi-accent" if accent else "kpi-value"
    sub_html  = f"<div class='kpi-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='{val_class}'>{value}</div>
        <div class='kpi-label'>{label}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def fmt_chart(fig, title=""):
    fig.update_layout(**CHART_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color="#1a1a2e"), x=0))
    fig.update_xaxes(showgrid=False, linecolor="#e0e4ef", tickfont=dict(color="#555"))
    fig.update_yaxes(gridcolor="#e0e4ef", linecolor="#e0e4ef", tickfont=dict(color="#555"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PATIENT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Patient Overview":
    st.title("📊 Patient Overview")
    st.markdown(f"""
    <div class='section-note'>
    Analysis of <strong>{len(df):,} patients</strong> drawn from three sources: DASS-42 OpenPsychometrics
    survey (n=8,027), Kaggle Sleep Health dataset (n=374), and a Lebanon/MENA conflict-calibrated
    simulation (n=5,000). Filters apply to all charts below.
    </div>
    """, unsafe_allow_html=True)

    # ── 6 KPIs ──────────────────────────────────────────────────────────────
    total    = len(df)
    affected = len(df[df["Sleep_Disorder"] != "No Disorder"])
    prev_pct = f"{affected / total * 100:.1f}%" if total > 0 else "N/A"

    avg_stress  = f"{df['Stress_Score'].mean():.1f} / 10"
    avg_anxiety = f"{df['Anxiety_Score'].mean():.1f} / 21"
    avg_sleep   = f"{df['Sleep_Duration_Hrs'].mean():.1f} hrs"
    conf_pct    = f"{df['Conflict_Exposed'].mean() * 100:.1f}%"
    top_dis     = (df[df["Sleep_Disorder"] != "No Disorder"]["Sleep_Disorder"]
                   .value_counts().idxmax() if affected > 0 else "N/A")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1: kpi("Disorder Prevalence", prev_pct, f"{affected:,} of {total:,}", accent=True)
    with r1c2: kpi("Avg Stress Score", avg_stress, "1–10 scale")
    with r1c3: kpi("Avg Anxiety (GAD-7)", avg_anxiety, "0–21 scale")
    st.markdown("<br>", unsafe_allow_html=True)
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1: kpi("Avg Sleep Duration", avg_sleep, "per night")
    with r2c2: kpi("Conflict-Exposed", conf_pct, "war/crisis exposure")
    with r2c3: kpi("Top Disorder", top_dis, "most prevalent")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Disorder Distribution")
        counts = (df[df["Sleep_Disorder"] != "No Disorder"]["Sleep_Disorder"]
                  .value_counts().reset_index())
        counts.columns = ["Disorder", "Count"]
        fig = px.pie(
            counts, names="Disorder", values="Count",
            color="Disorder", color_discrete_map=PALETTE, hole=0.50
        )
        fig.update_layout(**CHART_LAYOUT, legend=dict(orientation="h", y=-0.12))
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Stress Score by Disorder")
        fig2 = px.box(
            df, x="Sleep_Disorder", y="Stress_Score",
            color="Sleep_Disorder", color_discrete_map=PALETTE,
            labels={"Sleep_Disorder": "", "Stress_Score": "Stress (1–10)"}
        )
        fmt_chart(fig2)
        fig2.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Conflict Exposure by Disorder")
        cross = (df[df["Sleep_Disorder"] != "No Disorder"]
                 .groupby(["Sleep_Disorder", "Conflict_Exposed"]).size()
                 .reset_index(name="Count"))
        cross["Exposure"] = cross["Conflict_Exposed"].map({0: "Not Exposed", 1: "Conflict Exposed"})
        fig3 = px.bar(
            cross, x="Sleep_Disorder", y="Count", color="Exposure",
            barmode="group",
            color_discrete_sequence=["#ADB5BD", "#F72585"],
            labels={"Sleep_Disorder": "", "Count": "Patients"}
        )
        fmt_chart(fig3)
        fig3.update_layout(legend_title="", legend=dict(orientation="h", y=1.08),
                           xaxis_tickangle=-15)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Age Distribution by Disorder")
        fig4 = px.box(
            df[df["Sleep_Disorder"] != "No Disorder"],
            x="Sleep_Disorder", y="Age",
            color="Sleep_Disorder", color_discrete_map=PALETTE,
            labels={"Sleep_Disorder": "", "Age": "Age (years)"}
        )
        fmt_chart(fig4)
        fig4.update_layout(showlegend=False, xaxis_tickangle=-15)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Disorder Summary Table")
    summary = df.groupby("Sleep_Disorder").agg(
        Patients        = ("Age", "count"),
        Avg_Age         = ("Age",            lambda x: round(x.mean(), 1)),
        Pct_Female      = ("Gender",          lambda x: f"{(x=='Female').mean()*100:.1f}%"),
        Avg_Stress      = ("Stress_Score",    lambda x: round(x.mean(), 1)),
        Avg_Anxiety     = ("Anxiety_Score",   lambda x: round(x.mean(), 1)),
        Avg_Sleep_Hrs   = ("Sleep_Duration_Hrs", lambda x: round(x.mean(), 1)),
        Conflict_Exposed= ("Conflict_Exposed",lambda x: f"{x.mean()*100:.1f}%"),
    ).reset_index().sort_values("Patients", ascending=False)
    summary.columns = ["Disorder", "Patients", "Avg Age", "% Female",
                       "Avg Stress", "Avg Anxiety (GAD-7)",
                       "Avg Sleep (hrs)", "Conflict Exposed %"]
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — GLOBAL BURDEN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Global Burden":
    st.title("🌍 Global & Lebanon Burden")
    st.markdown("""
    <div class='section-note'>
    Countries affected by active or recent armed conflict rank among the highest globally in sleep
    disorder prevalence. Lebanon (55.5%), Palestine (59.6%), and Syria (53.1%) far exceed the global
    average of 16.2%. Lebanon's rate has surged <strong>+17 percentage points</strong> since 2018,
    driven by economic collapse (2019), the Beirut port explosion (2020), and two successive wars.
    </div>
    """, unsafe_allow_html=True)

    # ── 6 KPIs ──────────────────────────────────────────────────────────────
    lbn_2026  = trend[trend["Year"] == 2026]["Lebanon_Prevalence"].values[0]
    lbn_2018  = trend[trend["Year"] == 2018]["Lebanon_Prevalence"].values[0]
    lbn_surge = lbn_2026 - lbn_2018

    pal_prev  = country[country["Country"] == "Palestine"]["Any_Sleep_Disorder_Pct"].values[0]
    mena_2026 = trend[trend["Year"] == 2026]["MENA_Prevalence"].values[0]
    glob_2026 = trend[trend["Year"] == 2026]["Global_Prevalence"].values[0]
    n_countries = len(country)

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1: kpi("Lebanon Prevalence", f"{lbn_2026:.1f}%", "2026 estimate", accent=True)
    with r1c2: kpi("Palestine Prevalence", f"{pal_prev:.1f}%", "highest globally", accent=True)
    with r1c3: kpi("MENA Average", f"{mena_2026:.1f}%", "regional 2026")
    st.markdown("<br>", unsafe_allow_html=True)
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1: kpi("Global Average", f"{glob_2026:.1f}%", "worldwide 2026")
    with r2c2: kpi("Lebanon War Surge", f"+{lbn_surge:.1f} pp", "vs 2018 baseline")
    with r2c3: kpi("Countries Analyzed", str(n_countries), "across all regions")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── World map ─────────────────────────────────────────────────────────────
    st.subheader("Global Sleep Disorder Prevalence Map")
    fig_map = px.choropleth(
        country, locations="ISO3", color="Any_Sleep_Disorder_Pct",
        hover_name="Country",
        hover_data={"ISO3": False, "Region": True,
                    "Any_Sleep_Disorder_Pct": ":.1f",
                    "Conflict_Index": ":.2f"},
        color_continuous_scale="Blues",
        projection="natural earth",
        labels={"Any_Sleep_Disorder_Pct": "Prevalence (%)"}
    )
    fig_map.update_layout(
        paper_bgcolor="white",
        geo=dict(showframe=False, showcoastlines=True, coastlinecolor="#ccc",
                 showland=True, landcolor="#f8f9fc"),
        margin=dict(t=0, b=0, l=0, r=0),
        coloraxis_colorbar=dict(title="Prevalence (%)", tickfont=dict(size=10))
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 15 Countries by Prevalence")
        top15 = country.nlargest(15, "Any_Sleep_Disorder_Pct").sort_values("Any_Sleep_Disorder_Pct")
        fig_top = px.bar(
            top15, x="Any_Sleep_Disorder_Pct", y="Country", orientation="h",
            color="Conflict_Index", color_continuous_scale="Reds",
            text="Any_Sleep_Disorder_Pct",
            labels={"Any_Sleep_Disorder_Pct": "Prevalence (%)", "Conflict_Index": "Conflict Index"}
        )
        fig_top.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fmt_chart(fig_top)
        fig_top.update_layout(coloraxis_colorbar=dict(title="Conflict", tickfont=dict(size=9)))
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        st.subheader("Conflict Index vs Sleep Disorder Prevalence")
        fig_sc = px.scatter(
            country, x="Conflict_Index", y="Any_Sleep_Disorder_Pct",
            color="Region", size="Any_Sleep_Disorder_Pct",
            hover_name="Country", trendline="ols",
            labels={"Any_Sleep_Disorder_Pct": "Prevalence (%)", "Conflict_Index": "Conflict Index"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fmt_chart(fig_sc)
        fig_sc.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Trend chart ───────────────────────────────────────────────────────────
    st.subheader("Lebanon vs Global Trends (2000–2026)")
    colors_t = {
        "Global_Prevalence":      "#ADB5BD",
        "MENA_Prevalence":        "#4361EE",
        "MENA_Conflict_Subgroup": "#FF6B35",
        "Lebanon_Prevalence":     "#F72585",
    }
    labels_t = {
        "Global_Prevalence":      "Global Average",
        "MENA_Prevalence":        "MENA Region",
        "MENA_Conflict_Subgroup": "MENA Conflict Subgroup",
        "Lebanon_Prevalence":     "Lebanon",
    }
    widths_t = {
        "Global_Prevalence": 1.6, "MENA_Prevalence": 1.8,
        "MENA_Conflict_Subgroup": 2.0, "Lebanon_Prevalence": 2.8
    }

    fig_trend = go.Figure()
    for col, color in colors_t.items():
        fig_trend.add_trace(go.Scatter(
            x=trend["Year"], y=trend[col],
            name=labels_t[col],
            line=dict(color=color, width=widths_t[col]),
            mode="lines+markers",
            marker=dict(size=4 if col != "Lebanon_Prevalence" else 5)
        ))

    for yr, label in LB_EVENTS.items():
        leb_val = trend[trend["Year"] == yr]["Lebanon_Prevalence"].values[0]
        fig_trend.add_annotation(
            x=yr, y=leb_val + 3.5, text=label,
            showarrow=True, arrowhead=2, arrowcolor="#F72585",
            arrowsize=0.8, ax=0, ay=-30,
            font=dict(size=9.5, color="#F72585"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#F72585", borderwidth=1, borderpad=3
        )
        fig_trend.add_vline(x=yr, line_dash="dot", line_color="#F72585", opacity=0.28)

    fig_trend.update_layout(
        **{**CHART_LAYOUT, "margin": dict(t=60, b=40, l=10, r=10)},
        xaxis_title="Year",
        yaxis_title="Estimated Prevalence (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        yaxis=dict(range=[8, 70]),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Country table ─────────────────────────────────────────────────────────
    st.subheader("Country Data Table")
    display_cols = ["Country", "Region", "Any_Sleep_Disorder_Pct",
                    "Insomnia_Prev_Pct", "Sleep_Apnea_Prev_Pct",
                    "Anxiety_Prev_Pct", "Stress_Index", "Conflict_Index"]
    st.dataframe(
        country[display_cols]
        .sort_values("Any_Sleep_Disorder_Pct", ascending=False)
        .rename(columns={
            "Any_Sleep_Disorder_Pct": "Any Disorder (%)",
            "Insomnia_Prev_Pct":      "Insomnia (%)",
            "Sleep_Apnea_Prev_Pct":   "Sleep Apnea (%)",
            "Anxiety_Prev_Pct":       "Anxiety (%)",
            "Stress_Index":           "Stress Index",
            "Conflict_Index":         "Conflict Index"
        }),
        use_container_width=True, hide_index=True
    )

    st.divider()
    st.caption(
        "Data sources: WHO GHO · IHME GBD · Hallit et al. (2020) · BMC Public Health (2025) · "
        "Research Square (2025) · Tandfonline (2025) · JOGH (2025) | "
        "Dashboard: MSBA 382, AUB Olayan School of Business, Summer 2026"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK PREDICTOR (BONUS)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.title("🤖 Sleep Disorder Risk Predictor")
    st.markdown("""
    <div class='section-note'>
    <strong>Bonus component.</strong> An XGBoost classifier trained on 13,401 patient records predicts
    individual sleep disorder risk. SHAP (SHapley Additive exPlanations) values explain which factors
    drive each prediction — making the model interpretable and clinically relevant.
    </div>
    """, unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    import xgboost as xgb
    import shap

    @st.cache_resource(show_spinner="Training XGBoost model on 13,401 records…")
    def train_model():
        df_ml = patients.copy()
        df_ml["Target"] = (df_ml["Sleep_Disorder"] != "No Disorder").astype(int)

        pa_map  = {"None_PA": 0, "Low": 1, "Moderate": 2, "High": 3}
        occ_map = {"Low": 0, "Medium": 1, "High": 2}
        df_ml["PA_enc"]     = df_ml["Physical_Activity"].map(pa_map).fillna(1)
        df_ml["Occ_enc"]    = df_ml["Occupation_Stress"].map(occ_map).fillna(1)
        df_ml["Gender_enc"] = (df_ml["Gender"] == "Male").astype(int)

        features = ["Age", "Gender_enc", "BMI", "Stress_Score", "Anxiety_Score",
                    "Sleep_Duration_Hrs", "Caffeine_Daily_Cups", "Screen_Time_Hrs",
                    "PA_enc", "Conflict_Exposed", "Occ_enc",
                    "Smoking", "Alcohol_Use", "Chronic_Pain"]

        X = df_ml[features]
        y = df_ml["Target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        )
        model.fit(X_train, y_train)
        return model, features, X_test, y_test

    model, features, X_test, y_test = train_model()

    FEATURE_LABELS = ["Age", "Gender", "BMI", "Stress", "Anxiety (GAD-7)",
                      "Sleep Duration (hrs)", "Caffeine", "Screen Time",
                      "Physical Activity", "Conflict Exposed",
                      "Job Stress", "Smoking", "Alcohol", "Chronic Pain"]

    tab1, tab2 = st.tabs(["🎯 Personal Risk Assessment", "📊 Model Performance"])

    # ── Tab 1: Risk Assessment ────────────────────────────────────────────────
    with tab1:
        st.subheader("Enter Patient Profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            age_in     = st.slider("Age", 18, 80, 32)
            gender_in  = st.selectbox("Gender", ["Male", "Female"])
            bmi_in     = st.slider("BMI", 16.0, 50.0, 25.0, 0.5)
            stress_in  = st.slider("Stress Score (1–10)", 1, 10, 5)
            anxiety_in = st.slider("Anxiety (GAD-7, 0–21)", 0, 21, 7)
        with c2:
            sleep_in    = st.slider("Avg Sleep Duration (hrs)", 2.0, 10.0, 7.0, 0.5)
            caffeine_in = st.slider("Daily Caffeine (cups)", 0, 6, 2)
            screen_in   = st.slider("Screen Time (hrs/day)", 0.0, 12.0, 4.0, 0.5)
            pa_in       = st.selectbox("Physical Activity", ["None", "Low", "Moderate", "High"])
            occ_in      = st.selectbox("Occupation Stress", ["Low", "Medium", "High"])
        with c3:
            conflict_in = st.selectbox("Conflict/War Exposure", ["No", "Yes"])
            smoking_in  = st.selectbox("Smoking", ["No", "Yes"])
            alcohol_in  = st.selectbox("Alcohol Use", ["No", "Yes"])
            pain_in     = st.selectbox("Chronic Pain", ["No", "Yes"])

        input_df = pd.DataFrame([{
            "Age": age_in,
            "Gender_enc": 1 if gender_in == "Male" else 0,
            "BMI": bmi_in,
            "Stress_Score": stress_in,
            "Anxiety_Score": anxiety_in,
            "Sleep_Duration_Hrs": sleep_in,
            "Caffeine_Daily_Cups": caffeine_in,
            "Screen_Time_Hrs": screen_in,
            "PA_enc": {"None": 0, "Low": 1, "Moderate": 2, "High": 3}[pa_in],
            "Conflict_Exposed": 1 if conflict_in == "Yes" else 0,
            "Occ_enc": {"Low": 0, "Medium": 1, "High": 2}[occ_in],
            "Smoking": 1 if smoking_in == "Yes" else 0,
            "Alcohol_Use": 1 if alcohol_in == "Yes" else 0,
            "Chronic_Pain": 1 if pain_in == "Yes" else 0,
        }])

        prob     = model.predict_proba(input_df)[0][1]
        risk_pct = round(prob * 100, 1)

        st.divider()
        r1, r2 = st.columns([1, 2])

        with r1:
            if risk_pct < 30:
                color, label, icon = "#27ae60", "Low Risk", "✅"
            elif risk_pct < 60:
                color, label, icon = "#f39c12", "Moderate Risk", "⚠️"
            else:
                color, label, icon = "#e74c3c", "High Risk", "🚨"

            st.markdown(f"""
            <div style='background:#f8f9fc; border:2px solid {color}; border-radius:14px;
                        padding:2rem 1.5rem; text-align:center; margin-top:0.5rem;'>
                <div style='font-size:3rem; line-height:1;'>{icon}</div>
                <div style='font-size:3rem; font-weight:800; color:{color}; line-height:1.1;'>
                    {risk_pct}%</div>
                <div style='font-size:1rem; color:{color}; font-weight:600; margin-top:0.3rem;'>
                    {label}</div>
                <div style='font-size:0.78rem; color:#888; margin-top:0.6rem; line-height:1.4;'>
                    Estimated probability of<br>sleep disorder presence
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.subheader("SHAP: What Drives This Prediction")
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(input_df)
            shap_df   = pd.DataFrame({
                "Feature": FEATURE_LABELS,
                "SHAP Value": shap_vals[0]
            }).sort_values("SHAP Value", key=abs, ascending=True)

            fig_shap = px.bar(
                shap_df, x="SHAP Value", y="Feature", orientation="h",
                color="SHAP Value", color_continuous_scale="RdBu_r",
                color_continuous_midpoint=0,
                title="Red = increases risk  |  Blue = decreases risk"
            )
            fmt_chart(fig_shap)
            fig_shap.update_layout(coloraxis_showscale=False,
                                   title=dict(font=dict(size=11), x=0))
            st.plotly_chart(fig_shap, use_container_width=True)

    # ── Tab 2: Model Performance ──────────────────────────────────────────────
    with tab2:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = round(roc_auc_score(y_test, y_prob), 3)
        acc    = round((y_pred == y_test).mean(), 3)

        st.markdown(
            f"Trained on **{len(patients):,} records** (80/20 split) · "
            f"Accuracy **{acc:.1%}** · ROC-AUC **{auc}**"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=["No Disorder", "Has Disorder"],
                y=["No Disorder", "Has Disorder"]
            )
            fmt_chart(fig_cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        with m2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                         name=f"AUC = {auc}",
                                         line=dict(color="#4361EE", width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                         line=dict(dash="dot", color="#ADB5BD"),
                                         showlegend=False))
            fig_roc.update_layout(**CHART_LAYOUT,
                                  xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate",
                                  legend=dict(x=0.55, y=0.1))
            st.plotly_chart(fig_roc, use_container_width=True)

        with m3:
            st.subheader("Feature Importance")
            imp = pd.DataFrame({
                "Feature": FEATURE_LABELS,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Blues")
            fmt_chart(fig_imp)
            fig_imp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
