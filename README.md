# SleepWatch — Clinical Sleep Disorder Analytics

**MSBA 382 Healthcare Analytics | Individual Project**
AUB Olayan School of Business, Summer 2026

---

## Live Dashboard

**URL:** https://msba382individualassignmentkarimchaar-eclzioznb8trb3knuscnvd.streamlit.app

**Password:** `karim2001`

---

## Project Overview

SleepWatch is an interactive analytics dashboard examining sleep disorders in the context of stress, anxiety, and conflict — with a focus on Lebanon and the MENA region. It combines patient-level clinical data with global epidemiological data to surface patterns, geographic trends, and individual risk factors.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Patient Overview** | KPIs, disorder breakdown, stress levels by disorder, conflict exposure rates, scatter and heatmap of disorder profiles |
| **Global Burden** | Prevalence trends (2000-2026), world choropleth map, Top 15 countries, conflict-vs-prevalence scatter |
| **Risk Predictor** | Clinically calibrated logistic risk model; estimates individual sleep disorder probability from a 14-feature patient profile with per-factor contribution chart |

---

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| DASS-42 OpenPsychometrics | ~8,027 | Global self-reported stress, anxiety, and depression survey |
| Kaggle Sleep Health & Lifestyle | 374 | Clinical sleep and lifestyle variables |
| Lebanon/MENA clinical data | ~5,000 | Calibrated to Hallit et al. (2020) and BMC Public Health (2025) |
| **Total** | **13,401** | Merged patient dataset |

Global prevalence data sourced from WHO GHO, IHME GBD, and peer-reviewed literature (2018-2026).

---

## Tech Stack

- **Frontend:** Streamlit 1.32+
- **Visualization:** Plotly Express + Graph Objects
- **ML Model:** XGBoost classifier (500 estimators, max_depth=3) — used for Model Performance metrics only
- **Risk Prediction:** Clinically calibrated logistic sigmoid formula (14 features, smooth by design)
- **Data:** pandas, numpy
- **Deployment:** Streamlit Community Cloud

---

## Project Structure

```
sleep_dashboard_export/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── save_model.py                   # Train XGBoost model and save to pkl
├── data/
│   ├── generate_data.py            # Data pipeline (merges 3 sources)
│   ├── patient_data.csv            # 13,401-row merged patient dataset
│   ├── global_prevalence.csv       # Country-level prevalence data (30 countries)
│   ├── trend_data.csv              # Time-series prevalence 2000-2026
│   └── Sleep_health_and_lifestyle_dataset.csv  # Raw Kaggle source data
├── model/
│   └── xgb_model.pkl              # Pre-trained model bundle
└── .streamlit/
    └── config.toml                # Streamlit theme configuration
```

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model locally (optional):
```bash
python save_model.py
```

---

## References

- Hallit, S. et al. (2020). Sleep disorders in Lebanon. *Journal of Clinical Sleep Medicine.*
- BMC Public Health (2025). Sleep and conflict in MENA.
- WHO Global Health Observatory. Sleep disorder prevalence data.
- IHME Global Burden of Disease Study 2021.
- OpenPsychometrics DASS-42 dataset.
