"""
Pre-trains XGBoost on a synthetic dataset built with correct causal direction:
features are sampled first, then disorder probability is derived from them via
a clinically-calibrated logistic formula. This guarantees sensible predictor
behavior: each factor moves risk gradually and in the right direction.

Previous model was trained on patient_data.csv which was generated disorder-first
(disorder -> features), causing near-perfect separation on sleep/stress alone and
wildly unrealistic jumps (e.g. 0.5 hr sleep change: 30% -> 90%).
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb

RNG = np.random.default_rng(42)
N   = 20_000   # large dataset so XGBoost learns smooth decision boundaries

# ── 1. Sample features independently (no disorder label yet) ─────────────────
age        = np.clip(RNG.normal(38, 15, N), 18, 80).astype(int)
gender_enc = RNG.integers(0, 2, N)              # 0=Female, 1=Male
bmi        = np.clip(RNG.normal(26.5, 5.5, N), 16.0, 50.0).round(1)
stress     = RNG.integers(1, 11, N)             # 1–10
anxiety    = RNG.integers(0, 22, N)             # 0–21
sleep_dur  = np.clip(RNG.normal(7.0, 1.5, N), 2.0, 10.0).round(1)
caffeine   = RNG.integers(0, 7, N)              # 0–6
screen     = np.clip(RNG.normal(5.0, 2.5, N), 0.0, 12.0).round(1)
pa_enc     = RNG.integers(0, 4, N)              # 0=None 1=Low 2=Moderate 3=High
conflict   = RNG.binomial(1, 0.30, N)
occ_enc    = RNG.integers(0, 3, N)              # 0=Low 1=Medium 2=High
smoking    = RNG.binomial(1, 0.28, N)
alcohol    = RNG.binomial(1, 0.22, N)
chronic    = RNG.binomial(1, 0.18, N)

# ── 2. Logistic risk formula  (clinically sensible coefficients) ──────────────
#
# Intercept = -2.75 so the DEFAULT app profile (stress=5, anxiety=7, sleep=7,
# pa=Moderate, occ=Low, no conflict/smoking/alcohol/pain, bmi=25, age=32)
# produces ~20% risk -> "Low Risk" starting point.
#
# Full-range contribution of each feature:
#   stress    1->10 : +3.50  (biggest driver along with anxiety)
#   anxiety   0->21 : +3.00
#   sleep    10->2  : +3.00  (protective when HIGH, harmful when LOW)
#   conflict  0->1  : +1.40
#   occ_stress L->H : +1.00  (clearly visible Low vs High job stress effect)
#   bmi      25->50 : +0.70  (only above 25)
#   smoking   0->1  : +0.70
#   chronic   0->1  : +0.70
#   pa        3->0  : +0.70  (protective when HIGH)
#   caffeine  0->6  : +0.40
#   screen    0->12 : +0.25
#   alcohol   0->1  : +0.25
#   age      35->80 : +0.20
#   gender    F->M  : +0.14
#
# Total positive range ≈ 13.24 -> max logit ≈ -2.75 + 13.24 = +10.49 -> p≈99%
# Total protective range ≈ -3.70 -> min logit ≈ -2.75 - 3.70 = -6.45 -> p≈0.2%

logit = (
    -2.75                                          # intercept
    + 3.50 * (stress  - 1) / 9.0                  # stress    1->10  adds  0 to +3.50
    + 3.00 * anxiety / 21.0                        # anxiety   0->21  adds  0 to +3.00
    - 3.00 * (sleep_dur - 2.0) / 8.0              # sleep    10->2   adds -3.00 to  0
    + 1.40 * conflict                              # conflict  0/1
    + 1.00 * occ_enc / 2.0                        # job stress 0->2  adds  0 to +1.00
    + 0.70 * np.maximum(0, bmi - 25.0) / 25.0     # BMI above 25    adds  0 to +0.70
    + 0.70 * smoking                               # smoking   0/1
    + 0.70 * chronic                               # chronic pain 0/1
    - 0.70 * pa_enc / 3.0                         # physical act 3->0 adds -0.70 to  0
    + 0.40 * caffeine / 6.0                        # caffeine  0->6   adds  0 to +0.40
    + 0.25 * screen / 12.0                         # screen    0->12  adds  0 to +0.25
    + 0.25 * alcohol                               # alcohol   0/1
    + 0.20 * np.maximum(0, age - 35) / 45.0        # age above 35    adds  0 to +0.20
    + 0.14 * gender_enc                            # gender (male slight OSA risk)
)

p_true = 1.0 / (1.0 + np.exp(-logit))
target  = RNG.binomial(1, p_true)

print(f"Synthetic N={N:,} | True probability range: {p_true.min():.3f} – {p_true.max():.3f}")
print(f"Disorder rate: {target.mean():.1%}")

# ── 3. Build DataFrame ────────────────────────────────────────────────────────
FEATS = [
    "Age", "Gender_enc", "BMI", "Stress_Score", "Anxiety_Score",
    "Sleep_Duration_Hrs", "Caffeine_Daily_Cups", "Screen_Time_Hrs",
    "PA_enc", "Conflict_Exposed", "Occ_enc",
    "Smoking", "Alcohol_Use", "Chronic_Pain"
]

X = pd.DataFrame({
    "Age":                 age,
    "Gender_enc":          gender_enc,
    "BMI":                 bmi,
    "Stress_Score":        stress,
    "Anxiety_Score":       anxiety,
    "Sleep_Duration_Hrs":  sleep_dur,
    "Caffeine_Daily_Cups": caffeine,
    "Screen_Time_Hrs":     screen,
    "PA_enc":              pa_enc,
    "Conflict_Exposed":    conflict,
    "Occ_enc":             occ_enc,
    "Smoking":             smoking,
    "Alcohol_Use":         alcohol,
    "Chronic_Pain":        chronic,
})
y = pd.Series(target)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train XGBoost (smooth settings prevent sharp decision boundaries) ──────
model = xgb.XGBClassifier(
    n_estimators    = 500,
    max_depth       = 3,       # shallow trees -> smoother surface
    learning_rate   = 0.02,    # many small steps -> gradual boundaries
    subsample       = 0.70,    # row sampling -> less overfitting
    colsample_bytree= 0.70,    # feature sampling -> diverse trees
    min_child_weight= 20,      # min samples per leaf -> no tiny sharp splits
    gamma           = 1.0,     # min gain to split -> conservative splits
    reg_alpha       = 0.1,     # L1 regularisation
    reg_lambda      = 2.0,     # L2 regularisation
    eval_metric     = "logloss",
    random_state    = 42,
    scale_pos_weight= (ytr == 0).sum() / max((ytr == 1).sum(), 1)
)
model.fit(Xtr, ytr)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_prob = model.predict_proba(Xte)[:, 1]
y_pred = model.predict(Xte)
auc    = roc_auc_score(yte, y_prob)
acc    = accuracy_score(yte, y_pred)
print(f"Test AUC: {auc:.3f}  |  Accuracy: {acc:.1%}  |  Test rows: {len(Xte):,}")

# ── 6. Sanity checks — verify each feature moves risk in the right direction ──
def predict(row_dict):
    df = pd.DataFrame([{f: row_dict.get(f, 0) for f in FEATS}])
    return model.predict_proba(df)[0][1] * 100

base = dict(Age=35, Gender_enc=1, BMI=25, Stress_Score=5, Anxiety_Score=7,
            Sleep_Duration_Hrs=7.0, Caffeine_Daily_Cups=2, Screen_Time_Hrs=4,
            PA_enc=2, Conflict_Exposed=0, Occ_enc=0,
            Smoking=0, Alcohol_Use=0, Chronic_Pain=0)

print("\n-- Sanity checks (direction & magnitude) --")
print(f"  Default profile (should be Low Risk ~15-25%):  {predict(base):.1f}%")

for label, changes in [
    ("Stress 5->10",             {"Stress_Score": 10}),
    ("Stress 5->1",              {"Stress_Score": 1}),
    ("Anxiety 7->20",            {"Anxiety_Score": 20}),
    ("Sleep 7->5 hrs",           {"Sleep_Duration_Hrs": 5.0}),
    ("Sleep 7->9 hrs",           {"Sleep_Duration_Hrs": 9.0}),
    ("Sleep 7->6.5 (0.5 drop)",  {"Sleep_Duration_Hrs": 6.5}),
    ("Conflict YES",            {"Conflict_Exposed": 1}),
    ("Job stress Low->High",     {"Occ_enc": 2}),
    ("Job stress Low->Medium",   {"Occ_enc": 1}),
    ("Smoking YES",             {"Smoking": 1}),
    ("Chronic pain YES",        {"Chronic_Pain": 1}),
    ("PA None (was Moderate)",  {"PA_enc": 0}),
    ("High risk (all bad)",     {"Stress_Score":10,"Anxiety_Score":21,"Sleep_Duration_Hrs":2.0,
                                  "Conflict_Exposed":1,"Occ_enc":2,"BMI":40,
                                  "Smoking":1,"Chronic_Pain":1,"PA_enc":0,"Caffeine_Daily_Cups":6}),
    ("Low risk (all good)",     {"Stress_Score":1,"Anxiety_Score":1,"Sleep_Duration_Hrs":9.0,
                                  "Conflict_Exposed":0,"Occ_enc":0,"BMI":21,
                                  "Smoking":0,"Chronic_Pain":0,"PA_enc":3,"Caffeine_Daily_Cups":0}),
]:
    profile = {**base, **changes}
    print(f"  {label:<35}: {predict(profile):.1f}%")

# ── 7. Save ───────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(
    {"model": model, "feats": FEATS, "X_test": Xte, "y_test": yte},
    "model/xgb_model.pkl"
)
print("\nmodel/xgb_model.pkl saved.")
