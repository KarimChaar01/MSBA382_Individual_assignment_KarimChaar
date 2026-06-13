"""Pre-trains the XGBoost model locally and saves it for fast cloud loading."""
import pandas as pd, numpy as np, joblib, os
from sklearn.model_selection import train_test_split
import xgboost as xgb

patients = pd.read_csv("data/patient_data.csv")
df = patients.sample(n=4000, random_state=42).reset_index(drop=True).copy()
df["Target"]     = (df["Sleep_Disorder"] != "No Disorder").astype(int)
df["PA_enc"]     = df["Physical_Activity"].map({"None_PA":0,"Low":1,"Moderate":2,"High":3}).fillna(1).astype(int)
df["Occ_enc"]    = df["Occupation_Stress"].map({"Low":0,"Medium":1,"High":2}).fillna(1).astype(int)
df["Gender_enc"] = (df["Gender"] == "Male").astype(int)

FEATS = ["Age","Gender_enc","BMI","Stress_Score","Anxiety_Score","Sleep_Duration_Hrs",
         "Caffeine_Daily_Cups","Screen_Time_Hrs","PA_enc","Conflict_Exposed","Occ_enc",
         "Smoking","Alcohol_Use","Chronic_Pain"]

X, y = df[FEATS], df["Target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                           eval_metric="logloss", random_state=42,
                           scale_pos_weight=(ytr==0).sum()/max((ytr==1).sum(),1))
model.fit(Xtr, ytr)

os.makedirs("model", exist_ok=True)
joblib.dump({"model": model, "feats": FEATS, "X_test": Xte, "y_test": yte}, "model/xgb_model.pkl")
print(f"Saved. AUC check done on {len(Xte)} test rows.")
