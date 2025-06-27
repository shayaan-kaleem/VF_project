
# Final integrated and filtered version of the visual field progression pipeline
# Includes: SITA type filtering, 24-2 pattern enforcement, and robust ML pipeline

import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from tqdm import tqdm

tqdm.pandas()

# Set your directory path
vf_directory = "/Users/shayaan/Downloads/VF_tags1-2100.txt"

# Define 24-2 coordinates (54 points) for validation
x_vals = np.array(
    [-27, -21, -15, -9, -3, 3, 9, 15, 21, 27] * 5
)
y_vals = np.array(
    [21] * 10 + [15] * 10 + [9] * 10 + [3] * 10 + [-3] * 10
    + [-9] * 10 + [-15] * 10 + [-21] * 10
)
x_vals = x_vals[:54]
y_vals = y_vals[:54]

# Helper to check if a set of coordinates includes all 24-2 points
def has_24_2(coords):
    vf_coords = set(tuple(row) for row in coords)
    template_coords = set(zip(x_vals, y_vals))
    return template_coords.issubset(vf_coords)

# Parse the visual field text file
def parse_visual_fields(filepath):
    with open(filepath, "r", errors="ignore") as file:
        contents = file.read()

    entries = re.split(r"
(?=Patient[ _]?ID[ :])", contents)
    all_data = []

    for entry in entries:
        try:
            patient_id = re.search(r'Patient[ _]?ID[ :]*([A-Za-z0-9_-]+)', entry).group(1)
            date_match = re.search(r'Study Date[ :]*([\d-]+)', entry)
            test_date = pd.to_datetime(date_match.group(1)) if date_match else None

            coords = re.findall(r"X: (-?\d+)\s+Y: (-?\d+)\s+Threshold: (\d+)", entry)
            coords = [(int(x), int(y), int(val)) for x, y, val in coords]
            if len(coords) == 0:
                continue

            coords_array = np.array(coords)
            sensitivity_points = coords_array.shape[0]

            sita_match = re.search(r'SITA[ -]?(\w+)', entry)
            sita_type = sita_match.group(0).strip() if sita_match else "Unknown"

            pattern_match = re.search(r'Pattern Type[ :]*([\d\w-]+)', entry)
            pattern_type = pattern_match.group(1).strip() if pattern_match else "Unknown"

            if pattern_type != "24-2":
                continue
            if sita_type not in ["SITA-Fast", "SITA-Faster"]:
                continue

            coords_only = coords_array[:, :2]
            if not has_24_2(coords_only):
                continue

            thresholds = coords_array[:, 2]

            record = {
                "Patient_ID": patient_id,
                "Study_Date": test_date,
                "SITA_Type": sita_type,
                "Pattern_Type": pattern_type,
            }

            for i, val in enumerate(thresholds):
                record[f"f{i+1}"] = val

            all_data.append(record)
        except Exception:
            continue

    df = pd.DataFrame(all_data)
    return df

# Load and preprocess data
print("--- Data Overview ---")
vf_df = parse_visual_fields(vf_directory)
print(f"Step 0: All parsed visual fields: {len(vf_df)}")

# Filter for patients with ≥5 VFs
vf_counts = vf_df["Patient_ID"].value_counts()
valid_ids = vf_counts[vf_counts >= 5].index
vf_df = vf_df[vf_df["Patient_ID"].isin(valid_ids)]
print(f"Step 1: All unique patients with 54-point fields: {vf_df['Patient_ID'].nunique()}")

# Compute slope labels using linear regression
def compute_slope_label(group):
    group = group.sort_values("Study_Date")
    fields = group[[f"f{i+1}" for i in range(54)]].to_numpy()
    dates = (group["Study_Date"] - group["Study_Date"].min()).dt.days.to_numpy()
    if len(fields) < 3:
        return pd.Series()
    slopes = np.polyfit(dates, fields.mean(axis=1), 1)[0]
    return pd.Series({"slope": slopes, "label": int(slopes < -0.2)})

slopes = vf_df.groupby("Patient_ID").apply(compute_slope_label).dropna()
vf_df = vf_df[vf_df["Patient_ID"].isin(slopes.index)]
print(f"Step 2: Patients with slope labels: {len(slopes)}")

# Merge slope labels
vf_df = vf_df.merge(slopes, left_on="Patient_ID", right_index=True)

# Take first 2 visual fields for modeling
vf_df_sorted = vf_df.sort_values(by=["Patient_ID", "Study_Date"])
vf_early = vf_df_sorted.groupby("Patient_ID").head(2)
patients_with_2 = vf_early["Patient_ID"].value_counts()
vf_early = vf_early[vf_early["Patient_ID"].isin(patients_with_2[patients_with_2 == 2].index)]
print(f"Step 3: Patients with 2 early fields for ML: {vf_early['Patient_ID'].nunique()}")

# Pivot into wide format: f1_1 to f54_1 and f1_2 to f54_2
features = []
labels = []
days_between = []

for pid, group in vf_early.groupby("Patient_ID"):
    if group.shape[0] != 2:
        continue
    row = {}
    group_sorted = group.sort_values("Study_Date")
    date1, date2 = group_sorted["Study_Date"].tolist()
    days = (date2 - date1).days
    days_between.append(days)
    for i, (_, r) in enumerate(group_sorted.iterrows()):
        for j in range(1, 55):
            row[f"f{j}_{i+1}"] = r.get(f"f{j}", np.nan)
    row["label"] = group_sorted["label"].iloc[0]
    row["days_between"] = days
    features.append(row)

train_df = pd.DataFrame(features).dropna()
print(f"Patients used in final model: {len(train_df)}")

# Train-test split
X = train_df.drop("label", axis=1)
y = train_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversample with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_scaled, y_train)

# Train LightGBM and XGBoost
lgb_model = lgb.LGBMClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

lgb_cal = CalibratedClassifierCV(lgb_model, method="isotonic", cv=5)
xgb_cal = CalibratedClassifierCV(xgb_model, method="isotonic", cv=5)

lgb_cal.fit(X_res, y_res)
xgb_cal.fit(X_res, y_res)

# Average predictions
preds_lgb = lgb_cal.predict_proba(X_test_scaled)[:, 1]
preds_xgb = xgb_cal.predict_proba(X_test_scaled)[:, 1]
avg_preds = (preds_lgb + preds_xgb) / 2

# Determine optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, avg_preds)
youden_index = np.argmax(tpr - fpr)
optimal_threshold = thresholds[youden_index]
print(f"Optimal threshold based on Youden's J: {optimal_threshold:.3f}")

# Final predictions
y_pred = (avg_preds >= optimal_threshold).astype(int)

# Evaluation
print("
Confusion Matrix:
", confusion_matrix(y_test, y_pred))
print("
Classification Report:
", classification_report(y_test, y_pred))
print("
ROC AUC Score:", roc_auc_score(y_test, avg_preds))

# Optional: Visualize ROC
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, avg_preds):.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
