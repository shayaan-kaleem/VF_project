import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
 # --- Rest of the script logic follows ---
import os
import re
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
vf_directory = '/Users/shayaan/Downloads'  # Update this path if your VF .txt files are elsewhere

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# --- Parse VF files to extract Patient_ID and Study_Date ---
vf_files = glob.glob(os.path.join(vf_directory, '*.txt'))
for vf_file in vf_files:
    try:
        with open(os.path.join(vf_directory, vf_file), 'r', encoding='utf-8') as file:
            contents = file.read()
    except UnicodeDecodeError:
        with open(os.path.join(vf_directory, vf_file), 'r', encoding='ISO-8859-1') as file:
            contents = file.read()
records = []
for vf_file in vf_files:
            contents = file.read()
    patient_id_match = re.search(r'Patient[ _]?ID[ :]*([A-Za-z0-9_-]+)', contents)
    study_date_match = re.search(r'Study[ _]?Date[ :]*([0-9]{4}-[0-9]{2}-[0-9]{2})', contents)
    if not patient_id_match or not study_date_match:
        continue  # Skip file if essential info is missing
    patient_id = patient_id_match.group(1)
    study_date = study_date_match.group(1)
    vf_record = {
        'Patient_ID': patient_id,
        'Study_Date': study_date,
        'Raw_Text': contents
    }
    records.append(vf_record)

# --- SITA Type & Pattern Extraction ---
def extract_sita_and_pattern(text):
    sita_match = re.search(r'Strategy[ :]*SITA[- ]?(Standard|Fast(er)?)', text, re.IGNORECASE)
    pattern_match = re.search(r'Pattern[ _]?Type[ :]*(\d{2}-\d{2})', text)
    sita_type = sita_match.group(1).capitalize() if sita_match else 'Unknown'
    if sita_type.lower() == 'faster':
        sita_type = 'Fast'  # Normalize Faster as Fast
    pattern_type = pattern_match.group(1) if pattern_match else 'Unknown'
    return sita_type, pattern_type

# Apply extraction
vf_df[['SITA_Type', 'Pattern_Type']] = vf_df['Raw_Text'].apply(lambda x: pd.Series(extract_sita_and_pattern(x)))

# --- Drop unknown SITA types and keep only 24-2 fields ---
vf_df = vf_df[(vf_df['SITA_Type'] != 'Unknown') & (vf_df['Pattern_Type'] == '24-2')]
vf_df = pd.DataFrame(records)
vf_df['Study_Date'] = pd.to_datetime(vf_df['Study_Date'])
# --- Reproducibility and Environment Logging ---
import sys
import random
import sklearn
import lightgbm
print("Python version:", sys.version)
print("scikit-learn version:", sklearn.__version__)
print("lightgbm version:", lightgbm.__version__)
# Fix seeds for reproducibility
np.random.seed(42)
random.seed(42)
from sklearn.metrics import roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
# Define canonical 54-point 24-2 grid (excluding blind spot)
x_vals = np.array([-27, -21, -15, -9, -3, 3, 9, 15, 21, 27])
y_vals = np.array([-21, -15, -9, -3, 3, 9, 15, 21])
canonical_grid = [(x, y) for y in y_vals for x in x_vals if not (x == 0 and y == 3)]
canonical_points = np.array(canonical_grid)
snap_cache = {}
def snap_to_grid(x, y, grid, tolerance=1.5):
    key = (x, y)
    if key in snap_cache:
        return snap_cache[key]
    dists = np.sqrt((grid[:, 0] - x) ** 2 + (grid[:, 1] - y) ** 2)
    min_idx = np.argmin(dists)
    result = tuple(grid[min_idx]) if dists[min_idx] <= tolerance else None
    snap_cache[key] = result
    return result
# Region definitions
nasal_pts = [(x, y) for x, y in canonical_grid if x > 9]
arcuate_sup = [(x, y) for x, y in canonical_grid if y > 0 and abs(x) >= 9]
arcuate_inf = [(x, y) for x, y in canonical_grid if y < 0 and abs(x) >= 9]
sup_hem_pts = [(x, y) for x, y in canonical_grid if y > 0]
inf_hem_pts = [(x, y) for x, y in canonical_grid if y < 0]
temporal_pts = [(x, y) for x, y in canonical_grid if x < -9]
paracentral_pts = [(x, y) for x, y in canonical_grid if abs(x) <= 9 and abs(y) <= 9]
# Regular expressions for parsing
re_date = re.compile(r"Tag: \(0008, 0020\), Name: Study Date, Value: (\d{8})")
re_dob = re.compile(r"Tag: \(0010, 0030\), Name: Patient's Birth Date, Value: (\d{8})")
re_pid = re.compile(r"Tag: \(0010, 0020\), Name: Patient ID, Value: (\S+)")
re_eye = re.compile(r"Tag: \(0020, 0060\), Name: Laterality, Value: (\S+)")
re_coord = re.compile(r"FL: ([\d\.\-]+)")
# Ensure AGE_NORMATIVE covers all points
AGE_NORMATIVE = {
    pt: {"base": norm, "slope": slope}
    for pt, norm, slope in zip(
        canonical_grid,
        [
            33.4, 33.0, 32.6, 32.2, 31.8, 31.8, 32.2, 32.6, 33.0, 33.4,
            33.8, 33.3, 32.9, 32.5, 32.1, 32.1, 32.5, 32.9, 33.3, 33.8,
            34.2, 33.7, 33.3, 32.9, 32.5, 32.5, 32.9, 33.3, 33.7, 34.2,
            34.6, 34.1, 33.7, 33.3, 32.9, 32.9, 33.3, 33.7, 34.1, 34.6,
            35.0, 34.5, 34.1, 33.7, 33.3, 33.3, 33.7, 34.1, 34.5, 35.0,
            35.4, 34.9, 34.5, 34.1,
        ],
        [0.06] * 54,
    )
}
AGE_NORMATIVE.update({pt: {"base": 32.0, "slope": 0.06} for pt in canonical_grid if pt not in AGE_NORMATIVE})
file_path = "/Users/shayaan/Downloads/VF_tags1-2100.txt"
    content = f.read()
records = content.split("Tag: (0024, 0089)")
def parse_vf_text_block(block):
    date_match = re_date.search(block)
    dob_match = re_dob.search(block)
    if not date_match or not dob_match:
        return None
    date = pd.to_datetime(date_match.group(1), format="%Y%m%d", errors="coerce")
    dob = pd.to_datetime(dob_match.group(1), format="%Y%m%d", errors="coerce")
    if pd.isna(date) or pd.isna(dob):
    age = (date - dob).days / 365.25
    pid_match = re_pid.search(block)
    patient_id = pid_match.group(1) if pid_match else None
    eye_match = re_eye.search(block)
    eye = eye_match.group(1).upper() if eye_match else "RE"
    if eye not in ["RE", "LE", "OD", "OS", "R", "L"]:
        eye = "RE"
    flip_eye = eye in ["LE", "OS", "L"]
    patient_eye_id = f"{patient_id}_{eye}"
    x, y, sens = None, None, None
    points = []
    for line in block.splitlines():
        if "(0024, 0090)" in line:
            x = float(re_coord.search(line).group(1))
        elif "(0024, 0091)" in line:
            y = float(re_coord.search(line).group(1))
        elif "(0024, 0094)" in line:
            s_match = re_coord.search(line)
            if s_match:
                sens = float(s_match.group(1))
                if x is not None and y is not None:
                    if flip_eye:
                        x = -x
                    points.append((x, y, sens))
                    x, y, sens = None, None, None
    snapped_coords = [(snap_to_grid(x, y, canonical_points), s) for x, y, s in points]
    snapped_coords = [(pt, s) for pt, s in snapped_coords if pt is not None]
    if len(set(pt for pt, _ in snapped_coords)) != 54:
    coord_map = dict(snapped_coords)
    sensitivities = [coord_map.get(pt, 0) for pt in canonical_grid]
    expected = [
        AGE_NORMATIVE[pt]["base"] - AGE_NORMATIVE[pt]["slope"] * (age - 50)
        for pt in canonical_grid
    ]
    td = [s - e for s, e in zip(sensitivities, expected)]
    calc_md = np.mean(td)
    calc_psd = np.std([d - calc_md for d in td])
    nasal_avg = np.mean([coord_map.get(pt, 0) for pt in nasal_pts])
    arc_sup_avg = np.mean([coord_map.get(pt, 0) for pt in arcuate_sup])
    arc_inf_avg = np.mean([coord_map.get(pt, 0) for pt in arcuate_inf])
    sup_inf_diff = abs(arc_sup_avg - arc_inf_avg)
    sup_avg = np.mean([coord_map.get(pt, 0) for pt in sup_hem_pts])
    inf_avg = np.mean([coord_map.get(pt, 0) for pt in inf_hem_pts])
    hemi_diff = abs(sup_avg - inf_avg)
    temp_avg = np.mean([coord_map.get(pt, 0) for pt in temporal_pts])
    para_avg = np.mean([coord_map.get(pt, 0) for pt in paracentral_pts])
    entry = {
        "Date": date,
        "Patient_ID": patient_eye_id,
        "Age": age,
        "MD": calc_md,
        "PSD": calc_psd,
        "Nasal_Sens": nasal_avg,
        "ArcSup_Sens": arc_sup_avg,
        "ArcInf_Sens": arc_inf_avg,
        "SupInf_Diff": sup_inf_diff,
        "SupHem_Avg": sup_avg,
        "InfHem_Avg": inf_avg,
        "Hemi_Diff": hemi_diff,
        "Temporal_Sens": temp_avg,
        "Paracentral_Sens": para_avg,
    entry.update({f"Point_{i+1}": val for i, val in enumerate(sensitivities)})
    return entry
vf_records = [parse_vf_text_block(r) for r in records if parse_vf_text_block(r)]
vf_df = pd.DataFrame(vf_records).sort_values(by="Date").reset_index(drop=True)
print("--- Data Overview ---")
print("Step 0: All parsed visual fields:", len(vf_df))
vf_df["Date"] = pd.to_datetime(vf_df["Date"])
vf_df["MD"] = pd.to_numeric(vf_df["MD"], errors="coerce")
vf_df["Days"] = vf_df.groupby("Patient_ID")["Date"].transform(lambda x: (x - x.min()).dt.days)
vf_counts = vf_df.groupby("Patient_ID").size()
eligible_ids = vf_counts[vf_counts >= 5].index
vf_df = vf_df[vf_df["Patient_ID"].isin(eligible_ids)]
print("Step 1: All unique patients with 54-point fields:", vf_df["Patient_ID"].nunique())
print("Step 2: Patients with ≥5 visual fields:", len(eligible_ids))
vf_df["Progressor_Label"] = None
for pid, group in vf_df.groupby("Patient_ID"):
    group = group.dropna(subset=["MD", "Days"])
    if group.shape[0] < 3:
        continue
    X = group["Days"].values.reshape(-1, 1)
    y = group["MD"].values
 # Define features and target
if 'Progression_Label' in train_df.columns:
    y = train_df['Progression_Label']
    X = train_df.drop(columns=['Progression_Label', 'VF1_Date', 'VF2_Date'], errors='ignore')
else:
    raise ValueError('Progression_Label column missing in training data')
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0] * 365.25
    label = int(slope < -1.0)
    vf_df.loc[vf_df["Patient_ID"] == pid, "Progressor_Label"] = label
vf_df = vf_df.dropna(subset=["Progressor_Label"])
print("Step 3: Patients with slope labels:", vf_df["Patient_ID"].nunique())
first_two = vf_df.sort_values(["Patient_ID", "Date"]).groupby("Patient_ID").head(2)
records = {}
record_list = []
for pid, group in first_two.groupby("Patient_ID"):
    if group.shape[0] < 2:
    row = {"Patient_ID": pid, "Progressor_Label": group.iloc[0]["Progressor_Label"]}
    for i, (_, row_data) in enumerate(group.iterrows(), start=1):
        for pt in range(1, 55):
            row[f"VF{i}_Point_{pt}"] = row_data[f"Point_{pt}"]
        row[f"Age{i}"] = row_data["Age"]
        row[f"MD{i}"] = row_data["MD"]
        row[f"PSD{i}"] = row_data["PSD"]
        row[f"Nasal{i}"] = row_data["Nasal_Sens"]
        row[f"ArcSup{i}"] = row_data["ArcSup_Sens"]
        row[f"ArcInf{i}"] = row_data["ArcInf_Sens"]
        row[f"SupInfDiff{i}"] = row_data["SupInf_Diff"]
        row[f"SupHem{i}"] = row_data["SupHem_Avg"]
        row[f"InfHem{i}"] = row_data["InfHem_Avg"]
        row[f"HemiDiff{i}"] = row_data["Hemi_Diff"]
        row[f"Temporal{i}"] = row_data["Temporal_Sens"]
        row[f"Paracentral{i}"] = row_data["Paracentral_Sens"]
        record_list.append(row)
train_df = pd.DataFrame(record_list)
print("Step 4: Patients with 2 early fields for ML:", train_df.shape[0])
print("Patients used in final model:", train_df.shape[0])
feature_cols = [col for col in train_df.columns if col not in ["Patient_ID", "Progressor_Label"]]
X = train_df[feature_cols]
y = train_df["Progressor_Label"].astype(int)
# --- Imports for interpretability, evaluation, and calibration ---
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
# --- Add Delta Features ---
train_df["MD_Change"] = train_df["MD2"] - train_df["MD1"]
train_df["PSD_Change"] = train_df["PSD2"] - train_df["PSD1"]
for i in range(1, 55):
    col1 = f"VF1_Point_{i}"
    col2 = f"VF2_Point_{i}"
    delta_col = f"VF_Change_Point_{i}"
    train_df[delta_col] = train_df[col2] - train_df[col1]
# --- Add Time Delta as Feature (days between VF1 and VF2) ---
# --- Merge Days_Between_VFs from extracted visual field metadata ---
train_df["Patient_ID"] = train_df["Patient_ID"].astype(str)
# --- Compute Days_Between_VFs directly from raw visual field text file ---
vf_records = []
    for line in f:
        if "Tag: (0010, 0020)" in line and "Patient ID" in line:
            pid_match = re.search(r"Value:\s*(\\d+)", line)
            if pid_match:
                current_record = {"Patient_ID": pid_match.group(1)}
        elif "Tag: (0008, 0020)" in line and "Study Date" in line:
            date_match = re.search(r"Value:\s*(\\d{8})", line)
            if date_match:
                current_record["Study_Date"] = datetime.strptime(date_match.group(1), "%Y%m%d")
                if "Patient_ID" in current_record and "Study_Date" in current_record:
                    vf_records.append(current_record)
                    current_record = {}
vf_df = pd.DataFrame(vf_records)
vf_df_sorted = vf_df.sort_values(by=["Patient_ID", "Study_Date"])
vf_first_two = vf_df_sorted.groupby("Patient_ID").head(2).copy()
vf_first_two["VF_Order"] = vf_first_two.groupby("Patient_ID").cumcount() + 1
vf_pivot = vf_first_two.pivot(index="Patient_ID", columns="VF_Order", values="Study_Date").reset_index()
vf_pivot.columns = ["Patient_ID", "VF1_Date", "VF2_Date"]
vf_pivot["Days_Between_VFs"] = (vf_pivot["VF2_Date"] - vf_pivot["VF1_Date"]).dt.days
vf_pivot["Patient_ID"] = vf_pivot["Patient_ID"].astype(str)
train_df = train_df.merge(vf_pivot[["Patient_ID", "Days_Between_VFs"]], on="Patient_ID", how="left")
# --- Final Feature Set ---
feature_cols = [col for col in train_df.columns if col not in ["Patient_ID", "Progressor_Label", "VF1_Date", "VF2_Date"]]
# Train/test split (stratified by label)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define LightGBM model with class weights
scale_weight = (y == 0).sum() / (y == 1).sum()
lgb_model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    class_weight={0: 1, 1: scale_weight},
    random_state=42
# Calibrate probabilities
calibrated_lgb = CalibratedClassifierCV(estimator=lgb_model, method="isotonic", cv=5)
calibrated_lgb.fit(X_train_scaled, y_train)
# --- Predict Probabilities and Threshold Using Youden’s J ---
y_prob = calibrated_lgb.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_j = tpr - fpr
optimal_threshold = thresholds[youden_j.argmax()]
y_pred = (y_prob >= optimal_threshold).astype(int)
# --- Evaluation Metrics ---
print(f"\nOptimal threshold based on Youden's J: {optimal_threshold:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("Brier Score:", brier_score_loss(y_test, y_prob))
# --- Stratified 5-Fold Cross-Validation AUC ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(calibrated_lgb, scaler.transform(X), y, cv=cv, scoring="roc_auc")
print("Stratified 5-Fold CV AUCs:", cv_auc)
print("Mean CV AUC:", cv_auc.mean())
# --- Calibration Curve Plot ---
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("Calibration Curve")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.grid()
plt.tight_layout()
plt.savefig("calibration_curve.png")
# --- SHAP Interpretation ---
explainer = shap.Explainer(calibrated_lgb.base_estimator_, X_train_scaled)
shap_values = explainer(X_test_scaled)
shap.summary_plot(shap_values, features=X_test_scaled, feature_names=X.columns, show=False)
plt.savefig("shap_summary_plot.png")