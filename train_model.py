"""
train_model.py  –  Run once to produce model.pkl, scaler.pkl, ohe.pkl, le.pkl
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv("loan_approval_data.csv")
df = df.drop("Applicant_ID", axis=1)

# ── 2. Impute ──────────────────────────────────────────────────────────────────
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols   = df.select_dtypes(include=["number"]).columns

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# ── 3. Encode target ───────────────────────────────────────────────────────────
le = LabelEncoder()
df["Loan_Approved"] = le.fit_transform(df["Loan_Approved"])   # No=0, Yes=1

# ── 4. Label-encode Education_Level ───────────────────────────────────────────
le_edu = LabelEncoder()
df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

# ── 5. One-hot encode remaining categoricals ───────────────────────────────────
ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded     = ohe.fit_transform(df[ohe_cols])
encoded_df  = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

# ── 6. Feature engineering ────────────────────────────────────────────────────
df["DTI_Ratio_sq"]          = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"]       = df["Credit_Score"] ** 2
df["Applicant_Income_log"]  = np.log1p(df["Applicant_Income"])

X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]

# ── 7. Split & scale ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 8. Train Naive Bayes (best precision) ─────────────────────────────────────
model = GaussianNB()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print("=== Naive Bayes (final model) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# ── 9. Save artifacts ─────────────────────────────────────────────────────────
joblib.dump(model,   "model.pkl")
joblib.dump(scaler,  "scaler.pkl")
joblib.dump(ohe,     "ohe.pkl")
joblib.dump(le_edu,  "le_edu.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\n✅  Saved: model.pkl, scaler.pkl, ohe.pkl, le_edu.pkl, feature_names.pkl")
