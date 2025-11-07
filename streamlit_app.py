# app_streamlit.py
"""
MCQ / Telco Churn Prediction Streamlit App
- Provides inputs for required features (10-ish) + optional ones with defaults.
- Converts optional free-text customer review -> sentiment score with a lightweight lexicon.
- Preprocesses inputs to match a trained model's expected features.
- Attempts to load model & preprocessing artifacts from ./artifacts/.
- Falls back to a dummy heuristic model if artifacts are missing.
- Has an Analysis pane to show simple EDA if the user uploads a dataset CSV.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------
# App config & helpers
# ---------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📱")
st.title("📈 Telco Churn Predictor — Demo App")
st.markdown(
    """
Provide customer information (some fields required, others optional).  
You can also paste a short customer review — the app will convert it to a simple sentiment score (no external models required).
"""
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# ---------------------
# Simple sentiment converter (rule-based)
# ---------------------
POS_WORDS = {
    "good", "great", "excellent", "satisfied", "happy", "love", "recommend",
    "reliable", "easy", "best", "positive", "pleased", "fast", "impressed"
}
NEG_WORDS = {
    "bad", "terrible", "awful", "unhappy", "angry", "hate", "disappointed",
    "slow", "worst", "problem", "issue", "complain", "complaint", "expensive"
}


def simple_sentiment_score(text: str) -> float:
    """
    Very small lexicon-based sentiment score in [-1, +1].
    Designed to be deterministic and offline (no heavy models).
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    txt = text.lower()
    # tokenization (simple)
    words = [w.strip(".,!?()[]\"'") for w in txt.split()]
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    if pos == 0 and neg == 0:
        # fallback: length-based mild positive bias for longer reviews
        return 0.0
    score = (pos - neg) / max(1, (pos + neg))
    # normalize to 0..1 if your dataset uses that (we keep -1..1)
    return float(np.clip(score, -1.0, 1.0))


# ---------------------
# Model artifact loader
# ---------------------
def load_artifacts():
    """
    Attempts to load a saved sklearn model, scaler, and feature columns JSON.
    Returns: dict with keys: model, scaler, feat_cols, loaded(bool)
    """
    info = {"model": None, "scaler": None, "feat_cols": None, "loaded": False}
    try:
        if MODEL_PATH.exists():
            info["model"] = joblib.load(MODEL_PATH)
        if SCALER_PATH.exists():
            info["scaler"] = joblib.load(SCALER_PATH)
        if FEATURES_PATH.exists():
            info["feat_cols"] = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        # set loaded True if at least model and feat_cols present
        if info["model"] is not None and info["feat_cols"] is not None:
            info["loaded"] = True
    except Exception as e:
        st.warning(f"Could not load artifacts: {e}")
    return info


class DummyHeuristicModel:
    """
    Simple fallback model:
    - returns a churn prob based on tenure and monthly charges heuristics
    """
    def predict_proba(self, X):
        # X is DataFrame
        probs = []
        for _, r in X.iterrows():
            score = 0.05
            # tenure effect
            if "tenure" in X.columns:
                if r["tenure"] < 12:
                    score += 0.35
                elif r["tenure"] < 24:
                    score += 0.15
            # monthly charges effect
            if "MonthlyCharges" in X.columns:
                if r["MonthlyCharges"] > 80:
                    score += 0.25
                elif r["MonthlyCharges"] > 50:
                    score += 0.10
            # paperless billing minor
            if "PaperlessBilling" in X.columns and r.get("PaperlessBilling", 0) == 1:
                score += 0.03
            # online security reduces churn
            if "OnlineSecurity" in X.columns and r.get("OnlineSecurity", 0) == 1:
                score -= 0.10
            score = float(np.clip(score, 0.01, 0.99))
            probs.append([1 - score, score])  # [prob_no, prob_yes]
        return np.array(probs)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------
# Preprocessing helper (to align features)
# ---------------------
DEFAULT_FEATURE_COLUMNS = [
    # numeric + binary + example one-hot columns that model might expect
    "SeniorCitizen", "tenure", "MonthlyCharges", "feedback_length", "sentiment",
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "PaperlessBilling", "HasInternet",
    # example one-hot columns (if your saved model used them)
    "InternetService_fiber optic", "InternetService_no", "Contract_one year", "Contract_two year",
    "PaymentMethod_credit card (automatic)", "PaymentMethod_electronic check",
    "PaymentMethod_mailed check", "gender_male"
]


def preprocess_input(df_input: pd.DataFrame, feat_cols: List[str] = None, scaler=None) -> pd.DataFrame:
    """
    - Clean internet-service alias values
    - Create HasInternet
    - Replace 'yes'/'no' with 1/0 for binary columns we expect
    - One-hot for multi-cat columns keeping only expected columns (feat_cols)
    - Scale numeric features if scaler provided
    """
    df = df_input.copy()

    # normalize strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # replace 'no internet service' -> 'no' and 'no phone service' -> 'no'
    internet_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace({"no internet service": "no"})

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"no phone service": "no"})

    # HasInternet
    if "InternetService" in df.columns:
        df["HasInternet"] = df["InternetService"].apply(lambda v: 0 if str(v).lower() == "no" else 1)

    # binary mapping
    yesno_cols = ['Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for col in yesno_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0, "Yes": 1, "No": 0, True: 1, False: 0}).fillna(0).astype(int)

    # SeniorCitizen sometimes numeric already; ensure numeric
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # One-hot encode multi-cat columns (safe)
    multi_cols = [c for c in ["InternetService", "Contract", "PaymentMethod", "gender"] if c in df.columns]
    if len(multi_cols) > 0:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    # If feature columns provided, align to them, adding missing columns filled with 0
    if feat_cols is None:
        feat_cols = DEFAULT_FEATURE_COLUMNS
    # ensure all numeric columns expected exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0

    # Reorder to feat_cols
    df_out = df[feat_cols].copy()

    # Scale numerics if scaler given (assume scaler expects numeric_cols)
    if scaler is not None:
        # try to scale 'tenure', 'MonthlyCharges', 'feedback_length', 'sentiment' if present
        numeric_cols = [c for c in ["tenure", "MonthlyCharges", "feedback_length", "sentiment"] if c in df_out.columns]
        if numeric_cols:
            try:
                scaled = scaler.transform(df_out[numeric_cols])
                df_out[numeric_cols] = scaled
            except Exception as e:
                st.warning(f"Scaler present but failed to transform numeric cols: {e}")

    return df_out


# ---------------------
# Load model / artifacts
# ---------------------
artifacts = load_artifacts()
if artifacts["loaded"]:
    st.success("✅ Loaded model artifacts from ./artifacts/")
    model = artifacts["model"]
    model_feat_cols = artifacts["feat_cols"]
    scaler = artifacts.get("scaler")
else:
    st.info("Model artifacts not found in ./artifacts/. App will use a fallback heuristic model.")
    model = DummyHeuristicModel()
    model_feat_cols = DEFAULT_FEATURE_COLUMNS
    scaler = None

# ---------------------
# Sidebar: quick dataset upload & settings
# ---------------------
st.sidebar.header("Options & Data")
uploaded = st.sidebar.file_uploader("Upload dataset CSV for Analysis (optional)", type=["csv"])
show_artifacts_info = st.sidebar.checkbox("Show expected artifact filenames", value=False)
if show_artifacts_info:
    st.sidebar.write("Expected artifact files (optional):")
    st.sidebar.write("- artifacts/model.pkl  (sklearn-like model with predict_proba)")
    st.sidebar.write("- artifacts/scaler.pkl (sklearn StandardScaler)")
    st.sidebar.write("- artifacts/feature_columns.json (list of feature column names)")

# ---------------------
# Main: Input form for one customer
# ---------------------
st.header("Customer input (single) — required fields + optional defaults")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=29, step=1)
        monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=1000.0, value=70.0, step=0.5)
        total_charges = st.number_input("TotalCharges (optional)", min_value=0.0, value=1397.0, step=1.0)
    with col2:
        senior = st.selectbox("SeniorCitizen", options=[0, 1], index=0)
        partner = st.selectbox("Partner", options=["no", "yes"], index=0)
        dependents = st.selectbox("Dependents", options=["no", "yes"], index=0)
    with col3:
        internet = st.selectbox("Internet Service", options=["fiber optic", "dsl", "no"], index=1)
        contract = st.selectbox("Contract", options=["month-to-month", "one year", "two year"], index=0)
        paperless = st.selectbox("PaperlessBilling", options=["yes", "no"], index=0)

    st.markdown("#### Optional extras (kept as defaults if not changed)")
    col4, col5 = st.columns(2)
    with col4:
        online_security = st.selectbox("OnlineSecurity", options=["yes", "no", "no internet service"], index=1)
        tech_support = st.selectbox("TechSupport", options=["yes", "no", "no internet service"], index=1)
        phone_service = st.selectbox("PhoneService", options=["yes", "no"], index=0)
    with col5:
        multiple_lines = st.selectbox("MultipleLines", options=["no", "yes", "no phone service"], index=0)
        streaming_tv = st.selectbox("StreamingTV", options=["no", "yes", "no internet service"], index=0)
        payment_method = st.selectbox("PaymentMethod", options=["electronic check", "mailed check", "bank transfer (automatic)", "credit card (automatic)"], index=0)

    # customer review -> sentiment
    st.markdown("#### Customer review (optional)")
    review_text = st.text_area("Customer review (optional). We'll convert to a sentiment score automatically.", height=120)
    # optional inputs for feedback_length if you want to override:
    feedback_length_input = st.number_input("Feedback length (optional; auto-calculated if left 0)", min_value=0, value=0, step=1)

    submitted = st.form_submit_button("🔮 Predict churn")

# ---------------------
# When submitted: assemble row, preprocess, predict
# ---------------------
if submitted:
    # assemble dataframe single row
    row = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "StreamingTV": streaming_tv,
        "StreamingMovies": "no",  # default
        "DeviceProtection": "no",
        "gender": "male",  # default but you could expose
        "PaymentMethod": payment_method
    }
    # sentiment
    sentiment_score = simple_sentiment_score(review_text)
    # Compute feedback_length
    feedback_length = feedback_length_input if feedback_length_input > 0 else (len(review_text.split()) if review_text else 0)
    row["feedback_length"] = feedback_length
    row["sentiment"] = sentiment_score

    df_row = pd.DataFrame([row])

    # Preprocess / align
    X_row = preprocess_input(df_row, feat_cols=model_feat_cols, scaler=scaler)

    # Prediction
    try:
        proba = float(model.predict_proba(X_row)[0, 1])
        pred = int(proba >= 0.5)
    except Exception:
        # fallback if model doesn't have predict_proba
        pred = int(model.predict(X_row)[0])
        proba = None

    # Show results
    st.subheader("Prediction")
    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Churn (0 = stay, 1 = churn)", pred)
        if proba is not None:
            st.progress(proba)
            st.write(f"Churn probability: **{proba:.2%}**")
    with colB:
        st.info("Interpretation / recommended actions:")
        # simple rules to explain
        reasons = []
        if tenure < 12:
            reasons.append("New customer — consider onboarding offers & quick wins to retain.")
        if monthly > 80:
            reasons.append("High monthly charges — consider targeted discounts or value bundles.")
        if paperless == "yes":
            reasons.append("Paperless billing -> slight churn risk (monitor).")
        if online_security == "no":
            reasons.append("Lack of OnlineSecurity -> add security-focused upsell to reduce churn.")
        if sentiment_score < 0:
            reasons.append("Negative review sentiment — follow up with customer support.")
        if len(reasons) == 0:
            reasons = ["No major red flags detected — keep monitoring."]
        for r in reasons:
            st.write("- " + r)

    # If model has feature importances, show top features in bar chart
    try:
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=model_feat_cols)
            top = importances.sort_values(ascending=False).head(8)
            st.subheader("Top Model Features (if using a tree model)")
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=top.values, y=top.index, ax=ax)
            ax.set_xlabel("Importance")
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not show feature importances: {e}")

# ---------------------
# Analysis tab: quick EDA if dataset uploaded
# ---------------------
st.header("Analysis (EDA) — Upload dataset CSV to reproduce project visuals")
if uploaded is not None:
    try:
        df_upload = pd.read_csv(uploaded)
        st.write("Uploaded dataset preview:")
        st.dataframe(df_upload.head(6))

        # Basic target distribution
        if "Churn" in df_upload.columns:
            st.subheader("Churn distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df_upload, x="Churn", palette="viridis", ax=ax)
            st.pyplot(fig)

        # Correlation with churn heatmap (encoded)
        st.subheader("Feature correlations with Churn (ranked)")
        # safe encode
        cols_for_encoding = [c for c in df_upload.columns if df_upload[c].dtype != "O"]
        # We'll do one-hot for a few categorical cols automatically
        df_enc = pd.get_dummies(df_upload, drop_first=False)
        if "Churn" in df_enc.columns:
            corr = df_enc.corr()[["Churn"]].sort_values(by="Churn", ascending=False)
            fig, ax = plt.subplots(figsize=(6, min(0.3 * len(corr), 10)))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=False, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No 'Churn' column found in the uploaded dataset for correlation analysis.")

        st.markdown("### Quick recommended actions (automatically suggested):")
        st.write(
            "1. Investigate high churn segments (month-to-month contracts, high MonthlyCharges, low tenure).<br>"
            "2. Improve low-value features like OnlineSecurity adoption for vulnerable groups.<br>"
            "3. Run A/B pilots for promotions that convert month-to-month -> 1-year contracts.<br>"
            "4. Use sentiment monitoring: flag negative reviews and follow up quickly."
            , unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Could not analyze uploaded file: {e}")
else:
    st.info("Upload dataset CSV on the left to run EDA visualizations (optional).")

# ---------------------
# Footer / instructions
# ---------------------
st.markdown("---")
st.markdown(
    """
**How to use your trained model with this app**
1. Train and export your sklearn-like model to `artifacts/model.pkl`. Use `joblib.dump(model, 'artifacts/model.pkl')`.
2. Save a fitted `StandardScaler` to `artifacts/scaler.pkl` if you scaled numerics during training.
3. Save a JSON list of feature column names your model expects (ordered) to `artifacts/feature_columns.json`.
   Example:)
