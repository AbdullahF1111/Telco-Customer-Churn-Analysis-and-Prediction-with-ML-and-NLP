"""
📊 Telco Customer Churn Prediction App
- Predicts churn probability using your trained ML model (Random Forest).
- Accepts ~10 key customer inputs + optional feedback (converted to sentiment).
- Automatically aligns inputs with model’s training features.
- Includes dynamic feature importance and optional dataset EDA upload.
"""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===========================
# ⚙️ App Configuration
# ===========================
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📱")
st.title("📈 Telco Customer Churn Prediction — Interactive App")
st.markdown("""
Welcome to the Telco Churn Prediction Dashboard!  
Enter customer details below to predict churn risk and get actionable insights.  
Optionally, you can upload a dataset for quick EDA.
""")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# ===========================
# 💬 Simple Sentiment Logic
# ===========================
POS_WORDS = {"good","great","excellent","happy","satisfied","love","recommend","reliable","best","positive","pleased"}
NEG_WORDS = {"bad","terrible","awful","angry","hate","disappointed","slow","worst","problem","complaint","expensive"}

def simple_sentiment_score(text: str) -> float:
    """Tiny rule-based sentiment analyzer."""
    if not text or len(text.strip()) == 0:
        return 0.0
    words = text.lower().split()
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return round((pos - neg) / (pos + neg), 2)

# ===========================
# 🧠 Load Artifacts
# ===========================
def load_artifacts():
    info = {"model": None, "scaler": None, "feat_cols": None, "loaded": False}
    try:
        if MODEL_PATH.exists():
            info["model"] = joblib.load(MODEL_PATH)
        if SCALER_PATH.exists():
            info["scaler"] = joblib.load(SCALER_PATH)
        if FEATURES_PATH.exists():
            info["feat_cols"] = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        if info["model"] is not None and info["feat_cols"] is not None:
            info["loaded"] = True
    except Exception as e:
        st.warning(f"Could not load model artifacts: {e}")
    return info

artifacts = load_artifacts()
if artifacts["loaded"]:
    st.success("✅ Model & artifacts loaded successfully.")
else:
    st.warning("⚠️ Could not find model artifacts — ensure they’re in the `artifacts/` folder.")

model = artifacts["model"]
scaler = artifacts.get("scaler")
model_feat_cols = artifacts["feat_cols"] if artifacts["feat_cols"] else []

# ===========================
# 🧩 Preprocessing Function
# ===========================
def preprocess_input(df_input: pd.DataFrame, feat_cols: List[str], scaler=None):
    df = df_input.copy()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # Clean up text-based "no internet/phone service"
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].replace({"no internet service": "no"})
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"no phone service": "no"})

    # Binary encoding
    yesno_cols = ['Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yesno_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

    # HasInternet
    if "InternetService" in df.columns:
        df["HasInternet"] = df["InternetService"].apply(lambda v: 0 if v == "no" else 1)

    # SeniorCitizen
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # One-hot encode categorical features
    multi_cols = [c for c in ["InternetService","Contract","PaymentMethod","gender"] if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    # Align to model feature list
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feat_cols]

    # Scale numeric features
    num_cols = [c for c in ['tenure','MonthlyCharges','TotalCharges','feedback_length','sentiment'] if c in df.columns]
    if scaler is not None:
        try:
            df[num_cols] = scaler.transform(df[num_cols])
        except Exception as e:
            st.warning(f"Scaling failed: {e}")
    return df

# ===========================
# 🧍 Input Form
# ===========================
st.header("🎯 Customer Input Form")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 24)
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 75.0)
        total = st.number_input("Total Charges ($)", 0.0, 9000.0, 1500.0)
    with col2:
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["no", "yes"])
        dependents = st.selectbox("Dependents", ["no", "yes"])
    with col3:
        internet = st.selectbox("Internet Service", ["fiber optic", "dsl", "no"])
        contract = st.selectbox("Contract", ["month-to-month", "one year", "two year"])
        paperless = st.selectbox("Paperless Billing", ["yes", "no"])

    st.markdown("### Optional Features")
    col4, col5 = st.columns(2)
    with col4:
        online_security = st.selectbox("Online Security", ["yes", "no", "no internet service"])
        tech_support = st.selectbox("Tech Support", ["yes", "no", "no internet service"])
        phone_service = st.selectbox("Phone Service", ["yes", "no"])
    with col5:
        multiple_lines = st.selectbox("Multiple Lines", ["no", "yes", "no phone service"])
        streaming_tv = st.selectbox("Streaming TV", ["no", "yes", "no internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["no", "yes", "no internet service"])
        device_protection = st.selectbox("Device Protection", ["no", "yes", "no internet service"])
        payment = st.selectbox("Payment Method", [
            "electronic check", "mailed check", "bank transfer (automatic)", "credit card (automatic)"
        ])

    review_text = st.text_area("Customer Review (optional for sentiment analysis)", height=100)
    submitted = st.form_submit_button("🔮 Predict Churn")

# ===========================
# 🔍 Prediction Section
# ===========================
if submitted:
    sentiment = simple_sentiment_score(review_text)
    feedback_length = len(review_text.split()) if review_text else 0

    row = {
        "SeniorCitizen": senior, "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Partner": partner, "Dependents": dependents, "InternetService": internet,
        "Contract": contract, "PaperlessBilling": paperless,
        "OnlineSecurity": online_security, "TechSupport": tech_support, "PhoneService": phone_service,
        "MultipleLines": multiple_lines, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "DeviceProtection": device_protection,
        "PaymentMethod": payment, "gender": "male",
        "feedback_length": feedback_length, "sentiment": sentiment
    }

    df_input = pd.DataFrame([row])
    X_input = preprocess_input(df_input, feat_cols=model_feat_cols, scaler=scaler)

    proba = float(model.predict_proba(X_input)[0, 1])
    pred = int(proba >= 0.5)

    st.subheader("🧾 Prediction Result")
    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Churn Prediction", f"{pred} ({'Yes' if pred==1 else 'No'})")
        st.progress(proba)
        st.write(f"**Churn Probability:** {proba:.2%}")
    with colB:
        st.info("Recommended Business Actions:")
        if tenure < 12:
            st.write("- Offer welcome retention program for new customers.")
        if monthly > 80:
            st.write("- Consider loyalty discounts for high spenders.")
        if online_security == "no":
            st.write("- Promote online security bundles to reduce churn risk.")
        if sentiment < 0:
            st.write("- Negative feedback detected: prioritize customer support outreach.")
        else:
            st.write("- No strong churn indicators detected — maintain engagement.")

    # ===========================
    # 🔥 Feature Importance (Dynamic)
    # ===========================
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=model_feat_cols)
        top_features = importances.sort_values(ascending=False).head(10)
        st.subheader("Feature Importance (Top 10)")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_features.values, y=top_features.index, palette="viridis", ax=ax)
        st.pyplot(fig)

# ===========================
# 📊 Optional Dataset EDA
# ===========================
st.header("📈 Quick EDA — Upload Dataset")
uploaded = st.file_uploader("Upload your Telco dataset (CSV)", type=["csv"])
if uploaded is not None:
    df_up = pd.read_csv(uploaded)
    st.write("Sample Data Preview:")
    st.dataframe(df_up.head(5))

    if "Churn" in df_up.columns:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Churn", data=df_up, palette="coolwarm", ax=ax)
        st.pyplot(fig)

        df_enc = pd.get_dummies(df_up, drop_first=True)
        corr = df_enc.corr()["Churn"].sort_values(ascending=False)
        st.subheader("Feature Correlation with Churn")
        fig, ax = plt.subplots(figsize=(6, 10))
        sns.heatmap(corr.to_frame(), annot=True, cmap="coolwarm", cbar=False)
        st.pyplot(fig)

# ===========================
# 🧾 Footer
# ===========================
st.markdown("---")
st.caption("Developed by Abdullah Fahlo — Data Science Portfolio Project")
