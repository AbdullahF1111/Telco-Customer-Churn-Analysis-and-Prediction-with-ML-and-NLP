# ===============================================
# TELCO CUSTOMER CHURN PREDICTION APP
# ===============================================
# ✅ Uses your trained ML model and scaler
# ✅ Supports optional customer review → sentiment
# ✅ Includes EDA section for uploaded datasets
# ===============================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. App configuration
# ----------------------------
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📊", layout="wide")
st.title("📈 Telco Customer Churn Predictor")
st.markdown("""
This demo predicts whether a telecom customer is likely to **churn** based on their profile and usage data.  
You can also add an optional customer review — the app converts it into a **sentiment score**.
""")

# ----------------------------
# 2. Load your saved artifacts
# ----------------------------
MODEL_PATH = "artifacts/model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
FEATURES_PATH = "artifacts/feature_columns.json"

st.sidebar.header("Model Artifacts")
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "r") as f:
        model_feat_cols = json.load(f)
    st.sidebar.success("✅ Model, Scaler, and Features loaded successfully!")
    st.sidebar.write(f"**Model type:** {type(model).__name__}")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model artifacts: {e}")
    st.stop()

# ----------------------------
# 3. Simple sentiment analyzer (offline)
# ----------------------------
POS_WORDS = {"good","great","excellent","satisfied","happy","love","recommend","reliable","best","positive","pleased","fast"}
NEG_WORDS = {"bad","terrible","awful","unhappy","angry","hate","disappointed","slow","worst","problem","issue","complaint","expensive"}

def simple_sentiment_score(text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    words = text.lower().split()
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    if pos + neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / (pos + neg), -1, 1))

# ----------------------------
# 4. Preprocessing function
# ----------------------------
def preprocess_input(df, feat_cols, scaler):
    df = df.copy()
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

    # Replace "no internet service" → "no"
    internet_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for c in internet_cols:
        if c in df.columns:
            df[c] = df[c].replace("no internet service","no")

    # Replace "no phone service"
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace("no phone service","no")

    # Add HasInternet
    if "InternetService" in df.columns:
        df["HasInternet"] = df["InternetService"].apply(lambda v: 0 if v == "no" else 1)

    # Encode yes/no
    yesno_cols = ['Partner','Dependents','PhoneService','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for c in yesno_cols:
        if c in df.columns:
            df[c] = df[c].map({"yes":1,"no":0}).fillna(0).astype(int)

    # One-hot encode multi-category features
    df = pd.get_dummies(df, columns=["InternetService","Contract","PaymentMethod","gender"], drop_first=False)

    # Align with model columns
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feat_cols]

    # Scale numeric columns
    numeric_cols = [c for c in ["tenure","MonthlyCharges","feedback_length","sentiment"] if c in df.columns]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

# ----------------------------
# 5. Input section
# ----------------------------
st.header("🧾 Customer Information")

with st.form("customer_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=24)
        monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=1000.0, value=70.0)
        senior = st.selectbox("Senior Citizen", options=[0, 1])
    with col2:
        partner = st.selectbox("Partner", ["yes","no"])
        dependents = st.selectbox("Dependents", ["yes","no"])
        internet = st.selectbox("Internet Service", ["dsl","fiber optic","no"])
    with col3:
        contract = st.selectbox("Contract Type", ["month-to-month","one year","two year"])
        payment = st.selectbox("Payment Method", ["electronic check","mailed check","bank transfer (automatic)","credit card (automatic)"])
        paperless = st.selectbox("Paperless Billing", ["yes","no"])

    st.markdown("### Optional Internet Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        online_security = st.selectbox("Online Security", ["no","yes","no internet service"])
        device_protection = st.selectbox("Device Protection", ["no","yes","no internet service"])
    with c2:
        tech_support = st.selectbox("Tech Support", ["no","yes","no internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["no","yes","no internet service"])
    with c3:
        streaming_movies = st.selectbox("Streaming Movies", ["no","yes","no internet service"])
        phone_service = st.selectbox("Phone Service", ["yes","no"])
        multiple_lines = st.selectbox("Multiple Lines", ["no","yes","no phone service"])

    # Optional review
    st.markdown("### 🗣️ Optional Customer Review")
    review = st.text_area("Enter a short customer review (optional)")
    sentiment = simple_sentiment_score(review)
    feedback_length = len(review.split()) if review else 0

    submitted = st.form_submit_button("🔮 Predict Churn")

# ----------------------------
# 6. Predict when submitted
# ----------------------------
if submitted:
    row = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "OnlineSecurity": online_security,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "gender": "male",
        "feedback_length": feedback_length,
        "sentiment": sentiment
    }

    df_input = pd.DataFrame([row])
    X_input = preprocess_input(df_input, feat_cols=model_feat_cols, scaler=scaler)

    proba = model.predict_proba(X_input)[0, 1]
    pred = int(proba >= 0.5)

    st.subheader("🎯 Prediction Result")
    colA, colB = st.columns([1,2])
    with colA:
        st.metric("Predicted Churn", "Yes" if pred==1 else "No")
        st.progress(proba)
        st.write(f"**Churn probability:** {proba:.1%}")
    with colB:
        st.info("**Recommended Actions:**")
        if tenure < 12:
            st.write("- New customer: offer onboarding discounts.")
        if monthly > 80:
            st.write("- High monthly charge: consider loyalty rewards.")
        if paperless == "yes":
            st.write("- Monitor paperless customers — slight churn trend.")
        if online_security == "no":
            st.write("- Offer security add-ons to improve satisfaction.")
        if sentiment < 0:
            st.write("- Negative review: flag for customer service follow-up.")
        if sentiment == 0 and feedback_length == 0:
            st.write("- No feedback available; keep monitoring satisfaction.")

# ----------------------------
# 7. Optional Analysis / EDA
# ----------------------------
st.header("📊 Analysis (EDA)")
uploaded = st.file_uploader("Upload a dataset CSV for quick analysis", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df.head())

    if "Churn" in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Churn", palette="coolwarm", ax=ax)
        st.pyplot(fig)

        df_enc = pd.get_dummies(df, drop_first=True)
        corr = df_enc.corr()["Churn"].sort_values(ascending=False)
        st.bar_chart(corr.head(10))
else:
    st.info("Upload a dataset to view analysis (optional).")

st.markdown("---")
st.caption("Developed by Abdullah Fahlo — Telco Churn ML Project 🚀")
