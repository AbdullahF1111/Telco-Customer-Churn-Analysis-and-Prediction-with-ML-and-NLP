"""
Telco Customer Churn Prediction Streamlit App
- Clean & Professional UI
- Dynamic inputs (hide multiple lines when no phone service)
- Optional "Additional Info" expander for extra fields
- Uses trained Random Forest model and scaler from /artifacts
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------------------- App Setup ---------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📊")
st.title("📈 Telco Customer Churn Prediction App")
st.markdown(
    """
Enter key customer details below — optional fields can be expanded.  
You can also add a customer review for automatic sentiment scoring.
"""
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# --------------------- Sentiment Converter ---------------------
POS_WORDS = {"good","great","excellent","satisfied","happy","love","recommend","reliable","best","positive","pleased","fast"}
NEG_WORDS = {"bad","terrible","awful","unhappy","angry","hate","disappointed","slow","worst","problem","issue","complain","expensive"}

def simple_sentiment_score(text: str) -> float:
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    txt = text.lower()
    words = [w.strip(".,!?()[]\"'") for w in txt.split()]
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return float(np.clip((pos - neg) / max(1, (pos + neg)), -1, 1))

# --------------------- Load Artifacts ---------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feat_cols = json.loads(FEATURES_PATH.read_text())
        st.success("✅ Loaded model, scaler, and feature columns successfully.")
        return model, scaler, feat_cols
    except Exception as e:
        st.error(f"❌ Could not load model artifacts: {e}")
        return None, None, None

model, scaler, model_feat_cols = load_artifacts()

# --------------------- Preprocessing ---------------------
def preprocess_input(df_input: pd.DataFrame, feat_cols=None, scaler=None):
    df = df_input.copy()

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    internet_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].replace({"no internet service": "no"})

    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"no phone service": "no"})

    if "InternetService" in df.columns:
        df["HasInternet"] = df["InternetService"].apply(lambda v: 0 if str(v).lower() == "no" else 1)

    yesno_cols = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yesno_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

    for c in ["tenure","MonthlyCharges","TotalCharges","feedback_length","sentiment"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    multi_cols = [c for c in ["InternetService","Contract","PaymentMethod","gender"] if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    if feat_cols is not None:
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[feat_cols]
    else:
        df = df.select_dtypes(include=np.number)

    if scaler is not None:
        num_cols = [c for c in ["tenure","MonthlyCharges","TotalCharges","feedback_length","sentiment"] if c in df.columns]
        try:
            df[num_cols] = scaler.transform(df[num_cols])
        except Exception as e:
            st.warning(f"Scaler transform failed: {e}")

    return df

# --------------------- Input Form ---------------------
st.header("🧾 Customer Input")

with st.form("customer_input"):
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.number_input("Tenure (months)", 0, 100, 24)
        monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 75.0)
        total = st.number_input("Total Charges", 0.0, 10000.0, 1800.0)
    with c2:
        internet = st.selectbox("Internet Service", ["fiber optic","dsl","no"], 1)
        contract = st.selectbox("Contract", ["month-to-month","one year","two year"], 0)
        paperless = st.selectbox("Paperless Billing", ["yes","no"], 0)
    with c3:
        payment = st.selectbox("Payment Method", ["electronic check","mailed check","bank transfer (automatic)","credit card (automatic)"], 0)
        phone_service = st.selectbox("Phone Service", ["yes","no"], 0)
        online_security = st.selectbox("Online Security", ["yes","no","no internet service"], 1)

    # --- Conditional: Hide MultipleLines if no phone service ---
    if phone_service == "yes":
        multiple_lines = st.selectbox("Multiple Lines", ["no","yes","no phone service"], 0)
    else:
        multiple_lines = "no phone service"

    # --- Optional Fields in Expander ---
    with st.expander("📂 Enter Additional Info (optional)"):
        device_protection = st.selectbox("Device Protection", ["yes","no","no internet service"], 1)
        streaming_tv = st.selectbox("Streaming TV", ["yes","no","no internet service"], 1)
        streaming_movies = st.selectbox("Streaming Movies", ["yes","no","no internet service"], 1)
        tech_support = st.selectbox("Tech Support", ["yes","no","no internet service"], 1)
        partner = st.selectbox("Partner", ["yes","no"], 1)
        dependents = st.selectbox("Dependents", ["yes","no"], 1)
        senior = st.selectbox("Senior Citizen", [0, 1], 0)

    review_text = st.text_area("📝 Customer Review (optional)")
    submitted = st.form_submit_button("🔮 Predict Churn")

# --------------------- Prediction ---------------------
if submitted:
    sentiment = simple_sentiment_score(review_text)
    feedback_len = len(review_text.split()) if review_text else 0

    row = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Partner": partner,
        "Dependents": dependents,
        "InternetService": internet,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security,
        "TechSupport": tech_support,
        "DeviceProtection": device_protection,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "PaymentMethod": payment,
        "feedback_length": feedback_len,
        "sentiment": sentiment,
        "gender": "male"
    }

    df_input = pd.DataFrame([row])
    X_input = preprocess_input(df_input, feat_cols=model_feat_cols, scaler=scaler)
    X_input = X_input.astype(float).fillna(0)

    try:
        proba = float(model.predict_proba(X_input)[0, 1])
        pred = int(proba >= 0.5)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("🧠 Prediction Result")
    st.metric("Churn Prediction (1 = Yes)", pred)
    st.progress(proba)
    st.write(f"**Churn Probability:** {proba:.2%}")

    st.markdown("#### 📊 Recommended Actions")
    tips = []
    if tenure < 12: tips.append("🟡 New customer — offer onboarding incentives.")
    if monthly > 80: tips.append("🔴 High monthly charge — suggest loyalty discount.")
    if online_security == "no": tips.append("🟠 Offer Online Security package.")
    if sentiment < 0: tips.append("🔴 Negative sentiment — follow up with support.")
    if not tips: tips.append("🟢 Customer appears stable — continue monitoring.")
    for t in tips:
        st.write("- " + t)

# -------------------- EDA Section ---------------------
st.markdown("---")
st.header("📊 Exploratory Data Analysis & Insights")

st.markdown("""
### 🧩 Overview
The analysis explored relationships between customer demographics, contract types, service usage, and churn behavior.  
The following visuals summarize **feature importance**, **distribution patterns**, and **service correlations**.

Below are the highlights of the analysis and model performance.
""")

# --- Display Uploaded Figures ---
eda_imgs = {
    "Feature Importance": "8fb85dc9-32bf-417c-b665-47c56d95e396.png",
    "Numerical vs Churn": "e45ce6c2-2d3d-4206-a72d-d38dec9b673a.png",
    "Categorical vs Churn (Set 1)": "9cd83850-3bbf-4bcf-950b-22208fa704f2.png",
    "Numerical vs Churn (Alt)": "a71e5bb7-7ff0-472e-9465-7c917c9e5949.png",
    "Categorical vs Churn (Set 2)": "66b85777-7c13-4bf1-afa5-92393b355cc8.png",
}

for title, img_id in eda_imgs.items():
    st.subheader(f"📌 {title}")
    st.image(f"/mnt/data/{img_id}", use_container_width=True)

# --- Markdown Summary ---
st.markdown("""
### 🧠 Key Findings

- **Churn rate:** ~26%, showing a notable loss of customers.
- **High churn** among customers with:
  - **Month-to-month contracts**
  - **Electronic check payments**
  - **No tech support or online security**
  - **Fiber optic internet**
- **Higher monthly charges** and **shorter tenure** → greater churn risk.
- **Sentiment analysis** shows that negative feedback correlates strongly with churn.

---

### ⚙️ Model Comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|:------|:----------|:-----------|:--------|:----------|:---------|
| **Random Forest** | 0.899 | 0.877 | 0.722 | 0.792 | 0.946 |
| **Logistic Regression** | 0.895 | 0.834 | 0.754 | 0.792 | 0.945 |
| **XGBoost** | 0.898 | 0.836 | 0.765 | 0.799 | 0.941 |

✅ **Random Forest** performed best overall, balancing interpretability and accuracy.

---

### 💼 Expected Business Impact

- **Early churn prediction → targeted retention strategy**
- **Improved marketing ROI → focused engagement on at-risk customers**
- **Estimated churn reduction:** ~15–25% with proactive offers & follow-up actions

---

### 📊 Business Insights & Recommendations

- Focus retention campaigns on **month-to-month** customers.
- Offer discounts or loyalty programs to **high-charge customers**.
- Incentivize users to switch from **electronic check to auto-pay**.
- Promote **Online Security** and **Tech Support** services.
- Monitor **negative sentiment reviews** for early churn warning signals.

---

📘 *This section summarizes the analytical findings and model evaluation from the Telco Customer Churn project.*
""")
