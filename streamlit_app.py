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
### 🧩 Project Overview
This analysis explores customer churn patterns using machine learning and NLP techniques. 
The model identifies at-risk customers and provides actionable insights for retention strategies.
""")

# Display the actual charts from your analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Customer Churn Distribution")
    st.image("churn_distribution.png", use_container_width=True)
    st.caption("Baseline churn rate: 26% - indicating significant customer retention opportunity")

with col2:
    st.subheader("🎯 ROC Curve - Random Forest")
    st.image("ROC_curve.png", use_container_width=True)
    st.caption("AUC = 0.95 - Excellent predictive performance")

# Feature Importance
st.subheader("🔍 Top Predictive Features")
st.image("feature_corr.png", use_container_width=True)
st.caption("Sentiment analysis and tenure are the strongest churn predictors")

# Numerical Features Analysis
st.subheader("📊 Numerical Features vs Churn")
st.image("num_features.png", use_container_width=True)
st.caption("Key patterns: Higher monthly charges + shorter tenure = increased churn risk")

# --- Business Insights & Model Performance ---
st.markdown("""
## 🎯 Business Insights & Strategic Findings

### 📈 Key Churn Drivers Identified
""")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("""
    **🚨 High-Risk Segments:**
    - Month-to-month contracts: **45% churn rate**
    - Electronic check users: **45% higher churn**
    - No Tech Support: **42% churn rate**
    - Fiber optic + no security: **53% churn rate**
    - High charges + low tenure: **48% churn rate**
    """)

with insight_col2:
    st.markdown("""
    **📊 Customer Behavior:**
    - Shorter tenure (<12 months): High vulnerability
    - Monthly charges >$75: Price sensitivity
    - Negative sentiment: Strong churn correlation
    - Paperless billing: Slight risk increase
    """)

# Model Performance Table
st.subheader("⚙️ Model Performance Comparison")

performance_data = {
    'Model': ['Random Forest', 'Logistic Regression', 'XGBoost'],
    'Accuracy': [0.899, 0.895, 0.898],
    'Precision': [0.877, 0.834, 0.836],
    'Recall': [0.722, 0.754, 0.765],
    'F1-Score': [0.792, 0.792, 0.799],
    'ROC-AUC': [0.946, 0.945, 0.941]
}

performance_df = pd.DataFrame(performance_data)
st.dataframe(performance_df.style.format({
    'Accuracy': '{:.3f}',
    'Precision': '{:.3f}', 
    'Recall': '{:.3f}',
    'F1-Score': '{:.3f}',
    'ROC-AUC': '{:.3f}'
}).background_gradient(subset=['ROC-AUC'], cmap='Blues'), use_container_width=True)

st.info("✅ **Random Forest** selected as production model for optimal balance of accuracy and interpretability")

# --- Strategic Recommendations ---
st.markdown("""
## 💼 Strategic Recommendations & Expected Impact

### 🎯 Priority Retention Initiatives
""")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("""
    **🎫 Contract Optimization**
    - Convert month-to-month to 1-year contracts
    - **Expected Impact:** 20% churn reduction
    - **Target:** 55% of customer base
    
    **💳 Payment Method Enhancement** 
    - Auto-pay incentives & digital wallet integration
    - **Expected Impact:** 10-15% churn reduction
    - **Target:** Electronic check users (33% of churn)
    """)

with rec_col2:
    st.markdown("""
    **🛡️ Service Bundle Strategy**
    - Security & support package promotions
    - **Expected Impact:** 8-12% churn reduction  
    - **Target:** Fiber optic & high-risk segments
    
    **📝 Proactive Monitoring**
    - NLP sentiment analysis + early intervention
    - **Expected Impact:** 5-8% churn reduction
    - **Target:** Negative feedback customers
    """)

# Financial Impact
st.markdown("""
### 📊 Expected Business Impact

| Metric | Current | Expected Improvement | Impact |
|--------|---------|---------------------|---------|
| **Churn Rate** | 26.5% | 16.5-18.5% | **35-45% reduction** |
| **Customers Saved** | - | 520-860 annually | **Revenue protection** |
| **Annual Revenue** | $6.2M at risk | $2.1-2.7M protected | **34-44% improvement** |
| **Implementation ROI** | - | 2.4x (Y1), 4.8x (Y2) | **Strong business case** |

---

### 🚀 Immediate Next Steps

1. **Launch contract conversion pilot** (Month 1-2)
2. **Implement payment optimization** (Month 1-3) 
3. **Deploy sentiment monitoring** (Month 2-4)
4. **Scale successful initiatives** (Month 5-12)

**🎯 Recommendation:** Proceed with phased retention program targeting highest-impact segments first.
""")

# Optional: Upload for custom analysis
st.markdown("---")
st.subheader("🔍 Upload Your Data for Custom Analysis")

uploaded_file = st.file_uploader("Upload customer data CSV for personalized insights", type=["csv"])
if uploaded_file is not None:
    try:
        custom_df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(custom_df)} records")
        
        # Quick analysis
        if 'Churn' in custom_df.columns:
            custom_churn_rate = custom_df['Churn'].mean() * 100
            st.metric("Your Dataset Churn Rate", f"{custom_churn_rate:.1f}%")
            
            if custom_churn_rate > 26:
                st.warning("⚠️ Higher than benchmark churn rate - urgent action recommended")
            else:
                st.success("✅ Below benchmark churn rate - maintain current strategies")
    except Exception as e:
        st.error(f"Error analyzing uploaded file: {e}")

st.markdown("""
---
*Analysis based on 7,043 customer records using Random Forest classification with 94.6% AUC performance.*
""")
