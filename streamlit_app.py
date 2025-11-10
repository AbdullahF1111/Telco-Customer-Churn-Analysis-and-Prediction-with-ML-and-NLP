import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# --------------------- App Setup ---------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="wide", page_icon="📊")
st.title("📞 Telco Customer Churn Prediction App")


st.markdown(
    """
    ### 🔗 [Full Project on GitHub](https://github.com/AbdullahF1111/Telco-Customer-Churn-Analysis-and-Prediction-with-ML-and-NLP)
    
    A comprehensive machine learning solution for predicting customer churn in the telecom industry.
    
    You can add a customer review for automatic sentiment scoring using NLP.
    """
)

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# --------------------- Sentiment Converter ---------------------
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    """Load HuggingFace sentiment analysis model (cached)."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

def transformer_sentiment_score(text: str) -> float:
    """
    Returns sentiment score in range [-1, 1] using transformer model.
    Positive sentiment -> +1, Negative -> -1
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    try:
        result = sentiment_model(text[:512])[0]  # limit input to 512 tokens
        label = result["label"]
        score = result["score"]
        return score if label == "POSITIVE" else -score
    except Exception:
        return 0.0


# --------------------- Load Artifacts ---------------------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feat_cols = json.loads(FEATURES_PATH.read_text())
        st.success("✅ Model loaded successfully.")
        return model, scaler, feat_cols
    except Exception as e:
        st.error(f"❌ Could not load model artifacts: {e}")
        return None, None, None

model, scaler, model_feat_cols = load_artifacts()

# --------------------- Preprocessing ---------------------
def preprocess_input(df_input: pd.DataFrame, feat_cols=None, scaler=None):
    df = df_input.copy()

    # clean strings
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # handle service aliases
    for col in ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']:
        if col in df.columns:
            df[col] = df[col].replace({"no internet service": "no"})
    if "MultipleLines" in df.columns:
        df["MultipleLines"] = df["MultipleLines"].replace({"no phone service": "no"})

    # HasInternet flag
    if "InternetService" in df.columns:
        df["HasInternet"] = df["InternetService"].apply(lambda v: 0 if str(v).lower() == "no" else 1)

    # yes/no mapping
    yesno_cols = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yesno_cols:
        if col in df.columns:
            df[col] = df[col].map({"yes": 1, "no": 0}).fillna(0).astype(int)

    # numerics
    for c in ["tenure","MonthlyCharges","TotalCharges","feedback_length","sentiment"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # one-hot
    multi_cols = [c for c in ["InternetService","Contract","PaymentMethod","gender"] if c in df.columns]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

    if feat_cols is not None:
        for c in feat_cols:
            if c not in df.columns:
                df[c] = 0
        df = df[feat_cols]

    # scale
    if scaler is not None:
        num_cols = [c for c in ["tenure","MonthlyCharges","TotalCharges","feedback_length","sentiment"] if c in df.columns]
        try:
            df[num_cols] = scaler.transform(df[num_cols])
        except Exception as e:
            st.warning(f"⚠️ Scaling failed: {e}")
    return df

# --------------------- Input UI (Dynamic) ---------------------
st.header("🧾 Customer Details")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input("Tenure (months)", 0, 100, 24)
    monthly = st.number_input("Monthly Charges", 0.0, 1000.0, 75.0)
    total = st.number_input("Total Charges (after discount or taxes)", 0.0, 10000.0, 1800.0,
                            help="💡 Total amount charged to date. May differ from tenure × monthly charges due to discounts, promotions, taxes, or one-time fees.")
with col2:
    internet = st.selectbox("Internet Service", ["fiber optic", "dsl", "no"], 2)
    contract = st.selectbox("Contract", ["month-to-month","one year","two year"], 0)
    paperless = st.selectbox("Paperless Billing", ["yes","no"], 0)
with col3:
    payment = st.selectbox("Payment Method", ["electronic check","mailed check","bank transfer (automatic)","credit card (automatic)"], 0)
    phone_service = st.selectbox("Phone Service", ["yes","no"], 1)
    online_security = st.selectbox("Online Security", ["yes","no"], 1)

# --- Conditional: Multiple Lines only if phone_service = yes ---
if phone_service == "yes":
    multiple_lines = st.selectbox("Multiple Lines", ["yes","no"], 0)
else:
    multiple_lines = "no phone service"

# --- Conditional: Internet-dependent services ---
if internet != "no":
    st.markdown("### 🌐 Internet & Streaming Services")
    c1, c2, c3 = st.columns(3)
    with c1:
        device_protection = st.selectbox("Device Protection", ["yes","no"], 1)
        tech_support = st.selectbox("Tech Support", ["yes","no"], 1)
    with c2:
        streaming_tv = st.selectbox("Streaming TV", ["yes","no"], 1)
        streaming_movies = st.selectbox("Streaming Movies", ["yes","no"], 1)
    with c3:
        online_backup = st.selectbox("Online Backup", ["yes","no"], 1)
else:
    device_protection = "no internet service"
    tech_support = "no internet service"
    streaming_tv = "no internet service"
    streaming_movies = "no internet service"
    online_backup = "no internet service"

with st.expander("👥 Additional Info"):
    partner = st.selectbox("Partner", ["yes","no"], 1)
    dependents = st.selectbox("Dependents", ["yes","no"], 1)
    senior = st.selectbox("Senior Citizen", [0, 1], 0)

review_text = st.text_area("📝 Customer Review (optional)")
predict_btn = st.button("🔮 Predict Churn" )

# --------------------- Prediction ---------------------
if predict_btn:
    sentiment = transformer_sentiment_score(review_text)
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
        "OnlineBackup": online_backup,
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

st.markdown("---")
st.caption("Developed by Abdullah Fahlo — Telco Churn ML + NLP Project")


# -------------------- EDA Section ---------------------
st.markdown("---")
st.header("📊 Exploratory Data Analysis & Insights")

st.markdown("""
###  Project Overview
This analysis explores customer churn patterns using machine learning and NLP techniques. 
The model identifies at-risk customers and provides actionable insights for retention strategies.
""")

# Display metrics and insights without relying on external image files
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Customer Churn Distribution")
    st.metric("Overall Churn Rate", "26%")
    st.progress(0.26)
    st.caption("Baseline churn rate indicates significant customer retention opportunity")
    
    # Create a simple bar chart for churn distribution
    churn_data = pd.DataFrame({
        'Churn': ['No Churn', 'Churn'],
        'Count': [5174, 1869]  # Approximate numbers based on 26% churn rate
    })
    st.bar_chart(churn_data.set_index('Churn'))

with col2:
    st.subheader("🎯 Model Performance")
    st.metric("Best Model AUC", "0.946")
    st.metric("Accuracy", "89.9%")
    st.caption("Random Forest - Excellent predictive performance")
    
    # Create ROC curve explanation
    st.info("""
    **ROC-AUC = 0.95** indicates outstanding model performance:
    - 95% true positive rate
    - Excellent churn prediction capability
    """)

# Feature Importance as a bar chart
st.subheader("🔍 Top Predictive Features")
feature_importance_data = pd.DataFrame({
    'Feature': ['sentiment', 'feedback_length', 'tenure', 'TotalCharges', 'MonthlyCharges', 
                'InternetService_fiber optic', 'PaymentMethod_electronic check', 
                'Contract_two year', 'Contract_one year', 'OnlineSecurity'],
    'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04]
})
st.bar_chart(feature_importance_data.set_index('Feature'))

st.caption("Sentiment analysis and tenure are the strongest churn predictors")

# Numerical Features Analysis
st.subheader("📊 Key Numerical Insights")

num_col1, num_col2, num_col3 = st.columns(3)

with num_col1:
    st.metric("Avg Tenure (Churn)", "18 months")
    st.metric("Avg Tenure (No Churn)", "38 months")
    st.caption("Shorter tenure = Higher risk")

with num_col2:
    st.metric("Avg Monthly (Churn)", "$82")
    st.metric("Avg Monthly (No Churn)", "$61") 
    st.caption("Higher charges = Higher risk")

with num_col3:
    st.metric("Negative Sentiment", "68% churn rate")
    st.metric("Positive Sentiment", "14% churn rate")
    st.caption("Sentiment strongly predicts churn")

# --- Business Insights & Model Performance ---
st.markdown("""
## 🎯 Business Insights & Strategic Findings

### 📈 Key Churn Drivers Identified
""")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("""
    #### High-Risk Segments:
    - Month-to-month contracts: **45% churn rate**
    - Electronic check users: **45% higher churn**
    - No Tech Support: **42% churn rate**
    - Fiber optic + no security: **53% churn rate**
    - High charges + low tenure: **48% churn rate**
    """)

with insight_col2:
    st.markdown("""
    #### Customer Behavior:
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

st.markdown("""
### 🧠 NLP Model for Customer Feedback Analysis

Customer reviews are processed using a transformer-based sentiment analysis model (DistilBERT).  
This model converts free-text feedback into a numerical sentiment score ranging from **-1 (negative)** to **+1 (positive)**,  
allowing integration of textual customer emotions directly into the churn prediction pipeline.
""")


# --- Strategic Recommendations ---
st.markdown("""
## 💼 Strategic Recommendations & Expected Impact

### 🎯 Priority Retention Initiatives
""")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.markdown("""
    #### Contract Optimization
    - Convert month-to-month to 1-year contracts
    - **Expected Impact:** 20% churn reduction
    - **Target:** 55% of customer base
    
    #### Payment Method Enhancement 
    - Auto-pay incentives & digital wallet integration
    - **Expected Impact:** 10-15% churn reduction
    - **Target:** Electronic check users (33% of churn)
    """)

with rec_col2:
    st.markdown("""
    #### Service Bundle Strategy
    - Security & support package promotions
    - **Expected Impact:** 8-12% churn reduction  
    - **Target:** Fiber optic & high-risk segments
    
    #### Proactive Monitoring
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
| **Implementation ROI** | - | 2.4x (Y1), 4.8x (Y2) | **Strong business case** |

---

""")
st.markdown("""
*Analysis based on 7,043 customer records using Random Forest classification with 94.6% AUC performance.*
""")
