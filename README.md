# 📞 Telco Customer Churn Prediction with ML & NLP

**A full-stack churn prediction solution combining EDA, ML, and NLP — deployed via Streamlit for real-time business use.**

### 🎯 Project Overview
An **end-to-end data science project** to predict **customer churn** in a telecom company using **Machine Learning, NLP sentiment analysis, and Streamlit** for interactive predictions.  
The project combines **structured data analysis**, **customer feedback sentiment scoring**, and **business insights** to help reduce churn rates and improve retention strategies.

---

### Key Objectives
- Predict whether a telecom customer will **churn or stay** using ensemble methods
- Extract customer sentiment from written feedback using Transformer-based NLP (DistilBERT) for contextual understanding.
- Provide **data-driven recommendations** to reduce churn by 35-45%
- Deploy an interactive **Streamlit app** for real-time predictions

---

### Tech Stack
| Category | Tools & Libraries |
|-----------|-------------------|
| **Data Analysis** | Pandas, NumPy, Seaborn, Matplotlib |
| **Machine Learning** | Scikit-learn, Random Forest, XGBoost, Logistic Regression |
| **NLP / Sentiment Analysis** | DistilBERT Transformer (Hugging Face), NLP Pipeline |
| **Model Deployment** | Streamlit, Joblib |
| **Version Control** | GitHub |
| **Environment** | Python 3.8+ |

---

### 📂 Project Structure
```
telco-customer-churn-analysis-and-prediction-with-ml-and-nlp/
│
├── artifacts/                 # Trained model & preprocessing
│   ├── model.pkl             # Random Forest classifier
│   ├── scaler.pkl            # StandardScaler for numeric features
│   └── feature_columns.json  # Expected feature columns
│
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
└── (Notebooks & raw data in separate analysis environment)
```

---

### 📊 Key Business Insights

| Insight | Impact | Recommendation |
|----------|---------|----------------|
| **26.5% overall churn rate** | Significant revenue loss | Implement targeted retention programs |
| **Month-to-month contracts: 45% churn** | Highest risk segment | Convert to 1-year contracts with incentives |
| **Electronic check users: 45% higher churn** | Payment method risk | Promote auto-pay with discounts |
| **Negative sentiment: 68% churn rate** | Strong predictor | Proactive sentiment monitoring |
| **High charges + low tenure: 48% churn** | Value perception issue | Loyalty discounts for at-risk customers |

---

### 🧮 Model Performance

**Random Forest** achieved the best balanced performance:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 0.899 | 0.877 | 0.722 | 0.792 | **0.946** |
| Logistic Regression | 0.895 | 0.834 | 0.754 | 0.792 | 0.945 |
| XGBoost | 0.898 | 0.836 | 0.765 | 0.799 | 0.941 |

✅ **Production Model:** Random Forest  
- Excellent AUC (0.946) for churn prediction  
- Handles mixed data types effectively  
- Provides feature importance for business interpretation

---

### 🔍 Feature Importance (Top 10)

1. **sentiment** - Customer feedback sentiment score
2. **feedback_length** - Length of customer review
3. **tenure** - Months with the company
4. **TotalCharges** - Total amount charged
5. **MonthlyCharges** - Monthly service cost
6. **InternetService_fiber optic** - Fiber optic service users
7. **PaymentMethod_electronic check** - Payment method
8. **Contract_two year** - Two-year contract
9. **Contract_one year** - One-year contract  
10. **OnlineSecurity** - Security service subscription

---

### 💬 Streamlit App Features

**Live Demo:** [https://telco-customer-churn-analysis-and-prediction-with-ml-and-nlp-w.streamlit.app/#customer-details]

The app provides:
-  **Interactive customer input form** with conditional fields
-  **Real-time churn predictions** with probability scores
-  **Automatic sentiment analysis** of customer reviews
-  **Actionable recommendations** based on risk factors
-  **EDA insights** and model performance metrics

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

### 🎯 Business Impact & ROI

**Expected Outcomes:**
- **35-45% reduction** in churn rate (from 26.5% to 16.5-18.5%)
- **$2.1-2.7M annual revenue** protected from churn
- **520-860 customers saved** annually through targeted interventions
- **2.4x ROI in Year 1**, 4.8x in Year 2

**Priority Initiatives:**
1. **Contract conversion program** (20% churn reduction)
2. **Payment method optimization** (10-15% reduction)  
3. **Service bundle promotions** (8-12% reduction)
4. **Sentiment monitoring & intervention** (5-8% reduction)

---

### 🌍 Future Enhancements

- **Real-time CRM integration** for automated predictions
- **SHAP explainability** for feature-level insights
- **A/B testing framework** for retention offers
- **Cloud deployment** on AWS/Azure with CI/CD pipeline

---

### 👨‍💻 About the Author

**Abdullah Fahlo**  
🎓 B.Sc. in Information Engineering — University of Aleppo (2025)  
💡 Data Analyst | ML & AI Enthusiast | NLP Learner  
📬 [abdullahfahlo.com@gmail.com](mailto:abdullahfahlo.com@gmail.com)  
🌍 [LinkedIn]((https://www.linkedin.com/in/abdullah-fahlo-77b28a29b))  
💼 [Portfolio Projects]([https://github.com/abdullahfahlo](https://github.com/AbdullahF1111))

---

### 🧾 License

This project is licensed under the **MIT License** — you're free to use and adapt it for educational and commercial purposes.

---

*Last updated: November 2025*

---

[*Dataset Source(Beata Faron)*](https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback)

