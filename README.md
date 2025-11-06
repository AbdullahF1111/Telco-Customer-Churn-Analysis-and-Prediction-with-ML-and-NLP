# 📞 Telco Customer Churn Prediction with ML & NLP

### 🎯 Project Overview
An **end-to-end data science project** to predict **customer churn** in a telecom company using **Machine Learning, NLP sentiment analysis, and Streamlit** for interactive predictions.  
The project combines **structured data analysis**, **customer feedback sentiment scoring**, and **business insights** to help reduce churn rates and improve retention strategies.

---

### 🚀 Key Objectives
- Predict whether a telecom customer will **churn or stay**.  
- Extract **customer sentiment** from written feedback using NLP.  
- Provide **business-driven recommendations** to reduce churn.  
- Deploy an interactive **Streamlit app** for real-time predictions.  

---

### 🧩 Tech Stack
| Category | Tools & Libraries |
|-----------|-------------------|
| **Data Analysis** | Pandas, NumPy, Seaborn, Matplotlib |
| **Machine Learning** | Scikit-learn, XGBoost, Random Forest |
| **NLP / Sentiment Analysis** | TextBlob / Vader / OpenAI API (optional) |
| **Model Deployment** | Streamlit |
| **Version Control** | GitHub |
| **Environment** | Google Colab / Jupyter Notebook |

---

### 📂 Project Structure
```

Telco-Customer-Churn-Prediction-with-ML-and-NLP/
│
├── data/
│   ├── telco_prep.csv
│   ├── telco_noisy_feedback_prep.csv
│
├── notebooks/
│   ├── 01_EDA_and_Insights.ipynb
│   ├── 02_Preprocessing_and_Modeling.ipynb
│
├── app/
│   ├── streamlit_app.py
│
├── models/
│   ├── best_model.pkl
│
├── README.md
└── requirements.txt

```

---

### 📊 Exploratory Data Analysis (EDA)
Key findings from the analysis:

| Insight | Business Impact |
|----------|------------------|
| Customers with **month-to-month contracts** churn more frequently. | Encourage customers to switch to **1-year or 2-year contracts** with incentives. |
| **Fiber optic users** show higher churn compared to DSL users. | Investigate network stability and service satisfaction for fiber customers. |
| **No OnlineSecurity or TechSupport** → higher churn rates. | Offer **free trials or discounts** for these add-ons. |
| Customers with **high MonthlyCharges** churn more. | Introduce **loyalty discounts or flexible plans**. |
| **Low tenure (<12 months)** has high churn. | Improve **onboarding experience and support** for new users. |

---

### 🧮 Modeling
Three ML models were compared:

| Model | Accuracy | Recall | ROC-AUC |
|--------|-----------|--------|----------|
| Random Forest | 0.898 | 0.72 | **0.947** |
| Logistic Regression | 0.896 | 0.76 | 0.945 |
| XGBoost | 0.893 | 0.74 | 0.941 |

✅ **Best Model:** Random Forest  
- Balanced between recall and accuracy.  
- Great for handling both numerical and categorical data.  
- Deployed in Streamlit app.

---

### 📈 Feature Importance (Top 10)
```

1. Contract type
2. Tenure
3. TechSupport
4. InternetService
5. OnlineSecurity
6. MonthlyCharges
7. PaymentMethod
8. StreamingTV
9. Dependents
10. PaperlessBilling

````

---

### 💬 Streamlit App
A simple, user-friendly web app where:
- You can **input customer details**.
- Optionally enter **customer feedback** → automatically converted to **sentiment score**.
- Get a **churn prediction (Yes/No)** with probability.

🔗 Try it locally:
```bash
streamlit run app/streamlit_app.py
````

---

### 🧠 Business Recommendations

✅ Focus on **customer retention** by improving:

1. Contract renewal incentives.
2. Technical support satisfaction.
3. Service reliability (especially fiber users).
4. Targeted retention campaigns for high-risk users.
5. AI-based feedback sentiment tracking.

---

### 🌍 Future Improvements

* Integrate **real customer reviews** from CRM.
* Automate retraining with new data (MLOps).
* Add **SHAP explainability** for feature-level insights.
* Cloud deploy on **Azure or Streamlit Cloud**.

---

### 👨‍💻 About the Author

**Abdullah Fahlo**
🎓 B.Sc. in Information Engineering — University of Aleppo (2025)
💡 Data Analyst | ML & AI Enthusiast | NLP Learner
📬 [abdullahfahlo.com@gmail.com](mailto:abdullahfahlo.com@gmail.com)
🌍 [LinkedIn Profile](#) | [Portfolio Projects](#)

---

### 🧾 License

This project is licensed under the **MIT License** — you’re free to use and adapt it.

---

```

---

## 🧠 Tips for Maximum Portfolio Impact
✅ Add **graphs as images** in your README (like churn distribution or feature importance).  
✅ Include a short **GIF or screenshot** of your Streamlit app interface.  
✅ Write a short **“Why this project matters”** section — 2 lines only.  
✅ Keep the README visually balanced: icons 🎯📊 make it readable but not childish.  

---

Would you like me to help you **generate the README visuals (graphs + app mockup layout)** next, so your GitHub repo looks professional and complete?
```
