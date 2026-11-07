# 💳 Credit Risk Prediction Model

An end-to-end credit risk model that predicts the probability of loan default, deployed as a Streamlit app for loan officers.

🔗 **Live app:** https://credit-prediction-01.streamlit.app/

> **Note:** This is a learning project built from the Codebasics ML course problem statement (a simulated NBFC client). The dataset is synthetic, not real customer data.

---

## 📌 Business Problem

A lender needs to identify risky borrowers before approving a loan, using a model the credit team can interpret. The goal is to predict whether a customer will default, using demographics, loan details and credit bureau history.

## 📊 Dataset

50,000 customers across 3 tables, merged on `cust_id`:

| Table | Content |
|---|---|
| `customers.csv` | Age, income, employment, residence, city, dependants |
| `loans.csv` | Loan purpose and type, amount, tenure, fees, outstanding principal |
| `bureau_data.csv` | Open and closed accounts, delinquent months, days past due, enquiries, credit utilization |

**Target:** `default` (1 = defaulted). Only about 8.6% of loans defaulted (4,297 of 50,000), so the data is heavily imbalanced.

## 🔧 Approach

1. **Data cleaning**
   - Removed processing-fee outliers (fees above 3% of loan amount).
   - Fixed an inconsistent label ("Personaal" → "Personal").
   - Imputed missing `residence_type` with the training-set mode, after the train/test split, to avoid leakage.
2. **Feature engineering**
   - `loan_to_income`
   - `delinquency_ratio` (delinquent months ÷ total loan months)
   - `avg_dpd_per_delinquency`
3. **Feature selection**
   - Dropped highly collinear features using VIF.
   - Kept features with Information Value (IV) above 0.02, which took 21 features down to 10.
   - The strongest by IV were credit utilization (2.35), delinquency ratio (0.72) and loan-to-income (0.48).
4. **Class imbalance:** Compared no balancing, random undersampling and SMOTE-Tomek.
5. **Modelling:** Logistic Regression and XGBoost, tuned with RandomizedSearchCV and Optuna.
6. **Evaluation:** Precision/recall, ROC-AUC, Gini, and rank ordering with KS by decile.

## 🏆 Results

Evaluated on 12,497 unseen loans, using the final tuned Logistic Regression with SMOTE-Tomek:

| Metric | Value |
|---|---|
| AUC | 0.98 |
| Gini | 0.97 |
| KS statistic | 85.9 |
| Recall (defaulters) | 94% |
| Precision (defaulters) | 56% |
| Accuracy | 93% |

The top 2 risk deciles capture 98.5% of all actual defaulters.

### Model comparison (defaulter class)

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression, no balancing | 0.72 | 0.85 | 0.78 |
| XGBoost, no balancing | 0.76 | 0.82 | 0.79 |
| Logistic Regression + Undersampling | 0.51 | 0.96 | 0.67 |
| XGBoost + Undersampling (tuned) | 0.57 | 0.97 | 0.72 |
| XGBoost + SMOTE-Tomek + Optuna | 0.68 | 0.89 | 0.77 |
| **Logistic Regression + SMOTE-Tomek + Optuna (final)** | **0.56** | **0.94** | **0.70** |

**Why Logistic Regression?** It gives high recall on defaulters, and its coefficients are directly interpretable, which matters in lending. In credit risk, missing a defaulter usually costs more than rejecting a good applicant, so I prioritised recall.

## 🖥️ Streamlit App

A loan officer enters borrower details (age, loan purpose, loan type, tenure, credit utilization, delinquency history and so on). The app returns the predicted default probability.

Run it locally:

```bash
git clone https://github.com/Faishal-sameer/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```
├── artifacts/
│   └── model_data.joblib      # model, scaler, feature list
├── Credit_Predicition.ipynb   # EDA, modelling, evaluation
├── app.py                     # Streamlit app
├── requirements.txt
└── README.md
```

## 🧰 Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · imbalanced-learn · Optuna · Statsmodels · Matplotlib · Seaborn · Streamlit

## ⚠️ Limitations and Future Work

- The data is synthetic, so the very high AUC likely overstates real-world performance.
- Precision on defaulters is moderate (56%), a trade-off made to maximise recall.
- Add a Poor / Average / Good / Excellent credit scorecard.
- Add model monitoring, such as drift and performance tracking.
- Add SHAP explanations to the app.

## 👤 Author

**Faishal Sameer**
[LinkedIn](https://linkedin.com/in/faishal-sameer-b13414244) · [GitHub](https://github.com/Faishal-sameer) · faishalsameer05@gmail.com
