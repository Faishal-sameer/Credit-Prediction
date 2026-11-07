# Credit-Prediction


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
| KS
