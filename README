# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or rejected, based on applicant details such as income, credit history, education, and property area.

---

## Problem Statement

Manual loan approval is time-consuming and inconsistent. This project builds a classification model that automates the decision — given an applicant's profile, predict whether their loan will be approved (Y) or rejected (N).

---

## Dataset

- **Source:** Analytics Vidhya Loan Prediction dataset
- **Size:** 614 rows, 13 features
- **Target:** `Loan_Status` (Y = Approved, N = Rejected)
- **Class Distribution:** 422 Approved (68.7%) / 192 Rejected (31.3%) — moderately imbalanced

| Feature           | Type                | Description                          |
| ----------------- | ------------------- | ------------------------------------ |
| Loan_ID           | Identifier          | Unique loan ID                       |
| Gender            | Binary Categorical  | Male / Female                        |
| Married           | Binary Categorical  | Yes / No                             |
| Dependents        | Ordinal Categorical | 0 / 1 / 2 / 3+                       |
| Education         | Binary Categorical  | Graduate / Not Graduate              |
| Self_Employed     | Binary Categorical  | Yes / No                             |
| ApplicantIncome   | Numerical           | Applicant's monthly income           |
| CoapplicantIncome | Numerical           | Co-applicant's monthly income        |
| LoanAmount        | Numerical           | Loan amount requested (in thousands) |
| Loan_Amount_Term  | Numerical           | Term of loan in months               |
| Credit_History    | Binary              | 1 = good history, 0 = bad history    |
| Property_Area     | Nominal Categorical | Urban / Semiurban / Rural            |
| Loan_Status       | Target              | Y = Approved, N = Rejected           |

---

## Project Structure

```
Loan-Approval-Prediction/
│
├── data/
│   ├── loan_data_set.csv       ← original raw data (never modified)
│   ├── df_clean.csv            ← cleaned data after preprocessing
│   ├── x_train.csv             ← processed training features
│   ├── x_test.csv              ← processed test features
│   ├── y_train.csv             ← training labels
│   └── y_test.csv              ← test labels
│
├── models/
│   ├── loan_model.pkl          ← trained Logistic Regression model
│   └── scaler.pkl              ← fitted RobustScaler
│
├── notebooks/
│   ├── 01_data_audit_eda.ipynb              ← data audit and exploratory analysis
│   ├── 02_data_cleaning_preprocessing.ipynb ← missing value handling
│   ├── 03_feature_engineering.ipynb         ← encoding, scaling, train/test split
│   └── 04_model_building.ipynb              ← model training and evaluation
│
├── app.py                      ← Flask app for deployment
└── README.md
```

---

## Workflow

### 1. Data Audit & EDA

- Identified missing values across 7 columns
- Classified features into binary, ordinal, nominal and numerical types
- Key EDA findings:
  - **Credit_History** is the strongest predictor — applicants with good credit history have ~80% approval rate vs ~8% for those without
  - **Property_Area** has moderate signal — Semiurban has the highest approval rate (~76.8%)
  - **ApplicantIncome** and **LoanAmount** are weak predictors on their own

### 2. Data Cleaning

- Categorical columns (Gender, Married, Dependents, Self_Employed) → imputed with **mode**
- Numerical columns (LoanAmount, Loan_Amount_Term) → imputed with **median**
- Credit_History → imputed with mode + created a `Credit_History_Missing` binary indicator column to preserve the missingness signal

### 3. Feature Engineering

- Dropped `Loan_ID` (identifier, not predictive)
- Stratified train/test split (80/20) on `Loan_Status` to preserve class balance
- Label encoding for binary categorical columns (Gender, Married, Education, Self_Employed)
- Ordinal encoding for Dependents (0/1/2/3)
- One-hot encoding for Property_Area (drop_first=True — Rural as reference)
- RobustScaler applied to numerical columns (chosen over StandardScaler due to outliers in income and loan amount)
- Target encoded: Y → 1, N → 0

### 4. Model Building

Three models were trained and compared:

| Model               | Train Accuracy | Test Accuracy | Notes                                           |
| ------------------- | -------------- | ------------- | ----------------------------------------------- |
| Logistic Regression | —              | **86%**       | Best overall performance                        |
| Decision Tree       | 100%           | 77.24%        | Severe overfitting                              |
| Random Forest       | High           | < 86%         | Reduced overfitting vs tree, but didn't beat LR |

- Logistic Regression hyperparameter tuning (varying `C`) showed stable performance for C ≥ 0.1
- 5-fold cross-validation: mean accuracy **79.84%**, std **1.63%** — consistent and stable
- GridSearchCV confirmed Logistic Regression as the best configuration

### Final Model: Logistic Regression

**Reasons for selection:**

- Highest test accuracy among all evaluated models
- Strong recall on the approval class (99%) — rarely misses a genuine approval
- Stable across cross-validation folds
- Interpretable coefficients aligned with EDA findings (Credit_History has highest weight)
- Simpler and more reliable than ensemble methods for this dataset size

**Known weakness:** Low recall on rejections (58%) — 16 out of 38 actual rejections were incorrectly approved. This is a known limitation of the class imbalance in the dataset.

---

## Results

```
              precision    recall  f1-score   support

           0       0.96      0.58      0.72        38
           1       0.84      0.99      0.91        85

    accuracy                           0.86       123
   macro avg       0.90      0.78      0.81       123
weighted avg       0.88      0.86      0.85       123
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/Tanmoy-Das-2002/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction

# Install dependencies
pip install pandas numpy scikit-learn flask joblib

# Run the app
python app.py
```

---

## Tech Stack

- **Python 3.13**
- **pandas, numpy** — data manipulation
- **scikit-learn** — preprocessing, modeling, evaluation
- **joblib** — model serialization
- **Flask** — web app for deployment
- **matplotlib, seaborn** — visualization
