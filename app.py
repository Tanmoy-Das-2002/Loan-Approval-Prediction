import streamlit as st
import joblib
import pandas as pd


model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Loan Prediction App")

st.title("Loan Approval Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Amount Term", min_value=0)

credit_history = st.selectbox("Credit History", [1, 0])

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

if st.button("Predict"):

    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    semiurban = 1 if property_area == "Semiurban" else 0
    urban = 1 if property_area == "Urban" else 0

    data = pd.DataFrame([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        0,
        semiurban,
        urban
    ]], columns=[
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Credit_History_Missing',
        'Property_Area_Semiurban',
        'Property_Area_Urban'
    ])

    cols_to_scale = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term'
    ]

    data[cols_to_scale] = scaler.transform(
        data[cols_to_scale]
    )

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    
    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
    
    st.write(f"Approval Probability: {probability:.2%}")