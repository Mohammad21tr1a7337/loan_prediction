import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load trained model
@st.cache_resource  # Updated from st.cache (deprecated for Streamlit >=1.18)
def load_model():
    try:
        return joblib.load("loan_model.pkl")
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'loan_model.pkl' exists.")
        st.stop()

model = load_model()

# Set up Streamlit page
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    layout="centered"
)

# Add background image
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1605902711622-cfb43c4437d1");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

# Title and form description
st.title("🏠 Dream Housing Loan Eligibility Prediction")
st.markdown("### Fill the form to check loan approval status")

# User input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=1)
credit_history = st.selectbox("Credit History", ["Meets guidelines", "Does not meet guidelines"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Input mapping
input_data = {
    "Gender": 1 if gender == "Male" else 0,
    "Married": 1 if married == "Yes" else 0,
    "Dependents": 3 if dependents == "3+" else int(dependents),
    "Education": 0 if education == "Graduate" else 1,
    "Self_Employed": 1 if self_employed == "Yes" else 0,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": 1 if credit_history == "Meets guidelines" else 0,
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
}

# Prediction logic
if st.button("Predict Loan Status"):
    input_df = pd.DataFrame([input_data])

    # Feature engineering
    input_df["TotalIncome"] = input_df["ApplicantIncome"] + input_df["CoapplicantIncome"]
    input_df["EMI"] = input_df["LoanAmount"] / input_df["Loan_Amount_Term"]
    input_df["Balance_Income"] = input_df["TotalIncome"] - (input_df["EMI"] * 1000)
    input_df.drop(["ApplicantIncome", "CoapplicantIncome"], axis=1, inplace=True)

    try:
        prediction = model.predict(input_df)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        st.subheader(result)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
