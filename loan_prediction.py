
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Load trained model (ensure to provide the correct path)
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("loan_model.pkl")

model = load_model()

# Set up page config with background
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    layout="centered"
)

# Background image using base64 encoded string
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

st.title("🏠 Dream Housing Loan Eligibility Prediction")
st.markdown("### Fill the form to check loan approval status")

# Input features
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.selectbox("Credit History", ["Meets guidelines", "Does not meet guidelines"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Map inputs
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
    "Property_Area": {"Urban":2, "Semiurban":1, "Rural":0}[property_area]
}

if st.button("Predict Loan Status"):
    input_df = pd.DataFrame([input_data])
    input_df["TotalIncome"] = input_df["ApplicantIncome"] + input_df["CoapplicantIncome"]
    input_df["EMI"] = input_df["LoanAmount"] / input_df["Loan_Amount_Term"]
    input_df["Balance_Income"] = input_df["TotalIncome"] - (input_df["EMI"] * 1000)
    input_df.drop(["ApplicantIncome", "CoapplicantIncome"], axis=1, inplace=True)

    prediction = model.predict(input_df)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
    st.subheader(result)
