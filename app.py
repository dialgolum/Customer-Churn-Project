import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression


# Load the trained model
# To keep it simple, we train the model again here (optional: later we can use joblib to save/load)

@st.cache_data
def load_model():
    model = joblib.load("churn_model.pkl")
    input_features = joblib.load("model_features.pkl")
    return model, input_features

model, input_features = load_model()

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="Customer Churn Prediction", layout="wide") 
st.title("📉 Customer Churn Prediction")
st.markdown("Fill in the customer details below to predict churn:")

# ------------------------
# User Input Form
# ------------------------

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (in months)", 0, 72, 12)

with col2:
    phone_service = st.selectbox("Phone Service", [0, 1])
    multiple_lines = st.selectbox("Multiple Lines_No phone service", [0, 1])
    internet_service_Fiber = st.selectbox("Internet Service: Fiber optic", [0, 1])
    internet_service_None = st.selectbox("Internet Service: None", [0, 1])
    online_security = st.selectbox("Online Security_No internet service", [0, 1])

st.markdown("### 📦 Services")
col3, col4 = st.columns(2)

with col3:
    online_backup = st.selectbox("Online Backup_No internet service", [0, 1])
    device_protection = st.selectbox("Device Protection_No internet service", [0, 1])
    tech_support = st.selectbox("Tech Support_No internet service", [0, 1])
    streaming_tv = st.selectbox("Streaming TV_No internet service", [0, 1])

with col4:
    streaming_movies = st.selectbox("Streaming Movies_No internet service", [0, 1])
    contract_type = st.selectbox("Contract Type (One/Two Year)", [0, 1])  # 1 = Two year
    paperless_billing = st.selectbox("Paperless Billing", [0, 1])
    payment_electronic = st.selectbox("Payment Method: Electronic Check", [0, 1])
    payment_mail = st.selectbox("Payment Method: Mailed Check", [0, 1])

st.markdown("### 💰 Billing Details")
monthly_charges = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
total_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 1500.0)

# ------------------------
# Prepare Input
# ------------------------

input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': 1 if partner == 'Yes' else 0,
    'Dependents': 1 if dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'PaperlessBilling': paperless_billing,
    'MultipleLines_No phone service': multiple_lines,
    'InternetService_Fiber optic': internet_service_Fiber,
    'InternetService_No': internet_service_None,
    'OnlineSecurity_No internet service': online_security,
    'OnlineBackup_No internet service': online_backup,
    'DeviceProtection_No internet service': device_protection,
    'TechSupport_No internet service': tech_support,
    'StreamingTV_No internet service': streaming_tv,
    'StreamingMovies_No internet service': streaming_movies,
    'Contract_Two year': contract_type,
    'PaymentMethod_Electronic check': payment_electronic,
    'PaymentMethod_Mailed check': payment_mail
}

# Add 0s for any missing features (important for model input)

for feat in input_features:
    if feat not in input_dict:
        input_dict[feat] = 0

# Reorder the features
input_df = pd.DataFrame([input_dict])[input_features]

# ------------------------
# Predict
# ------------------------

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"🔴 Likely to churn (Probability: {prob:.2f})")
    else:
        st.success(f"🟢 Likely to stay (Probability: {prob:.2f})")

