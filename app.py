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

st.title("📉 Customer Churn Prediction")
st.write("Fill in the customer information below:")

# Collect user input
user_inputs = {}
for feature in input_features:
    if "Yes" in feature or "No" in feature or "Female" in feature or "Male" in feature:
        user_inputs[feature] = st.selectbox(f"{feature}:", ["0", "1"])
    else:
        user_inputs[feature] = st.text_input(f"{feature}:", "0")

#Predict
if st.button("Predict Churn"):
    try:
        input_array = np.array([float(user_inputs[feat]) for feat in input_features]).reshape(1, -1)
        prediction = model.predict(input_array)
        result = "🔴 Yes, this customer is likely to churn." if prediction[0] == 1 else "🟢 No, this customer is likely to stay."
        st.subheader("Prediction Result:")
        st.success(result)
    except Exception as e:
        st.error(f"Invalid input or error in prediction: {e}")

