import streamlit as st
import pandas as pd
import joblib

from preprocess import preprocess_dataframe

# --------------------------------
# Load trained model
# --------------------------------
model = joblib.load("model/health_insurance_model.pkl")

# --------------------------------
# Streamlit UI
# --------------------------------
st.set_page_config(page_title="Health Insurance Prediction", layout="centered")

st.title("💊 Health Insurance Cost Prediction")
st.write("Enter the details below to predict insurance charges.")

# --------------------------------
# User Inputs
# --------------------------------
age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# --------------------------------
# Prediction
# --------------------------------
if st.button("Predict Insurance Cost"):
    # Create input dataframe
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    # Preprocess input
    processed_input = preprocess_dataframe(input_data)

    # Predict
    prediction = model.predict(processed_input)[0]

    st.success(f"💰 Estimated Insurance Cost: ₹ {prediction:,.2f}")
