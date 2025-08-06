import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Load model and selected feature list
model = joblib.load('model.pkl')
selected_features = joblib.load('selected_features.pkl')  # This is a list of feature names

st.title("Engagement Score Predictor")

# Create input fields dynamically
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])  # Create a DataFrame with correct feature names
    prediction = model.predict(input_df)
    st.success(f"Predicted Engagement Score: {prediction[0]:.2f}")
