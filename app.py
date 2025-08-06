import streamlit as st
import pickle
import pandas as pd
import joblib

with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Define the features used in the model
selected_features = ['gender_Male', 'state_Madhya Pradesh', 'language_preference_Hindi', 'device_type_Smartphone', 'subscription_type_Free', 'age', 'watch_time_hours', 'regional_relevance', 'family_friendly_score', 'view_completion_rate', 'user_rating', 'Day', 'Month', 'Year', 'Quarter']

# Streamlit App UI
st.title("📈 Engagement Score Predictions on an OTT platform")

st.markdown(
    """
    This app predicts the **Engagement Score** based on Ott Platform Data
    """
)

# Collect input data from user
input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert input into DataFrame
input_df = pd.DataFrame([input_data])

# Predict when button is pressed
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Engagement Score: **{prediction[0]:.2f}%**")
