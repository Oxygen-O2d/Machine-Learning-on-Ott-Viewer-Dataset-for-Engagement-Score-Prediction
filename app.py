import streamlit as st
import pandas as pd
import joblib

# Load trained model and RFE transformer
model = joblib.load("model.pkl")
rfe = joblib.load("rfe.pkl")

st.title("OTT Viewer Engagement Score Predictor")

# --- Form Input ---
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])  # Will be encoded as 0,1,2

    state = st.selectbox("State", ["Madhya Pradesh", "Other"])
    language = st.selectbox("Content Language", ["Hindi", "Other"])
    device_type = st.selectbox("Device Type", ["Smartphone", "Other"])
    subscription_type = st.selectbox("Subscription Type", ["Free", "Paid"])

    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    watch_time_hours = st.number_input("Watch Time (hours)", min_value=0.0, value=2.0)
    regional_relevance = st.slider("Regional Relevance", 0.0, 1.0, 0.5)
    family_friendly_score = st.slider("Family Friendly Score", 0.0, 1.0, 0.5)
    view_completion_rate = st.slider("View Completion Rate", 0.0, 1.0, 0.7)
    user_rating = st.slider("User Rating", 0.0, 5.0, 3.0)

    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    year = st.number_input("Year", min_value=2020, max_value=2025, value=2025)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])

    submit = st.form_submit_button("Predict")

# --- Prediction Logic ---
if submit:
    # Manual Encoding
    input_dict = {
        'state_Madhya Pradesh': 1 if state == "Madhya Pradesh" else 0,
        'device_type_Smartphone': 1 if device_type == "Smartphone" else 0,
        'subscription_type_Free': 1 if subscription_type == "Free" else 0,
        'content_language_Hindi': 1 if language == "Hindi" else 0,
        'age': age,
        'gender': {"Female": 0, "Male": 1, "Other": 2}[gender],
        'watch_time_hours': watch_time_hours,
        'regional_relevance': regional_relevance,
        'family_friendly_score': family_friendly_score,
        'view_completion_rate': view_completion_rate,
        'user_rating': user_rating,
        'Day': day,
        'Month': month,
        'Year': year,
        'Quarter': quarter
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # RFE transform
    input_df_rfe = rfe.transform(input_df)

    # Predict
    prediction = model.predict(input_df_rfe)

    # Display
    st.success(f"Predicted Engagement Score: {prediction[0]:.2f}")
