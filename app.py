import streamlit as st
import pandas as pd
import joblib

# Load trained model and RFE (if applicable)
model = joblib.load("model.pkl")
# rfe = joblib.load("rfe.pkl")  # Only if you're using RFE.transform()

# Selected features after RFE
selected_features = [
    'gender_Male', 'state_Madhya Pradesh', 'language_preference_Hindi',
    'device_type_Smartphone', 'subscription_type_Free', 'age',
    'watch_time_hours', 'regional_relevance', 'family_friendly_score',
    'view_completion_rate', 'user_rating', 'Day', 'Month', 'Year', 'Quarter'
]

# -------------------------
# 🚀 Streamlit App Starts
# -------------------------
st.title("OTT Viewer Engagement Score Predictor")

# Form UI
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    state = st.selectbox("State", ["Madhya Pradesh", "Other"])
    lang = st.selectbox("Language Preference", ["Hindi", "Other"])
    device = st.selectbox("Device Type", ["Smartphone", "Other"])
    sub_type = st.selectbox("Subscription Type", ["Free", "Paid"])
    
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    watch_time = st.number_input("Watch Time (in hours)", min_value=0.0, value=2.5)
    regional_relevance = st.slider("Regional Relevance", 0.0, 1.0, 0.5)
    family_score = st.slider("Family Friendly Score", 0.0, 1.0, 0.5)
    completion_rate = st.slider("View Completion Rate", 0.0, 1.0, 0.6)
    rating = st.slider("User Rating", 0.0, 5.0, 3.0)
    
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    year = st.number_input("Year", min_value=2020, max_value=2025, value=2025)
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    
    submit = st.form_submit_button("Predict")

# -------------------------
# 🧠 Prediction Logic
# -------------------------
if submit:
    # Manually map UI inputs to model-required features
    input_dict = {
        'gender_Male': 1 if gender == "Male" else 0,
        'state_Madhya Pradesh': 1 if state == "Madhya Pradesh" else 0,
        'language_preference_Hindi': 1 if lang == "Hindi" else 0,
        'device_type_Smartphone': 1 if device == "Smartphone" else 0,
        'subscription_type_Free': 1 if sub_type == "Free" else 0,
        'age': age,
        'watch_time_hours': watch_time,
        'regional_relevance': regional_relevance,
        'family_friendly_score': family_score,
        'view_completion_rate': completion_rate,
        'user_rating': rating,
        'Day': day,
        'Month': month,
        'Year': year,
        'Quarter': quarter
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(input_df)
    st.success(f"Predicted Engagement Score: {prediction[0]:.2f}")
