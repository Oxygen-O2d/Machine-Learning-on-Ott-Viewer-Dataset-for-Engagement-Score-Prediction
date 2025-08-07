import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Models & Feature List ---
try:
    model = joblib.load("model.pkl")
    rfe = joblib.load("rfe.pkl")
    original_features = joblib.load("original_feature_names.pkl") 
except FileNotFoundError:
    st.error("Model or feature files not found. Please ensure 'model.pkl', 'rfe.pkl', and 'original_feature_names.pkl' are in the same directory.")
    st.stop()


st.set_page_config(layout="wide")
st.title("OTT View Completion Rate Predictor")

# --- Form Input ---
with st.form("prediction_form"):
    st.header("User Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    with col3:
        state = st.selectbox("State", ["Madhya Pradesh", "Other"])

    st.header("Viewing Context")
    # --- CHANGE HERE: Added Season and removed Language ---
    col4, col5, col6 = st.columns(3)
    with col4:
        device_type = st.selectbox("Device Type", ["Smartphone", "Other"])
    with col5:
        subscription_type = st.selectbox("Subscription Type", ["Free", "Paid"])
    with col6:
        season = st.selectbox("Season", ["Spring", "Summer", "Fall", "Winter"])


    st.header("Viewing Habits & Ratings")
    col7, col8 = st.columns(2)
    with col7:
        watch_time_hours = st.number_input("Watch Time (hours)", min_value=0.0, value=2.0, step=0.5)
    with col8:
        user_rating = st.slider("User Rating", 0.0, 5.0, 3.0, step=0.1)
        
    st.header("Content Attributes")
    col10, col11 = st.columns(2)
    with col10:
        regional_relevance = st.slider("Regional Relevance", 0.0, 1.0, 0.5)
    with col11:
        family_friendly_score = st.slider("Family Friendly Score", 0.0, 1.0, 0.5)


    st.header("Date Information")
    col12, col13, col14, col15 = st.columns(4)
    with col12:
        day = st.number_input("Day", min_value=1, max_value=31, value=15)
    with col13:
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
    with col14:
        year = st.number_input("Year", min_value=2020, max_value=2025, value=2024)
    with col15:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)

    submit = st.form_submit_button("Predict View Completion Rate")

# --- Prediction Logic ---
if submit:
    # 1. Create a dictionary of the raw inputs from the form
    # --- CHANGE HERE: Updated dictionary to match new features ---
    input_data = {
        'age': age,
        'gender': {"Female": 1, "Male": 0, "Other": 2}[gender], # Adjusted to match notebook encoding
        'watch_time_hours': watch_time_hours,
        'regional_relevance': regional_relevance,
        'family_friendly_score': family_friendly_score,
        'user_rating': user_rating,
        'Day': day,
        'Month': month,
        'Year': year,
        'Quarter': quarter,
        'state_Madhya Pradesh': 1 if state == "Madhya Pradesh" else 0,
        'device_type_Smartphone': 1 if device_type == "Smartphone" else 0,
        'subscription_type_Premium': 1 if subscription_type == "Paid" else 0,
        'Season_Spring': 1 if season == "Spring" else 0,
        'Season_Summer': 1 if season == "Summer" else 0,
        'Season_Winter': 1 if season == "Winter" else 0,
    }

    # 2. Create a DataFrame from the dictionary
    input_df = pd.DataFrame([input_data])

    # 3. Reorder the DataFrame to match the training order.
    try:
        # Create a full DataFrame with all original columns, filled with 0
        full_input_df = pd.DataFrame(columns=original_features)
        full_input_df = pd.concat([full_input_df, input_df]).fillna(0)
        
        # Ensure the final order is correct
        input_df_reordered = full_input_df[original_features]

    except KeyError as e:
        st.error(f"A feature mismatch occurred. The model is missing the following feature from the input: {e}")
        st.stop()


    # 4. RFE transform
    input_df_rfe = rfe.transform(input_df_reordered)

    # 5. Predict
    prediction = model.predict(input_df_rfe)

    # 6. Display Result
    st.success(f"Predicted View Completion Rate: {prediction[0]:.2%}")
    st.balloons()
