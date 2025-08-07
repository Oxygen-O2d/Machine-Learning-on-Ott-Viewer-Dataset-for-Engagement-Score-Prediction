import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Models & Feature List ---
# Use a try-except block to handle potential file loading errors gracefully.
try:
    model = joblib.load("model.pkl")
    rfe = joblib.load("rfe.pkl")
    original_features = joblib.load("original_feature_names.pkl") 
except FileNotFoundError:
    st.error("Error: Model or feature files not found. Please ensure 'model.pkl', 'rfe.pkl', and 'original_feature_names.pkl' are in the same directory as your app.py file.")
    st.stop()


st.set_page_config(layout="wide")
st.title("OTT View Completion Rate Predictor")

# --- Form Input ---
# Use a form to gather all inputs before running the prediction.
with st.form("prediction_form"):
    st.header("User Demographics")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
    with col2:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    with col3:
        state = st.selectbox("State", ["Madhya Pradesh", "Other"]) # Example state

    st.header("Viewing Context")
    col4, col5, col6 = st.columns(3)
    with col4:
        device_type = st.selectbox("Device Type", ["Smartphone", "Tablet", "Laptop", "Connected TV"])
    with col5:
        subscription_type = st.selectbox("Subscription Type", ["Free", "Premium"])
    with col6:
        # Day of week input for the model
        day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])


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
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
    with col13:
        month = st.number_input("Month", min_value=1, max_value=12, value=6)
    with col14:
        year = st.number_input("Year", min_value=2020, max_value=2025, value=2024)
    with col15:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4], index=1)

    submit = st.form_submit_button("Predict View Completion Rate")

# --- Prediction Logic ---
if submit:
    # This is a robust way to create the input DataFrame for prediction.
    # 1. Create a DataFrame with all the columns the model was trained on, and fill with 0.
    input_df = pd.DataFrame(0, index=[0], columns=original_features)

    # 2. Update the DataFrame with the user's input from the form.
    # This method avoids errors if a column is missing or named incorrectly.
    
    # Simple numerical inputs
    input_df['age'] = age
    input_df['gender'] = {"Female": 1, "Male": 0, "Other": 2}[gender]
    input_df['watch_time_hours'] = watch_time_hours
    input_df['regional_relevance'] = regional_relevance
    input_df['family_friendly_score'] = family_friendly_score
    input_df['user_rating'] = user_rating
    input_df['Day'] = day
    input_df['Month'] = month
    input_df['Year'] = year
    input_df['Quarter'] = quarter

    # One-hot encoded features: only set the selected one to 1 if it exists in the columns
    # This prevents errors if the user selects a value that wasn't in the training data (e.g., 'Other')
    if state != 'Other' and f'state_{state}' in input_df.columns:
        input_df[f'state_{state}'] = 1
            
    if f'device_type_{device_type}' in input_df.columns:
        input_df[f'device_type_{device_type}'] = 1

    if subscription_type == 'Premium' and 'subscription_type_Premium' in input_df.columns:
        input_df['subscription_type_Premium'] = 1
    
    # Determine the season from the selected month
    if month in [12, 1, 2]: season = 'Winter'
    elif month in [3, 4, 5]: season = 'Spring'
    elif month in [6, 7, 8]: season = 'Summer'
    else: season = 'Fall'
    
    if f'Season_{season}' in input_df.columns:
        input_df[f'Season_{season}'] = 1

    if f'Day_of_Week_{day_of_week}' in input_df.columns:
        input_df[f'Day_of_Week_{day_of_week}'] = 1

    # 3. RFE transform and Predict
    try:
        # The input_df now has the exact same structure as the training data
        input_df_rfe = rfe.transform(input_df)
        
        prediction = model.predict(input_df_rfe)

        st.success(f"Predicted View Completion Rate: {prediction[0]:.2%}")
        st.balloons()

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
