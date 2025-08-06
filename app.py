import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Load Models & Feature List ---
# It's good practice to load all your artifacts at the start.
try:
    model = joblib.load("model.pkl")
    rfe = joblib.load("rfe.pkl")
    # --- CHANGE HERE ---
    # Load the list of ALL ORIGINAL feature names that the model was trained on.
    # You must create this file from your training notebook.
    # Example: joblib.dump(list(X_train.columns), 'original_feature_names.pkl')
    original_features = joblib.load("original_feature_names.pkl") 
except FileNotFoundError:
    st.error("Model or feature files not found. Please ensure 'model.pkl', 'rfe.pkl', and 'original_feature_names.pkl' are in the same directory.")
    st.stop()


st.set_page_config(layout="wide")
st.title("OTT Viewer Engagement Score Predictor")

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

    st.header("Content & Subscription Details")
    col4, col5, col6 = st.columns(3)
    with col4:
        language = st.selectbox("Content Language", ["Hindi", "Other"])
    with col5:
        device_type = st.selectbox("Device Type", ["Smartphone", "Other"])
    with col6:
        subscription_type = st.selectbox("Subscription Type", ["Free", "Paid"])

    st.header("Viewing Habits & Ratings")
    col7, col8, col9 = st.columns(3)
    with col7:
        watch_time_hours = st.number_input("Watch Time (hours)", min_value=0.0, value=2.0, step=0.5)
    with col8:
        view_completion_rate = st.slider("View Completion Rate", 0.0, 1.0, 0.7)
    with col9:
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


    submit = st.form_submit_button("Predict Engagement Score")

# --- Prediction Logic ---
if submit:
    # 1. Create a dictionary of the raw inputs from the form
    input_data = {
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
        'Quarter': quarter,
        'state_Madhya Pradesh': 1 if state == "Madhya Pradesh" else 0,
        'device_type_Smartphone': 1 if device_type == "Smartphone" else 0,
        'subscription_type_Free': 1 if subscription_type == "Free" else 0,
        'content_language_Hindi': 1 if language == "Hindi" else 0,
    }

    # 2. Create a DataFrame from the dictionary
    input_df = pd.DataFrame([input_data])

    # 3. Reorder the DataFrame to match the training order.
    # This is the crucial step. It ensures the columns are in the exact 
    # order the model expects, filling any missing ones with 0.
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
    # The transform step will now work because it receives a DataFrame with all original features.
    input_df_rfe = rfe.transform(input_df_reordered)

    # 5. Predict
    prediction = model.predict(input_df_rfe)

    # 6. Display Result
    st.success(f"Predicted Engagement Score: {prediction[0]:.2f}")
    st.balloons()
