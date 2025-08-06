import streamlit as st
import pickle
import pandas as pd
import joblib

with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

# Define the features used in the model
selected_features = ['gender_Female', 'location_Bhopal', 'location_Coimbatore', 'location_Kolhapur', 'location_Nanded', 'location_Ujjain', 'language_preference_Assamese', 'language_preference_Bengali', 'language_preference_Hindi', 'language_preference_Malayalam', 'language_preference_Marathi', 'language_preference_Tamil', 'device_type_Laptop', 'device_type_Smartphone', 'device_type_Tablet', 'subscription_type_Free', 'content_genre_Comedy', 'content_genre_Crime', 'content_genre_Drama', 'content_genre_Horror-Comedy', 'content_genre_Romance', 'content_genre_Sci-Fi', 'content_genre_Thriller', 'content_language_Assamese', 'content_language_Gujarati', 'content_language_Kannada', 'content_language_Malayalam', 'content_language_Marathi', 'content_language_Odia', 'content_language_Punjabi', 'content_language_Tamil', 'Week_Day_Monday', 'Week_Day_Sunday', 'Season_Post-Monsoon', 'watch_time_hours', 'family_friendly_score', 'view_completion_rate', 'Year', 'Quarter']

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

