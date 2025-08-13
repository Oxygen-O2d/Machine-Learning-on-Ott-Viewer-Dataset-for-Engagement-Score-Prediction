import streamlit as st
import pandas as pd
import numpy as np  # Import numpy for clipping
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import warnings

# Suppress warnings for a cleaner app interface
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="View Completion Rate Predictor",
    page_icon="🎬",
    layout="wide"
)

# --- Model Training Function with Caching ---
@st.cache_resource
def train_model():
    """
    Loads data, preprocesses, and trains the RFE model.
    Caches the model and the full list of training columns.
    """
    st.info("Initializing the prediction model... Please wait.")

    # Load the dataset
    df = pd.read_csv('ott_viewer_dataset.csv')

    # --- Data Preprocessing ---
    df = df.drop(['user_id', 'content_id'], axis=1)
    # Fill missing user_rating values
    df['user_rating'].fillna(df['user_rating'].mean(), inplace=True)
    # The 'view_completion_rate' has no missing values to fill
    
    df['watch_date'] = pd.to_datetime(df['watch_date'])
    df['Day'] = df['watch_date'].dt.day
    df['Month'] = df['watch_date'].dt.month
    df['Year'] = df['watch_date'].dt.year
    df['Quarter'] = df['watch_date'].dt.quarter
    seasons = {
        1: 'Winter', 2: 'Winter', 3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Post-Monsoon', 11: 'Post-Monsoon', 12: 'Winter'
    }
    df['Season'] = df['Month'].apply(lambda x: seasons[x])
    df = df.drop('watch_date', axis=1)
    df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns, drop_first=True)

    # --- Model Training ---
    X = df.drop(['engagement_score', 'view_completion_rate'], axis=1)
    y = df['view_completion_rate']
    
    all_training_columns = X.columns.tolist()

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=regressor, n_features_to_select=15)
    rfe.fit(X, y)

    return rfe, all_training_columns

# --- Load the Model and Columns ---
rfe_model, all_training_columns = train_model()

# --- User Interface ---
st.title("🎬 OTT View Completion Rate Predictor")
st.write(
    "This app predicts the **View Completion Rate** of content. "
    "Use the sidebar to enter details and see the model's prediction."
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Viewer and Content Details")

def get_user_input():
    # ... (This function remains the same as before) ...
    genders = ['Female', 'Male', 'Other']
    locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Guntur']
    sub_types = ['Free', 'Basic', 'Premium']
    genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Thriller', 'Romance']

    age = st.sidebar.slider('Age', 18, 100, 35)
    watch_time_hours = st.sidebar.slider('Watch Time (Hours)', 0.1, 24.0, 2.5)
    regional_relevance = st.sidebar.slider('Regional Relevance (0-10)', 0, 10, 5)
    family_friendly_score = st.sidebar.slider('Family Friendly Score (0-10)', 0, 10, 7)
    user_rating = st.sidebar.slider('User Rating (1-5)', 1.0, 5.0, 4.0)
    month = st.sidebar.slider('Month', 1, 12, 8)
    
    gender = st.sidebar.selectbox('Gender', genders)
    location = st.sidebar.selectbox('Location', locations)
    sub_type = st.sidebar.selectbox('Subscription Type', sub_types)
    content_genre = st.sidebar.selectbox('Content Genre', genres)

    input_dict = {
        'age': age,
        'watch_time_hours': watch_time_hours,
        'regional_relevance': regional_relevance,
        'family_friendly_score': family_friendly_score,
        'user_rating': user_rating,
        'Day': 13,
        'Month': month,
        'Year': 2025,
        'Quarter': (month - 1) // 3 + 1,
        'gender': gender,
        'location': location,
        'subscription_type': sub_type,
        'content_genre': content_genre,
    }
    return input_dict

user_input = get_user_input()

# --- Prediction Logic ---
if st.button('Predict Completion Rate'):
    # Create the dataframe for prediction (same as before)
    prediction_df = pd.DataFrame(columns=all_training_columns)
    prediction_df.loc[0] = 0

    for key, value in user_input.items():
        if key in prediction_df.columns:
            prediction_df[key] = value

    for cat_feature in ['gender', 'location', 'subscription_type', 'content_genre']:
        col_name = f"{cat_feature}_{user_input[cat_feature]}"
        if col_name in prediction_df.columns:
            prediction_df[col_name] = 1

    month = user_input['Month']
    if month in [6, 7, 8, 9]: season_col = 'Season_Monsoon'
    elif month in [10, 11]: season_col = 'Season_Post-Monsoon'
    elif month in [3, 4, 5]: season_col = 'Season_Summer'
    else: season_col = 'Season_Winter'
    
    if season_col in prediction_df.columns:
        prediction_df[season_col] = 1

    prediction_df = prediction_df.apply(pd.to_numeric)
    
    # Make the prediction
    prediction = rfe_model.predict(prediction_df)
    
    # **FIX 1: Clip the prediction to be within the 0-100 range**
    clipped_prediction = np.clip(prediction[0], 0, 100)
    
    # **FIX 2: Do NOT multiply by 100. Just round the value.**
    predicted_rate = round(clipped_prediction, 2)
    
    st.subheader('Prediction Result')
    st.success(f"The predicted View Completion Rate is **{predicted_rate}%**")
    st.balloons()
