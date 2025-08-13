import streamlit as st
import pandas as pd
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
    This function loads the data, preprocesses it, and trains the RFE model.
    Using @st.cache_resource ensures this expensive process runs only once.
    """
    st.info("Training model... This will only run once on the first load. Might Take Long so sit back and chill")

    # Load the dataset
    df = pd.read_csv('ott_viewer_dataset.csv')

    # --- Data Preprocessing ---
    df = df.drop(['user_id', 'content_id'], axis=1)
    df['user_rating'].fillna(df['user_rating'].mean(), inplace=True)
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
    
    # Initialize and fit RFE
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rfe = RFE(estimator=regressor, n_features_to_select=15)
    rfe.fit(X, y)

    # Get the list of selected feature names
    selected_columns = X.columns[rfe.support_].tolist()

    return rfe, selected_columns

# --- Load the Model ---
# This calls the cached function
rfe_model, selected_columns = train_model()

# --- User Interface ---
st.title("🎬 OTT View Completion Rate Predictor")
st.write(
    "This app predicts the **View Completion Rate** of content. "
    "Use the sidebar to enter details and see the model's prediction."
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Viewer and Content Details")

def user_input_features():
    # Create input fields for all features initially required for one-hot encoding
    age = st.sidebar.slider('Age', 18, 100, 35)
    watch_time_hours = st.sidebar.slider('Watch Time (Hours)', 0.1, 24.0, 2.5)
    regional_relevance = st.sidebar.slider('Regional Relevance (0-10)', 0, 10, 5)
    family_friendly_score = st.sidebar.slider('Family Friendly Score (0-10)', 0, 10, 7)
    user_rating = st.sidebar.slider('User Rating (1-5)', 1.0, 5.0, 4.0)
    day = st.sidebar.slider('Day of Month', 1, 31, 15)
    month = st.sidebar.slider('Month', 1, 12, 6)
    year = st.sidebar.slider('Year', 2022, 2024, 2023)
    
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
    location = st.sidebar.selectbox('Location', ('Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Guntur'))
    sub_type = st.sidebar.selectbox('Subscription Type', ('Basic', 'Premium'))
    content_genre = st.sidebar.selectbox('Content Genre', ('Action', 'Comedy', 'Drama', 'Romance'))
    
    # Process inputs into a DataFrame
    data = {
        'age': [age],
        'watch_time_hours': [watch_time_hours],
        'regional_relevance': [regional_relevance],
        'family_friendly_score': [family_friendly_score],
        'user_rating': [user_rating],
        'Day': [day],
        'Month': [month],
        'Year': [year],
        'Quarter': [(month - 1) // 3 + 1],
        'gender_Male': [1 if gender == 'Male' else 0],
        'gender_Other': [1 if gender == 'Other' else 0],
        'location_Guntur': [1 if location == 'Guntur' else 0],
        'subscription_type_Premium': [1 if sub_type == 'Premium' else 0],
        'content_genre_Romance': [1 if content_genre == 'Romance' else 0],
        'Season_Winter': [1 if month in [1, 2, 12] else 0]
    }
    
    features = pd.DataFrame(data)
    return features

# Get user input
input_df = user_input_features()

# Display user inputs
st.subheader("Your Selections")
st.write(input_df)

# Create a final DataFrame with the exact columns the model expects
final_df = pd.DataFrame(columns=selected_columns)
final_df = pd.concat([final_df, input_df]).fillna(0)

# Ensure order and columns match the training data
final_df = final_df[selected_columns]

# --- Prediction ---
if st.button('Predict Completion Rate'):
    prediction = rfe_model.predict(final_df)
    
    st.subheader('Prediction')
    predicted_rate = round(prediction[0] * 100, 2)
    st.success(f"The predicted View Completion Rate is **{predicted_rate}%**")
    st.balloons()