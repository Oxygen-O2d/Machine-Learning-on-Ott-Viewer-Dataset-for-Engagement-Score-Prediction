# 🎬 OTT Viewer Dataset — View Completion Rate Prediction

This project predicts **`view_completion_rate`** for OTT platform users using **synthetic viewer data**.  
It applies **machine learning** for regression tasks and is deployed as an **interactive Streamlit web app**.

---

## 📊 About the Project
The project uses a synthetic dataset of OTT viewers with features like:
- User demographics (age, gender, location)
- Device details
- Subscription type
- Content preferences
- Engagement metrics

The ML model predicts **how much of a video a user will watch** (`view_completion_rate`), helping OTT platforms understand and improve audience retention.

---

## 🚀 Features
- **Data Preprocessing**: Handling categorical & numerical features.
- **Feature Selection**: Recursive Feature Elimination (RFE) with Random Forest.
- **Model Training**: Random Forest Regressor for predicting completion rate.
- **Deployment**: Streamlit web application accessible online.
- **Interactive Predictions**: Input viewer details & get predictions instantly.

---

## 📂 Dataset
The dataset used is **synthetic** and created for experimentation.  
File: `ott_viewer_dataset.csv`

**Sample columns:**
- `state`, `device_type`, `subscription_type`, `content_language`, `age`, `gender`, `watch_time_hours`, `regional_relevance`, `family_friendly_score`, `user_rating`
- Target variable: `view_completion_rate`

---

## 🛠️ Installation
1. **Clone this repository**:
```bash
git clone https://github.com/your-username/ott-view-completion-prediction.git
cd ott-view-completion-prediction
