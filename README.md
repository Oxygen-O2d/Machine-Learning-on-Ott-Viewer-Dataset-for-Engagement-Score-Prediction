OTT Viewer Completion Rate PredictorA machine learning project that predicts the view completion rate for users on an Over-The-Top (OTT) media service. The project includes a data processing and model training pipeline in a Jupyter Notebook and an interactive web application built with Streamlit for real-time predictions.<!-- It's a good idea to add a screenshot of your app here -->🚀 FeaturesData Preprocessing: Cleans and prepares the raw OTT viewer data, handling missing values and engineering new features.Feature Engineering: Creates insightful features from raw data, such as Day_of_Week and Season, to improve model accuracy.Machine Learning Model: Utilizes a RandomForestRegressor to predict the view completion rate.Feature Selection: Employs Recursive Feature Elimination (RFE) to select the most impactful features for the model.Interactive Web App: A user-friendly interface built with Streamlit that allows for real-time predictions based on user-defined inputs.🛠️ Tech StackPython: Core programming language.Pandas: For data manipulation and analysis.NumPy: For numerical operations.Scikit-learn: For machine learning tasks, including model training (RandomForestRegressor) and feature selection (RFE).Joblib: For saving and loading the trained machine learning models.Streamlit: For building and deploying the interactive web application.Jupyter Notebook: For data exploration, model development, and training.📁 File StructureThe repository is organized as follows:.
├── 📄 app.py                    # The main Streamlit application file
├── 📓 machine learning on ott_viewer_dataset.ipynb  # Jupyter Notebook for data processing and model training
├── 💾 model.pkl                   # Saved trained RandomForestRegressor model
├── 💾 rfe.pkl                     # Saved RFE object for feature transformation
├── 💾 original_feature_names.pkl  # List of feature names used for training
├── 🗂️ ott_viewer_dataset.csv      # The raw dataset
├── 📋 requirements.txt           # List of Python dependencies
└── 📄 README.md                   # This file
⚙️ Setup and InstallationFollow these steps to set up the project locally.1. Clone the Repositorygit clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
2. Create a Virtual EnvironmentIt's recommended to use a virtual environment to manage project dependencies.# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install DependenciesInstall all the required Python packages from the requirements.txt file.pip install -r requirements.txt
(Note: You will need to create a requirements.txt file. You can generate one by running pip freeze > requirements.txt in your activated virtual environment after installing the packages listed in the Tech Stack.)🚀 Usage1. Train the Model (Optional)If you wish to retrain the model or experiment with different parameters, run the machine learning on ott_viewer_dataset.ipynb notebook from start to finish. This will regenerate the .pkl model files.2. Run the Streamlit ApplicationTo start the interactive web application, run the following command in your terminal:streamlit run app.py
This will open the application in your default web browser, where you can input various parameters to get a predicted view completion rate.🧠 Model DetailsModel: RandomForestRegressorTarget Variable: view_completion_rate (normalized to be a value between 0 and 1).Feature Selection: Recursive Feature Elimination (RFE) is used to select the top 15 most predictive features from the dataset.Key Engineered Features: The model's performance is enhanced by creating time-based features like Day_of_Week and Season from the watch_date.
