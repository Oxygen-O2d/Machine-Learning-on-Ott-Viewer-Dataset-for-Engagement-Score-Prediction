<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OTT Viewer Completion Rate Predictor</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 40px auto;
            padding: 0 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            color: #111;
        }
        h1 {
            font-size: 2em;
        }
        h2 {
            font-size: 1.5em;
            margin-top: 40px;
        }
        h3 {
            font-size: 1.2em;
            margin-top: 30px;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin-bottom: 10px;
        }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            background-color: #eee;
            padding: 2px 6px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        pre {
            background-color: #2d2d2d;
            color: #f1f1f1;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        strong {
            color: #111;
        }
    </style>
</head>
<body>

    <h1>OTT Viewer Completion Rate Predictor</h1>
    <p>
        A machine learning project that predicts the view completion rate for users on an Over-The-Top (OTT) media service. The project includes a data processing and model training pipeline in a Jupyter Notebook and an interactive web application built with Streamlit for real-time predictions.
    </p>

    <h2>🚀 Features</h2>
    <ul>
        <li><strong>Data Preprocessing</strong>: Cleans and prepares the raw OTT viewer data, handling missing values and engineering new features.</li>
        <li><strong>Feature Engineering</strong>: Creates insightful features from raw data, such as <code>Day_of_Week</code> and <code>Season</code>, to improve model accuracy.</li>
        <li><strong>Machine Learning Model</strong>: Utilizes a <code>RandomForestRegressor</code> to predict the view completion rate.</li>
        <li><strong>Feature Selection</strong>: Employs Recursive Feature Elimination (RFE) to select the most impactful features for the model.</li>
        <li><strong>Interactive Web App</strong>: A user-friendly interface built with Streamlit that allows for real-time predictions based on user-defined inputs.</li>
    </ul>

    <h2>🛠️ Tech Stack</h2>
    <ul>
        <li><strong>Python</strong>: Core programming language.</li>
        <li><strong>Pandas</strong>: For data manipulation and analysis.</li>
        <li><strong>NumPy</strong>: For numerical operations.</li>
        <li><strong>Scikit-learn</strong>: For machine learning tasks, including model training (RandomForestRegressor) and feature selection (RFE).</li>
        <li><strong>Joblib</strong>: For saving and loading the trained machine learning models.</li>
        <li><strong>Streamlit</strong>: For building and deploying the interactive web application.</li>
        <li><strong>Jupyter Notebook</strong>: For data exploration, model development, and training.</li>
    </ul>

    <h2>📁 File Structure</h2>
    <p>The repository is organized as follows:</p>
    <pre><code>.
├── 📄 app.py                    # The main Streamlit application file
├── 📓 machine learning on ott_viewer_dataset.ipynb  # Jupyter Notebook for data processing and model training
├── 💾 model.pkl                   # Saved trained RandomForestRegressor model
├── 💾 rfe.pkl                     # Saved RFE object for feature transformation
├── 💾 original_feature_names.pkl  # List of feature names used for training
├── 🗂️ ott_viewer_dataset.csv      # The raw dataset
├── 📋 requirements.txt           # List of Python dependencies
└── 📄 README.md                   # This file
</code></pre>

    <h2>⚙️ Setup and Installation</h2>
    <p>Follow these steps to set up the project locally.</p>
    
    <h3>1. Clone the Repository</h3>
    <pre><code>git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name</code></pre>

    <h3>2. Create a Virtual Environment</h3>
    <p>It's recommended to use a virtual environment to manage project dependencies.</p>
    <pre><code># For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate</code></pre>

    <h3>3. Install Dependencies</h3>
    <p>Install all the required Python packages from the <code>requirements.txt</code> file.</p>
    <pre><code>pip install -r requirements.txt</code></pre>
    <p><em>(Note: You will need to create a <code>requirements.txt</code> file. You can generate one by running <code>pip freeze > requirements.txt</code> in your activated virtual environment after installing the packages listed in the Tech Stack.)</em></p>

    <h2>🚀 Usage</h2>
    
    <h3>1. Train the Model (Optional)</h3>
    <p>If you wish to retrain the model or experiment with different parameters, run the <code>machine learning on ott_viewer_dataset.ipynb</code> notebook from start to finish. This will regenerate the <code>.pkl</code> model files.</p>

    <h3>2. Run the Streamlit Application</h3>
    <p>To start the interactive web application, run the following command in your terminal:</p>
    <pre><code>streamlit run app.py</code></pre>
    <p>This will open the application in your default web browser, where you can input various parameters to get a predicted view completion rate.</p>

    <h2>🧠 Model Details</h2>
    <ul>
        <li><strong>Model</strong>: <code>RandomForestRegressor</code></li>
        <li><strong>Target Variable</strong>: <code>view_completion_rate</code> (normalized to be a value between 0 and 1).</li>
        <li><strong>Feature Selection</strong>: Recursive Feature Elimination (RFE) is used to select the top 15 most predictive features from the dataset.</li>
        <li><strong>Key Engineered Features</strong>: The model's performance is enhanced by creating time-based features like <code>Day_of_Week</code> and <code>Season</code> from the <code>watch_date</code>.</li>
    </ul>

</body>
</html>
