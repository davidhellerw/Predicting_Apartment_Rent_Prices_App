<h1>Apartment Rent Price Prediction App</h1>

<h2>Overview</h2>
<p>This Streamlit app is the culmination of an extensive data science project aimed at predicting monthly rent prices for apartments across various states in the U.S. The project was designed not only to enhance machine learning skills but also to create a functional application that provides real value to users looking to understand real estate pricing dynamics. From initial data gathering to deploying a fully functional app, this project involved detailed data cleaning, comprehensive exploratory data analysis, and the application of numerous machine learning models to identify the one that best predicts rental prices.</p>

<a href="https://apartment-monthly-rent.streamlit.app/">App link</a>

<h2>Project Complexity</h2>
<p>The development of this application involved numerous stages, each requiring significant investment in time and effort:</p>
<ul>
  <li><strong>Data Cleaning:</strong> The raw data was meticulously cleaned, involving handling missing values, removing outliers, and transforming variables to ensure quality inputs for modeling.</li>
  <li><strong>Exploratory Data Analysis (EDA):</strong> An in-depth EDA was conducted to understand the underlying patterns and trends, which informed the feature selection and modeling strategy.</li>
  <li><strong>Model Experimentation:</strong> Various models were tested including linear regression, decision trees, and ensemble methods. The model experimentation phase was iterative, involving tuning and validation to optimize performance.</li>
  <li><strong>Model Selection:</strong> The final selection of 15 Random Forest models, some state-specific and others more general, was based on their RMSE performance, with a relentless focus on achieving the lowest possible error rates.</li>
</ul>

<h2>Features</h2>
<ul>
  <li><strong>Rent Price Prediction:</strong> Allows users to input apartment features to predict rental prices.</li>
  <li><strong>Data Analytics:</strong> Allows users to compare rental price distributions and averages across states.</li>
  <li><strong>Model Performance Metrics:</strong> Displays the RMSE of each model, emphasizing the accuracy of the predictions.</li>
</ul>

<h2>Project Structure</h2>
<pre>
predicting_apartment_rent_prices_app/
│
├── final_app.py                        # Main application script for Streamlit
├── requirements.txt                    # Dependencies to run the app
├── models_performance_rmse.csv         # Data on model performance
├── README.md                           # Documentation of the project
├── data/
│   ├── cleaned_apartment_df.csv        # Cleaned dataset used for predictions
│   └── zip_lat_long.csv                # ZIP code data for location analysis
├── notebooks/
│   ├── building_ml_models.ipynb        # Notebook for building ML models
│   ├── data_cleaning.ipynb             # Notebook for data cleaning process
│   ├── evaluating_models.ipynb         # Notebook for model evaluation
│   └── experimenting_with_ml_models.ipynb  # Notebook for ML model experimentation
└── models/
    ├── AR_model.pkl                    # Model for Arkansas
    ├── CA_model.pkl                    # Model for California
    ├── IA_model.pkl                    # Model for Iowa
    ├── ME_model.pkl                    # Model for Maine
    ├── NC_model.pkl                    # Model for North Carolina
    ├── ND_model.pkl                    # Model for North Dakota
    ├── NH_model.pkl                    # Model for New Hampshire
    ├── UT_model.pkl                    # Model for Utah
    ├── global_model*.pkl               # Model for various states
    └── combined_states_pipeline.pkl    # Model for various states
</pre>

<h2>Setup & Installation</h2>
<ol>
  <li><strong>Clone the Repository:</strong>
    <pre>git clone https://github.com/your-username/predicting_apartment_rent_prices_app.git
cd predicting_apartment_rent_prices_app</pre>
  </li>
  <li><strong>Create a Virtual Environment (Optional but recommended):</strong>
    <pre>python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`</pre>
  </li>
  <li><strong>Install Dependencies:</strong>
    <pre>pip install -r requirements.txt</pre>
  </li>
  <li><strong>Run the Application:</strong>
    <pre>streamlit run final_app.py</pre>
  </li>
</ol>

<h2>Usage</h2>
<p>After running the app, visit <a href="http://localhost:8501">http://localhost:8501</a> in your web browser to access the app. Follow the on-screen instructions to interact with the application.</p>

<h2>Data Sources</h2>
<ul>
  <li>Data provided by <a href="https://archive.ics.uci.edu/ml/index.php">UCI Machine Learning Repository</a></li>
  <li>ZIP code data from <a href="https://www.kaggle.com">Kaggle</a></li>
</ul>
