<h1>Apartment Rent Price Prediction App</h1>

<h2>Overview</h2>
<p>This Streamlit app predicts the monthly rent prices of apartments across various states in the U.S. The main goals are to enhance machine learning skills and to provide a real app that can predict rental prices accurately.</p>

<h2>Features</h2>
<ul>
  <li><strong>Rent Price Prediction:</strong> Users can input features of an apartment like number of bedrooms, bathrooms, and location to get predicted rent prices.</li>
  <li><strong>Data Analytics:</strong> Visual analytics showing the distribution of apartment prices across different states.</li>
  <li><strong>Model Performance Metrics:</strong> Displays the accuracy of the predictive models used in terms of Root Mean Squared Error (RMSE).</li>
</ul>

<h2>How It Works</h2>
<p>The app utilizes 15 Random Forest regression models to predict rent prices. Some models are tailored for specific states, while others are general models used across multiple states. The RMSE metric is used to measure the prediction error, with a lower RMSE indicating higher model accuracy.</p>

<h2>What is RMSE?</h2>
<p>Root Mean Squared Error (RMSE) is a standard metric used to measure the accuracy of a model in predicting quantitative data. Essentially, it represents the square root of the average of the squared differences between predicted values and actual values. In this app, an RMSE of $100 means the average difference between the predicted and actual rent prices is about $100, indicating the model's prediction accuracy.</p>

<h2>Project Structure</h2>
<pre>
predicting_apartment_rent_prices_app/
│
├── final_app.py          # Main application script for Streamlit
├── requirements.txt      # Dependencies to run the app
├── models_performance_rmse.csv   # Data on model performance
├── README.md             # Documentation of the project
└── data/
    ├── cleaned_apartment_df.csv # Cleaned dataset used for predictions
    └── zip_lat_long.csv         # ZIP code data for location analysis
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

<h2>Data Sources </h2>
<ul>
  <li>Data provided by <a href="https://archive.ics.uci.edu/ml/index.php">UCI Machine Learning Repository</a></li>
  <li>ZIP code data from <a href="https://www.kaggle.com">Kaggle</a></li>
</ul>
