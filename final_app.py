import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="US Apartment Rent Price Prediction and Data Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load an image from a file
image_path = 'house_prediction_image.jpg'
image = Image.open(image_path)

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the RMSE DataFrame from the CSV file
rmse_df_path = 'models_performance_rmse.csv'

# Adding error handling for loading the CSV file
try:
    rmse_df = pd.read_csv(rmse_df_path)
except FileNotFoundError:
    st.error(f"RMSE file not found at {rmse_df_path}. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the RMSE file: {e}")
    st.stop()

# Load the ZIP code, latitude, and longitude DataFrame from the CSV file
zip_lat_long_path = 'zip_lat_long.csv'

# Adding error handling for loading the CSV file
try:
    zip_lat_long_df = pd.read_csv(zip_lat_long_path)
except FileNotFoundError:
    st.error(f"ZIP code file not found at {zip_lat_long_path}. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the ZIP code file: {e}")
    st.stop()

# Load the cleaned apartment data
apartment_data_path = 'cleaned_apartment_df.csv'

# Adding error handling for loading the CSV file
try:
    df = pd.read_csv(apartment_data_path)
except FileNotFoundError:
    st.error(f"Apartment data file not found at {apartment_data_path}. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the apartment data file: {e}")
    st.stop()

# Define the global models and their corresponding states with explicit paths
global_model_states = {
    'global_model.pkl': ['AZ', 'CO', 'KS', 'KY', 'LA', 'MD', 'NE', 'NV', 'OH', 'OK', 'TN', 'TX', 'VA', 'WA', 'HI', 'AL'],
    'global_model_2.pkl': ['FL', 'SC'],
    'global_model_3_1.pkl': ['NM', 'WV', 'MO', 'NJ', 'IN', 'SD', 'WY', 'MT'],
    'global_model_4.pkl': ['AK', 'MS', 'VT'], 
    'global_model_5.pkl': ['DC', 'GA', 'OR'],
    'global_model_6.pkl': ['ID', 'MA', 'WI'], 
    'combined_states_pipeline.pkl': ['IL', 'PA', 'RI', 'MI', 'MN', 'NY', 'DE', 'CT']
}

# Define the individual model states separately as they do not include the column 'state'
model_states = {
    'AR_model.pkl': ['AR'],
    'CA_model.pkl': ['CA'],
    'IA_model.pkl': ['IA'],
    'NC_model.pkl': ['NC'],
    'ND_model.pkl': ['ND'],
    'NH_model.pkl': ['NH'],
    'UT_model.pkl': ['UT'],
    'ME_model.pkl': ['ME'],
}

# Helper function to load the correct model based on the state
def load_model(state):
    for model_path, states in global_model_states.items():
        if state in states:
            if os.path.exists(model_path):
                return joblib.load(model_path), True
            else:
                st.error(f"Model file {model_path} not found.")
                return None, None
    for model_path, states in model_states.items():
        if state in states:
            if os.path.exists(model_path):
                return joblib.load(model_path), False
            else:
                st.error(f"Model file {model_path} not found.")
                return None, None
    return None, None

# Function to display analytics tab
def display_analytics(df):
    st.title('Data Analytics 📊')
    
    # Top 10 Average Rent Prices by State
    average_prices = df.groupby('state')['price'].mean().sort_values(ascending=False).head(10)
    
    # Format the DataFrame for display
    formatted_average_prices = average_prices.reset_index().rename(columns={'price': 'Average Rent Price ($)'}).round(2)
    formatted_average_prices['Average Rent Price ($)'] = formatted_average_prices['Average Rent Price ($)'].map("{:.2f}".format)
    
    st.subheader('Top 10 States by Monthly Average Rent Prices')
    # Adjust index from 0-based to 1-based for display
    formatted_average_prices.index = range(1, len(formatted_average_prices) + 1)
    st.table(formatted_average_prices)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=average_prices.values, y=average_prices.index, palette='viridis')
    plt.xlabel('Average Rent Price ($)')
    plt.ylabel('State')
    plt.title('Top 10 States by Average Rent Prices')

    # Pass the figure to st.pyplot()
    st.pyplot()

    # 10 States with the Lowest Average Rent Prices 
    states_with_lowest_prices = df.groupby('state')['price'].mean().sort_values().head(10)
    
    # Format the DataFrame for display
    formatted_states_with_lowest_prices = states_with_lowest_prices.reset_index().rename(columns={'price': 'Average Rent Price ($)'}).round(2)
    formatted_states_with_lowest_prices['Average Rent Price ($)'] = formatted_states_with_lowest_prices['Average Rent Price ($)'].map("{:.2f}".format)
    
    st.subheader('10 States with the Lowest Monthly Average Rent Prices')
    # Adjust index from 0-based to 1-based for display
    formatted_states_with_lowest_prices.index = range(1, len(formatted_states_with_lowest_prices) + 1)
    st.table(formatted_states_with_lowest_prices)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=states_with_lowest_prices.values, y=states_with_lowest_prices.index, palette='plasma')
    plt.xlabel('Average Rent Price ($)')
    plt.ylabel('State')
    plt.title('10 States with the Lowest Average Rent Prices')

    # Pass the figure to st.pyplot()
    st.pyplot()

    # Compare Rent Prices between Two States with Box Plot
    st.subheader('Compare Average Monthly Rent Prices and Distributions Between Two States')

    # Get unique states sorted alphabetically
    states_sorted = sorted(df['state'].unique())

    state1 = st.selectbox('Select State 1', options=states_sorted)
    state2 = st.selectbox('Select State 2', options=states_sorted)

    if st.button('Compare'):
        # Filter data for the selected states
        state1_data = df[df['state'] == state1]['price']
        state2_data = df[df['state'] == state2]['price']

        # Calculate average rent prices for selected states
        state1_avg_price = state1_data.mean()
        state2_avg_price = state2_data.mean()

        st.markdown(f"**Average Rent Price for {state1}:** ${state1_avg_price:.2f}")
        st.markdown(f"**Average Rent Price for {state2}:** ${state2_avg_price:.2f}")

        # Create a box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='state', y='price', data=df[df['state'].isin([state1, state2])], palette='Set2')
        plt.xlabel('State')
        plt.ylabel('Price')
        plt.title(f'Rent Price Distribution Comparison between {state1} and {state2}')
        plt.grid(True)
        st.pyplot()

# Function to flatten lists
def flatten_list(list_of_lists):
    flattened_list = [item for sublist in list_of_lists for item in sublist]
    return sorted(flattened_list)

# Function to validate ZIP code input
def validate_zip_code(zip_code):
    if not zip_code.isdigit() or len(zip_code) != 5:
        return False
    return True

def display_model_errors():
    st.title('Model Performance Metrics 📉')
    
    st.markdown("""
    The main objective of this project was to build models that result in the lowest possible Root Mean Squared Error (RMSE) for each of the states to enhance prediction accuracy. For this reason, 15 random forest regression models were built. Some models are dedicated to only one specific state, while other models are used for several states.

    **What is RMSE?**
    Root Mean Squared Error (RMSE) is a standard way to measure the error of a model in predicting quantitative data. Essentially, it represents the square root of the average of the squared differences between predicted values and actual values. The RMSE value tells us how concentrated the data is around the line of best fit.

    In the context of predicting apartment rent prices, an RMSE of $100 means that the average difference between the predicted rent prices and the actual rent prices is about $100. This implies that typically, the predictions made by the model can be expected to be off by $100. Therefore, a lower RMSE value is better as it indicates more accurate predictions by the model.
    """)

    if 'rmse_df' in globals():
        # Ensure RMSE values are rounded to two decimal places for display
        rmse_df['RMSE'] = rmse_df['RMSE'].round(2)

        # Adjusting the index to start at 1
        rmse_df.index = range(1, len(rmse_df) + 1)
        
        # Displaying the DataFrame as a table with formatted RMSE values
        st.subheader('Models RMSE')
        st.dataframe(rmse_df.style.format({'RMSE': '{:.2f}'}))
    else:
        st.error("Model performance data is not loaded. Please check the file path and try again.")

# Function to display the "About" section
def display_about():
    st.title("About This App")
    st.markdown("""
    Welcome to the Apartment Rent Price Prediction and Data Analytics app! This project is designed to provide insights into apartment rent prices across different states in the United States. Whether you're looking to predict rent prices based on apartment features or explore data analytics showing rent trends, this app has you covered.
                
    **Features:**
    - Predict rent prices based on number of bathrooms, bedrooms, square feet, ZIP code, and state.
    - View data analytics including top 10 states by average rent prices and 10 states with the lowest average rent prices.
    
    **Data Sources:**
    - [Apartment data](https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified): The dataset, provided by the UCI Machine Learning Repository, contains 100,000 rows and 22 columns. It was cleaned before building the ML models.
    - [ZIP code, latitude, and longitude](https://www.kaggle.com/datasets/joeleichter/us-zip-codes-with-lat-and-long): This dataset, from Kaggle, is used to map the selected zip code to its corresponding longitude and latitude.
    
    **Models Used:**
    - Random Forest regressors trained on historical apartment rent data. There are 15 models. Depending on the state selected by the user, the appropiate model is called to make the prediction.
    
                
    ### About the Author
     
    Hello! I'm David Heller, a data science enthusiast looking to enhance my data analysis skills. This app represents my journey into applying data-driven insights to real-world scenarios. I'm passionate about machine learning, statistical modeling, and business analytics. 
    """)
                
    # Adding About the Author section with image
    author_image_path = 'my_image.png' 
    author_image = Image.open(author_image_path)
    st.image(author_image, width=250)           
                
    st.markdown("""
                
    ### Connect with Me

    I invite you to connect with me on [LinkedIn](https://www.linkedin.com/in/david-heller-w/) to stay updated on my latest projects and insights. Let's explore how data science can empower us to solve complex challenges and achieve impactful outcomes together!
    
    For more of my work and projects, you can also check my [GitHub](https://github.com/davidhellerw).
    """)



# Streamlit app
def main():
    st.title("🏠US Apartment Rent Price Prediction and Data Analytics🏠")

    # Sidebar navigation with radio buttons for tabs
    page = st.sidebar.radio("Select a page", ["Predict Rent", "Data Analytics", "Predictive Model Accuracy", "About"])
    st.sidebar.image(image, caption='Image generated with ChatGPT-4.0', use_column_width=True)
    
    # Create empty space to push the footer to the bottom
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("Author: David Heller")

    if page == "Predict Rent":
        st.title("Predict Monthly Rent Price 🎯")
        st.subheader("Select the apartment's features:")
        # User inputs for prediction
        bathrooms = st.slider("Number of Bathrooms", min_value=1.0, max_value=6.0, step=0.5)
        bedrooms = st.slider("Number of Bedrooms", min_value=0, max_value=6, step=1)
        square_feet = st.number_input("Square Feet", min_value=0, step=100, value=1000)
        zip_code = st.text_input("ZIP Code")
        
        # Flatten the list of states
        all_states = flatten_list(list(global_model_states.values()) + list(model_states.values()))
        
        state = st.selectbox("State", options=all_states)

        # Validate ZIP code
        if not validate_zip_code(zip_code):
            st.warning("Please enter a valid 5-digit ZIP Code.")
            st.stop()
        
        if st.button("Predict Monthly Rent"):
            # Load the correct model
            model, include_state = load_model(state)

            if model:
                # Look up latitude and longitude based on ZIP code
                zip_info = zip_lat_long_df[zip_lat_long_df['ZIP'] == int(zip_code)]
                if zip_info.empty:
                    st.error("ZIP Code not found.")
                else:
                    latitude = zip_info['LAT'].values[0]
                    longitude = zip_info['LNG'].values[0]
                    
                    # Prepare input data
                    if include_state:
                        # Include the 'state' feature for global models
                        input_data = {'bathrooms': bathrooms, 'bedrooms': bedrooms, 'square_feet': square_feet, 'latitude': latitude, 'longitude': longitude, 'state': state}
                    else:
                        # Exclude the 'state' feature for individual models
                        input_data = {'bathrooms': bathrooms, 'bedrooms': bedrooms, 'square_feet': square_feet, 'latitude': latitude, 'longitude': longitude}

                    # Convert input data to DataFrame
                    input_df = pd.DataFrame([input_data])

                    try:
                        # Placeholder for prediction logic
                        predicted_price = model.predict(input_df)[0]
                        st.subheader(f"The predicted monthly rent price is: ${predicted_price:.2f}")

                        # Get RMSE for the selected state and display the message
                        rmse = rmse_df[rmse_df['state'] == state]['RMSE'].values[0]
                        st.write(f"Please note that this prediction can be off by approximately ${rmse:.2f} due to variations in the data.")
                        st.write("The model was trained with data from 2019, so it may not reflect current market prices.")

                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
            else:
                st.write("No model available for the selected state.")

    elif page == "Data Analytics":
        if 'df' in globals():
            display_analytics(df)  
        else:
            st.error("Apartment data is not loaded. Please check the file path and try again.")

    elif page == "Predictive Model Accuracy":
        display_model_errors()

    elif page == "About":
        display_about()

if __name__ == '__main__':
    main()
