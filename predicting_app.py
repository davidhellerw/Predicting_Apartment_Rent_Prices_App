import os
import streamlit as st
import pandas as pd
import joblib

# Load the RMSE DataFrame from the CSV file
rmse_df_path = 'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/models_performance.csv'

# Adding error handling for loading the CSV file
try:
    rmse_df = pd.read_csv(rmse_df_path)
except FileNotFoundError:
    st.error(f"RMSE file not found at {rmse_df_path}. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the RMSE file: {e}")
    st.stop()

# Define the global models and their corresponding states with explicit paths
global_model_states = {
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/global_model.pkl': ['AZ', 'CO', 'KS', 'KY', 'LA', 'MD', 'NE', 'NV', 'OH', 'OK', 'TN', 'TX', 'VA', 'WA'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/global_model_2.pkl': ['FL', 'SC'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/global_model_3_1.pkl': ['NM', 'WV', 'MO', 'NJ', 'IN', 'SD', 'WY', 'MT'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/global_model_4.pkl': ['AK', 'MS', 'VT'], 
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/global_model_5.pkl': ['DC', 'GA', 'OR'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/global_model_6.pkl': ['ID', 'MA', 'MN', 'WI'], 
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/combined_states_pipeline.pkl': ['IL', 'OR', 'PA', 'WI', 'RI', 'MI', 'MN', 'NY', 'DE', 'CT']
}

# Define the individual model states separately as they do not include the column 'state'
model_states = {
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/AR_model.pkl': ['AR'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/CA_model.pkl': ['CA'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/IA_model.pkl': ['IA'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/NC_model.pkl': ['NC'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/ND_model.pkl': ['ND'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/NH_model.pkl': ['NH'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/UT_model.pkl': ['UT'],
    'C:/Users/david/OneDrive/Desktop/ih-labs/Final Project/ME_model.pkl': ['ME'],
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

# Streamlit app
st.title("Apartment Rent Price Prediction")

# User inputs
bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
square_feet = st.number_input("Square Feet", min_value=0, step=100)
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

# Create a sorted list of unique states from both global_model_states and model_states
all_states = sorted(set(sum(global_model_states.values(), []) + sum(model_states.values(), [])))

state = st.selectbox("State", options=all_states)

# Add a button to trigger the prediction
if st.button("Predict"):
    # Load the correct model
    model, include_state = load_model(state)

    if model:
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
            # Predict rent price
            predicted_price = model.predict(input_df)[0]
            st.write(f"The predicted rent price is: ${predicted_price:.2f}")

            # Get RMSE for the selected state and display the message
            rmse = rmse_df[rmse_df['state'] == state]['RMSE'].values[0]
            st.write(f"Please note that this prediction can be off by approximately ${rmse:.2f} due to variations in the data.")
            st.write("The model was trained with data from 2019, so it may not reflect current market prices.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.write("No model available for the selected state.")