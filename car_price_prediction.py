
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load necessary files
voting_model = joblib.load('Car_Sales_price_ML_Model.h5') # This is the trained model that predicts price and it includes the scaling numerical features, encoding categorical features
input_features = joblib.load('input_features.h5') # A list of column names used in the model input

# Dropdown lists for select boxes
fuel_type_list = joblib.load('fuel_type_list.h5')
transmission_list = joblib.load('transmission_list.h5')
drivetrain_list = joblib.load('drivetrain_list.h5')
exterior_color_list = joblib.load('exterior_color_list.h5')
interior_color_list = joblib.load('interior_color_list.h5')
manufacturer_list = joblib.load('manufacturer_list.h5')

# Prediction function
def predict(condition, mileage_mi, state, model_year, manufacturer, fuel_type, drivetrain, transmission,
            exterior_color, interior_color, accidents_or_damage, one_owner_vehicle): #This is a function you call when the user submits input
    test_df = pd.DataFrame(columns=input_features) # Creates an empty DataFrame with the same columns your model expects. It is like a blank sheet of paper with all the correct column names.

    test_df.at[0, 'condition'] = condition
    test_df.at[0, 'mileage_mi'] = mileage_mi
    test_df.at[0, 'state'] = state
    test_df.at[0, 'model_year'] = model_year
    test_df.at[0, 'manufacturer'] = manufacturer
    test_df.at[0, 'fuel_type'] = fuel_type
    test_df.at[0, 'drivetrain'] = drivetrain
    test_df.at[0, 'transmission'] = transmission
    test_df.at[0, 'exterior_color'] = exterior_color
    test_df.at[0, 'interior_color'] = interior_color
    test_df.at[0, 'accidents_or_damage'] = accidents_or_damage
    test_df.at[0, '1_owner_vehicle'] = one_owner_vehicle

    price = voting_model.predict(test_df) # Predict the car sales prices for the test dataset using the trained Voting Regressor model
    predicted_price = round(np.exp(price[0]), 2)
    return predicted_price # Return the predicted car sales price

# Streamlit app

# This function does the following:
# Creates a web page using Streamlit
# Adds sliders and dropdowns for user inputs
# When the user clicks Predict, it:
    # Calls the predict() function you built earlier
    # Displays the price prediction

def main():
    st.title('🚗 Car Price Prediction App')
    st.markdown("Predict the price of a car based on its features.")
    st.subheader("Enter Car Details:")

    condition = st.selectbox('Condition', ['New', 'Used', 'Certified Pre-Owned (CPO)'])
    mileage_mi = st.slider('Mileage (mi)', min_value=0, max_value=163000, step=1000, value=0)
    state = st.selectbox('State', ['Illinois', 'Indiana'])
    model_year = st.slider('Model Year', min_value=1982, max_value=2025, step=1, value=2000)
    manufacturer = st.selectbox('Manufacturer', manufacturer_list)
    fuel_type = st.selectbox('Fuel Type', fuel_type_list)
    drivetrain = st.selectbox('Drivetrain', drivetrain_list)
    transmission = st.selectbox('Transmission', transmission_list)
    exterior_color = st.selectbox('Exterior Color', exterior_color_list)
    interior_color = st.selectbox('Interior Color', interior_color_list)
    accidents_or_damage = st.selectbox('Accidents or Damage', ['No accidents/damage', 'had accident/damage'])
    one_owner_vehicle = st.selectbox('Ownership', ['First owner', 'Multiple owners'])

    if st.button("Predict Car Price"):
        predicted_price = predict(condition, mileage_mi, state, model_year, manufacturer, fuel_type,
                                  drivetrain, transmission, exterior_color, interior_color,
                                  accidents_or_damage, one_owner_vehicle)

        st.success(f"💰 Estimated Car Price is: **${predicted_price:,}**")

if __name__ == '__main__':
    main()
