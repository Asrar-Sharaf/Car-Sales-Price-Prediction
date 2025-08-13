
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Set wide layout
st.set_page_config(page_title="Car Sales Dashboard", layout="wide")

# Load data
df = pd.read_csv("cars_sales_cleaned_For_deployment.csv")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter the Data")

fuel_filter = st.sidebar.selectbox('Select Fuel Type', ["All"] + df['fuel_type'].dropna().unique().tolist())
condition_filter = st.sidebar.selectbox('Select Condition', ["All"] + df['condition'].dropna().unique().tolist())
transmission_filter = st.sidebar.selectbox('Select Transmission', ["All"] + df['transmission'].dropna().unique().tolist())
drivetrain_filter = st.sidebar.selectbox('Select Drivetrain', ["All"] + df['drivetrain'].dropna().unique().tolist())
exterior_color_filter = st.sidebar.selectbox('Select Exterior Color', ["All"] + df['exterior_color'].dropna().unique().tolist())
manufacturer_filter = st.sidebar.selectbox('Select Manufacturer', ["All"] + df['manufacturer'].dropna().unique().tolist())

# Apply filters
df = df.copy()
if fuel_filter != "All":
    df = df[df['fuel_type'] == fuel_filter]
if condition_filter != "All":
    df = df[df['condition'] == condition_filter]
if transmission_filter != "All":
    df = df[df['transmission'] == transmission_filter]
if drivetrain_filter != "All":
    df = df[df['drivetrain'] == drivetrain_filter]
if exterior_color_filter != "All":
    df = df[df['exterior_color'] == exterior_color_filter]
if manufacturer_filter != "All":
    df = df[df['manufacturer'] == manufacturer_filter]

# Styling
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    h1, h2, h3, .css-10trblm {color: #2E8B57;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: 600; color: #2E8B57; background-color: #f0f0f0;}
    </style>
""", unsafe_allow_html=True)

# Tabs
st.title("🚗 Car Sales Dashboard with ML Price Predictor")
tabs = st.tabs(["📈 Market Overview", "📊 Price Insights", "🔀 Multivariate Analysis", "🤖 Price Prediction"])

# TAB 1: Market Overview
with tabs[0]:
    st.header("📈 Market Overview")

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Univariate Analysis visual codes
        st.subheader("Distribution of Car Prices")
        price_dist_fig = ff.create_distplot([df['price']], group_labels=['Price'], colors=['#2E8B57'], show_hist=True, bin_size=10000)
        price_dist_fig.update_layout(width=750, height=500)
        st.plotly_chart(price_dist_fig, use_container_width=True)

        st.subheader("condition Category")
        condition_fig = px.pie(df, names='condition', title='condition Category', color_discrete_sequence=px.colors.sequential.Greens, width=600, height=400)
        condition_fig.update_traces(textinfo='percent+label')
        st.plotly_chart(condition_fig)

        st.subheader("Most Common Manufacturers")
        manufacturers_fig = px.histogram(df, x="manufacturer", title="Most Common Manufacturers", color_discrete_sequence=['#2E8B57'], nbins=30, width=700, height=500, marginal='violin')
        st.plotly_chart(manufacturers_fig, use_container_width=True)

        st.subheader("Fuel Type Distribution")
        fuel_types_fig = px.bar(df, x='fuel_type', color='fuel_type', title='Distribution of Fuel Types', color_discrete_sequence=["#2E8B57"])
        st.plotly_chart(fuel_types_fig, use_container_width=True)

        st.subheader("Exterior Color Preferences")
        exterior_color_fig = px.bar(df, x='exterior_color', title='Top Exterior Colors', color='exterior_color', color_discrete_sequence=["#2E8B57"], width=750, height=500)
        st.plotly_chart(exterior_color_fig, use_container_width=True)

        st.subheader("Drivetrain Types")
        drivetrain_fig = px.pie(df, names='drivetrain', title='Drivetrain Types', color_discrete_sequence=px.colors.sequential.Greens, width=600, height=400)
        drivetrain_fig.update_traces(textinfo='percent+label')
        st.plotly_chart(drivetrain_fig)

        st.subheader("Transmission Types")
        transmission = df['transmission'].value_counts().reset_index()
        transmission.columns = ['transmission', 'count']
        transmission_fig = px.line(transmission, x='transmission', y='count', title='Count of Vehicles by Transmission Type', markers=True, color_discrete_sequence=['#2E8B57'], width=750, height=500)
        st.plotly_chart(transmission_fig, use_container_width=True)

        st.subheader("Distribution of Cars Model Year")
        model_year_counts = df['model_year'].value_counts().sort_index().reset_index()
        model_year_fig = px.bar(model_year_counts, x='model_year', y='count', title='Number of Cars by Model Year', color_discrete_sequence=['#2E8B57'], width=750, height=500)
        st.plotly_chart(model_year_fig, use_container_width=True)


# TAB 2: Price Insight

with tabs[1]:
    st.header("📊 Price Insights")

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Bivariate Analysis visual codes
        st.subheader("Total Manufacturers Prices")
        manufacturer_pri_fig = px.bar(df, x="manufacturer", y="price", color="manufacturer", title="Manufacturers Prices", color_discrete_sequence=px.colors.sequential.Greens, width=1000, height=500)
        st.plotly_chart(manufacturer_pri_fig, use_container_width=True)

        st.subheader("Total Price by Vehicle Condition")
        price_by_condition = df.groupby("condition")["price"].sum().round(2).reset_index()
        price_by_condition_fig = px.pie(price_by_condition, names='condition', values='price', color_discrete_sequence=px.colors.sequential.Greens)
        price_by_condition_fig.update_traces(textinfo='percent+label')
        st.plotly_chart(price_by_condition_fig)

        st.subheader("Total Price by Drivetrain")
        sum_price_by_drivetrain = df.groupby("drivetrain")["price"].sum().reset_index().sort_values(by='price', ascending=False)
        sum_price_by_drivetrain_fig = px.line(sum_price_by_drivetrain, x='drivetrain', y='price', title='Car Prices by Drivetrain', markers=True, color_discrete_sequence=["#2E8B57"], width=750, height=500)
        st.plotly_chart(sum_price_by_drivetrain_fig)

        st.subheader("Top 10 Car Prices by Model Year")
        sum_price_by_year_model = df.groupby("model_year")["price"].sum().reset_index().sort_values(by='price', ascending=False).head(10)
        sum_price_by_year_model_fig = px.bar(sum_price_by_year_model, x="model_year", y="price", text="price", color_discrete_sequence=["#2E8B57"])
        st.plotly_chart(sum_price_by_year_model_fig, use_container_width=True)

        st.subheader("Total Price by Fuel Type")
        fuel_pri = df.groupby("fuel_type")["price"].sum().reset_index()
        fuel_pri_fig = px.pie(fuel_pri, names='fuel_type', values='price', color_discrete_sequence=px.colors.sequential.Greens)
        st.plotly_chart(fuel_pri_fig)

        st.subheader("Top Exterior Colors by Total Price")
        price_by_colors = df.groupby("exterior_color")["price"].sum().reset_index().sort_values(by='price', ascending=False).head(5)
        price_by_colors_fig = px.scatter(price_by_colors, x="exterior_color", y="price", text="price", title="Top Exterior Colors and Their Total Prices", color="exterior_color", color_discrete_sequence=px.colors.sequential.Greens, width=750, height=500, labels={"exterior_color": "Exterior Color", "price": "Total Price"})
        price_by_colors_fig.update_traces(textposition="top center", marker=dict(size=15, line=dict(width=2, color="DarkSlateGrey")))
        st.plotly_chart(price_by_colors_fig, use_container_width=True)

        st.subheader("Total Prices of Transmission Types")
        price_by_transmission= df.groupby("transmission")["price"].sum().reset_index().sort_values(by='price', ascending=False)
        price_by_transmission_fig = px.scatter(price_by_transmission, x="transmission", y="price", text="price", title="Top Exterior Colors and Their Total Prices", color="transmission", color_discrete_sequence=px.colors.sequential.Greens, width=750, height=500, labels={"transmission": "Transmission Types", "price": "Price"})
        price_by_transmission_fig.update_traces(textposition="top center", marker=dict(size=15, line=dict(width=2, color="DarkSlateGrey")))
        st.plotly_chart(price_by_transmission_fig, use_container_width=True)

        st.subheader("Price by Drivetrain & Fuel Type")
        price_by_drivetrain_fuel_type = df.groupby(['drivetrain', 'fuel_type'])['price'].sum().reset_index().sort_values(by='price', ascending=False)
        price_by_drivetrain_fuel_type_fig = px.bar(price_by_drivetrain_fuel_type, x='drivetrain', y='price', color='fuel_type', barmode='group', color_discrete_sequence=['#2E8B57', '#FFA500', '#F5F5DC', '#808080'])
        st.plotly_chart(price_by_drivetrain_fuel_type_fig, use_container_width=True)

        st.subheader("Total Price by Manufacturer & Condition")
        manufacturer_price_by_cond = df.groupby(["manufacturer", "condition"])["price"].sum().reset_index().sort_values(by='price', ascending=False).head(10)
        manufacturer_price_by_cond_fig = px.histogram(manufacturer_price_by_cond, x='manufacturer', y='price', color='condition', barmode='group', color_discrete_sequence=["#2E8B57", "#A9DFBF"])
        st.plotly_chart(manufacturer_price_by_cond_fig, use_container_width=True)

# TAB 3: Multivariate Analysis
with tabs[2]:
    st.header("🔀 Multivariate Analysis")

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Multivariate Analysis 
        numeric_cols = df.select_dtypes(include='number').columns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
        st.pyplot(fig)

        cat_cols= df.select_dtypes(include='O').columns
        for col in cat_cols:
            st.write(f"📌 Total Car Prices by '{col}'", (df.groupby(col)['price'].sum().sort_values(ascending=False)))


# TAB 4: ML Prediction (voting model (xgb and ridge))
with tabs[3]:

    # Load ML model and features
    voting_model = joblib.load('Car_Sales_price_ML_Model.h5')  # This is the trained model that predicts price and it includes the scaling numerical features, encoding categorical features
    input_features = joblib.load('input_features.h5') # A list of column names used in the model input

    # Dropdown lists for select boxes
    fuel_type_list = joblib.load('fuel_type_list.h5')
    transmission_list = joblib.load('transmission_list.h5')
    drivetrain_list = joblib.load('drivetrain_list.h5')
    exterior_color_list = joblib.load('exterior_color_list.h5')
    interior_color_list = joblib.load('interior_color_list.h5')
    manufacturer_list = joblib.load('manufacturer_list.h5')

    # Prediction function
    def predict(condition, mileage_mi, state, model_year, manufacturer, fuel_type, drivetrain, transmission, exterior_color, interior_color, accidents_or_damage, one_owner_vehicle):  #This is a function you call when the user submits input
        test_df = pd.DataFrame(columns=input_features) # Creates an empty DataFrame with the same columns. It is like a blank sheet of paper with all the correct column names.
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
        st.subheader("Please Enter Car Details:")

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

    main()         
# Footer
st.markdown("---")
st.caption("Developed by Asrar Sharaf")
