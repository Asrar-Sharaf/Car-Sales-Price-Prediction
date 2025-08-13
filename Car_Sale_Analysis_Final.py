
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


df = pd.read_csv("cars_sales_cleaned_For_deployment.csv")

st.set_page_config(page_title="Car Sales Dashboard", layout="wide")

# Sidebar Filters
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

# Setting Styling
st.markdown("""
    <style>
    .main {background-color: #F9F9F9;}
    h1, h2, h3, .css-10trblm {color: #2E8B57;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: 600; color: #2E8B57; background-color: #f0f0f0;}
    </style>
""", unsafe_allow_html=True)

# Title 
st.title("🚗 Car Sales Interactive Dashboard")
st.markdown("Explore key insights from the car sales dataset. Use the tabs below to navigate through different types of analysis.")

# Tabs names
tabs = st.tabs(["📈 Market Overview", "📊 Price Insights", "🔀 Multivariate Analysis"])


with tabs[0]:
    st.header("📈 Market Overview")

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
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



with tabs[1]:
    st.header("📊 Price Insights")

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Bivariate
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


with tabs[2]:
    st.header("🔀 Multivariate Analysis")

    if df.empty:
        st.warning("No data available for the selected filters.")
    else:

        numeric_cols = df.select_dtypes(include='number').columns
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, ax=ax)
        st.pyplot(fig)

        cat_cols= df.select_dtypes(include='O').columns
        for col in cat_cols:
            st.write(f"📌 Total Car Prices by '{col}'", (df.groupby(col)['price'].sum().sort_values(ascending=False)))



st.markdown("---")
st.caption("Developed by Asrar Sharaf")
