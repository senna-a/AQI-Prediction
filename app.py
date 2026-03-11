import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(
    page_title="AQI ML Dashboard",
    page_icon="🌫",
    layout="wide"
)

# Load model
model = joblib.load("model/model.pkl")

# Load dataset
df = pd.read_csv("data/clean_aqi.csv")

st.title("🌫 Air Quality Index Prediction Dashboard")
st.write("Machine Learning model to predict AQI using pollution parameters.")

# -----------------------
# PREDICTION SECTION
# -----------------------

st.header("AQI Prediction")

col1, col2 = st.columns(2)

with col1:

    pm25 = st.number_input("PM2.5", 0.0, 500.0)
    pm10 = st.number_input("PM10", 0.0, 500.0)
    no2 = st.number_input("NO2", 0.0, 200.0)
    so2 = st.number_input("SO2", 0.0, 200.0)
    co = st.number_input("CO", 0.0, 20.0)
    o3 = st.number_input("O3", 0.0, 300.0)

with col2:

    month = st.slider("Month", 1, 12)
    day = st.slider("Day", 1, 31)

if st.button("Predict AQI"):

    features = np.array([[pm25, pm10, no2, so2, co, o3, month, day]])

    prediction = model.predict(features)[0]

    st.subheader(f"Predicted AQI: {prediction:.2f}")

# -----------------------
# VISUALIZATION SECTION
# -----------------------

st.header("Dataset Visualizations")

col3, col4 = st.columns(2)

# AQI Distribution
with col3:

    st.subheader("AQI Distribution")

    fig, ax = plt.subplots()

    sns.histplot(df["AQI"], bins=30, kde=True, ax=ax)

    st.pyplot(fig)

# Correlation Heatmap
with col4:

    st.subheader("Pollution Correlation Heatmap")

    fig2, ax2 = plt.subplots()

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)

    st.pyplot(fig2)

# -----------------------
# DATA PREVIEW
# -----------------------

st.header("Dataset Preview")

st.dataframe(df.head(20))