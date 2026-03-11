import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Air Quality Index Predictor",
    page_icon="🌫️",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
model = joblib.load("model/model.pkl")
df = pd.read_csv("data/clean_aqi.csv")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("AQI Dashboard")
st.sidebar.markdown("Machine Learning Air Quality Monitoring System")

st.sidebar.header("Input Pollution Levels")

pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 50.0)
pm10 = st.sidebar.slider("PM10", 0.0, 500.0, 80.0)
no2 = st.sidebar.slider("NO2", 0.0, 200.0, 40.0)
so2 = st.sidebar.slider("SO2", 0.0, 200.0, 10.0)
co = st.sidebar.slider("CO", 0.0, 20.0, 1.0)
o3 = st.sidebar.slider("O3", 0.0, 300.0, 25.0)

month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)

# -----------------------------
# TITLE
# -----------------------------
st.title("🌫 Air Quality Index Prediction System")
st.markdown("Predict AQI using a Machine Learning Model trained on pollution data.")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "📁 Dataset"])

# -----------------------------
# PREDICTION TAB
# -----------------------------
with tab1:

    col1, col2, col3 = st.columns(3)

    if st.button("Predict AQI"):

        features = np.array([[pm25, pm10, no2, so2, co, o3, month, day]])
        prediction = model.predict(features)[0]

        col1.metric("Predicted AQI", round(prediction,2))

        if prediction <= 50:
            col2.success("Air Quality: Good")
        elif prediction <= 100:
            col2.info("Air Quality: Moderate")
        elif prediction <= 150:
            col2.warning("Sensitive Groups")
        elif prediction <= 200:
            col2.error("Unhealthy")
        elif prediction <= 300:
            col2.error("Very Unhealthy")
        else:
            col2.error("Hazardous")

        col3.metric("PM2.5 Level", pm25)

    st.markdown("---")

    st.subheader("Input Pollution Parameters")

    data = {
        "Parameter": ["PM2.5","PM10","NO2","SO2","CO","O3"],
        "Value": [pm25,pm10,no2,so2,co,o3]
    }

    chart_df = pd.DataFrame(data)

    fig = px.bar(chart_df,x="Parameter",y="Value",color="Parameter")
    st.plotly_chart(fig,use_container_width=True)

# -----------------------------
# ANALYTICS TAB
# -----------------------------
with tab2:

    st.subheader("AQI Distribution")

    fig1 = px.histogram(df,x="AQI",nbins=40,color_discrete_sequence=["orange"])
    st.plotly_chart(fig1,use_container_width=True)

    st.subheader("Pollution Correlation Heatmap")

    corr = df.corr()

    fig2 = px.imshow(corr,text_auto=True,color_continuous_scale="RdBu_r")
    st.plotly_chart(fig2,use_container_width=True)

# -----------------------------
# DATA TAB
# -----------------------------
with tab3:

    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("Dataset Statistics")
    st.write(df.describe())
