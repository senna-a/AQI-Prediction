#  Air Quality Index Prediction System

A Machine Learning-based Air Quality Index (AQI) Prediction System built using Python, Scikit-learn, Flask, and Streamlit.

This project analyzes pollution data, trains regression models, visualizes AQI trends, and predicts AQI levels based on pollutant concentrations.

---

##  Features

- AQI Prediction using Machine Learning
- Data Preprocessing Pipeline
- Outlier Removal & Feature Scaling
- Multiple ML Model Comparison
- Interactive Streamlit Dashboard
- REST API using Flask
- Data Visualization & Correlation Analysis
- Pollution Analytics Dashboard

---

##  Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Flask
- Plotly
- Matplotlib
- Seaborn

---

## Project Structure

```bash
AQI-Prediction-System/
│
├── app.py
├── api.py
├── preprocess.py
├── train_model.py
├── visualize.py
├── requirements.txt
│
├── data/
│   ├── city_day.csv
│   └── clean_aqi.csv
│
├── model/
│   └── model.pkl
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AQI-Prediction.git
cd AQI-Prediction
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

##  Model Training

Run preprocessing:

```bash
python preprocess.py
```

Train the model:

```bash
python train_model.py
```

---

##  Visualizations

Generate graphs and heatmaps:

```bash
python visualize.py
```

### Feature Correlation Heatmap

<img width="640" height="480" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/b8acb3a0-6bc5-48c3-992e-79ddb2dc6cdc" />

### Monthly AQI Trend
<img width="640" height="480" alt="monthly_aqi_trend" src="https://github.com/user-attachments/assets/444d5301-c528-4f5e-a579-25201bdb5edb" />


<img width="640" height="480" alt="aqi_distribution" src="https://github.com/user-attachments/assets/e642cd40-ae03-4f73-8509-fc941b7a8183" />


---

##  Streamlit Dashboard

Run the dashboard:

```bash
streamlit run app.py
```

Features:
- Real-time AQI prediction
- Interactive pollution sliders
- AQI analytics
- Dataset preview

---

##  Flask API

Run API server:

```bash
python api.py
```

### API Endpoint

```http
POST /predict
```

### Sample Request

```json
{
  "pm25": 50,
  "pm10": 80,
  "no2": 40,
  "so2": 10,
  "co": 1,
  "o3": 25,
  "month": 6,
  "day": 15
}
```

### Sample Response

```json
{
  "aqi": 112.45
}
```

---

##  Requirements

```txt
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
streamlit
joblib
```

---

##  Future Improvements

- Deploying using Docker
- Adding Deep Learning models
- Real-time AQI integration
- City-wise forecasting
- Mobile responsive dashboard

---

##  Author

Senna Krishna 


---

##  Support

If you found this project useful, give it a star, Thank You.
