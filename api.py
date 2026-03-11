from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("model/model.pkl")

@app.route("/")
def home():
    return "AQI Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        data["pm25"],
        data["pm10"],
        data["no2"],
        data["so2"],
        data["co"],
        data["o3"],
        data["month"],
        data["day"]
    ]])

    prediction = model.predict(features)

    return jsonify({"aqi": float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)