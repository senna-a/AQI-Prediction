import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/city_day.csv")
# Drop duplicates
df = df.drop_duplicates()

# Convert date column
df["Date"] = pd.to_datetime(df["Date"])

df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day
df["year"] = df["Date"].dt.year


# Remove rows with no target

df = df.dropna(subset=["AQI"])

# Select useful features
features = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "O3",
    "month",
    "day"
]

df = df[features + ["AQI"]]


# Handle missing values
imputer = SimpleImputer(strategy="median")
df[features] = imputer.fit_transform(df[features])


# Remove outliers using IQR
Q1 = df[features].quantile(0.25)
Q3 = df[features].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[features] < (Q1 - 1.5 * IQR)) |
          (df[features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature scaling
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Save clean dataset
df.to_csv("data/clean_aqi.csv", index=False)

print("Preprocessing complete")