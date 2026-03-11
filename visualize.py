import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/clean_aqi.csv")

# AQI distribution
sns.histplot(df["AQI"], bins=40)
plt.title("AQI Distribution")
plt.savefig("aqi_distribution.png")
plt.clf()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.savefig("correlation_heatmap.png")
plt.clf()

# Monthly AQI trend
df.groupby("month")["AQI"].mean().plot()
plt.title("Average AQI by Month")
plt.savefig("monthly_aqi_trend.png")

print("Visualizations saved")