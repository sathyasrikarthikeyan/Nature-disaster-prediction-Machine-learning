Natural disaster prediction aims to leverage machine learning (ML) to forecast events like earthquakes, floods, hurricanes, and wildfires. By analyzing historical data, satellite imagery, weather patterns, and geological signals, ML models can identify trends and provide early warnings.

Key components of an ML-based prediction system include:

Data Collection – Sources include seismic sensors, weather stations, satellite imagery, and historical disaster records.
Feature Engineering – Extracting key factors like temperature, humidity, pressure, wind speed, and seismic activity.
Model Selection – Common models include Random Forest, LSTMs, CNNs (for image-based data), and XGBoost for tabular data.
Training & Validation – Using historical data to train models and validate accuracy with real-world events.
Real-Time Prediction & Alerts – Deploying models in cloud environments to process incoming data and trigger alerts.
Challenges include data quality, model interpretability, and false alarms. However, ML-driven disaster prediction enhances preparedness, minimizes damage, and saves lives by enabling proactive decision-making


# python code

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = "/earthquake_data_with_additional_features.csv"
df = pd.read_excel(file_path, engine="openpyxl")

# Selecting relevant numerical features
features = ["depth", "Energy Released (J)", "Population Density", "Proximity to Fault Lines (km)", "Seismic Activity Index", "Historical Earthquake Count"]
target = "mag."

# Splitting data into training and testing sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Visualization of predictions vs actual values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("Actual vs Predicted Earthquake Magnitude")

# Add linear regression line
x_vals = np.linspace(min(y_test), max(y_test), 100)
y_vals = x_vals  # Ideal prediction line
plt.plot(x_vals, y_vals, color='red', linestyle='dashed', label='Ideal Fit')
plt.legend()
plt.show()

# Feature importance
importance = pd.Series(model.coef_, index=features).sort_values()
plt.figure(figsize=(8, 6))
importance.plot(kind='barh', color='skyblue')
plt.xlabel("Coefficient Value")
plt.title("Feature Importance in Predicting Magnitude")
plt.show()
