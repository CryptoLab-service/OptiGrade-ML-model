# File: models/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load training dataset
df = pd.read_csv("data/training_data.csv")

# Features and target
X = df[["credit_load", "study_hours", "GPA_last_semester", "current_CGPA"]]
y = df["target_CGPA"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("✅ Model trained and saved as models/model.pkl")
