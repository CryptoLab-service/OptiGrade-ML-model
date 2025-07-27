import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Define full set of feature names (MATCHING PREDICTION)
feature_names = [
    'GPA_last_semester',
    'credit_load',
    'current_CGPA',
    'study_hours',
    'attendance',  # Changed from 'Attendance' to match prediction
    'engagement',  # Changed from 'Lecture_Engagement'
    'midterm_score'  # Changed from 'Midterm_Score'
]

# Load training dataset
df = pd.read_csv("data/training_data.csv")

# Columns feature names
column_mapping = {
    'Attendance': 'attendance',
    'Lecture_Engagement': 'engagement',
    'Midterm_Score': 'midterm_score'
}
df = df.rename(columns=column_mapping)

# Required features
for feature in feature_names:
    if feature not in df.columns:
        df[feature] = 0.0  # Add missing features with default value

# Features and target
X = df[feature_names]
y = df["target_CGPA"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Confirming models directory
os.makedirs("models", exist_ok=True)

# Model's location - Save both model and feature names
joblib.dump({
    'model': model,
    'feature_names': feature_names
}, "models/model.pkl")

print(f"✅ Model trained with features: {feature_names}")
print("✅ Model saved as models/model.pkl")
