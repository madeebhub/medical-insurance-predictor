# predict.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# ---------------- Load Data ----------------
DataFrame = pd.read_csv("data.csv")

# ---------------- Encode categorical variables ----------------
# These mappings must match the API
DataFrame['sex'] = DataFrame['sex'].map({'male': 1, 'female': 0})
DataFrame['smoker'] = DataFrame['smoker'].map({'yes': 1, 'no': 0})

# Features and target
FEATURES = ['age', 'bmi', 'children', 'smoker', 'sex']
X = DataFrame[FEATURES]
y = DataFrame['charges']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Save model
joblib.dump(model, "medical_charges_model.joblib")
print("✅ Model trained & saved successfully")
