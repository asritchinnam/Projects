# Model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Sample data
df = pd.DataFrame({
    "Age": [25, 30, 35, 40, 45],
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Education Level": ["Bachelor", "Master", "PhD", "Bachelor", "Master"],
    "Job Title": ["Developer", "Manager", "Data Scientist", "Analyst", "Engineer"],
    "Years of Experience": [2, 5, 10, 6, 7],
    "Salary": [500000, 800000, 1500000, 750000, 950000]
})

X = df.drop("Salary", axis=1)
y = df["Salary"]

# Preprocessing
categorical = ["Gender", "Education Level", "Job Title"]
numerical = ["Age", "Years of Experience"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical)
], remainder="passthrough")

# Build pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

# Save the model
joblib.dump(model, "salary_prediction_pipeline.pkl")
print("✅ Model saved as 'salary_prediction_pipeline.pkl'")
