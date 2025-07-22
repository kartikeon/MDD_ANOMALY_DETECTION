# train_model.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import joblib
import os

# Load dataset
df = pd.read_csv("data/secom_merged.csv")
X = df.drop("label", axis=1)

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(contamination=0.07, random_state=42)
model.fit(X_imputed)

# Save model and imputer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(imputer, "model/imputer.pkl")
print("✅ Model trained and saved.")
