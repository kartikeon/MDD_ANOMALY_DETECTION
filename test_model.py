# test_model.py
import pandas as pd
import joblib

# Load data and model
df = pd.read_csv("data/secom_merged.csv")
X = df.drop("label", axis=1)
model = joblib.load("model/model.pkl")
imputer = joblib.load("model/imputer.pkl")

# Impute and predict
X_imputed = imputer.transform(X)
df["predicted_label"] = model.predict(X_imputed)  # 1 = normal, -1 = anomaly

# Save predictions
df.to_csv("data/test_predictions.csv", index=False)
print("✅ Predictions saved to data/test_predictions.csv")
