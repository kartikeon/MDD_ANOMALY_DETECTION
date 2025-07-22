# prepare_data.py
import pandas as pd

# Load raw files
data = pd.read_csv("data/secom.data", sep="\s+", header=None)
labels = pd.read_csv("data/secom_labels.data", sep="\s+", header=None)

# Rename columns
data.columns = [f"f{i}" for i in range(data.shape[1])]
data["label"] = labels[0]

# Save merged dataset
data.to_csv("data/secom_merged.csv", index=False)
print("✅ Data prepared and saved to data/secom_merged.csv")
