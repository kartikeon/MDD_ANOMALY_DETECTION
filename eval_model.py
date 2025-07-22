# eval_model.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load predictions
df = pd.read_csv("data/test_predictions.csv")

# Convert labels
y_true = df["label"].map({-1: 0, 1: 1})          # 0 = normal, 1 = defect
y_pred = df["predicted_label"].map({1: 0, -1: 1})  # 1 = normal, -1 = anomaly

# Print evaluation metrics
print("🔍 Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# --- Visualization Section ---

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Defect"], yticklabels=["Normal", "Defect"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("data/confusion_matrix.png")
plt.show()

# Plot prediction distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette="pastel")
plt.title("Predicted Label Distribution")
plt.xlabel("Predicted Class (0=Normal, 1=Defect)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/predicted_distribution.png")
plt.show()
