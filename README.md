# 🏭 Manufacturing Defect Detection using Anomaly Detection

This project applies **unsupervised anomaly detection** to identify defective products in a semiconductor manufacturing process using the **SECOM dataset**. It uses the **Isolation Forest** algorithm from scikit-learn to learn the "normal" sensor behavior and flag outliers as potential defects.

---

## 📌 What is Anomaly Detection?

**Anomaly Detection** refers to identifying patterns in data that do not conform to expected behavior. These patterns, or "anomalies", may indicate:

- Defective manufacturing processes  
- Fraudulent activity  
- Sensor malfunctions  
- Rare but critical events  

In this project, we apply anomaly detection to **detect faulty production runs** based solely on sensor data.

---

## 🗃 Dataset: SECOM

- Real-world dataset from a semiconductor manufacturing line  
- 1567 samples (production runs)  
- 590 sensor measurements per sample  
- Each sample labeled as:
  - `-1` → Normal (non-defective)
  - `1` → Defective

---

## ⚙️ Project Structure

```
manufacturing_defect_detection/
├── data/
│   ├── secom.data
│   ├── secom_labels.data
│   ├── secom_merged.csv
│   ├── test_predictions.csv
├── model/
│   ├── model.pkl
│   ├── imputer.pkl
├── prepare_data.py
├── train_model.py
├── test_model.py
├── eval_model.py
├── README.md
```

---

## 🚀 Getting Started

### 🔧 Step 1: Clone the Repository

Download or clone this project folder.

---

### 📥 Step 2: Download the Dataset

Download the following files from [UCI SECOM Repository](https://archive.ics.uci.edu/ml/datasets/SECOM):

- `secom.data`  
- `secom_labels.data`  

Place them inside the `data/` folder.

---

### 📦 Step 3: Install Requirements

```bash
pip install pandas scikit-learn joblib matplotlib seaborn
```

---

### 🛠 Step 4: Prepare the Data

```bash
python prepare_data.py
```

This merges `secom.data` and `secom_labels.data` into `secom_merged.csv`.

---

### 🤖 Step 5: Train the Model

```bash
python train_model.py
```

This trains an Isolation Forest model and saves it to the `model/` folder.

---

### 🧪 Step 6: Test the Model

```bash
python test_model.py
```

This generates predictions and stores them in `data/test_predictions.csv`.

---

### 📊 Step 7: Evaluate the Model

```bash
python eval_model.py
```

This prints classification metrics and shows visual plots:
- Confusion Matrix (as heatmap)
- Predicted Label Distribution

---

## 📈 Sample Output

### Confusion Matrix Example

```
                 Predicted
               Normal  Defect
Actual Normal     1368     95
       Defect       89     15
```

### F1 Score (Defect Class)

Approximately **14%**, which shows the model is good at detecting normal cases but struggles with defect recall — a known limitation of unsupervised models.

---

## 📚 References

- UCI SECOM Dataset: [https://archive.ics.uci.edu/ml/datasets/SECOM](https://archive.ics.uci.edu/ml/datasets/SECOM)
- Scikit-learn Isolation Forest: [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)


