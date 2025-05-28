# Task 3: Fraud Detection System

## 📌 Description
This project aims to build a machine learning-based fraud detection system using the Credit Card Fraud Detection dataset. The dataset is highly imbalanced and includes transactions labeled as fraud (1) or not fraud (0). We address this imbalance using SMOTE and train a Random Forest classifier to detect fraudulent activities.

---

## 📊 Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: Anonymized numeric features (`V1` to `V28`), `Amount`, and `Class`
- Target: `Class` (1 = Fraud, 0 = Not Fraud)

---

## 🔍 Steps Followed

### 1. **Data Preprocessing**
- Dropped unnecessary columns (`Time`)
- Scaled the `Amount` feature using `StandardScaler`

### 2. **Handling Class Imbalance**
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of fraudulent transactions

### 3. **Model Training**
- Trained a **Random Forest Classifier** on the balanced dataset
- Split data into 80% train, 20% test

### 4. **Evaluation**
- Evaluated using **Precision**, **Recall**, **F1-score**, and **Confusion Matrix**
- Visualization using Seaborn heatmap

### 5. **Testing Interface**
- Added a basic function to test predictions on new transaction data

---

## ✅ Results
- Achieved strong recall and F1-score for both fraud and non-fraud classes
- Demonstrated ability to detect frauds with improved performance after balancing

---

## ▶️ How to Run

1. Open the Colab notebook.
2. Upload `creditcard.csv` dataset to your session.
3. Run all cells in sequence.
4. Use `test_transaction()` with 29 feature values to test new transactions.

Example:
```python
test_transaction([0.1, -1.3, ..., 10.5])  # Replace with real 29 values
