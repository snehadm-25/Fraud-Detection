# Credit Card Fraud Detection Pipeline

This repository contains an end-to-end machine learning pipeline to detect fraudulent credit card transactions.

## Project Structure
- `eda.py`: Performs Exploratory Data Analysis and generates visualizations.
- `preprocess.py`: Scales data using `RobustScaler` and handles imbalance with SMOTE.
- `train.py`: Trains Logistic Regression and Random Forest models.
- `evaluate.py`: Generates performance metrics (AUPRC, Recall, Precision) and plots.
- `predict.py`: Provides an interface for making predictions on new transaction data.

## Performance Summary
| Model | AUPRC | Precision (Fraud) | Recall (Fraud) |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.734 | 0.06 | 0.90 |
| **Random Forest** | **0.796** | **0.51** | **0.84** |

## Quick Start
1. **Explore Data**: `python eda.py`
2. **Preprocess State**: `python preprocess.py`
3. **Train Models**: `python train.py`
4. **Evaluate Performance**: `python evaluate.py`
5. **Run Inference**:
```python
from predict import predict_fraud
result = predict_fraud(transaction_features)
print(result)
```

## Visualizations
Check the following generated files in the root directory:
- `class_distribution.png`
- `precision_recall_curve.png`
- `confusion_matrix_random_forest.png`
