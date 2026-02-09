import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data():
    print("Loading data for preprocessing...")
    df = pd.read_csv('creditcard.csv')
    
    # Scaling Time and Amount
    print("Scaling Amount and Time using RobustScaler...")
    rob_scaler = RobustScaler()
    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Shuffling data
    df = df.sample(frac=1).reset_index(drop=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("Splitting data into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Handling Class Imbalance with SMOTE on Training Data
    print("Applying SMOTE to training data...")
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"Resampled training set size: {X_train_res.shape[0]}")
    print(f"Fraud count after SMOTE: {sum(y_train_res == 1)}")
    
    # Saving preprocessed data and scaler
    print("Saving scaler and splits...")
    joblib.dump(rob_scaler, 'robust_scaler.pkl')
    
    # We'll save the splits for training and evaluation
    data_splits = {
        'X_train_res': X_train_res,
        'y_train_res': y_train_res,
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(data_splits, 'data_splits.pkl')
    print("Preprocessing completed and data splits saved to data_splits.pkl")

if __name__ == "__main__":
    preprocess_data()
