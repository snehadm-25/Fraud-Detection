import joblib
import pandas as pd
import numpy as np

def predict_fraud(transaction_data):
    """
    Predicts if a transaction is fraudulent.
    transaction_data: list or numpy array of 30 features (V1-V28, scaled_amount, scaled_time)
    """
    model = joblib.load('rf_model.pkl')
    
    if isinstance(transaction_data, list):
        transaction_data = np.array(transaction_data).reshape(1, -1)
    
    prediction = model.predict(transaction_data)
    probability = model.predict_proba(transaction_data)[:, 1]
    
    return {
        'is_fraud': bool(prediction[0]),
        'fraud_probability': float(probability[0])
    }

if __name__ == "__main__":
    # Example usage with a dummy transaction (V1-V28, scaled_amount, scaled_time)
    dummy_transaction = [0] * 30 
    result = predict_fraud(dummy_transaction)
    print(f"Prediction Result: {result}")
