import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import time

def train_models():
    print("Loading preprocessed data...")
    data = joblib.load('data_splits.pkl')
    X_train = data['X_train_res']
    y_train = data['y_train_res']
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression model...")
    start_time = time.time()
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    print(f"Logistic Regression trained in {time.time() - start_time:.2f} seconds.")
    joblib.dump(lr_model, 'lr_model.pkl')
    
    # 2. Random Forest (More robust)
    # Reducing estimators for speed in this environment
    print("Training Random Forest model (this might take a few minutes)...")
    start_time = time.time()
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print(f"Random Forest trained in {time.time() - start_time:.2f} seconds.")
    joblib.dump(rf_model, 'rf_model.pkl')
    
    print("Models trained and saved to disk.")

if __name__ == "__main__":
    train_models()
