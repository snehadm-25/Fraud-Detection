from flask import Flask, render_template, request, jsonify, url_for
import joblib
import numpy as np
import os
import shutil

app = Flask(__name__)

# Load models and scaler
print("Loading Model and Scaler...")
rf_model = joblib.load('rf_model.pkl')
print("Model loaded.")
rob_scaler = joblib.load('robust_scaler.pkl')
print("Scaler loaded.")

def setup():
    # Move plots to static folder if they exist
    print("Ensuring static folder and copying plots...")
    os.makedirs('static', exist_ok=True)
    plots = ['precision_recall_curve.png', 'confusion_matrix_random_forest.png']
    for plot in plots:
        if os.path.exists(plot):
            shutil.copy(plot, os.path.join('static', plot))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Preprocess Amount (using the saved scaler)
    # The scaler was fitted on 2D array [Amount]
    scaled_amount = rob_scaler.transform(np.array([[data['amount']]]))[0][0]
    scaled_time = 0 # Placeholder for time
    
    # Prepare feature vector (V1-V28, scaled_amount, scaled_time)
    # V1 and V2 from user, others set to 0 for demo
    v1 = data['v1']
    v2 = data['v2']
    features = [v1, v2] + [0] * 26 + [scaled_amount, scaled_time]
    
    # Reshape for prediction
    features_arr = np.array(features).reshape(1, -1)
    
    prediction = rf_model.predict(features_arr)[0]
    probability = rf_model.predict_proba(features_arr)[0][1]
    
    return jsonify({
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability)
    })

if __name__ == '__main__':
    setup()
    print("SentryForce Dashboard running at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
