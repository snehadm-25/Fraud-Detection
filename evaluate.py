import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, average_precision_score

def evaluate_models():
    print("Loading models and test data...")
    data = joblib.load('data_splits.pkl')
    X_test = data['X_test']
    y_test = data['y_test']
    
    lr_model = joblib.load('lr_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Classification Report
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()
        
        # Precision-Recall Curve (Crucial for imbalanced data)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        ap = average_precision_score(y_test, y_probs)
        
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.4f}, AP = {ap:.4f})')
        print(f"AUPRC for {name}: {pr_auc:.4f}")

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig('precision_recall_curve.png')
    plt.close()
    
    print("\nEvaluation completed. Reports printed and plots saved.")

if __name__ == "__main__":
    evaluate_models()
