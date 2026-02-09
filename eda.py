import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    print("Loading dataset...")
    df = pd.read_csv('creditcard.csv')
    
    # 1. Class Distribution
    print("Visualizing class distribution...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, hue='Class', palette='viridis', legend=False)
    plt.title('Class Distribution (0: No Fraud, 1: Fraud)')
    plt.savefig('class_distribution.png')
    plt.close()
    
    fraud_count = df[df['Class'] == 1].shape[0]
    total_count = df.shape[0]
    print(f"Total Transactions: {total_count}")
    print(f"Fraudulent Transactions: {fraud_count} ({100*fraud_count/total_count:.4f}%)")
    
    # 2. Amount and Time Distribution
    print("Analyzing Time and Amount distributions...")
    fig, ax = plt.subplots(1, 2, figsize=(18, 4))

    sns.histplot(df['Amount'], ax=ax[0], color='r', kde=True)
    ax[0].set_title('Distribution of Transaction Amount')
    ax[0].set_xlim([min(df['Amount']), max(df['Amount'])])

    sns.histplot(df['Time'], ax=ax[1], color='b', kde=True)
    ax[1].set_title('Distribution of Transaction Time')
    ax[1].set_xlim([min(df['Time']), max(df['Time'])])

    plt.savefig('amount_time_distribution.png')
    plt.close()
    
    # 3. Correlation Matrix
    print("Generating correlation matrix...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap='coolwarm_r', annot_kws={'size': 20})
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("EDA completed. Plots saved: class_distribution.png, amount_time_distribution.png, correlation_matrix.png")

if __name__ == "__main__":
    run_eda()
