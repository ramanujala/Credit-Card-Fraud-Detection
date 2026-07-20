import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def main():
    """
    Main function to load credit card data, preprocess it, handle class imbalance,
    train a Random Forest model, evaluate it, and save the model assets + test samples.
    """
    print("Loading dataset...")
    # Attempt to load the credit card fraud dataset from the workspace root.
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: creditcard.csv not found in the current directory.")
        return

    # Print dataset characteristics
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

    # Separate the highly imbalanced dataset into fraud (Class 1) and legitimate (Class 0) transactions.
    # The dataset contains only 492 fraud cases out of 284,807 total rows.
    df_fraud = df[df['Class'] == 1]
    df_legit = df[df['Class'] == 0]

    # Split the 492 fraud cases into a training split (392 cases) and a testing split (100 cases).
    df_fraud_train, df_fraud_test = train_test_split(df_fraud, test_size=100, random_state=42)
    
    # Split the 284,315 legitimate cases into training and testing partitions.
    # We allocate 50,000 cases to the test set to simulate a realistic, highly imbalanced deployment environment.
    df_legit_train, df_legit_test = train_test_split(df_legit, test_size=50000, random_state=42)

    # To address the severe class imbalance and prevent model bias (and speed up training),
    # we downsample the legitimate training partition to 10,000 random samples.
    df_legit_train_downsampled = df_legit_train.sample(n=10000, random_state=42)

    # Combine training and test partitions and shuffle them to randomize row order.
    df_train = pd.concat([df_fraud_train, df_legit_train_downsampled]).sample(frac=1, random_state=42)
    df_test = pd.concat([df_fraud_test, df_legit_test]).sample(frac=1, random_state=42)

    print(f"Training set size: {df_train.shape} (Fraud: {len(df_fraud_train)}, Legit: {len(df_legit_train_downsampled)})")
    print(f"Testing set size: {df_test.shape} (Fraud: {len(df_fraud_test)}, Legit: {len(df_legit_test)})")

    # Define feature columns (Time, Amount, V1 to V28) and target label column (Class).
    feature_cols = [c for c in df.columns if c not in ['Class']]
    X_train = df_train[feature_cols]
    y_train = df_train['Class']
    X_test = df_test[feature_cols]
    y_test = df_test['Class']

    # Standardize numerical features using standard scaling (mean = 0, variance = 1).
    # This ensures that Time, Amount, and the PCA variables have comparable ranges.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test set using training scale parameters to avoid data leakage.
    X_test_scaled = scaler.transform(X_test)

    # Initialize a Random Forest Classifier.
    # We restrict max_depth to 15 to prevent overfitting, and use n_jobs=-1 for multi-threaded speed.
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    model.fit(X_train_scaled, y_train)

    # Generate predictions and prediction probabilities for model evaluation.
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Print out detailed model performance metrics.
    print("\nModel Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Calculate and store baseline statistics (means and standard deviations on scaled data)
    # for legitimate and fraud transactions. This forms the baseline profile.
    # When a new transaction is processed, we compare its feature values to these baselines
    # using Z-scores to pinpoint which specific indicators make it anomalous.
    print("\nComputing statistical profiles for explanations...")
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_train_scaled_df['Class'] = y_train.values

    stats = {
        # Mean and standard deviation profile for legitimate transactions
        'mean_legit': X_train_scaled_df[X_train_scaled_df['Class'] == 0][feature_cols].mean().to_dict(),
        'std_legit': X_train_scaled_df[X_train_scaled_df['Class'] == 0][feature_cols].std().to_dict(),
        # Mean and standard deviation profile for fraud transactions
        'mean_fraud': X_train_scaled_df[X_train_scaled_df['Class'] == 1][feature_cols].mean().to_dict(),
        'std_fraud': X_train_scaled_df[X_train_scaled_df['Class'] == 1][feature_cols].std().to_dict(),
    }

    # Bundle all model artifacts, preprocessors, and statistics into a dictionary and save to disk.
    assets = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'stats': stats,
        'metrics': {
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    }
    joblib.dump(assets, 'fraud_model_assets.joblib')
    print("Saved model assets to 'fraud_model_assets.joblib'")

    # Extract and save a small balanced subset of test samples (50 legit and 50 fraud).
    # These samples are stored as a CSV to allow the user to easily pick realistic transactions
    # from a selectbox in the Streamlit interface to verify prediction and Gen AI explanation.
    print("Saving test samples for Streamlit UI...")
    sample_legit = df_test[df_test['Class'] == 0].sample(n=50, random_state=42)
    sample_fraud = df_test[df_test['Class'] == 1].sample(n=50, random_state=42)
    sample_df = pd.concat([sample_legit, sample_fraud]).sample(frac=1, random_state=42)
    sample_df.to_csv('sample_test_data.csv', index=False)
    print("Saved sample test data to 'sample_test_data.csv'")

if __name__ == '__main__':
    main()
