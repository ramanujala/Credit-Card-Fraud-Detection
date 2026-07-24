import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def main():
    
    print("Loading dataset...")
    
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: creditcard.csv not found in the current directory.")
        return

    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")

   
    df_fraud = df[df['Class'] == 1]
    df_legit = df[df['Class'] == 0]

   
    df_fraud_train, df_fraud_test = train_test_split(df_fraud, test_size=100, random_state=42)
    
   
    df_legit_train, df_legit_test = train_test_split(df_legit, test_size=50000, random_state=42)

   
    df_legit_train_downsampled = df_legit_train.sample(n=10000, random_state=42)

   
    df_train = pd.concat([df_fraud_train, df_legit_train_downsampled]).sample(frac=1, random_state=42)
    df_test = pd.concat([df_fraud_test, df_legit_test]).sample(frac=1, random_state=42)

    print(f"Training set size: {df_train.shape} (Fraud: {len(df_fraud_train)}, Legit: {len(df_legit_train_downsampled)})")
    print(f"Testing set size: {df_test.shape} (Fraud: {len(df_fraud_test)}, Legit: {len(df_legit_test)})")

   
    feature_cols = [c for c in df.columns if c not in ['Class']]
    X_train = df_train[feature_cols]
    y_train = df_train['Class']
    X_test = df_test[feature_cols]
    y_test = df_test['Class']

   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
  
    X_test_scaled = scaler.transform(X_test)

   
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
    model.fit(X_train_scaled, y_train)

    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

  
    print("\nModel Evaluation:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

   
    print("\nComputing statistical profiles for explanations...")
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_train_scaled_df['Class'] = y_train.values

    stats = {
      
        'mean_legit': X_train_scaled_df[X_train_scaled_df['Class'] == 0][feature_cols].mean().to_dict(),
        'std_legit': X_train_scaled_df[X_train_scaled_df['Class'] == 0][feature_cols].std().to_dict(),
      
        'mean_fraud': X_train_scaled_df[X_train_scaled_df['Class'] == 1][feature_cols].mean().to_dict(),
        'std_fraud': X_train_scaled_df[X_train_scaled_df['Class'] == 1][feature_cols].std().to_dict(),
    }

   
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

   
    print("Saving test samples for Streamlit UI...")
    sample_legit = df_test[df_test['Class'] == 0].sample(n=50, random_state=42)
    sample_fraud = df_test[df_test['Class'] == 1].sample(n=50, random_state=42)
    sample_df = pd.concat([sample_legit, sample_fraud]).sample(frac=1, random_state=42)
    sample_df.to_csv('sample_test_data.csv', index=False)
    print("Saved sample test data to 'sample_test_data.csv'")

if __name__ == '__main__':
    main()
