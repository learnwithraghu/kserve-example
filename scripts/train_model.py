"""
Banking Churn Model Training Script
This script trains a Random Forest classifier for banking customer churn prediction.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import json

def load_data(filepath):
    """Load the training data from CSV file."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded successfully. Shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess the data for model training.
    
    Steps:
    1. Separate features and target
    2. Encode categorical variables
    3. Scale numerical features
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Drop customer_id as it's not a feature
    if 'customer_id' in df_processed.columns:
        df_processed = df_processed.drop('customer_id', axis=1)
    
    # Separate features and target
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical features: {categorical_cols}")
    print(f"Numerical features: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"✓ Encoded '{col}': {list(le.classes_)}")
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    print(f"\n✓ Scaled numerical features")
    
    return X, y, scaler, label_encoders

def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Initialize Random Forest with optimized parameters
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("\nModel parameters:")
    print(json.dumps(model.get_params(), indent=2, default=str))
    
    print("\nTraining Random Forest classifier...")
    model.fit(X_train, y_train)
    print("✓ Model training complete!")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate the trained model and display metrics."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save metrics to file
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'feature_importance': feature_importance.to_dict('records')
    }
    
    return metrics

def save_model_artifacts(model, scaler, label_encoders, metrics, feature_names):
    """Save model and preprocessing artifacts."""
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)
    
    # Create model directory
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained model
    model_path = os.path.join(model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save the scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")
    
    # Save label encoders
    encoders_path = os.path.join(model_dir, 'label_encoders.joblib')
    joblib.dump(label_encoders, encoders_path)
    print(f"✓ Label encoders saved to: {encoders_path}")
    
    # Save feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.json')
    with open(feature_names_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"✓ Feature names saved to: {feature_names_path}")
    
    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Save model metadata for KServe
    metadata = {
        'model_type': 'sklearn',
        'model_class': 'RandomForestClassifier',
        'framework': 'scikit-learn',
        'features': feature_names,
        'target': 'churn',
        'classes': [0, 1],
        'class_names': ['No Churn', 'Churn']
    }
    
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")
    
    print(f"\n✓ All artifacts saved to '{model_dir}/' directory")

def main():
    """Main training pipeline."""
    print("="*60)
    print("BANKING CHURN PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Check if data exists
    data_path = 'data/banking_churn_train.csv'
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Training data not found at '{data_path}'")
        print("Please run 'python scripts/generate_data.py' first to generate the data.")
        return
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess data
    X, y, scaler, label_encoders = preprocess_data(df)
    
    # Split data into train and validation sets
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_val, y_val, X.columns.tolist())
    
    # Save model and artifacts
    save_model_artifacts(model, scaler, label_encoders, metrics, X.columns.tolist())
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review model metrics in 'model/metrics.json'")
    print("2. Deploy the model to Kubernetes using KServe")
    print("3. Follow the README.md for deployment instructions")

if __name__ == "__main__":
    main()
