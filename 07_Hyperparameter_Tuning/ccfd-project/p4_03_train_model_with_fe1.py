"""
Credit Card Fraud Detection - Training with Feature Engineering (FE1)

Uses sklearn Pipeline for preprocessing + model in ONE object.

This script loads the FEATURE ENGINEERED CSV (with amount_ratio) and trains.

DATA SOURCE: data/credit_card_transactions_fe1_amount_ratio.csv

OUTPUT: 
 - models/p4_03_fe1_amount_ratio/model_latest.pkl
 - models/p4_03_fe1_amount_ratio/metadata_latest.json
 - results/logistic_regression_fe1_*.json
"""

import os
import pandas as pd
from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, save_metrics_json

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - TRAINING WITH FE1")
print("=" * 70)

# =============================================================================
# Configuration
# =============================================================================
FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'

# =============================================================================
# STEP 1: Load Feature Engineered Data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 1: Loading Data")
print("-" * 70)

if not os.path.exists(FE_DATA_FILE):
 print(f"ERROR: {FE_DATA_FILE} not found!")
 print(f"Run p4_01_feature_engineering_fe1.py first.")
 exit(1)

df = pd.read_csv(FE_DATA_FILE)
print(f"Loaded {len(df):,} transactions from {FE_DATA_FILE}")

# =============================================================================
# STEP 2: Verify amount_ratio Exists
# =============================================================================
print("\n" + "-" * 70)
print("STEP 2: Verifying Features")
print("-" * 70)

if 'amount_ratio' not in df.columns:
 print(f"ERROR: amount_ratio column not found!")
 exit(1)

print(f"Verified: amount_ratio column exists")

# =============================================================================
# STEP 3: Split Data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 3: Splitting Data")
print("-" * 70)

X_train, X_test, y_train, y_test, feature_info = split_data(df)

print(f"\n Feature Information:")
print(f" • Total features: {feature_info['n_features']}")
print(f" • Includes amount_ratio: ")

# =============================================================================
# STEP 4: Create Pipeline
# =============================================================================
print("\n" + "-" * 70)
print("STEP 4: Creating Pipeline")
print("-" * 70)

# Model parameters (sklearn 1.8+ compatible)
model_params = {
 'C': 1.0,
 'max_iter': 5000,
 'random_state': 42,
 'class_weight': 'balanced',
 'solver': 'saga', # Required for l1_ratio support
 'l1_ratio': 0 # 0 = L2 regularization
}

pipeline = create_pipeline(
 categorical_cols=feature_info['categorical_cols'],
 numerical_cols=feature_info['numerical_cols'],
 model_params=model_params
)

print("Pipeline created with FE1 features")

# =============================================================================
# STEP 5: Train Pipeline
# =============================================================================
print("\n" + "-" * 70)
print("STEP 5: Training Pipeline")
print("-" * 70)

print("Training pipeline...")
pipeline.fit(X_train, y_train)
print("Training complete!")

# =============================================================================
# STEP 6: Evaluate Performance
# =============================================================================
print("\n" + "-" * 70)
print("STEP 6: Evaluating Performance")
print("-" * 70)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

metrics = print_metrics('Logistic Regression (FE1: amount_ratio)', y_test, y_pred, y_pred_proba)

# =============================================================================
# STEP 7: Save Pipeline
# =============================================================================
print("\n" + "-" * 70)
print("STEP 7: Saving Pipeline")
print("-" * 70)

save_pipeline(
 pipeline=pipeline,
 model_name='p4_03_fe1_amount_ratio',
 metrics={
 'f1': metrics['f1'],
 'precision': metrics['precision'],
 'recall': metrics['recall'],
 'net_benefit': metrics['business_metrics']['net_benefit']
 },
 metadata={
 'feature_info': feature_info,
 'model_params': model_params,
 'fe_features': ['amount_ratio'],
 'data_source': FE_DATA_FILE,
 'version': 'p4_03_fe1_amount_ratio'
 }
)

# =============================================================================
# STEP 8: Save Metrics to JSON
# =============================================================================
dataset_info = {
 'train_size': len(X_train),
 'test_size': len(X_test),
 'train_fraud_rate': float(y_train.mean()),
 'test_fraud_rate': float(y_test.mean()),
 'total_features': feature_info['n_features'],
 'feature_names': feature_info['feature_names'],
 'fe_features': ['amount_ratio'],
 'data_source': FE_DATA_FILE,
 'version': 'p4_03_fe1_amount_ratio'
}

save_metrics_json(
 metrics=metrics,
 algorithm_name='Logistic Regression FE1 Amount Ratio',
 model_params=model_params,
 dataset_info=dataset_info
)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"\n Model: Logistic Regression (FE1)")
print(f"Features: {feature_info['n_features']} (includes amount_ratio)")
print(f"\n F1 Score: {metrics['f1']:.4f}")
print(f"Net Benefit: ${metrics['business_metrics']['net_benefit']:,.0f}")
print(f"\n Model saved to: models/p4_03_fe1_amount_ratio/")

print("\n" + "=" * 70)
print("FE1 TRAINING COMPLETE!")
print("=" * 70)
