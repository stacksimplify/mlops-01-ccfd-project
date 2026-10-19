"""
Credit Card Fraud Detection - Baseline Model Training (Basics)

Uses sklearn Pipeline for preprocessing + model in ONE object.
Compatible with MLflow, KServe, SageMaker out of the box!

STEPS:
1. Load raw data
2. Split into train/test (stratified)
3. Create Pipeline (preprocessor + model)
4. Train Pipeline
5. Evaluate with technical AND business metrics
6. Save Pipeline

OUTPUT:
    - models/p3_01_baseline/model_latest.pkl
    - models/p3_01_baseline/metadata_latest.json
    - results/logistic_regression_p3_01_baseline_*.json
"""

from utils.data_utils import load_data, split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, save_metrics_json

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION - BASELINE MODEL TRAINING")
print("=" * 70)
print("\nModel: Logistic Regression (Baseline)")
print("Features: Base features only (NO feature engineering)")
print("Approach: sklearn Pipeline (MLflow/KServe compatible)")

# =============================================================================
# STEP 1: Load Raw Data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 1: Loading Data")
print("-" * 70)

df = load_data()

# =============================================================================
# STEP 2: Split Data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 2: Splitting Data")
print("-" * 70)

X_train, X_test, y_train, y_test, feature_info = split_data(df)

print(f"\nFeature Information:")
print(f"   Total features: {feature_info['n_features']}")
print(f"   Categorical: {feature_info['categorical_cols']}")
print(f"   Numerical: {feature_info['numerical_cols']}")

# =============================================================================
# STEP 3: Create Pipeline
# =============================================================================
print("\n" + "-" * 70)
print("STEP 3: Creating Pipeline")
print("-" * 70)

# Model parameters (sklearn 1.8+ compatible)
# Note: 'penalty' parameter is deprecated in sklearn 1.8+
#       Use 'l1_ratio' instead: 0=L2, 1=L1, 0<x<1=ElasticNet
model_params = {
    'C': 1.0,
    'max_iter': 5000,
    'random_state': 42,
    'class_weight': 'balanced',
    'solver': 'saga',   # Required for l1_ratio support
    'l1_ratio': 0       # 0 = L2 regularization
}

pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols'],
    model_params=model_params
)

print("Pipeline created:")
print(f"   1. Preprocessor (OneHotEncoder + StandardScaler)")
print(f"   2. Model (LogisticRegression)")

# =============================================================================
# STEP 4: Train Pipeline
# =============================================================================
print("\n" + "-" * 70)
print("STEP 4: Training Pipeline")
print("-" * 70)

print("Training pipeline on raw data...")
pipeline.fit(X_train, y_train)
print("Training complete!")

# =============================================================================
# STEP 5: Evaluate Performance
# =============================================================================
print("\n" + "-" * 70)
print("STEP 5: Evaluating Performance")
print("-" * 70)

# Pipeline handles preprocessing internally!
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Print comprehensive metrics
metrics = print_metrics('Logistic Regression (Baseline)', y_test, y_pred, y_pred_proba)

# =============================================================================
# STEP 6: Save Pipeline
# =============================================================================
print("\n" + "-" * 70)
print("STEP 6: Saving Pipeline")
print("-" * 70)

save_pipeline(
    pipeline=pipeline,
    model_name='p3_01_baseline',
    metrics={
        'f1': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'net_benefit': metrics['business_metrics']['net_benefit']
    },
    metadata={
        'feature_info': feature_info,
        'model_params': model_params,
        'version': 'p3_01_baseline'
    }
)

# =============================================================================
# STEP 7: Save Metrics to JSON
# =============================================================================
dataset_info = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'train_fraud_rate': float(y_train.mean()),
    'test_fraud_rate': float(y_test.mean()),
    'total_features': feature_info['n_features'],
    'feature_names': feature_info['feature_names'],
    'version': 'p3_01_baseline'
}

save_metrics_json(
    metrics=metrics,
    algorithm_name='Logistic Regression P3_01 Baseline',
    model_params=model_params,
    dataset_info=dataset_info
)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print(f"\n   Model: Logistic Regression (Baseline)")
print(f"   Features: {feature_info['n_features']} (base features, no FE)")
print(f"   Train size: {len(X_train):,}")
print(f"   Test size: {len(X_test):,}")
print(f"\n   F1 Score: {metrics['f1']:.4f}")
print(f"   Net Benefit: ${metrics['business_metrics']['net_benefit']:,.0f}")
print(f"\n   Model saved to: models/p3_01_baseline/")
print(f"   Metrics saved to: results/")

print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE!")
print("=" * 70)
