import os
import pandas as pd
from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, save_metrics_json

FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'


# Step-1: Read the FE1 feature-engineered CSV
df = pd.read_csv(FE_DATA_FILE)


# Step-2: Train/test split (80/20 stratified)
X_train, X_test, y_train, y_test, feature_info = split_data(df)


# Step-3: Build Pipeline with hardcoded model_params (preprocessor + LogisticRegression)
model_params = {
    'C': 1.0,
    'max_iter': 5000,
    'random_state': 42,
    'class_weight': 'balanced',
    'solver': 'saga',
    'l1_ratio': 0
}
pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols'],
    model_params=model_params
)


# Step-4: Train pipeline (single .fit() with the hardcoded params)
pipeline.fit(X_train, y_train)


# Step-5: Evaluate on test set (predict + predict_proba + metrics)
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
metrics = print_metrics('Logistic Regression (FE1: amount_ratio)', y_test, y_pred, y_pred_proba)


# Step-6: Save trained pipeline + metadata pickle
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


# Step-7: Save metrics + dataset info to JSON results
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
