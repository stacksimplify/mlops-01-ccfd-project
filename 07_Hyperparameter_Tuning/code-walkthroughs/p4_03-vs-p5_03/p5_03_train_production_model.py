import os
import pandas as pd
import yaml

from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, save_metrics_json

FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'
INPUT_ARTIFACT = 'artifacts/p5_02_best_config.yaml'


# Step-0: Load winning config from p5_02's YAML artifact (production handoff)
with open(INPUT_ARTIFACT) as f:
    config = yaml.safe_load(f)
best_params = config['best_params']


# Step-1: Read the FE1 feature-engineered CSV
df = pd.read_csv(FE_DATA_FILE)


# Step-2: Train/test split (80/20 stratified)
X_train, X_test, y_train, y_test, feature_info = split_data(df)


# Step-3: Build Pipeline with LOADED model_params (from p5_02 artifact)
pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols'],
    model_params=best_params
)


# Step-4: Train pipeline (single .fit() with the loaded params)
pipeline.fit(X_train, y_train)


# Step-5: Evaluate on test set (predict + predict_proba + metrics)
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
metrics = print_metrics('Production Model (p5_03)', y_test, y_pred, y_pred_proba)


# Step-6: Save trained pipeline + metadata pickle
save_pipeline(
    pipeline=pipeline,
    model_name='p5_03_production',
    metrics={
        'f1': metrics['f1'],
        'net_benefit': metrics['business_metrics']['net_benefit'],
    },
    metadata={
        'feature_info': feature_info,
        'best_params': best_params,
        'config_source': INPUT_ARTIFACT,
        'config_source_method': config.get('method', 'unknown'),
        'production_role': 'deployable model from re-training job',
        'version': 'p5_03_production'
    }
)


# Step-7: Save metrics + dataset info to JSON results
dataset_info = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'data_source': FE_DATA_FILE,
    'config_source': INPUT_ARTIFACT,
    'version': 'p5_03_production'
}

save_metrics_json(
    metrics=metrics,
    algorithm_name='Logistic Regression p5_03 Production',
    model_params=best_params,
    dataset_info=dataset_info
)
