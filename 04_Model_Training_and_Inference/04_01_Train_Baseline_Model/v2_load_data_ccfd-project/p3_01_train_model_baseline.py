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

from utils.data_utils import load_data

# Step-1: Load Raw Data
print("Step 1: Loading Data")
df = load_data()
