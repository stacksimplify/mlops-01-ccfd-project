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


from utils.data_utils import load_data, split_data
from utils.model_utils import create_pipeline

# STEP 1: Load Raw Data
print("STEP 1: Loading Data")
df = load_data()

# STEP 2: Split Data
print("STEP 2: Splitting Data")
X_train, X_test, y_train, y_test, feature_info = split_data(df)

print(f"\nFeature Information:")
print(f"   Total features: {feature_info['n_features']}")
print(f"   Categorical: {feature_info['categorical_cols']}")
print(f"   Numerical: {feature_info['numerical_cols']}")


# STEP 3: Create Pipeline
print("STEP 3: Creating Pipeline")

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


# STEP 4: Train Pipeline
print("STEP 4: Training Pipeline")
print("Training pipeline on raw data...")
pipeline.fit(X_train, y_train)
print("Training complete!")





