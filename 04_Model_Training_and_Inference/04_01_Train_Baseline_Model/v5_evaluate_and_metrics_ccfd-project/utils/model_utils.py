"""
Credit Card Fraud Detection - Model Utilities (Basics)

Pipeline-based approach for sklearn compatibility.

Functions:
- create_pipeline: Create sklearn Pipeline with preprocessor + model
- save_pipeline: Save Pipeline to disk as model_*.pkl
- load_pipeline: Load Pipeline from disk
- get_available_pipelines: List saved models
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from utils.data_utils import create_preprocessor


"""
    Create an sklearn Pipeline with preprocessing and model.

    Pipeline structure:
    1. Preprocessor (ColumnTransformer) - handles encoding + scaling
    2. Model (LogisticRegression by default)

    Args:
        categorical_cols (list): Categorical column names
        numerical_cols (list): Numerical column names
        model: sklearn model instance (default: LogisticRegression)
        model_params (dict): Model parameters (ignored if model provided)

    Returns:
        Pipeline: sklearn Pipeline ready for fit()
"""

def create_pipeline(categorical_cols, numerical_cols, model=None, model_params=None):
    # Create preprocessor
    preprocessor = create_preprocessor(categorical_cols, numerical_cols)

    # Create model
    if model is None:
        # sklearn 1.8+ compatible defaults (penalty deprecated, use l1_ratio instead)
        default_params = {
            'C': 1.0,
            'max_iter': 5000,
            'random_state': 42,
            'class_weight': 'balanced',
            'solver': 'saga',    # saga supports l1_ratio
            'l1_ratio': 0        # 0 = L2 regularization (replaces penalty='l2')
        }
        if model_params:
            default_params.update(model_params)
        model = LogisticRegression(**default_params)

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

