"""
Credit Card Fraud Detection - Model Utilities (Basics)

Pipeline-based approach for sklearn compatibility.

Functions:
- create_pipeline: Create sklearn Pipeline with preprocessor + model
- save_pipeline: Save Pipeline to disk as model_*.pkl
- load_pipeline: Load Pipeline from disk
- get_available_pipelines: List saved models
"""

import pickle
import json
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from utils.data_utils import create_preprocessor


def create_pipeline(categorical_cols, numerical_cols, model=None, model_params=None):
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


def save_pipeline(pipeline, model_name, save_dir='models', metrics=None, metadata=None):
    """
    Save sklearn Pipeline to disk.

    Saves:
    - model_latest.pkl: The Pipeline object
    - metadata_latest.json: Metrics and metadata

    Args:
        pipeline: Fitted sklearn Pipeline
        model_name (str): Name identifier (e.g., 'baseline', 'p3_01_baseline')
        save_dir (str): Base directory for models
        metrics (dict): Performance metrics to save
        metadata (dict): Additional metadata to save

    Returns:
        tuple: (model_path, metadata_path)
    """
    # Create model-specific directory
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model (timestamped)
    model_timestamped = os.path.join(model_dir, f'model_{timestamp}.pkl')
    with open(model_timestamped, 'wb') as f:
        pickle.dump(pipeline, f)

    # Save model (latest)
    model_latest = os.path.join(model_dir, 'model_latest.pkl')
    with open(model_latest, 'wb') as f:
        pickle.dump(pipeline, f)

    # Prepare metadata
    meta = {
        'model_name': model_name,
        'model_type': type(pipeline.named_steps['model']).__name__,
        'created_at': datetime.now().isoformat(),
        'timestamp': timestamp,
        'pipeline_version': '1.0',
        'metrics': metrics or {},
        'metadata': metadata or {}
    }

    # Try to extract model params
    try:
        meta['model_params'] = pipeline.named_steps['model'].get_params()
    except Exception:
        meta['model_params'] = {}

    # Save metadata (timestamped)
    meta_timestamped = os.path.join(model_dir, f'metadata_{timestamp}.json')
    with open(meta_timestamped, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    # Save metadata (latest)
    meta_latest = os.path.join(model_dir, 'metadata_latest.json')
    with open(meta_latest, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nModel Saved: {model_name}")
    print(f"   Location: {model_dir}/")
    print(f"   Model: {meta['model_type']}")
    print(f"   Timestamp: {timestamp}")

    return model_latest, meta_latest


def load_pipeline(model_name, save_dir='models'):
    """
    Load sklearn Pipeline from disk.

    Args:
        model_name (str): Name identifier (e.g., 'baseline', 'p3_01_baseline')
        save_dir (str): Base directory for models

    Returns:
        tuple: (pipeline, metadata)
    """
    model_dir = os.path.join(save_dir, model_name)

    # Load model
    model_path = os.path.join(model_dir, 'model_latest.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Available models: {get_available_pipelines(save_dir)}"
        )

    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)

    # Load metadata
    meta_path = os.path.join(model_dir, 'metadata_latest.json')
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

    print(f"\nModel Loaded: {model_name}")
    print(f"   Model: {metadata.get('model_type', 'Unknown')}")
    print(f"   Created: {metadata.get('created_at', 'Unknown')}")

    return pipeline, metadata


def get_available_pipelines(save_dir='models'):
    """
    List all available saved pipelines.

    Returns:
        list: Names of available pipelines
    """
    if not os.path.exists(save_dir):
        return []

    pipelines = []
    for item in os.listdir(save_dir):
        item_path = os.path.join(save_dir, item)
        if os.path.isdir(item_path):
            model_path = os.path.join(item_path, 'model_latest.pkl')
            if os.path.exists(model_path):
                pipelines.append(item)

    return pipelines
