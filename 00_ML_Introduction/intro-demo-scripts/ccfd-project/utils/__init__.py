"""
Credit Card Fraud Detection - Utilities Package

Pipeline-based approach for sklearn compatibility.

Modules:
- data_utils: Data loading and preprocessing
- model_utils: Pipeline creation, save/load
- metrics_utils: Evaluation metrics (technical + business)
- feature_engineering_utils: Feature creation
"""

from utils.data_utils import (
    load_data,
    get_feature_columns,
    create_preprocessor,
    split_data
)

from utils.model_utils import (
    create_pipeline,
    save_pipeline,
    load_pipeline,
    get_available_pipelines
)

from utils.metrics_utils import (
    print_metrics,
    calculate_business_metrics,
    save_metrics_json
)

from utils.feature_engineering_utils import (
    apply_feature_engineering
)

__all__ = [
    # Data utilities
    'load_data',
    'get_feature_columns',
    'create_preprocessor',
    'split_data',
    # Model utilities
    'create_pipeline',
    'save_pipeline',
    'load_pipeline',
    'get_available_pipelines',
    # Metrics utilities
    'print_metrics',
    'calculate_business_metrics',
    'save_metrics_json',
    # Feature engineering
    'apply_feature_engineering',
]
