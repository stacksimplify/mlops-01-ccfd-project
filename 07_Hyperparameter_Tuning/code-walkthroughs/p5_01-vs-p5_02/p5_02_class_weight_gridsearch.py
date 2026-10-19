import os
import warnings
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, calculate_business_metrics, save_metrics_json

FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'
ARTIFACT_FILE = 'artifacts/p5_02_best_config.yaml'


# Step-1: Read the FE1 feature-engineered CSV
df = pd.read_csv(FE_DATA_FILE)


# Step-2: Train/test split (80/20 stratified)
X_train, X_test, y_train, y_test, feature_info = split_data(df)


# Step-3: Build Pipeline (NO model_params - the search will set them)
pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols']
)


# Step-4a: Custom business scorer (rank by dollar Net Benefit, not F1)
def business_value_scorer(y_true, y_pred):
    business_metrics = calculate_business_metrics(y_true, y_pred)
    net_benefit = business_metrics['net_benefit']
    return net_benefit

business_scorer = make_scorer(business_value_scorer, greater_is_better=True)


# Step-4b: Run GridSearchCV - test ALL 60 combos, 5-fold CV each = 300 fits
param_grid = {
    'model__C': [0.01, 0.1, 1.0, 10.0],
    'model__l1_ratio': [0, 0.5, 1.0],
    'model__solver': ['saga'],
    'model__max_iter': [5000],
    'model__class_weight': [
        'balanced',
        {0: 1, 1: 30},
        {0: 1, 1: 60},
        {0: 1, 1: 100},
        {0: 1, 1: 200},
    ],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=business_scorer,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)


# Step-4c: Extract winner - best_params_, best_estimator_ (auto-refit on full training data)
best_params_clean = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
best_pipeline = grid_search.best_estimator_


# Step-5: Evaluate winning pipeline on test set
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
metrics = print_metrics('p5_02 GridSearch (cw in grid, t=0.5)', y_test, y_pred, y_pred_proba)


# Step-6: Save winning config as YAML artifact (production handoff - p5_03 reads this)
os.makedirs('artifacts', exist_ok=True)
artifact = {
    'method': 'GridSearchCV_with_money_weighted_class_weight',
    'best_params': best_params_clean,
    'best_cv_score': float(grid_search.best_score_),
    'test_f1': float(metrics['f1']),
    'test_net_benefit': int(metrics['business_metrics']['net_benefit']),
    'grid_size': 60,
    'cv_folds': 5,
}
with open(ARTIFACT_FILE, 'w') as f:
    yaml.safe_dump(artifact, f, sort_keys=False)


# Step-7: Save best_pipeline + search metadata pickle
save_pipeline(
    pipeline=best_pipeline,
    model_name='p5_02_class_weight_gridsearch',
    metrics={
        'f1': metrics['f1'],
        'net_benefit': metrics['business_metrics']['net_benefit'],
        'best_cv_score': float(grid_search.best_score_)
    },
    metadata={
        'feature_info': feature_info,
        'best_params': best_params_clean,
        'tuning_method': 'GridSearchCV (cw in grid)',
        'grid_size': 60,
        'version': 'p5_02_class_weight_gridsearch'
    }
)


# Step-8: Save metrics + dataset info to JSON results
dataset_info = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'data_source': FE_DATA_FILE,
    'method': 'GridSearchCV (cw in grid)',
    'version': 'p5_02_class_weight_gridsearch'
}
save_metrics_json(
    metrics=metrics,
    algorithm_name='Logistic Regression p5_02 GridSearch (cw in grid)',
    model_params=best_params_clean,
    dataset_info=dataset_info
)
