import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, calculate_business_metrics, save_metrics_json

FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'


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


# Step-4b: Run RandomizedSearchCV - sample 20 of 180 combos, 5-fold CV each = 100 fits
param_distributions = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'model__solver': ['saga'],
    'model__max_iter': [2000, 5000, 10000],
    'model__class_weight': ['balanced', None]
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring=business_scorer,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)


# Step-4c: Extract winner - best_params_, best_estimator_ (auto-refit on full training data)
best_params_clean = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}
best_pipeline = random_search.best_estimator_


# Step-5: Evaluate winning pipeline on test set
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
metrics = print_metrics('p5_01 RandomSearch (default threshold 0.5)', y_test, y_pred, y_pred_proba)


# Step-6: Save best_pipeline + search metadata pickle
save_pipeline(
    pipeline=best_pipeline,
    model_name='p5_01_randomsearch',
    metrics={
        'f1': metrics['f1'],
        'net_benefit': metrics['business_metrics']['net_benefit'],
        'best_cv_score': float(random_search.best_score_)
    },
    metadata={
        'feature_info': feature_info,
        'best_params': best_params_clean,
        'tuning_method': 'RandomizedSearchCV',
        'n_iter': 20,
        'cv_folds': 5,
        'version': 'p5_01_randomsearch'
    }
)


# Step-7: Save metrics + dataset info to JSON results
dataset_info = {
    'train_size': len(X_train),
    'test_size': len(X_test),
    'total_features': feature_info['n_features'],
    'feature_names': feature_info['feature_names'],
    'data_source': FE_DATA_FILE,
    'tuning_info': {
        'method': 'RandomizedSearchCV',
        'n_iter': 20,
        'cv_folds': 5,
        'scoring': 'custom_business_value (Net Benefit)',
        'best_cv_score': float(random_search.best_score_),
        'best_params': best_params_clean
    },
    'version': 'p5_01_randomsearch'
}

save_metrics_json(
    metrics=metrics,
    algorithm_name='Logistic Regression p5_01 RandomSearch',
    model_params=best_params_clean,
    dataset_info=dataset_info
)
