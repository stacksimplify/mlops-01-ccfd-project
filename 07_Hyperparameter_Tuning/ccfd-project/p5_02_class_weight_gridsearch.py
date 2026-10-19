"""
Credit Card Fraud Detection - Step 2: GridSearchCV with money-weighted class_weight

p5_01 (Random search) showed that a generic search space did NOT help. The miss
was the search space itself - it did not include the right class_weight values.

This script does proper GridSearchCV with an expanded search grid that includes
money-weighted class_weight values like {0:1, 1:50} ... {0:1, 1:500}. The grid
finds the best params and writes them to a YAML artifact for p5_03 to consume.

YAML is the right format for this artifact because class_weight is a dict with
integer keys (e.g. {0: 1, 1: 100}) and YAML stores int keys natively. The
artifact is exactly what sklearn needs - no transformation on save or load.

DATA SOURCE: data/credit_card_transactions_fe1_amount_ratio.csv

OUTPUT:
 - models/p5_02_class_weight_gridsearch/model_latest.pkl
 - results/logistic_regression_p5_02_class_weight_gridsearch_*.json
 - artifacts/p5_02_best_config.yaml (consumed by p5_03 for production training)

Improvement Journey position: STAGE 4 (GridSearch finds the winning params).
This is what p5_03 (production model) reads from.
"""

import os
import warnings

import pandas as pd
import yaml   # YAML preserves int dict keys (used by class_weight).
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# saga + aggressive class_weight values like {0:1, 1:200} can hit max_iter
# before fully converging. Predictions are still good. The warnings are
# noisy and would flood the terminal during the search. Silence them.
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, calculate_business_metrics, save_metrics_json

print("=" * 70)
print("p5_02 - GridSearchCV with money-weighted class_weight")
print("=" * 70)

# =============================================================================
# Configuration + Improvement Journey reference values
# =============================================================================
FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'
ARTIFACT_FILE = 'artifacts/p5_02_best_config.yaml'

TRUE_BASELINE_NET_BENEFIT = 39625
FE1_NET_BENEFIT = 39700

# =============================================================================
# STEP 1: Load + split
# =============================================================================
print("\n" + "-" * 70)
print("STEP 1: Loading Data")
print("-" * 70)

if not os.path.exists(FE_DATA_FILE):
    print(f"ERROR: {FE_DATA_FILE} not found.")
    exit(1)

df = pd.read_csv(FE_DATA_FILE)
X_train, X_test, y_train, y_test, feature_info = split_data(df)
print(f"Loaded {len(df):,} transactions")

# =============================================================================
# STEP 2: Build Pipeline (same create_pipeline helper)
# =============================================================================
print("\n" + "-" * 70)
print("STEP 2: Building Pipeline (preprocessing + LogisticRegression)")
print("-" * 70)

pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols']
)
print("Pipeline created via create_pipeline() helper (same as p3_01 / p4_03 / p5_01)")

# =============================================================================
# STEP 3: Custom business scorer (same as p5_01)
# =============================================================================
print("\n" + "-" * 70)
print("STEP 3: Defining Business-Value Scorer (dollars, not F1)")
print("-" * 70)

def business_value_scorer(y_true, y_pred):
    """Custom scorer - returns dollar Net Benefit for the search to rank by."""

    # For understanding what calculate_business_metrics returns - it gives back a full dict
    # with net_benefit, fraud_caught, fraud_missed, false_alarms, total_saved, total_cost.
    business_metrics = calculate_business_metrics(y_true, y_pred)
    # Uncomment the line below to see the full metrics dict during search (noisy - one dict per fold).
    # print(f"[scorer] business_metrics: {business_metrics}")

    # The actual return value the search uses to rank candidates. Printed live below
    # so each fold's Net Benefit appears interleaved with the [CV] END lines.
    net_benefit = business_metrics['net_benefit']
    print(f"[scorer] net_benefit: ${net_benefit:,}")
    return net_benefit

business_scorer = make_scorer(business_value_scorer, greater_is_better=True)
print("Scorer ready: GridSearchCV will rank candidates by Net Benefit, not F1 or accuracy.")

# =============================================================================
# STEP 4: Grid with money-weighted class_weight values included
# =============================================================================
print("\n" + "-" * 70)
print("STEP 4: Defining Search Grid (wider class_weight than p5_01)")
print("-" * 70)

param_grid = {
    'model__C': [0.01, 0.1, 1.0, 10.0],
    'model__l1_ratio': [0, 0.5, 1.0],
    'model__solver': ['saga'],
    'model__max_iter': [5000],
    # THE WIDER PART - 5 values including 4 money-weighted dicts.
    # p5_01 had only ['balanced', None] here - this is the single change that
    # turns the section's loss into a win.
    'model__class_weight': [
        'balanced',
        {0: 1, 1: 30},
        {0: 1, 1: 60},
        {0: 1, 1: 100},
        {0: 1, 1: 200},
    ],
}

# =============================================================================
# STEP 5: Run GridSearchCV
# =============================================================================
# THE SEARCH SPACE
#   4 C  x  3 l1_ratio  x  1 solver  x  1 max_iter  x  5 class_weight  =  60 combinations
#   Compare to p5_01's space of 180 - we have a SMALLER grid here, but a much
#   better-chosen one (the class_weight values are the ones that actually matter).
#
# THE PLAN
#   GridSearchCV tests ALL 60 combinations. No random sampling.
#   Use this when the space is small enough that exhaustive coverage is feasible.
#
# THE WORK
#   Each of the 60 gets 5-fold cross-validation = 60 x 5 = 300 training jobs.
#   sklearn confirms this with its "Fitting 5 folds for each of 60 candidates..." line.
#   Takes 1-3 minutes depending on CPU.
#
# verbose=2 below makes sklearn print one [CV] END line per fit so you see the
# search exploring the grid in real time.
# =============================================================================
print("\n" + "-" * 70)
print("STEP 5: Running GridSearchCV (this will take 1-3 minutes)")
print("-" * 70)

print("The Space:  4 C  x  3 l1_ratio  x  1 solver  x  1 max_iter  x  5 class_weight  =  60 combinations")
print("  -> class_weight grid is WIDER than p5_01: 'balanced' + 4 money-weighted dicts {0:1, 1:30..200}")
print("The Plan:   test ALL 60 combinations  (GridSearchCV - no random sampling)")
print("The Work:   each of the 60 gets 5-fold cross-validation")
print("            60 x 5 = 300 training jobs")
print("\nverbose=2 below - you will see one line per fit as the search runs.\n")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,                      # THE WORK - 5-fold CV per combo = 300 total fits
    scoring=business_scorer,   # custom Net Benefit scorer (dollars, not F1)
    n_jobs=-1,                 # use all CPU cores
    verbose=2                  # one [CV] END line per fit so you see the search working
)
grid_search.fit(X_train, y_train)

best_params_clean = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
print(f"\nBEST PARAMETERS:")
for k, v in best_params_clean.items():
    print(f"  {k}: {v}")
print(f"\nBest CV Net Benefit: ${grid_search.best_score_:,.0f}")
print(f"  - averaged across the 5 CV folds (each scored on ~12 frauds = low absolute value)")
print(f"  - this is a RANKING signal for the search, not the number you would ship to production")
print(f"  - the real test-set Net Benefit appears in STEP 6 (and will be higher)")

best_pipeline = grid_search.best_estimator_

# =============================================================================
# STEP 6: Evaluate on test set (threshold 0.5)
# =============================================================================
print("\n" + "-" * 70)
print("STEP 6: Evaluating on Test Set")
print("-" * 70)

y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
metrics = print_metrics('p5_02 GridSearch (cw in grid, t=0.5)', y_test, y_pred, y_pred_proba)

# =============================================================================
# STEP 7: Save artifact + model + metrics
# =============================================================================
print("\n" + "-" * 70)
print("STEP 7: Saving Pipeline + Config")
print("-" * 70)

# Save best config artifact - this is what p5_03 (production) reads.
# best_params_clean is written as-is; YAML stores class_weight={0:1, 1:100}
# with its int keys intact so p5_03 can pass it straight to sklearn.
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
print(f"Saved artifact: {ARTIFACT_FILE}")

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

# =============================================================================
# Improvement Journey block
# =============================================================================
print("\n" + "=" * 70)
print("IMPROVEMENT JOURNEY SO FAR")
print("=" * 70)
nb = metrics['business_metrics']['net_benefit']
print(f"\n Stage 1 - Baseline (p3_01, no FE):           Net Benefit=${TRUE_BASELINE_NET_BENEFIT:,}")
print(f" Stage 2 - + FE1 (p4_03, + amount_ratio):     Net Benefit=${FE1_NET_BENEFIT:,}")
print(f" Stage 3 - + RandomSearch (p5_01):            (failed - too narrow grid)")
print(f" Stage 4 - + GridSearch w/ cw (p5_02):        Net Benefit=${nb:,}   (+${nb - FE1_NET_BENEFIT:,} vs FE1)")
print(f"\n Total change vs Stage 1 baseline:            +${nb - TRUE_BASELINE_NET_BENEFIT:,}")
print(f"\n p5_02's best config is saved to: {ARTIFACT_FILE}")
print(f" This YAML file is what p5_03 will read to train the production model.")
print(f"\n Run next: python p5_03_train_production_model.py (final deployable model)")
print("=" * 70)
