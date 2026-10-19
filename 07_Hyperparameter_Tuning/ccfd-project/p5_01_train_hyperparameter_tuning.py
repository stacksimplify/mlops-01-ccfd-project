"""
Credit Card Fraud Detection - Step 1: First-Pass Hyperparameter Tuning (RandomizedSearchCV)

The first attempt at hyperparameter tuning. Uses RandomizedSearchCV to sample
20 random combinations from a standard parameter space with the default
class_weight options ('balanced', None).

This script INTENTIONALLY shows that a generic hyperparameter search
can fail to beat the FE1 baseline. The reason - and the fix - come in p5_02.

DATA SOURCE: data/credit_card_transactions_fe1_amount_ratio.csv

OUTPUT:
 - models/p5_01_randomsearch/model_latest.pkl
 - results/logistic_regression_p5_01_randomsearch_*.json

Improvement Journey position: STAGE 3 (parameter tuning - typically hurts).
"""

import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, calculate_business_metrics, save_metrics_json

print("=" * 70)
print("p5_01 - HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("=" * 70)

# =============================================================================
# Configuration + Improvement Journey reference values
# =============================================================================
FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'

# Improvement Journey hardcoded references (for the comparison block at end)
TRUE_BASELINE_F1 = 0.2329
TRUE_BASELINE_NET_BENEFIT = 39625

FE1_F1 = 0.2347
FE1_NET_BENEFIT = 39700

# =============================================================================
# STEP 1: Load FE1 data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 1: Loading Data")
print("-" * 70)

if not os.path.exists(FE_DATA_FILE):
    print(f"ERROR: {FE_DATA_FILE} not found.")
    print(f"Run p4_01_feature_engineering_fe1.py first.")
    exit(1)

df = pd.read_csv(FE_DATA_FILE)
print(f"Loaded {len(df):,} transactions")

# =============================================================================
# STEP 2: Split data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 2: Splitting Data")
print("-" * 70)

X_train, X_test, y_train, y_test, feature_info = split_data(df)

# =============================================================================
# STEP 3: Build Pipeline (same create_pipeline helper as p3_01 / p4_03)
# =============================================================================
print("\n" + "-" * 70)
print("STEP 3: Creating Pipeline for Tuning")
print("-" * 70)

# Note: create_pipeline names the model step 'model' internally, so the
# search space below uses the 'model__' prefix to target it.
pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols']
)
print("Pipeline created via create_pipeline() helper (same as p3_01 / p4_03)")

# =============================================================================
# STEP 4: Custom Business Scorer - optimise dollars (Net Benefit), not F1
# =============================================================================
print("\n" + "-" * 70)
print("STEP 4: Defining Business-Value Scorer (dollars, not F1)")
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
print("Scorer ready: RandomizedSearchCV will rank candidates by Net Benefit, not F1 or accuracy.")

# =============================================================================
# Hyperparameter Search Space
# =============================================================================
# This is a STANDARD parameter space - the kind you would write on first attempt.
# Five hyperparameters of LogisticRegression, with these candidate values.
param_distributions = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
    'model__solver': ['saga'],
    'model__max_iter': [2000, 5000, 10000],
    # THE NARROW PART - only 2 values. This is exactly why p5_01 will lose money.
    # p5_02 widens this to 5 money-weighted class_weight values.
    'model__class_weight': ['balanced', None]
}

# =============================================================================
# STEP 5: Run RandomizedSearchCV
# =============================================================================
# THE SEARCH SPACE
#   The product of the number of values per knob defines how many unique
#   combinations exist in this grid:
#     6 C  x  5 l1_ratio  x  1 solver  x  3 max_iter  x  2 class_weight  =  180 combinations
#
# THE PLAN
#   RandomizedSearchCV does NOT try all 180. It samples 20 random combos
#   (n_iter=20 below). Fast first pass.
#
# THE WORK
#   Each of the 20 picks gets 5-fold cross-validation = 20 x 5 = 100 training jobs.
#   sklearn confirms this with its "Fitting 5 folds for each of 20 candidates..." line.
#
# verbose=2 below makes sklearn print one [CV] END line per fit so you see the
# search exploring the parameter space in real time.
# =============================================================================
print("\n" + "-" * 70)
print("STEP 5: Running Hyperparameter Search")
print("-" * 70)

print("The Space:  6 C  x  5 l1_ratio  x  1 solver  x  3 max_iter  x  2 class_weight  =  180 combinations")
print("  -> class_weight = ['balanced', None] only  -  this is the NARROW part")
print("The Plan:   pick 20 of those 180 randomly  (RandomizedSearchCV)")
print("The Work:   each of the 20 picks gets 5-fold cross-validation")
print("            20 x 5 = 100 training jobs")
print("\nverbose=2 below - you will see one line per fit as the search runs.\n")

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,                 # THE PLAN - sample 20 of the 180 combinations
    cv=5,                      # THE WORK - 5-fold CV per sample = 100 total fits
    scoring=business_scorer,   # custom Net Benefit scorer (dollars, not F1)
    n_jobs=-1,                 # use all CPU cores
    verbose=2,                 # one [CV] END line per fit so you see the search working
    random_state=42            # reproducibility
)

random_search.fit(X_train, y_train)

# =============================================================================
# STEP 6: Show Best Parameters
# =============================================================================
print("\n" + "-" * 70)
print("STEP 6: Best Parameters Found")
print("-" * 70)

print(f"\nBEST PARAMETERS:")
for param, value in random_search.best_params_.items():
    clean_param = param.replace('model__', '')
    print(f"  {clean_param}: {value}")

print(f"\nBest CV Net Benefit: ${random_search.best_score_:,.0f}")
print(f"  - averaged across the 5 CV folds (each scored on ~12 frauds = low absolute value)")
print(f"  - this is a RANKING signal for the search, not the number you would ship to production")
print(f"  - the real test-set Net Benefit appears in STEP 7 (and will be higher)")

best_pipeline = random_search.best_estimator_

# =============================================================================
# STEP 7: Evaluate on Test Set
# =============================================================================
print("\n" + "-" * 70)
print("STEP 7: Evaluating on Test Set")
print("-" * 70)

y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

metrics = print_metrics('p5_01 RandomSearch (default threshold 0.5)', y_test, y_pred, y_pred_proba)

# =============================================================================
# STEP 8: Save Pipeline + Metrics
# =============================================================================
print("\n" + "-" * 70)
print("STEP 8: Saving Pipeline")
print("-" * 70)

best_params_clean = {k.replace('model__', ''): v for k, v in random_search.best_params_.items()}

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

# =============================================================================
# Improvement Journey block
# =============================================================================
print("\n" + "=" * 70)
print("IMPROVEMENT JOURNEY SO FAR")
print("=" * 70)

tuned_f1 = metrics['f1']
tuned_nb = metrics['business_metrics']['net_benefit']
fe1_delta = FE1_NET_BENEFIT - TRUE_BASELINE_NET_BENEFIT
tuned_delta_vs_fe1 = tuned_nb - FE1_NET_BENEFIT
tuned_delta_vs_baseline = tuned_nb - TRUE_BASELINE_NET_BENEFIT

print(f"\n Stage 1 - Baseline (p3_01, no FE):           F1={TRUE_BASELINE_F1:.4f}   Net Benefit=${TRUE_BASELINE_NET_BENEFIT:,}")
print(f" Stage 2 - + FE1 (p4_03, + amount_ratio):     F1={FE1_F1:.4f}   Net Benefit=${FE1_NET_BENEFIT:,}   ({'+' if fe1_delta >= 0 else ''}${fe1_delta:,} vs Stage 1)")
print(f" Stage 3 - + RandomSearch (p5_01):            F1={tuned_f1:.4f}   Net Benefit=${tuned_nb:,}   ({'+' if tuned_delta_vs_fe1 >= 0 else ''}${tuned_delta_vs_fe1:,} vs Stage 2)")
print(f"\n Total change vs Stage 1 baseline:            {'+' if tuned_delta_vs_baseline >= 0 else ''}${tuned_delta_vs_baseline:,}")

if tuned_delta_vs_fe1 > 0:
    print(f"\n RandomSearch added ${tuned_delta_vs_fe1:,} on top of FE1.")
else:
    print(f"\n RandomSearch did NOT improve on FE1 - lost ${abs(tuned_delta_vs_fe1):,}.")
    print(f" The search space here is too narrow on class_weight.")
    print(f" p5_02 will show the fix.")

print(f"\n Run next: python p5_02_class_weight_gridsearch.py")
print("=" * 70)
