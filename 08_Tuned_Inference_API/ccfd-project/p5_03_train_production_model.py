"""
Credit Card Fraud Detection - Step 3: Train Production Model (final, deployable)

Loads the winning config from p5_02 (GridSearchCV with money-weighted
class_weight). Trains the production model. No HP search. No threshold magic.
No search loop. Same result every run. Easy to put in a CI job.

DESIGN: this is the script you would put in a daily / weekly re-training job.
It reads its parameters from artifacts/p5_02_best_config.yaml, so re-running
p5_02 on fresh data automatically updates what this script uses - no manual
copy-paste required.

The artifact is YAML so class_weight (a dict with int keys like {0:1, 1:100})
round-trips cleanly between p5_02 and p5_03 - it lands here exactly as sklearn
expects it.

DATA SOURCE: data/credit_card_transactions_fe1_amount_ratio.csv
INPUT ARTIFACT: artifacts/p5_02_best_config.yaml  (produced by p5_02)
OUTPUT:
 - models/p5_03_production/model_latest.pkl
 - models/p5_03_production/metadata_latest.json
 - results/logistic_regression_p5_03_production_*.json

Improvement Journey position: STAGE 5 (final production model, deployable).
"""

import os
import pandas as pd
import yaml   # YAML preserves int dict keys (used by class_weight).

from utils.data_utils import split_data, get_feature_columns
from utils.model_utils import create_pipeline, save_pipeline
from utils.metrics_utils import print_metrics, save_metrics_json

print("=" * 70)
print("p5_03 - TRAIN PRODUCTION MODEL (uses p5_02 winning config)")
print("=" * 70)

# =============================================================================
# Configuration
# =============================================================================
FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'
INPUT_ARTIFACT = 'artifacts/p5_02_best_config.yaml'

TRUE_BASELINE_NET_BENEFIT = 39625
FE1_NET_BENEFIT = 39700

# =============================================================================
# STEP 1: Load config from p5_02 artifact
# =============================================================================
print("\n" + "-" * 70)
print("STEP 1: Loading Best Config from p5_02")
print("-" * 70)

if not os.path.exists(INPUT_ARTIFACT):
    print(f"ERROR: {INPUT_ARTIFACT} not found.")
    print(f"Run p5_02_class_weight_gridsearch.py first.")
    exit(1)

with open(INPUT_ARTIFACT) as f:
    config = yaml.safe_load(f)

# best_params comes back exactly as p5_02 saved it - class_weight={0:1, 1:100}
# with int keys, ready to hand to sklearn as-is.
best_params = config['best_params']

print(f"Production config (from p5_02):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# =============================================================================
# STEP 2: Load + split data
# =============================================================================
print("\n" + "-" * 70)
print("STEP 2: Loading + Splitting Data")
print("-" * 70)

df = pd.read_csv(FE_DATA_FILE)
X_train, X_test, y_train, y_test, feature_info = split_data(df)
print(f"Loaded {len(df):,} transactions")

# =============================================================================
# STEP 3: Train with loaded config (same create_pipeline helper, no search)
# =============================================================================
print("\n" + "-" * 70)
print("STEP 3: Training Production Model (no search, just fit)")
print("-" * 70)

pipeline = create_pipeline(
    categorical_cols=feature_info['categorical_cols'],
    numerical_cols=feature_info['numerical_cols'],
    model_params=best_params
)
pipeline.fit(X_train, y_train)
print("Production model trained.")

# =============================================================================
# STEP 4: Evaluate on test set (threshold 0.5 - no threshold magic)
# =============================================================================
print("\n" + "-" * 70)
print("STEP 4: Evaluating Production Model")
print("-" * 70)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
metrics = print_metrics('Production Model (p5_03)', y_test, y_pred, y_pred_proba)
print("\nEvaluation complete.")

# =============================================================================
# STEP 5: Save production model + metrics
# =============================================================================
print("\n" + "-" * 70)
print("STEP 5: Saving Production Model")
print("-" * 70)

# Save the trained pipeline plus an audit record of what trained it.
# save_pipeline / save_metrics_json write human-readable audit logs under
# models/ and results/ - these are written once for the record, not read
# back by code.
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

# =============================================================================
# Final summary - full Improvement Journey
# =============================================================================
print("\n" + "=" * 70)
print("FULL IMPROVEMENT JOURNEY (final)")
print("=" * 70)
nb = metrics['business_metrics']['net_benefit']
print(f"""
  Stage 1 - Baseline (p3_01, no FE):           ${TRUE_BASELINE_NET_BENEFIT:,}
  Stage 2 - + FE1 (p4_03, amount_ratio):       ${FE1_NET_BENEFIT:,}   (+${FE1_NET_BENEFIT - TRUE_BASELINE_NET_BENEFIT:,})
  Stage 3 - + RandomSearch (p5_01):             typically HURT (narrow grid)
  Stage 4 - + GridSearch w/ cw (p5_02):         ${config['test_net_benefit']:,}   (finds the right params)
  Stage 5 - Production model (p5_03, this):    ${nb:,}

  Total change vs Stage 1 baseline:            +${nb - TRUE_BASELINE_NET_BENEFIT:,}

  The production model is saved at models/p5_03_production/
  Re-train it any time by re-running p5_02 (updates artifact) then this script.
""")
print("=" * 70)
