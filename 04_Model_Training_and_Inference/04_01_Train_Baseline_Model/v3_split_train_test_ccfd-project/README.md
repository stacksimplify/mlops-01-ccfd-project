# v3 - Split Train/Test

## Progress

```
  ✅ STEP 1: Load Data
  🟡 STEP 2: Split Train/Test       ← NEW IN THIS VERSION
  ⬜ STEP 3: Create Pipeline
  ⬜ STEP 4: Train Pipeline
  ⬜ STEP 5: Evaluate & Metrics
  ⬜ STEP 6: Save Pipeline
  ⬜ STEP 7: Save Metrics JSON
```

## What's New

| File | Change |
|------|--------|
| `utils/data_utils.py` | Implemented `split_data()` and `get_feature_columns()` |
| `utils/__init__.py` | Exports `split_data`, `get_feature_columns` |
| `p3_01_train_model_baseline.py` | Added STEP 2: stratified train/test split |

## New Functions

```python
# utils/data_utils.py
def get_feature_columns(df):
    """Return categorical and numerical column lists."""

def split_data(df, test_size=0.2, random_state=42):
    """Stratified train/test split preserving fraud ratio."""
```

## Key Concept: Stratified Split

We use `stratify=y` to ensure the fraud ratio (~3%) is preserved in both train and test sets. Without this, the test set might have 0% or 10% fraud by random chance.

## How to Run

```bash
cd v3_split_train_test_ccfd-project/

# PRE-REQUISITE: Generate data first
python p1_01_generate_initial_dataset.py

# Run training script
python p3_01_train_model_baseline.py
```

**Expected Output:**
```
STEP 1: Loading Data
Loaded 10,000 transactions from data/credit_card_transactions_latest.csv

STEP 2: Splitting Data
Split: 8,000 train | 2,000 test
   Fraud rate - Train: 2.99% | Test: 2.95%

Feature Information:
   Total features: 8
   Categorical: ['merchant_category', 'card_present', 'international']
   Numerical: ['transaction_amount', 'transaction_hour', 'days_since_last_txn', 'avg_transaction_amount', 'transaction_count_24h']
```

## Next Version

**v4** → Implement `create_preprocessor()` and `create_pipeline()` - build and train the sklearn Pipeline.
