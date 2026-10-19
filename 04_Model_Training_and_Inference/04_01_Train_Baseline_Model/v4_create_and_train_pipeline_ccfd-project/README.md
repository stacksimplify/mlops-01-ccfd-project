# v4 - Create & Train Pipeline

## Progress

```
  ✅ STEP 1: Load Data
  ✅ STEP 2: Split Train/Test
  🟡 STEP 3: Create Pipeline         ← NEW IN THIS VERSION
  🟡 STEP 4: Train Pipeline          ← NEW IN THIS VERSION
  ⬜ STEP 5: Evaluate & Metrics
  ⬜ STEP 6: Save Pipeline
  ⬜ STEP 7: Save Metrics JSON
```

## What's New

| File | Change |
|------|--------|
| `utils/data_utils.py` | Implemented `create_preprocessor()` - ColumnTransformer |
| `utils/model_utils.py` | Implemented `create_pipeline()` - sklearn Pipeline |
| `utils/__init__.py` | Exports `create_preprocessor`, `create_pipeline` |
| `p3_01_train_model_baseline.py` | Added STEP 3 (create) + STEP 4 (train) |

## New Functions

```python
# utils/data_utils.py
def create_preprocessor(categorical_cols, numerical_cols):
    """ColumnTransformer: OneHotEncoder + StandardScaler."""

# utils/model_utils.py
def create_pipeline(categorical_cols, numerical_cols, model_params=None):
    """Build sklearn Pipeline = preprocessor + LogisticRegression."""
```

## Key Concept: Why Pipeline?

```
WITHOUT Pipeline (bad):                WITH Pipeline (good):
─────────────────────                  ─────────────────────
encoder.fit(X_train)                   pipeline.fit(X_train, y_train)
X_train_enc = encoder.transform(...)   pipeline.predict(X_test)
scaler.fit(X_train_enc)               # That's it! One object does everything.
X_train_scaled = scaler.transform(...)
model.fit(X_train_scaled, y_train)
# Easy to leak data, hard to deploy
```

Pipeline = **one object** that handles preprocessing + model. Works with MLflow, KServe, SageMaker out of the box.

## How to Run

```bash
cd v4_create_and_train_pipeline_ccfd-project/

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

STEP 3: Creating Pipeline
Pipeline created:
   1. Preprocessor (OneHotEncoder + StandardScaler)
   2. Model (LogisticRegression)

STEP 4: Training Pipeline
Training pipeline on raw data...
Training complete!
```

## Next Version

**v5** → Implement `print_metrics()` - evaluate the trained model with technical AND business metrics.
