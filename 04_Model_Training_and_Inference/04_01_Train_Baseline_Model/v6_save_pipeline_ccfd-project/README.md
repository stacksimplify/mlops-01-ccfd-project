# v6 - Save Pipeline

## Progress

```
  ✅ STEP 1: Load Data
  ✅ STEP 2: Split Train/Test
  ✅ STEP 3: Create Pipeline
  ✅ STEP 4: Train Pipeline
  ✅ STEP 5: Evaluate & Metrics
  🟡 STEP 6: Save Pipeline           ← NEW IN THIS VERSION
  ⬜ STEP 7: Save Metrics JSON
```

## What's New

| File | Change |
|------|--------|
| `utils/model_utils.py` | Implemented `save_pipeline()` - pickle + metadata JSON |
| `utils/__init__.py` | Exports `save_pipeline` |
| `p3_01_train_model_baseline.py` | Added STEP 6: save trained pipeline to disk |

## New Function

```python
# utils/model_utils.py
def save_pipeline(pipeline, model_name, metrics=None, metadata=None):
    """Save Pipeline pickle + metadata JSON to models/ directory."""
```

## What Gets Saved

```
models/p3_01_baseline/
├── model_latest.pkl              # Trained Pipeline (pickle)
├── model_20260323_143052.pkl     # Timestamped backup
├── metadata_latest.json          # Metrics + model info
└── metadata_20260323_143052.json # Timestamped backup
```

The **latest** files are always the most recent - used by the inference API later. Timestamped files provide version history.

## Key Concept: Manual Model Saving

This is the **manual way** to save models. Notice what we have to manage ourselves:
- File paths and directories
- Pickle serialization
- Metadata JSON with metrics
- Timestamped backups

> **Good to know:** tools like MLflow (not part of this course) replace all of this with `mlflow.sklearn.log_model(pipeline)` - one line instead of 30+.

## How to Run

```bash
cd v6_save_pipeline_ccfd-project/

# PRE-REQUISITE: Generate data first
python p1_01_generate_initial_dataset.py

# Run training script
python p3_01_train_model_baseline.py
```

**Expected Output (STEP 6):**
```
STEP 6: Saving Pipeline

Model Saved: p3_01_baseline
   Location: models/p3_01_baseline/
   Model: LogisticRegression
   Timestamp: 20260323_143052
```

## Next Version

**v7** → Implement `save_metrics_json()` - save detailed metrics to a JSON file for tracking.
