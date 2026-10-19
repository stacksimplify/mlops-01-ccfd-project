# ccfd-project - Complete Reference

## Progress

```
  ✅ STEP 1: Load Data
  ✅ STEP 2: Split Train/Test
  ✅ STEP 3: Create Pipeline
  ✅ STEP 4: Train Pipeline
  ✅ STEP 5: Evaluate & Metrics
  ✅ STEP 6: Save Pipeline
  ✅ STEP 7: Save Metrics JSON
```

**All steps complete.** This is the final reference version with full docstrings on every function.

## What's Different From v7

| Aspect | v7 | ccfd-project |
|--------|-----|-------------|
| Code | Identical functionality | Identical functionality |
| Docstrings | Minimal | Full docstrings with Args, Returns, examples |
| Purpose | Learning (incremental build) | Reference (production-ready) |

## Project Structure

```
ccfd-project/
├── p1_01_generate_initial_dataset.py    # Generate synthetic dataset
├── p3_01_train_model_baseline.py        # Training script (7 steps)
├── requirements.txt                     # Python dependencies
└── utils/
    ├── __init__.py                      # Package exports
    ├── data_utils.py                    # load_data, split_data, create_preprocessor
    ├── model_utils.py                   # create_pipeline, save_pipeline
    └── metrics_utils.py                 # print_metrics, save_metrics_json
```

## All Functions

### data_utils.py
| Function | Purpose |
|----------|---------|
| `load_data()` | Load CSV → DataFrame |
| `get_feature_columns()` | Return categorical + numerical column lists |
| `split_data()` | Stratified train/test split with feature info |
| `create_preprocessor()` | ColumnTransformer (OneHotEncoder + StandardScaler) |

### model_utils.py
| Function | Purpose |
|----------|---------|
| `create_pipeline()` | Build sklearn Pipeline (preprocessor + model) |
| `save_pipeline()` | Save pipeline pickle + metadata JSON |

### metrics_utils.py
| Function | Purpose |
|----------|---------|
| `print_metrics()` | Print technical + business metrics |
| `calculate_business_metrics()` | Compute net benefit, fraud caught/missed |
| `save_metrics_json()` | Save metrics to results/ as JSON |

## How to Run

```bash
cd ccfd-project/

# 1. Generate data
python p1_01_generate_initial_dataset.py

# 2. Train model
python p3_01_train_model_baseline.py
```

## Next Section

| Next | Topic | What You'll Do |
|------|-------|---------------|
| [04_02_Inference_API](../../04_02_Inference_API/) | Serve Predictions | Build FastAPI server, test via browser + scripts |
