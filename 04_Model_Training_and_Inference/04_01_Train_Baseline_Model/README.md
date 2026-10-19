---
title: "04_01 - Train Baseline Model"
description: "Train a baseline Logistic Regression model for credit card fraud detection using sklearn Pipeline - built incrementally from v1 to v7."
---

# 04_01 - Train Baseline Model

Train a baseline Logistic Regression model for credit card fraud detection using sklearn Pipeline with business metrics.

Students build the training script **incrementally** across 7 versions - each version adds exactly ONE concept.


## Pre-requisite: Python Environment Setup

```bash
# Create conda environment
conda create -n mlops-env1 python=3.14 -c conda-forge -y

# Activate environment
conda activate mlops-env1

# Install dependencies (locked versions)
cd ccfd-project
pip install -r requirements.txt
```

> **Note:** All sections in this course use the same `mlops-env1` environment. You only need to create it once. After that, just activate it with `conda activate mlops-env1` before running any scripts.

---

---

## Incremental Build: v1 → v7

```
TRAINING PIPELINE - BUILD IT STEP BY STEP
==========================================

  v1          v2          v3           v4            v5            v6           v7
  Skeleton    Load        Split        Pipeline      Evaluate      Save         Save
  (stubs)     Data        Train/Test   & Train       & Metrics     Pipeline     Metrics
                                                                                JSON
  ┌─────┐   ┌─────┐    ┌─────┐     ┌─────┐       ┌─────┐      ┌─────┐      ┌─────┐
  │     │   │STEP1│    │STEP1│     │STEP1│       │STEP1│      │STEP1│      │STEP1│
  │     │   │Load │───>│Load │───> │Load │───>   │Load │───>  │Load │───>  │Load │
  │     │   │     │    │STEP2│     │STEP2│       │STEP2│      │STEP2│      │STEP2│
  │ ALL │   │     │    │Split│───> │Split│───>   │Split│───>  │Split│───>  │Split│
  │STUBS│   │     │    │     │     │STEP3│       │STEP3│      │STEP3│      │STEP3│
  │     │   │     │    │     │     │Pipe │───>   │Pipe │───>  │Pipe │───>  │Pipe │
  │     │   │     │    │     │     │STEP4│       │STEP4│      │STEP4│      │STEP4│
  │     │   │     │    │     │     │Train│───>   │Train│───>  │Train│───>  │Train│
  │     │   │     │    │     │     │     │       │STEP5│      │STEP5│      │STEP5│
  │     │   │     │    │     │     │     │       │Eval │───>  │Eval │───>  │Eval │
  │     │   │     │    │     │     │     │       │     │      │STEP6│      │STEP6│
  │     │   │     │    │     │     │     │       │     │      │Save │───>  │Save │
  │     │   │     │    │     │     │     │       │     │      │Model│      │Model│
  │     │   │     │    │     │     │     │       │     │      │     │      │STEP7│
  │     │   │     │    │     │     │     │       │     │      │     │      │Save │
  │     │   │     │    │     │     │     │       │     │      │     │      │JSON │
  └─────┘   └─────┘    └─────┘     └─────┘       └─────┘      └─────┘      └─────┘
```

---

## Version Summary

| Version | Folder | What's New | New Functions |
|---------|--------|------------|---------------|
| **v1** | `v1_base_ccfd-project/` | Project skeleton - all files are stubs | - |
| **v2** | `v2_load_data_ccfd-project/` | STEP 1: Load CSV data | `load_data()` |
| **v3** | `v3_split_train_test_ccfd-project/` | STEP 2: Stratified train/test split | `split_data()`, `get_feature_columns()` |
| **v4** | `v4_create_and_train_pipeline_ccfd-project/` | STEP 3-4: Create & train sklearn Pipeline | `create_preprocessor()`, `create_pipeline()` |
| **v5** | `v5_evaluate_and_metrics_ccfd-project/` | STEP 5: Evaluate with technical + business metrics | `print_metrics()`, `calculate_business_metrics()` |
| **v6** | `v6_save_pipeline_ccfd-project/` | STEP 6: Save trained pipeline to disk | `save_pipeline()` |
| **v7** | `v7_save_metrics_ccfd-project/` | STEP 7: Save metrics JSON for tracking | `save_metrics_json()` |
| **Final** | `ccfd-project/` | Complete reference with full docstrings | All functions |

---

## Project Structure (Final - ccfd-project)

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

---

## How to Run (Any Version)

```bash
# 1. Navigate to any version folder
cd v3_split_train_test_ccfd-project/       # or any v1-v7, or ccfd-project/

# 2. Generate data (required first - creates data/ folder)
python p1_01_generate_initial_dataset.py

# 3. Run the training script
python p3_01_train_model_baseline.py
```

> **Note:** v1 is stubs only - nothing runs. v2+ will execute the steps implemented so far.

---

## What the Pipeline Does

```
Input (8 raw features)
    │
    ▼
ColumnTransformer (Preprocessor)
├── OneHotEncoder  → 3 categorical (merchant_category, card_present, international)
└── StandardScaler → 5 numerical   (transaction_amount, transaction_hour, etc.)
    │
    ▼
LogisticRegression (C=1.0, class_weight='balanced')
    │
    ▼
Output: FRAUD / LEGITIMATE (with probability)
```

---

## Utils Module - Function Inventory

### data_utils.py

| Function | Added In | Purpose |
|----------|----------|---------|
| `load_data()` | v2 | Load CSV → DataFrame |
| `get_feature_columns()` | v3 | Return categorical + numerical column lists |
| `split_data()` | v3 | Stratified train/test split with feature info |
| `create_preprocessor()` | v4 | ColumnTransformer (OneHotEncoder + StandardScaler) |

### model_utils.py

| Function | Added In | Purpose |
|----------|----------|---------|
| `create_pipeline()` | v4 | Build sklearn Pipeline (preprocessor + model) |
| `save_pipeline()` | v6 | Save pipeline pickle + metadata JSON |

### metrics_utils.py

| Function | Added In | Purpose |
|----------|----------|---------|
| `print_metrics()` | v5 | Print technical + business metrics |
| `calculate_business_metrics()` | v5 | Compute net benefit, fraud caught/missed |
| `save_metrics_json()` | v7 | Save metrics to results/ as JSON |

---

## Expected Output (v7 / ccfd-project)

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
Training complete!

STEP 5: Evaluating Performance
   F1 Score:     0.2329
   Precision:    0.1373
   Recall:       0.7667
   ROC AUC:      0.8837
   Net Benefit:  $39,625

STEP 6: Saving Pipeline
   Model saved to: models/p3_01_baseline/

STEP 7: Saving Metrics
   Metrics saved to: results/

======================================================================
BASELINE TRAINING COMPLETE!
======================================================================
```

**Files Created (v7):**

| File | Description |
|------|-------------|
| `data/credit_card_transactions_latest.csv` | Generated by p1_01 |
| `models/p3_01_baseline/model_latest.pkl` | Trained sklearn Pipeline |
| `models/p3_01_baseline/metadata_latest.json` | Model metadata + metrics |
| `results/logistic_regression_*.json` | Detailed metrics JSON |

---

## Understanding the Metrics

### Why Recall Matters Most in Fraud Detection

```
Cost of missing real fraud:   $1,500  (average fraud amount)
Cost of a false alarm:        $25     (investigation cost)

Missing fraud is 60x more expensive than a false alarm!
```

That's why we use `class_weight='balanced'` - prioritize catching fraud even at the cost of more false alarms.

### Business Value (Net Benefit)

```
Net Benefit = (Fraud Caught × $1,475) - (False Alarms × $25) - (Fraud Missed × $1,500)
```

Even with low precision (13.7%), the model generates **positive net benefit** because catching fraud far outweighs false alarm costs.

---

## Next Steps

| Next | Topic | What You'll Do |
|------|-------|---------------|
| [04_02_Inference_API](../04_02_Inference_API/) | Serve Predictions | Build FastAPI server, test via browser + scripts |
