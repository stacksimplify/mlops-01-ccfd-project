# v2 - Load Data

## Progress

```
  🟡 STEP 1: Load Data              ← NEW IN THIS VERSION
  ⬜ STEP 2: Split Train/Test
  ⬜ STEP 3: Create Pipeline
  ⬜ STEP 4: Train Pipeline
  ⬜ STEP 5: Evaluate & Metrics
  ⬜ STEP 6: Save Pipeline
  ⬜ STEP 7: Save Metrics JSON
```

## What's New

| File | Change |
|------|--------|
| `utils/data_utils.py` | Implemented `load_data()` - reads CSV into DataFrame |
| `utils/__init__.py` | Exports `load_data` |
| `p3_01_train_model_baseline.py` | Added STEP 1: calls `load_data()` |

## New Function

```python
# utils/data_utils.py
def load_data(filepath='data/credit_card_transactions_latest.csv'):
    """Load the credit card transactions dataset."""
    df = pd.read_csv(filepath)
    return df
```

## How to Run

```bash
cd v2_load_data_ccfd-project/

# PRE-REQUISITE: Generate data first
python p1_01_generate_initial_dataset.py

# Run training script
python p3_01_train_model_baseline.py
```

**Expected Output:**
```
STEP 1: Loading Data
Loaded 10,000 transactions from data/credit_card_transactions_latest.csv
```

## Next Version

**v3** → Implement `split_data()` and add STEP 2 (stratified train/test split).
