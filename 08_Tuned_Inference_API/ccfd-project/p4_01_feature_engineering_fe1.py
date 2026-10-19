"""
Credit Card Fraud Detection - Feature Engineering Script

This script applies feature engineering and SAVES the result to a CSV file.

DATA SOURCE: Uses data/credit_card_transactions_latest.csv

INPUT: data/credit_card_transactions_latest.csv (8 features + target)
OUTPUT: data/credit_card_transactions_fe1_amount_ratio.csv (9 features + target)

NEW FEATURE: amount_ratio = transaction_amount / avg_transaction_amount
"""

# -----------------------------------------------------------------------------
# Step 1: Imports
# - pandas: load CSV into a DataFrame
# - apply_feature_engineering: utility we wrote in feature_engineering_utils.py
# -----------------------------------------------------------------------------
import pandas as pd
from utils.feature_engineering_utils import apply_feature_engineering

# -----------------------------------------------------------------------------
# Header banner (printed at script start)
# -----------------------------------------------------------------------------
print("="*70)
print("FEATURE ENGINEERING: Creating amount_ratio Feature")
print("="*70)

# =============================================================================
# Configuration - file paths (Step 0)
# =============================================================================
# Step 0a: INPUT - raw 8-feature dataset (the CSV from p1_01)
INPUT_FILE = 'data/credit_card_transactions_latest.csv'
# Step 0b: OUTPUT - 9-feature dataset with amount_ratio added (what p4_03 reads)
OUTPUT_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'

# =============================================================================
# STEP 1: Load Raw Data
# =============================================================================
# Print step banner
print("\n" + "-"*70)
print("STEP 1: Loading Raw Data")
print("-"*70)

# Step 1a: Read CSV into pandas DataFrame
df_raw = pd.read_csv(INPUT_FILE)

# Step 1b: Print row + column count to confirm data loaded correctly
print(f"Loaded {len(df_raw):,} transactions ({len(df_raw.columns)} columns)")

# =============================================================================
# STEP 2: Apply Feature Engineering
# =============================================================================
# Print step banner
print("\n" + "-"*70)
print("STEP 2: Applying Feature Engineering")
print("-"*70)

# Step 2a: Call utility function - creates amount_ratio column AND saves to CSV
df_fe = apply_feature_engineering(df_raw, output_filepath=OUTPUT_FILE)

# =============================================================================
# STEP 3: Verify the New Feature
# =============================================================================
# Print step banner
print("\n" + "-"*70)
print("STEP 3: Verifying Results")
print("-"*70)

# Step 3a: Compare column counts (raw=9 vs FE=10, where 10 = 9 features + is_fraud target)
print(f"Raw columns: {len(df_raw.columns)} → FE columns: {len(df_fe.columns)}")

# Step 3b: Confirm the new feature name
print(f"New feature added: amount_ratio")

# Step 3c: Print closing banner
print("\n" + "="*70)
print("FEATURE ENGINEERING COMPLETE!")
print("="*70)

# Step 3d: Print output file location for reference
print(f"\n Output: {OUTPUT_FILE}")
