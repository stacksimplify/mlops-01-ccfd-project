"""
Credit Card Fraud Detection - Feature Engineering Utilities

This module contains functions to create NEW features from existing data.

Functions:
- apply_feature_engineering: Add engineered features and SAVE to CSV
"""

# -----------------------------------------------------------------------------
# Step 1: Imports
# - pandas: DataFrame operations
# - numpy: numerical helpers (kept for future feature transforms)
# - os: filesystem helpers (create directories, save files)
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os


# -----------------------------------------------------------------------------
# Step 2: Define the apply_feature_engineering function (WITH TYPE HINTS)
# - Add type hints for parameters and return value
#   (enables IDE autocomplete like df.copy(), doubles as documentation)
# Parameters:
#   - df: pd.DataFrame - Raw dataset with original features
#   - output_filepath: str | None - Path to save feature-engineered CSV (default None)
# Returns:
#   - pd.DataFrame - Dataset with new engineered features added
# -----------------------------------------------------------------------------
def apply_feature_engineering(df: pd.DataFrame, output_filepath: str | None = None) -> pd.DataFrame:
    """
    Apply feature engineering transformations and optionally save to CSV

    New Features Added:
    - amount_ratio: transaction_amount / avg_transaction_amount
      (High ratio = suspicious behavior - spending more than usual)

    Args:
        df (pd.DataFrame): Raw dataset with original features
        output_filepath (str, optional): Path to save feature-engineered CSV

    Returns:
        pd.DataFrame: Dataset with new engineered features added
    """
    # -------------------------------------------------------------------------
    # Step 3: Make a defensive copy of the DataFrame
    # (so we don't accidentally modify the original passed by the caller)
    # -------------------------------------------------------------------------
    df = df.copy()

    # -------------------------------------------------------------------------
    # Step 4: FE1 - Create amount_ratio feature
    # Formula: amount_ratio = transaction_amount / avg_transaction_amount
    # Use .replace(0, 1) on the denominator to avoid divide-by-zero
    # -------------------------------------------------------------------------
    df['amount_ratio'] = df['transaction_amount'] / df['avg_transaction_amount'].replace(0, 1)

    # -------------------------------------------------------------------------
    # Step 5: Count total features (everything except target column 'is_fraud')
    # Used in the print message below to confirm a new column was added
    # -------------------------------------------------------------------------
    n_features = len([col for col in df.columns if col != 'is_fraud'])
    print(f"Added: amount_ratio ({n_features} total features)")

    # -------------------------------------------------------------------------
    # Step 6: Save to CSV (only if output_filepath was provided)
    # - Create parent directory if it doesn't exist
    # - Save WITHOUT the auto-generated row index column
    # -------------------------------------------------------------------------
    if output_filepath:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        df.to_csv(output_filepath, index=False)
        print(f"Saved: {output_filepath}")

    # -------------------------------------------------------------------------
    # Step 7: Return the DataFrame with the new engineered feature
    # -------------------------------------------------------------------------
    return df
