"""
Credit Card Fraud Detection - Data Utilities (Basics)

Pipeline-based approach for sklearn compatibility.

Functions:
- load_data: Load credit card transactions from CSV
- get_feature_columns: Get categorical and numerical column names
- create_preprocessor: Create ColumnTransformer for preprocessing
- split_data: Split data into train/test sets
"""

import pandas as pd

# Load Data 
def load_data(filepath='data/credit_card_transactions_latest.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions from {filepath}")
    return df 


