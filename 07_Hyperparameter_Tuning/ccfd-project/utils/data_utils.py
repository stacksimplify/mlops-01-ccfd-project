"""
Credit Card Fraud Detection - Data Utilities

Pipeline-based approach for sklearn/MLflow/KServe compatibility.

Functions:
- load_data: Load credit card transactions from CSV
- get_feature_columns: Get categorical and numerical column names
- create_preprocessor: Create ColumnTransformer for preprocessing
- split_data: Split data into train/test sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# =============================================================================
# Feature Column Definitions
# =============================================================================
# Base features (8 features)
BASE_CATEGORICAL_COLS = ['merchant_category', 'card_present', 'international']
BASE_NUMERICAL_COLS = ['transaction_amount', 'transaction_hour', 'days_since_last_txn',
                       'avg_transaction_amount', 'transaction_count_24h']

# Feature engineered features (9 features - adds amount_ratio)
FE1_NUMERICAL_COLS = BASE_NUMERICAL_COLS + ['amount_ratio']


def load_data(filepath='data/credit_card_transactions_latest.csv'):
    """
    Load the credit card transactions dataset.
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df):,} transactions from {filepath}")
    return df


def get_feature_columns(df):
    """
    Identify categorical and numerical columns based on dataframe.
    
    Automatically detects if amount_ratio (FE1) is present.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        tuple: (categorical_cols, numerical_cols)
    """
    categorical_cols = BASE_CATEGORICAL_COLS.copy()
    
    # Check if FE1 feature is present
    if 'amount_ratio' in df.columns:
        numerical_cols = FE1_NUMERICAL_COLS.copy()
    else:
        numerical_cols = BASE_NUMERICAL_COLS.copy()
    
    return categorical_cols, numerical_cols


def create_preprocessor(categorical_cols, numerical_cols):
    """
    Create a ColumnTransformer for preprocessing.
    
    This preprocessor:
    - Encodes categorical features with OneHotEncoder (nominal categories)
    - Scales numerical features with StandardScaler

    Args:
        categorical_cols (list): List of categorical column names
        numerical_cols (list): List of numerical column names

    Returns:
        ColumnTransformer: Preprocessing transformer
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    return preprocessor


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Uses stratified split to preserve fraud ratio in both sets.
    
    Args:
        df (pd.DataFrame): Dataset with 'is_fraud' target column
        test_size (float): Fraction for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_info)
    """
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Get feature column info
    categorical_cols, numerical_cols = get_feature_columns(df)
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state, 
        stratify=y
    )
    
    print(f"✅ Split: {len(X_train):,} train | {len(X_test):,} test")
    print(f"   Fraud rate - Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")
    
    # Feature info for tracking
    feature_info = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'feature_names': categorical_cols + numerical_cols,
        'n_features': len(categorical_cols) + len(numerical_cols)
    }
    
    return X_train, X_test, y_train, y_test, feature_info
