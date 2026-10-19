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
from sklearn.model_selection import train_test_split

# Load Data
def load_data(filepath='data/credit_card_transactions_latest.csv'):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions from {filepath}")
    return df


# Feature Column Definitions
# Base features (8 features)
BASE_CATEGORICAL_COLS = ['merchant_category', 'card_present', 'international']
BASE_NUMERICAL_COLS = ['transaction_amount', 'transaction_hour', 'days_since_last_txn',
                       'avg_transaction_amount', 'transaction_count_24h']

def get_feature_columns(df):
    categorical_cols = BASE_CATEGORICAL_COLS.copy()
    numerical_cols = BASE_NUMERICAL_COLS.copy()
    return categorical_cols, numerical_cols


def split_data(df, test_size=0.2, random_state=42):
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

    print(f"Split: {len(X_train):,} train | {len(X_test):,} test")
    print(f"   Fraud rate - Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # Feature info for tracking
    feature_info = {
        'categorical_cols': categorical_cols,
        'numerical_cols': numerical_cols,
        'feature_names': categorical_cols + numerical_cols,
        'n_features': len(categorical_cols) + len(numerical_cols)
    }
    return X_train, X_test, y_train, y_test, feature_info


