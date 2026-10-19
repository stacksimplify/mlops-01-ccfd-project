"""
Credit Card Fraud Detection - Dataset Generation

Generates synthetic credit card transaction data with realistic fraud patterns.

Dataset Characteristics:
- 10,000 transactions (9,800 legitimate, 200 fraud)
- 2% fraud rate (realistic)
- Feature overlap between classes (challenging but solvable)
- 5% noise added for realism
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

np.random.seed(42)


def generate_legitimate_transactions(n=9800):
    """Generate legitimate credit card transactions"""
    
    hour_probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                           0.03, 0.05, 0.06, 0.08, 0.08, 0.08,
                           0.08, 0.07, 0.06, 0.06, 0.06, 0.07,
                           0.08, 0.07, 0.05, 0.03, 0.02, 0.02])
    hour_probs = hour_probs / hour_probs.sum()
    
    n_low_value = int(n * 0.85)
    n_high_value = n - n_low_value
    
    low_value_amounts = np.random.gamma(shape=2, scale=80, size=n_low_value).clip(5, 1000)
    high_value_amounts = np.random.gamma(shape=3, scale=300, size=n_high_value).clip(500, 5000)
    
    transaction_amounts = np.concatenate([low_value_amounts, high_value_amounts])
    np.random.shuffle(transaction_amounts)
    
    merchant_categories = np.random.choice(
        ['grocery', 'gas_station', 'restaurant', 'retail', 'online_shopping',
         'pharmacy', 'entertainment', 'utilities', 'travel', 'electronics', 'furniture', 'jewelry'],
        size=n,
        p=[0.20, 0.11, 0.18, 0.12, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.01]
    )
    
    return pd.DataFrame({
        'transaction_amount': transaction_amounts,
        'transaction_hour': np.random.choice(range(24), size=n, p=hour_probs),
        'days_since_last_txn': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 14, 30], 
                                                 size=n,
                                                 p=[0.05, 0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]),
        'avg_transaction_amount': np.random.gamma(shape=2, scale=90, size=n).clip(20, 1500),
        'merchant_category': merchant_categories,
        'card_present': np.random.choice(['yes', 'no'], size=n, p=[0.70, 0.30]),
        'international': np.random.choice(['no', 'yes'], size=n, p=[0.92, 0.08]),
        'transaction_count_24h': np.random.choice([1, 2, 3, 4, 5], 
                                                   size=n,
                                                   p=[0.50, 0.30, 0.12, 0.05, 0.03]),
        'is_fraud': [0] * n
    })


def generate_fraudulent_transactions(n=200):
    """Generate fraudulent credit card transactions"""
    
    hour_probs = np.array([0.08, 0.08, 0.07, 0.06, 0.05, 0.04,
                           0.03, 0.03, 0.04, 0.05, 0.05, 0.05,
                           0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                           0.05, 0.04, 0.04, 0.03, 0.03, 0.03])
    hour_probs = hour_probs / hour_probs.sum()
    
    n_test_txn = int(n * 0.20)
    n_large_txn = n - n_test_txn
    
    test_amounts = np.random.uniform(5, 100, size=n_test_txn)
    large_amounts = np.random.uniform(500, 3000, size=n_large_txn)
    
    transaction_amounts = np.concatenate([test_amounts, large_amounts])
    np.random.shuffle(transaction_amounts)
    
    merchant_categories = np.random.choice(
        ['electronics', 'jewelry', 'online_shopping', 'travel', 
         'retail', 'gas_station', 'grocery', 'restaurant', 'entertainment'],
        size=n,
        p=[0.25, 0.15, 0.20, 0.08, 0.12, 0.08, 0.05, 0.05, 0.02]
    )
    
    return pd.DataFrame({
        'transaction_amount': transaction_amounts,
        'transaction_hour': np.random.choice(range(24), size=n, p=hour_probs),
        'days_since_last_txn': np.random.choice([0, 1, 2, 3, 4, 7], 
                                                 size=n,
                                                 p=[0.40, 0.25, 0.15, 0.10, 0.05, 0.05]),
        'avg_transaction_amount': np.random.gamma(shape=2, scale=70, size=n).clip(20, 500),
        'merchant_category': merchant_categories,
        'card_present': np.random.choice(['no', 'yes'], size=n, p=[0.70, 0.30]),
        'international': np.random.choice(['yes', 'no'], size=n, p=[0.30, 0.70]),
        'transaction_count_24h': np.random.choice([2, 3, 4, 5, 6, 7, 8], 
                                                   size=n,
                                                   p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05]),
        'is_fraud': [1] * n
    })


def add_noise(df, noise_level=0.05):
    """Add realistic noise to simulate real-world data"""
    df = df.copy()
    
    noise = np.random.normal(1.0, noise_level, size=len(df))
    df['transaction_amount'] = (df['transaction_amount'] * noise).clip(5, 5000)
    
    hour_noise = np.random.randint(-1, 2, size=len(df))
    df['transaction_hour'] = (df['transaction_hour'] + hour_noise).clip(0, 23)
    
    n_mislabel = int(len(df) * 0.01)
    mislabel_idx = np.random.choice(df.index, size=n_mislabel, replace=False)
    df.loc[mislabel_idx, 'is_fraud'] = 1 - df.loc[mislabel_idx, 'is_fraud']
    
    return df


def generate_dataset():
    """Generate complete realistic dataset"""
    
    print("="*70)
    print("🚀 CREDIT CARD FRAUD DETECTION - DATASET GENERATION")
    print("="*70)
    
    # Generate transactions
    print("\n⏳ Generating transactions...")
    legit_df = generate_legitimate_transactions(n=9800)
    print(f"   ✅ Legitimate: 9,800 transactions")
    
    fraud_df = generate_fraudulent_transactions(n=200)
    print(f"   ✅ Fraudulent: 200 transactions")
    
    # Combine and shuffle
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Add noise
    df = add_noise(df, noise_level=0.05)
    print(f"   ✅ Applied 5% noise for realism")
    
    # Round numerical columns
    df['transaction_amount'] = df['transaction_amount'].round(2)
    df['avg_transaction_amount'] = df['avg_transaction_amount'].round(2)
    
    # Calculate stats
    fraud_rate = df['is_fraud'].mean()
    legit_amounts = df[df['is_fraud']==0]['transaction_amount']
    fraud_amounts = df[df['is_fraud']==1]['transaction_amount']
    
    # Save dataset
    os.makedirs('data', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'data/credit_card_transactions_{timestamp}.csv'
    latest_filename = 'data/credit_card_transactions_latest.csv'
    
    df.to_csv(filename, index=False)
    df.to_csv(latest_filename, index=False)
    
    # Print summary
    print(f"\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    print(f"   Total Transactions: {len(df):,}")
    print(f"   Fraud Rate:         {fraud_rate*100:.2f}%")
    print(f"")
    print(f"   Legitimate: avg ${legit_amounts.mean():.2f} | range ${legit_amounts.min():.2f} - ${legit_amounts.max():.2f}")
    print(f"   Fraudulent: avg ${fraud_amounts.mean():.2f} | range ${fraud_amounts.min():.2f} - ${fraud_amounts.max():.2f}")
    
    print(f"\n" + "="*70)
    print("💾 FILES SAVED")
    print("="*70)
    print(f"   Timestamped: {filename}")
    print(f"   Latest:      {latest_filename}")
    
    print(f"\n" + "="*70)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*70)
    
    return df


if __name__ == "__main__":
    df = generate_dataset()
