"""
Credit Card Fraud Detection - EDA for Feature Engineering 1: Amount Ratio

Analyzes the NEW feature: amount_ratio
Run AFTER p4_01_feature_engineering_fe1.py which creates the FE CSV.

DATA SOURCE: data/credit_card_transactions_fe1_amount_ratio.csv

SECTIONS:
1. Load FE Data
2. Distribution Analysis
3. Correlation Comparison

OUTPUT: Images 07_fe1_*, 08_fe1_*, 09_fe1_* in 'eda_plots/' folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def main():
    """Run FE1 amount_ratio EDA pipeline."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    os.makedirs('eda_plots', exist_ok=True)

    # =========================================================================
    # HEADER
    # =========================================================================
    print("="*70)
    print("EDA FOR FEATURE ENGINEERING 1: AMOUNT RATIO")
    print("="*70)

    FE_DATA_FILE = 'data/credit_card_transactions_fe1_amount_ratio.csv'


    # =========================================================================
    # SECTION 1: LOAD FE DATA
    # =========================================================================
    print_section("SECTION 1: LOADING FE DATA")

    if not os.path.exists(FE_DATA_FILE):
        print(f"ERROR: {FE_DATA_FILE} not found!")
        exit(1)

    df = pd.read_csv(FE_DATA_FILE)
    print(f"Loaded {len(df):,} transactions")

    if 'amount_ratio' not in df.columns:
        print(f"ERROR: amount_ratio column not found!")
        exit(1)

    print(f"Verified: amount_ratio column exists")


    # =========================================================================
    # SECTION 2: DISTRIBUTION ANALYSIS
    # =========================================================================
    print_section("SECTION 2: DISTRIBUTION ANALYSIS")

    fraud_df = df[df['is_fraud'] == 1]
    legit_df = df[df['is_fraud'] == 0]

    print(f"\n Legitimate mean: {legit_df['amount_ratio'].mean():.2f}")
    print(f"Fraud mean: {fraud_df['amount_ratio'].mean():.2f}")

    mean_diff = fraud_df['amount_ratio'].mean() - legit_df['amount_ratio'].mean()
    print(f"Difference: {mean_diff:.2f} {'GOOD separation!' if mean_diff > 1 else ''}")

    # Plot: Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(legit_df['amount_ratio'].clip(0, 20), bins=50, alpha=0.7, label='Legitimate', color='#2ecc71')
    ax1.hist(fraud_df['amount_ratio'].clip(0, 20), bins=50, alpha=0.7, label='Fraud', color='#e74c3c')
    ax1.set_xlabel('amount_ratio (clipped at 20)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('FE1: amount_ratio Distribution by Class')
    ax1.legend()
    ax1.axvline(x=legit_df['amount_ratio'].mean(), color='#27ae60', linestyle='--')
    ax1.axvline(x=fraud_df['amount_ratio'].mean(), color='#c0392b', linestyle='--')

    ax2 = axes[1]
    df_plot = df.copy()
    df_plot['Class'] = df_plot['is_fraud'].map({0: 'Legitimate', 1: 'Fraud'})
    df_plot['amount_ratio_clipped'] = df_plot['amount_ratio'].clip(0, 30)
    sns.boxplot(x='Class', y='amount_ratio_clipped', data=df_plot, ax=ax2,
                hue='Class', palette=['#2ecc71', '#e74c3c'], legend=False)
    ax2.set_ylabel('amount_ratio (clipped at 30)')
    ax2.set_title('FE1: amount_ratio Box Plot by Class')

    plt.tight_layout()
    plt.savefig('eda_plots/07_fe1_amount_ratio_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/07_fe1_amount_ratio_distribution.png")


    # =========================================================================
    # SECTION 3: CORRELATION COMPARISON
    # =========================================================================
    print_section("SECTION 3: CORRELATION COMPARISON")

    corr_amount = df['transaction_amount'].corr(df['is_fraud'])
    corr_ratio = df['amount_ratio'].corr(df['is_fraud'])

    print(f"\n transaction_amount correlation: {corr_amount:.4f}")
    print(f"amount_ratio correlation: {corr_ratio:.4f}")

    if abs(corr_ratio) > abs(corr_amount):
        print(f"amount_ratio has STRONGER correlation with fraud!")

    # Plot: Fraud rate by amount_ratio bins
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    df['ratio_bin'] = pd.cut(df['amount_ratio'], bins=[0, 1, 2, 3, 5, 10, 100],
                             labels=['0-1', '1-2', '2-3', '3-5', '5-10', '10+'])
    fraud_by_bin = df.groupby('ratio_bin', observed=False)['is_fraud'].mean() * 100
    bars = ax1.bar(fraud_by_bin.index.astype(str), fraud_by_bin.values, color='steelblue')
    ax1.set_xlabel('amount_ratio Range')
    ax1.set_ylabel('Fraud Rate (%)')
    ax1.set_title('FE1: Fraud Rate by amount_ratio Range')
    ax1.axhline(y=df['is_fraud'].mean()*100, color='red', linestyle='--', label='Overall Avg')
    ax1.legend()

    for bar, val in zip(bars, fraud_by_bin.values):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', fontsize=9)

    ax2 = axes[1]
    sample_legit = legit_df.sample(min(500, len(legit_df)), random_state=42)
    sample_fraud = fraud_df.sample(min(100, len(fraud_df)), random_state=42)
    ax2.scatter(sample_legit['amount_ratio'].clip(0, 20), sample_legit['transaction_amount'].clip(0, 3000),
                alpha=0.5, label='Legitimate', color='#2ecc71', s=20)
    ax2.scatter(sample_fraud['amount_ratio'].clip(0, 20), sample_fraud['transaction_amount'].clip(0, 3000),
                alpha=0.7, label='Fraud', color='#e74c3c', s=30)
    ax2.set_xlabel('amount_ratio (clipped at 20)')
    ax2.set_ylabel('transaction_amount (clipped at 3000)')
    ax2.set_title('FE1: amount_ratio vs transaction_amount')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('eda_plots/08_fe1_amount_ratio_fraud_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/08_fe1_amount_ratio_fraud_comparison.png")

    # Correlation heatmap
    df_encoded = df.drop('ratio_bin', axis=1).copy()
    categorical_cols = ['merchant_category', 'card_present', 'international']
    for col in categorical_cols:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

    correlations = df_encoded.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    sns.heatmap(correlations, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap (with amount_ratio)')
    plt.tight_layout()
    plt.savefig('eda_plots/09_fe1_amount_ratio_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/09_fe1_amount_ratio_correlation.png")


    # =========================================================================
    # COMPLETE
    # =========================================================================
    print(f"\n{'='*70}")
    print("FE1 EDA COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
