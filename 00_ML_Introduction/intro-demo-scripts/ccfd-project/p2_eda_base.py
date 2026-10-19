"""
Credit Card Fraud Detection - Exploratory Data Analysis (EDA)

Comprehensive analysis of the credit card transactions dataset.

SECTIONS:
1. Dataset Overview
2. Target Distribution (Class Imbalance)
3. Feature Analysis
4. Fraud Pattern Analysis
5. Correlation Analysis

OUTPUT: Visualizations saved to 'eda_plots/' folder
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
    """Run complete EDA pipeline."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    os.makedirs('eda_plots', exist_ok=True)

    # =========================================================================
    # SECTION 1: DATASET OVERVIEW
    # =========================================================================
    print_section("SECTION 1: DATASET OVERVIEW")

    df = pd.read_csv('data/credit_card_transactions_latest.csv')

    print(f"\n Transactions: {len(df):,}")
    print(f"Features: {df.shape[1]}")
    print(f"Missing: {df.isnull().sum().sum()}")


    # =========================================================================
    # SECTION 2: TARGET DISTRIBUTION
    # =========================================================================
    print_section("SECTION 2: TARGET DISTRIBUTION")

    fraud_counts = df['is_fraud'].value_counts()
    fraud_pct = df['is_fraud'].value_counts(normalize=True) * 100

    print(f"\n Legitimate: {fraud_counts[0]:,} ({fraud_pct[0]:.1f}%)")
    print(f"Fraud: {fraud_counts[1]:,} ({fraud_pct[1]:.1f}%)")
    print(f"Imbalance: {fraud_counts[0]/fraud_counts[1]:.0f}:1")

    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#2ecc71', '#e74c3c']
    axes[0].bar(['Legitimate', 'Fraud'], fraud_counts.values, color=colors)
    axes[0].set_ylabel('Count')
    axes[0].set_title('Transaction Count by Class')
    for i, v in enumerate(fraud_counts.values):
        axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

    axes[1].pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
                colors=colors, explode=[0, 0.1], shadow=True)
    axes[1].set_title('Fraud Distribution')

    plt.tight_layout()
    plt.savefig('eda_plots/01_target_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/01_target_distribution.png")


    # =========================================================================
    # SECTION 3: FEATURE ANALYSIS
    # =========================================================================
    print_section("SECTION 3: FEATURE ANALYSIS")

    categorical_cols = ['merchant_category', 'card_present', 'international']
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['is_fraud']]

    print(f"\n Categorical: {categorical_cols}")
    print(f"Numerical: {numerical_cols}")

    # Numerical distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        df[col].hist(bins=50, ax=ax, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'{col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        stats_text = f'Mean: {df[col].mean():.1f}\nMedian: {df[col].median():.1f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('eda_plots/02_numerical_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/02_numerical_distributions.png")

    # Categorical distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        value_counts = df[col].value_counts()
        ax.bar(range(len(value_counts)), value_counts.values, color='steelblue')
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_title(f'{col}')
        ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('eda_plots/03_categorical_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/03_categorical_distributions.png")


    # =========================================================================
    # SECTION 4: FRAUD PATTERN ANALYSIS
    # =========================================================================
    print_section("SECTION 4: FRAUD PATTERN ANALYSIS")

    fraud_df = df[df['is_fraud'] == 1]
    legit_df = df[df['is_fraud'] == 0]

    print(f"\n {'Feature':<25} {'Legitimate':>12} {'Fraud':>12} {'Diff':>10}")
    print(f" {'-'*60}")

    for col in numerical_cols:
        legit_mean = legit_df[col].mean()
        fraud_mean = fraud_df[col].mean()
        diff_pct = ((fraud_mean - legit_mean) / legit_mean) * 100 if legit_mean != 0 else 0
        print(f" {col:<25} {legit_mean:>12.2f} {fraud_mean:>12.2f} {diff_pct:>+9.1f}%")

    # Box plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fraud_colors = {0: '#2ecc71', 1: '#e74c3c'}  # Green=Legit, Red=Fraud

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        sns.boxplot(x='is_fraud', y=col, data=df, ax=ax,
                    palette=fraud_colors, hue='is_fraud', legend=False)
        ax.set_title(f'{col}')
        ax.set_xlabel('Is Fraud (0=Legitimate, 1=Fraud)')
        ax.set_ylabel(col)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('eda_plots/04_fraud_vs_legitimate_numerical.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n Saved: eda_plots/04_fraud_vs_legitimate_numerical.png")

    # Fraud rate by category
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        fraud_rate = df.groupby(col)['is_fraud'].mean() * 100
        fraud_rate = fraud_rate.sort_values(ascending=False)
        colors = plt.cm.RdYlGn_r(fraud_rate.values / fraud_rate.max())
        ax.bar(range(len(fraud_rate)), fraud_rate.values, color=colors)
        ax.set_xticks(range(len(fraud_rate)))
        ax.set_xticklabels(fraud_rate.index, rotation=45, ha='right')
        ax.set_title(f'Fraud Rate by {col}')
        ax.set_ylabel('Fraud Rate (%)')
        ax.axhline(y=df['is_fraud'].mean()*100, color='red', linestyle='--')

    plt.tight_layout()
    plt.savefig('eda_plots/05_fraud_rate_by_category.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eda_plots/05_fraud_rate_by_category.png")


    # =========================================================================
    # SECTION 5: CORRELATION ANALYSIS
    # =========================================================================
    print_section("SECTION 5: CORRELATION ANALYSIS")

    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = pd.factorize(df_encoded[col])[0]

    correlations = df_encoded.corr()

    print("\n Correlation with Fraud:")
    fraud_corr = correlations['is_fraud'].drop('is_fraud').sort_values(key=abs, ascending=False)
    for feature, corr in fraud_corr.items():
        strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
        print(f" {feature:<25}: {corr:>+.4f} ({strength})")

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    sns.heatmap(correlations, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_plots/06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n Saved: eda_plots/06_correlation_heatmap.png")


    # =========================================================================
    # COMPLETE
    # =========================================================================
    print(f"\n{'='*70}")
    print("EDA COMPLETE! Check 'eda_plots/' folder for visualizations.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
