"""
Auto-EDA with Sweetviz (Best for Dataset Comparisons)

Sweetviz specializes in SIDE-BY-SIDE DATASET COMPARISONS - perfect for
our fraud-vs-legitimate use case. Two lines of code generates an
interactive HTML report that shows:
  - Every feature's distribution for class A vs class B overlaid
  - Highlights features with strongest target correlation
  - Auto-flags missing values, skew, cardinality
  - Beautiful, self-contained HTML

This script produces 2 reports:
  1. General feature analysis with target annotation
  2. Fraud vs Legit side-by-side comparison (the MAIN report)

=============================================================================
Environment Setup
=============================================================================
Sweetviz lives in the SAME dedicated `eda-tools-env` environment we used for
fg-data-profiling (see p2_02). This keeps ALL auto-EDA tools isolated from
the core `mlops-env1` environment - your locked requirements.txt stays
clean and untouched.

If you already created `eda-tools-env` (in the fg-data-profiling step),
just activate it and add sweetviz on top:

  conda activate eda-tools-env

  # ─────────────────────────────────────────────────────────────────────────
  # Pick ONE of the following THREE install approaches:
  # (No need to install pandas separately - it's already in eda-tools-env.)
  # ─────────────────────────────────────────────────────────────────────────
  #
  # OPTION A - install the LATEST version (NOT RECOMMENDED for this course)
  #   ⚠️  UI/output may differ from the screenshots and reports shown in lectures.
  pip install sweetviz
  #
  # OPTION B - install the EXACT version used while recording this course
  #   (use this if you want a 1:1 match with the lectures, but only for THIS tool)
  pip install sweetviz==2.3.3
  #
  # OPTION C - RECOMMENDED - install ALL auto-EDA tools at once (locked versions)
  #   (one shot - installs fg-data-profiling AND sweetviz at versions used in this course)
  pip install -r requirements-auto-eda-tools.txt
  # ─────────────────────────────────────────────────────────────────────────

If you have NOT created `eda-tools-env` yet, follow the setup steps in
p2_02_auto_eda_ydata_profiling.py first, then come back here.

=============================================================================
Usage
=============================================================================
  # Step 1: Generate data (run once from main env)
  conda activate mlops-env1
  python p1_01_generate_initial_dataset.py

  # Step 2: Switch env and run auto-EDA
  conda activate eda-tools-env
  python p2_03_auto_eda_sweetviz.py

  # Step 3: View reports
  open auto_eda_reports/sweetviz/*.html

=============================================================================
Note on Sweetviz Type Detection
=============================================================================
Sweetviz uses CARDINALITY-based type detection. Low-unique-value integer
columns (like transaction_hour with 24 values, or transaction_count_24h
with ~6 values, or is_fraud with 2 values) are classified as CATEGORICAL
even though they're int64 in pandas.

This is usually correct for ML: transaction_hour=14 isn't "twice as much"
as transaction_hour=7 - it's a different CATEGORY of time-of-day.
"""

import os
import time
from pathlib import Path

import pandas as pd
import sweetviz as sv

DATA_FILE = 'data/credit_card_transactions_latest.csv'
OUTPUT_DIR = 'auto_eda_reports/sweetviz'

assert os.path.exists(DATA_FILE), f"Run p1_01_generate_initial_dataset.py first - missing {DATA_FILE}"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AUTO-EDA: Sweetviz - fraud vs legit side-by-side")
print("=" * 70)

df = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(df):,} transactions, fraud rate {df['is_fraud'].mean():.2%}")

# ---------------------------------------------------------------------------
# Report 1: General feature analysis with target annotation
# ---------------------------------------------------------------------------
print(f"\n[1/2] Generating general feature analysis report...")
t0 = time.time()

report1 = sv.analyze(
    [df, "CCFD Dataset"],
    target_feat='is_fraud',
)
report1_path = os.path.join(OUTPUT_DIR, 'ccfd_general_analysis.html')
report1.show_html(report1_path, open_browser=False)

print(f"   Saved: {report1_path} ({os.path.getsize(report1_path) // 1024} KB, {time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Report 2: COMPARE fraud vs legit - the main Section 05 story
# ---------------------------------------------------------------------------
print(f"\n[2/2] Generating fraud vs legit comparison report...")
t0 = time.time()

# Drop is_fraud - it's constant in each slice (all 1s or all 0s) which
# creates misleading 'perfect' associations in the Associations matrix.
fraud_df = df[df['is_fraud'] == 1].drop(columns=['is_fraud']).copy()
legit_df = df[df['is_fraud'] == 0].drop(columns=['is_fraud']).copy()

print(f"       fraud rows: {len(fraud_df):,}")
print(f"       legit rows: {len(legit_df):,}")

report2 = sv.compare(
    [fraud_df, "Fraud"],
    [legit_df, "Legitimate"],
)
report2_path = os.path.join(OUTPUT_DIR, 'ccfd_fraud_vs_legit_comparison.html')
report2.show_html(report2_path, open_browser=False)

print(f"   Saved: {report2_path} ({os.path.getsize(report2_path) // 1024} KB, {time.time()-t0:.1f}s)")

print(f"\n{'=' * 70}")
print(f"Sweetviz complete.")
print(f"View reports:")
print(f"  open {report1_path}")
print(f"  open {report2_path}")
print(f"{'=' * 70}")
