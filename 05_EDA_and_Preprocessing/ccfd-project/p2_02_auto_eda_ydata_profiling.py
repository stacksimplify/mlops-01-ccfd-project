"""
Auto-EDA with data-profiling (Industry Reference Tool)

NOTE: data-profiling is the new name for ydata-profiling
      (see PACKAGE RENAME NOTE section below for details).

data-profiling is the industry-standard Python auto-EDA tool. One line of
code generates a comprehensive HTML report with:
  - Dataset overview (rows, columns, memory, duplicates, missing values)
  - Per-variable distributions (histograms, quantiles, extreme values)
  - Correlation matrices (Pearson, Spearman, Kendall, phi-k, Cramer's V)
  - Auto-flagged warnings (high cardinality, skewness, constant columns)
  - Duplicate row detection
  - Sample rows

This script produces 2 reports:
  1. Full dataset profile (all features, target-agnostic)
  2. Fraud vs Legit comparison (two profiles side-by-side)

=============================================================================
PACKAGE RENAME NOTE (2025-2026)
=============================================================================
This tool has been through TWO renames over its 9-year history:

  pandas-profiling  (2016-2023, original)
        |
        v
  ydata-profiling   (2023-2025, ydata-labs took stewardship)
        |
        v
  fg-data-profiling (2025+, current, under Data-Centric-AI-Community org)
                    Python import is just `data_profiling` (no fg_ prefix)

Same project. Same 13.5k-star codebase. Same MIT license. Same maintainers.
Just renamed twice.

Old `ydata-profiling` is deprecated and will stop receiving updates.

GitHub repo: https://github.com/Data-Centric-AI-Community/fg-data-profiling
PyPI page:   https://pypi.org/project/fg-data-profiling/

=============================================================================
IMPORTANT: Environment Setup
=============================================================================
fg-data-profiling requires Python 3.13 (or older) due to pydantic.v1
compatibility issues with Python 3.14. It also needs setuptools<81 because
the package still imports pkg_resources (removed in newer setuptools
versions).

Create a dedicated environment for this tool:

  conda create -n eda-tools-env python=3.13 -c conda-forge -y
  conda activate eda-tools-env

  # ─────────────────────────────────────────────────────────────────────────
  # Pick ONE of the following THREE install approaches:
  # (No need to install pandas separately - it's pulled in as a dependency.)
  # ─────────────────────────────────────────────────────────────────────────
  #
  # OPTION A - install the LATEST version (NOT RECOMMENDED for this course)
  #   ⚠️  UI/output may differ from the screenshots and reports shown in lectures.
  pip install "setuptools<81"
  pip install fg-data-profiling
  #
  # OPTION B - install the EXACT version used while recording this course
  #   (use this if you want a 1:1 match with the lectures, but only for THIS tool)
  pip install "setuptools<81"
  pip install fg-data-profiling==4.19.1
  #
  # OPTION C - RECOMMENDED - install ALL auto-EDA tools at once (locked versions)
  #   (one shot - installs fg-data-profiling AND sweetviz at versions used in this course)
  pip install -r requirements-auto-eda-tools.txt
  # ─────────────────────────────────────────────────────────────────────────

Note the package name split:
  - PyPI install: `fg-data-profiling`  (with fg- prefix)
  - Python import: `data_profiling`    (no fg_ prefix)

=============================================================================
Usage
=============================================================================
  # Step 1: Generate data (run once from main env)
  conda activate mlops-env1
  python p1_01_generate_initial_dataset.py

  # Step 2: Switch env and run auto-EDA
  conda activate eda-tools-env
  python p2_02_auto_eda_ydata_profiling.py

  # Step 3: View reports
  open auto_eda_reports/data_profiling/*.html
"""

import os
import time
from pathlib import Path

import pandas as pd
from data_profiling import ProfileReport

DATA_FILE = 'data/credit_card_transactions_latest.csv'
OUTPUT_DIR = 'auto_eda_reports/data_profiling'

assert os.path.exists(DATA_FILE), f"Run p1_01_generate_initial_dataset.py first - missing {DATA_FILE}"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("AUTO-EDA: data-profiling (Industry Reference Tool)")
print("=" * 70)

df = pd.read_csv(DATA_FILE)
print(f"\nLoaded {len(df):,} transactions, fraud rate {df['is_fraud'].mean():.2%}")

# ---------------------------------------------------------------------------
# Report 1: Full dataset profile
# ---------------------------------------------------------------------------
print(f"\n[1/2] Generating full dataset profile...")
t0 = time.time()

profile1 = ProfileReport(
    df,
    title="CCFD Dataset - Full Profile",
    explorative=True,
)
report1_path = os.path.join(OUTPUT_DIR, 'ccfd_full_profile.html')
profile1.to_file(report1_path)

print(f"   Saved: {report1_path} ({os.path.getsize(report1_path) // 1024} KB, {time.time()-t0:.1f}s)")

# ---------------------------------------------------------------------------
# Report 2: Fraud vs Legit comparison
# ---------------------------------------------------------------------------
print(f"\n[2/2] Generating Fraud vs Legit comparison...")
t0 = time.time()

fraud_df = df[df['is_fraud'] == 1].drop(columns=['is_fraud'])
legit_df = df[df['is_fraud'] == 0].drop(columns=['is_fraud'])

print(f"       fraud rows: {len(fraud_df):,}")
print(f"       legit rows: {len(legit_df):,}")

profile_fraud = ProfileReport(fraud_df, title="CCFD - Fraud Only")
profile_legit = ProfileReport(legit_df, title="CCFD - Legitimate Only")
comparison = profile_fraud.compare(profile_legit)

report2_path = os.path.join(OUTPUT_DIR, 'ccfd_fraud_vs_legit_comparison.html')
comparison.to_file(report2_path)

print(f"   Saved: {report2_path} ({os.path.getsize(report2_path) // 1024} KB, {time.time()-t0:.1f}s)")

print(f"\n{'=' * 70}")
print(f"data-profiling complete.")
print(f"View reports:")
print(f"  open {report1_path}")
print(f"  open {report2_path}")
print(f"{'=' * 70}")
