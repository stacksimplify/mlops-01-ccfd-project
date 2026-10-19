# p4_03 vs p5_01 Code Walkthrough

Read-only stripped versions of `p4_03_train_model_with_fe1.py` (from Section 06) and `p5_01_train_hyperparameter_tuning.py` (this section). Docstrings, comments, and prints removed - only `# Step-N: <one-line description>` markers remain.

**Purpose:** open both files side-by-side in VS Code to see exactly what's NEW in p5_01 vs p4_03 - without distractions.

**Step alignment:**

| Step | p4_03 | p5_01 |
|---|---|---|
| 1 | Read CSV | Read CSV |
| 2 | Train/test split | Train/test split |
| 3 | Build Pipeline (hardcoded model_params) | Build Pipeline (no model_params) |
| 4 | Train (`pipeline.fit`) | Train via search - **4a** scorer, **4b** RandomizedSearchCV, **4c** extract winner |
| 5 | Evaluate | Evaluate |
| 6 | Save pipeline pickle | Save pipeline pickle |
| 7 | Save metrics JSON | Save metrics JSON |

**For actual execution** (running data gen, training, inference), use the full project at `../../ccfd-project/`. This folder is NOT meant to be run from.
