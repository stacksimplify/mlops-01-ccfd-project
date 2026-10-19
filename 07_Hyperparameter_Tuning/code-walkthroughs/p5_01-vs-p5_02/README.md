# p5_01 vs p5_02 Code Walkthrough

Read-only stripped versions of `p5_01_train_hyperparameter_tuning.py` (RandomSearch - **the failure case**) and `p5_02_class_weight_gridsearch.py` (GridSearch with money-weighted class_weight - **the fix**). Docstrings, comments, and prints removed - only `# Step-N: <one-line description>` markers remain.

**Purpose:** open both files side-by-side in VS Code to see exactly what changed between the failed first attempt and the winning second attempt.

**Step alignment:**

| Step | p5_01 (RandomSearch - HURT $2,675) | p5_02 (GridSearch - WIN +$10,925) |
|---|---|---|
| 1 | Read CSV | Read CSV |
| 2 | Train/test split | Train/test split |
| 3 | Build Pipeline (no model_params) | Build Pipeline (no model_params) |
| 4a | Custom business scorer | Custom business scorer (same) |
| 4b | **RandomizedSearchCV(n_iter=20)** with narrow class_weight grid `['balanced', None]` | **GridSearchCV** (all 60 combos) with **wider grid** including 4 money-weighted dicts `{0:1, 1:30}` through `{0:1, 1:200}` |
| 4c | Extract winner | Extract winner |
| 5 | Evaluate | Evaluate |
| 6 | Save pipeline pickle | **Save `artifacts/p5_02_best_config.yaml` for p5_03 to consume** (NEW - the production handoff) |
| 7 | Save metrics JSON | Save pipeline pickle |
| 8 | (none) | Save metrics JSON |

**Key changes between the two scripts:**
- `RandomizedSearchCV` → `GridSearchCV` (and `n_iter=20` removed - tests all combos)
- `class_weight` grid widened from 2 values to 5 values (4 money-weighted dicts added)
- Extra Step-6: writes a YAML artifact that `p5_03_train_production_model.py` reads (so p5_02's pickle and metrics steps become Step-7 and Step-8)

**For actual execution** use the full project at `../../ccfd-project/`. This folder is NOT meant to be run from.
