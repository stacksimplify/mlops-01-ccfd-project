# p4_03 vs p5_03 Code Walkthrough

Read-only stripped versions of `p4_03_train_model_with_fe1.py` (from Section 06 - **baseline training with hardcoded params**) and `p5_03_train_production_model.py` (this section - **production retraining with params loaded from a YAML artifact**). Docstrings, comments, and prints removed - only `# Step-N: <one-line description>` markers remain.

**Purpose:** open both files side-by-side in VS Code to see the production handoff pattern - p5_03 has the SAME training shape as p4_03, but reads its params from a file instead of hardcoding them.

**Step alignment:**

| Step | p4_03 (baseline training) | p5_03 (production retraining) |
|---|---|---|
| 0 | (none) | **Load winning config from `artifacts/p5_02_best_config.yaml`** (NEW - the production handoff) |
| 1 | Read CSV | Read CSV |
| 2 | Train/test split | Train/test split |
| 3 | Build Pipeline with **hardcoded** model_params | Build Pipeline with **loaded** model_params (from p5_02 artifact) |
| 4 | Train (`pipeline.fit`) | Train (`pipeline.fit`) - same |
| 5 | Evaluate | Evaluate |
| 6 | Save pipeline pickle | Save pipeline pickle |
| 7 | Save metrics JSON | Save metrics JSON |

**The only real difference:**
- p4_03: params live IN the script (hardcoded dict)
- p5_03: params live in a YAML file that the script reads at startup

This is the MLOps production pattern - re-running `p5_02` on fresh data updates the YAML artifact automatically, so re-running `p5_03` picks up the new params without anyone editing code.

**For actual execution** use the full project at `../../ccfd-project/`. p5_03 requires `artifacts/p5_02_best_config.yaml` to exist (produced by running p5_02 first). This folder is NOT meant to be run from.
