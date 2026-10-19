---
title: "Machine Learning End-to-End: Credit Card Fraud Detection"
description: "Build a credit card fraud detection model in Python, end to end: EDA, feature engineering, training, business-metric tuning and a FastAPI inference API."
---

# Machine Learning End-to-End: Credit Card Fraud Detection

Build a real fraud model in Python: EDA, feature engineering, training, business-metric tuning & a FastAPI inference API.

- **Get the course:** https://stacksimplify.com/courses/machine-learning-end-to-end
- **Course website:** https://machine-learning-end-to-end.stacksimplify.com/

This is the code companion for the course. You build one complete machine learning project, a credit card fraud detector, from raw data to your own prediction API. It is written as plain Python scripts, not notebooks, organized the way real projects are. Every model is judged in dollars saved for the bank, not only by accuracy.

## The Course in Pictures

**Our project: from a card swipe to a decision**

![Our project, Credit Card Fraud Detection, from a card swipe to a decision: a customer swipes a card; the transaction data (amount, time, merchant and more) goes to the ML fraud detection model we build, a logistic regression behind a FastAPI inference API; the model gives a fraud probability; a high probability means flag the transaction for review as FRAUD, and a low one means approve it as LEGITIMATE](images/00-01-00-OurProject.png)

**The end result: two APIs, one real fraud from our test data**

![The intro demo page, Credit Card Fraud Detection: Baseline vs Tuned, with the two course web pages side by side, the baseline ML model on port 8000 and the final tuned ML model on port 8001, on test row 7382, a 363 dollar gas station purchase that was really fraud, loaded with the Real Fraud from Test Data button: the baseline says LEGITIMATE with 38.7 percent fraud probability, the tuned model says FRAUD, high risk, with 67.3 percent. Below them, the Net Benefit before tuning is 39,625 dollars and after tuning 50,550 dollars, a gain of 10,925 dollars](images/00-01-05-BaselineVsTunedDemo.png)

The baseline API (port 8000) lets this fraud through; the tuned API (port 8001) catches it. Getting from the left model to the right one, and proving it in dollars, is what this course is about.

**Run it yourself in about 2 minutes.** One script builds both models, starts both APIs and opens this page (you need the course environment from Section 02):

```bash
conda activate mlops-env1
cd 00_ML_Introduction/intro-demo-scripts
bash run_intro_demo.sh
```

All the steps: [Intro Demo Scripts](00_ML_Introduction/intro-demo-scripts/).

**The machine learning workflow**

![The machine learning workflow in seven stages. Stages 1 to 6 are core machine learning, built in this course: data collection, data preprocessing, feature engineering, model training, model evaluation, and model deployment, which here means the API only, no infrastructure. Stage 7, monitoring and retraining, is MLOps: explained, not built. A feedback loop goes from stage 7 back to data preprocessing: monitoring feeds fresh data back in, and the model is retrained. Why we stop at stage 6: stage 7 is MLOps, the next layer after the core machine learning job. Running the API on infrastructure (Docker, Kubernetes, cloud) and monitoring it need other tools. Section 01 explains them, but we do not build them](images/00-03-01-MLWorkflow.png)

**What you learn, and in what order**

![The course map: 00 Introduction, 01 Machine Learning Fundamentals, 02 Python Environment Setup, 03 Synthetic Dataset Generation, 04 A Model Training and Evaluation, 04 B Model Serving with an Inference API, 05 Exploratory Data Analysis, 06 Feature Engineering, 07 Hyperparameter Tuning, 08 Model Serving with the Tuned Inference API, about 15 hours of video](images/00-01-02-CourseMap.png)

**How the project follows the machine learning workflow**

![How our project follows the workflow: p1_01 generates the data in Section 03, p3_01 trains and evaluates the baseline in Section 04 A, p3_02 and p3_03 serve the baseline API in Section 04 B, p2 explores the data in Section 05, p4 scripts do feature engineering in Section 06, p5_01 to p5_03 do hyperparameter tuning and train the tuned model in Section 07, and p5_04 and p5_05 serve the tuned API in Section 08](images/00-03-02-OurProjectMap.png)

**The journey in dollars**

![The journey in dollars, as a table with the columns Section, What we do and Net Benefit. Section 04 A: Baseline Model, Logistic Regression on 8 features, 39,625 dollars. Section 06: plus Feature Engineering, one new feature amount_ratio adds only 75 dollars, 39,700 dollars. Section 07: plus Hyperparameter Tuning using Random Search, RandomizedSearchCV over a narrow class_weight grid actually loses 2,675 dollars, 37,025 dollars. Section 07: plus Hyperparameter Tuning using Grid Search, GridSearchCV over a wider, money-weighted class_weight grid catches 57 of 60 test frauds, 50,550 dollars. Net Benefit goes up by about 11,000 dollars over the baseline. Your exact numbers may differ a little by machine; Section 07 explains why](images/00-01-03-JourneyInDollars.png)

**The tools you will use**

![The tools you will use, in six groups. Setup in Section 02: Python 3.14, Miniconda and pip, one main environment mlops-env1 installed from requirements.txt. Data in Section 03: pandas and NumPy generate 10,000 card transactions into a CSV file. Explore in Sections 05 and 06: Matplotlib and Seaborn, plus the optional fg-data-profiling and Sweetviz reports. Machine learning in Sections 04 A, 06 and 07: scikit-learn with Pipeline, LogisticRegression, RandomizedSearchCV and GridSearchCV. Serve as an API in Sections 04 B and 08: FastAPI, Uvicorn and Pydantic, with the inference, health, docs and ui endpoints. Project and code in Sections 03 to 08: Python scripts, not notebooks, with shared utils modules, and models, metrics and settings saved with pickle, JSON and PyYAML. Every tool is explained the first time we use it. Not in this course: Docker, Kubernetes, MLflow and cloud deployment](images/00-01-06-ToolsYouWillUse.png)

## What You Will Learn

| Section | What you learn | Video | Website |
|---|---|---|---|
| [00. Introduction](00_ML_Introduction/) | The finished project, the machine learning workflow, the tools you will use | ~20 min | [Website](https://machine-learning-end-to-end.stacksimplify.com/00-introduction/) |
| [01. Machine Learning Fundamentals](01_ML_Fundamentals/) | What AI is, every stage of the machine learning workflow, and how our project follows it | ~81 min | [Website](https://machine-learning-end-to-end.stacksimplify.com/01-machine-learning-fundamentals/) |
| [02. Python Environment Setup](02_Setup_Python/) | Miniconda, the course environment, installing the packages | 17 min | [Website](https://machine-learning-end-to-end.stacksimplify.com/02-setup-python-environment/) |
| [03. Synthetic Dataset Generation](03_Generate_Dataset/) | Generate 10,000 card transactions (about 3% fraud) | 13 min | [Website](https://machine-learning-end-to-end.stacksimplify.com/03-generate-synthetic-dataset/) |
| [04 A. Model Training and Evaluation](04_Model_Training_and_Inference/04_01_Train_Baseline_Model/) | Build the training script step by step (v1 to v7): split, scikit-learn pipeline, metrics, Net Benefit, saving the model | 3h 38m | [Website](https://machine-learning-end-to-end.stacksimplify.com/04-model-training-and-inference/04-01-train-baseline-model/) |
| [04 B. Model Serving - Inference API](04_Model_Training_and_Inference/04_02_Inference_API/) | Build a FastAPI inference API with validation, a health check and a web page | 2h 17m | [Website](https://machine-learning-end-to-end.stacksimplify.com/04-model-training-and-inference/04-02-inference-api/) |
| [05. Exploratory Data Analysis](05_EDA_and_Preprocessing/) | Find the fraud patterns with plots, plus two auto-EDA tools | 3h 00m | [Website](https://machine-learning-end-to-end.stacksimplify.com/05-eda-and-preprocessing/) |
| [06. Feature Engineering](06_Feature_Engineering/) | Create `amount_ratio`, analyze it and retrain | 1h 25m | [Website](https://machine-learning-end-to-end.stacksimplify.com/06-feature-engineering/) |
| [07. Hyperparameter Tuning](07_Hyperparameter_Tuning/) | RandomizedSearchCV, GridSearchCV over `class_weight`, a YAML handoff and the production model | 2h 22m | [Website](https://machine-learning-end-to-end.stacksimplify.com/07-hyperparameter-tuning/) |
| [08. Model Serving - Tuned Inference API](08_Tuned_Inference_API/) | Serve the tuned model and compare it with the baseline | 13 min | [Website](https://machine-learning-end-to-end.stacksimplify.com/08-tuned-inference-api/) |

About 15 hours of video. Every section adds one piece to the same fraud detection project.

## What You Build

```
Data (p1_01) -> EDA (p2) -> Train baseline (p3_01) -> Serve baseline API (p3_02, port 8000)
            -> Feature engineering (p4_01..p4_03) -> Tune (p5_01, p5_02) -> Production model (p5_03)
            -> Serve tuned API (p5_04, port 8001)
```

At the end you have two FastAPI inference APIs running on your own machine. Each has `/inference`, `/health`, `/docs` and a `/ui` web page, and each says FRAUD or LEGITIMATE for any card transaction you send it.

### Results - baseline vs tuned

| | Baseline (Section 04) | Tuned (Sections 07 and 08) |
|---|---|---|
| Model | LogisticRegression, `class_weight='balanced'` | LogisticRegression, `class_weight={0:1, 1:100}` |
| Features | 8 | 9 (+ `amount_ratio`) |
| Decision threshold | 0.5 | 0.5 |
| Frauds caught (of 60 in the test set) | 46 (recall 77%) | 57 (recall 95%) |
| False alarms (of 1,940 honest transactions) | 289 | 1,161 |
| Net Benefit | $39,625 | $50,550 |

**The improvement journey:**

| Stage | Net Benefit | Change |
|---|---|---|
| Baseline (p3_01) | $39,625 | - |
| + Feature engineering (p4_03) | $39,700 | +$75 |
| + Random search, narrow grid (p5_01) | $37,025 | -$2,675 |
| + GridSearchCV over money-weighted `class_weight` (p5_02 -> p5_03) | $50,550 | +$10,925 vs baseline |

> **Your numbers may differ.** These numbers come from an Apple Silicon Mac. They were reproduced exactly on an Apple M5 on 2026-09-29. Most of the videos were recorded on an Intel Mac, where the same tuning picks `class_weight={0:1, 1:200}` and shows $46,525. Your own results depend on your processor and its math library. Section 07 explains why.

### How Net Benefit is calculated

```
Net Benefit = (Fraud Caught x $1,475) - (False Alarms x $25) - (Fraud Missed x $1,500)

Baseline:  (46 x $1,475) - (  289 x $25) - (14 x $1,500) = $39,625
Tuned:     (57 x $1,475) - (1,161 x $25) - ( 3 x $1,500) = $50,550
```

## What Is in This Repo

- **One folder per section** (`00_...` to `08_...`). Each has a README that teaches the section, and from Section 03 on its own `ccfd-project/` folder with every script that section needs, so you can jump into any section on its own.
- **`04_Model_Training_and_Inference/04_01_Train_Baseline_Model/v1_...` to `v7_...`:** the training script at each build step. Use them as answer keys while you build along.
- **`07_Hyperparameter_Tuning/code-walkthroughs/`:** side-by-side copies of the scripts, to see exactly what changed from one step to the next.
- **`ccfd-project-main/`:** an optional shared folder (see [Shared Project Folder (Alternative)](#shared-project-folder-alternative)).
- **`images/`:** the diagrams used in the READMEs.

**Data does not carry over between sections.** Inside each section's `ccfd-project/`, generate the data first (`p1_01_generate_initial_dataset.py`), then run that section's scripts.

### Shared Project Folder (Alternative)

Optional. Instead of the per-section folders, you can use `ccfd-project-main/` at the repo root: copy each section's scripts into it as you go, and everything runs from one place with shared data and models.

| Approach | Folder | Best for |
|---|---|---|
| Per-section | `03_Generate_Dataset/ccfd-project/` etc. | Jumping to any section, self-contained |
| Shared main | `ccfd-project-main/` | Following the course in order, one data and models location |

## Quick Start

Run every block below **from the repo root** (the folder you cloned). Open a new terminal, or `cd` back to the root, before the next block.

### 1. Set up the environment (once)

```bash
conda create -n mlops-env1 python=3.14 pip -c conda-forge -y
conda activate mlops-env1
pip install -r ccfd-project-main/requirements.txt
```

The video in Section 02 shows the same command without `pip`. Adding `pip` makes sure the new environment has its own `pip`. If `pip` is ever missing, run `conda install pip` inside the environment.

### 2. Generate the data

```bash
cd 03_Generate_Dataset/ccfd-project
python p1_01_generate_initial_dataset.py
```

**Expected:** 10,000 transactions with a fraud rate of 2.98%, saved as a CSV in `data/`.

### 3. Train the baseline model

```bash
cd 04_Model_Training_and_Inference/04_01_Train_Baseline_Model/ccfd-project
python p1_01_generate_initial_dataset.py
python p3_01_train_model_baseline.py
```

**Expected:** `Net Benefit: $39,625`, and the model in `models/p3_01_baseline/`.

### 4. Start the baseline API (terminal 1) and test it (terminal 2)

```bash
cd 04_Model_Training_and_Inference/04_02_Inference_API/ccfd-project
python p1_01_generate_initial_dataset.py
python p3_01_train_model_baseline.py
python p3_02_inference_api_baseline.py
```

```bash
cd 04_Model_Training_and_Inference/04_02_Inference_API/ccfd-project
python p3_03_inference_test_baseline.py
```

**Expected:** the API starts on port 8000 (`Starting Fraud Detection API (P3_01 Baseline)...`), and the test prints `Tests passed: 3/3`. Web page: http://127.0.0.1:8000/ui

### 5. Explore the data

```bash
cd 05_EDA_and_Preprocessing/ccfd-project
python p1_01_generate_initial_dataset.py
python p2_eda_base.py
```

**Expected:** 6 plots saved in `eda_plots/`.

### 6. Feature engineering

```bash
cd 06_Feature_Engineering/ccfd-project
python p1_01_generate_initial_dataset.py
python p4_01_feature_engineering_fe1.py
python p4_03_train_model_with_fe1.py
```

**Expected:** `Net Benefit: $39,700`.

### 7. Hyperparameter tuning and the production model

```bash
cd 07_Hyperparameter_Tuning/ccfd-project
python p1_01_generate_initial_dataset.py
python p4_01_feature_engineering_fe1.py
python p5_02_class_weight_gridsearch.py
python p5_03_train_production_model.py
```

**Expected:** the best config saved to `artifacts/p5_02_best_config.yaml`, and the production model with `Net Benefit: $50,550` (on Apple Silicon; see the note above).

### 8. The tuned API

```bash
cd 08_Tuned_Inference_API/ccfd-project
python p1_01_generate_initial_dataset.py
python p4_01_feature_engineering_fe1.py
python p5_02_class_weight_gridsearch.py
python p5_03_train_production_model.py
python p5_04_inference_api.py
```

**Expected:** the tuned API on port 8001. Web page: http://127.0.0.1:8001/ui. In a second terminal (same folder), `python p5_05_inference_test.py` sends 3 test transactions.

## Key Facts

| | |
|---|---|
| Python | 3.14 (conda environment `mlops-env1`) |
| Main libraries | pandas, NumPy, scikit-learn 1.8, FastAPI, pydantic, uvicorn (versions pinned in `requirements.txt`) |
| Dataset | 10,000 synthetic transactions, 298 fraud (2.98%) |
| Features | 8 base + 1 engineered (`amount_ratio`) |
| Model | LogisticRegression in a scikit-learn Pipeline |
| APIs | Baseline on port 8000, tuned on port 8001 |

## Requirements

- **Python basics:** functions, lists, dictionaries and simple classes.
- **No machine learning experience needed.** Everything is explained as we go.
- **A computer** (Mac or Windows) that can run Miniconda and VS Code.
- **An internet connection** to install the packages.

## Who This Course Is For

![Who this course is for. Made for: DevOps, SRE, and cloud engineers moving into machine learning; MLOps and AIOps engineers who want the machine learning foundation; developers, data engineers, and students with some Python and no machine learning yet; Python learners ready for their first machine learning project. Before you start: Python basics, no machine learning needed, a Mac, Windows, or Linux computer, free tools and no cloud account. We explain each machine learning idea with our project's own numbers](images/00-01-04-WhoThisIsFor.png)

- **DevOps, SRE and cloud engineers** moving into machine learning.
- **MLOps and AIOps engineers** who want the machine learning foundation.
- **Developers, data engineers and students** who want one real machine learning project, end to end.
- **Python learners** ready for their first machine learning project.

Good to know: the data is synthetic (we generate it ourselves, so every pattern is clear and safe to share), and the API runs on your own machine. Containers and cloud deployment are not part of this course.

## Troubleshooting

| Issue | Solution |
|---|---|
| `conda: command not found` | Restart the terminal after installing Miniconda |
| `pip` missing in the new environment | `conda activate mlops-env1`, then `conda install pip` |
| `No module named sklearn` (or pandas) | Check that `mlops-env1` is active, then `pip install -r ccfd-project-main/requirements.txt` |
| Package conflicts | `conda env remove -n mlops-env1`, then create it again |
| API does not start | The port may be in use. Mac / Linux: `lsof -i :8000` (or `:8001`). Windows: `netstat -ano \| findstr :8000` |
| Model file not found | Run the scripts in order inside the same section folder (p1 -> p3 -> p4 -> p5) |
| Test script cannot connect | Start the API first in another terminal |

## Get the Full Course

This repo is the code companion. The complete step-by-step course, videos plus written lessons, is on:

- **Get the course:** https://stacksimplify.com/courses/machine-learning-end-to-end
- **Course website:** https://machine-learning-end-to-end.stacksimplify.com/

---

(c) StackSimplify / Kalyan Reddy Daida. All rights reserved.
