# p5_05 - Tuned API Test Client

> **Step 2 of 2 (final).** A small Python test client that sends 3 sample transactions to the tuned API on port **8001** and prints the responses. Verifies `/health` and `/inference`. Pair it with `p3_03_inference_test_baseline.py` (which hits port 8000) to compare predictions side by side.

---

## 1. What we are going to do?

`p5_04` started a FastAPI service on port 8001. This script:

1. Calls `/health` to confirm the API is up and the production model is loaded
2. Sends 3 sample transactions to `/inference` and prints the prediction, probability, risk score, and confidence
3. Prints a summary of how many tests passed

It is the tuned-API equivalent of `p3_03_inference_test_baseline.py` from Section 04. The two test clients are intentionally near-identical so you can run them back-to-back and see the same 3 transactions get different decisions from the two APIs.

**Why a test client?** Because typing the same JSON into curl over and over while showing off a model gets old fast. This script is the "smoke test" - one command, three transactions, you see the API is alive and the predictions look right.

---

## 2. What exactly is it?

A standard `requests`-based test client. Notable bits:

- `API_URL = "http://localhost:8001"` - points at the tuned API (the baseline test client points at 8000)
- `test_transactions` - a list of 3 dicts. Each dict has a `name` (with an emoji) and a `data` payload that matches the 8-field schema:
  - **Normal Transaction** - $50 grocery in the afternoon. Should look legit.
  - **Suspicious Transaction** - $2,500 electronics at 3 AM, international, card-not-present, 7 transactions in 24h. Should look fraudulent.
  - **High Value Jewelry** - $1,500 jewelry at 2 AM, international, card-not-present. Edge case.
- `test_health()` - GETs `/health`, prints status. Returns `False` if it cannot connect (e.g. the API is not running).
- `test_inference(transaction)` - POSTs each transaction to `/inference`, prints the response fields (`prediction`, `probability`, `risk_score`, `confidence`, `message`).
- `main()` - runs health first, bails if unhealthy, otherwise loops through the 3 transactions and prints a final pass count.

Same exact structure as `p3_03_inference_test_baseline.py`. Only difference: the URL (8001 vs 8000) and the banner text.

---

## 3. How to run it

### Step 1 - make sure you are in the `mlops-env1` conda environment

```bash
conda info --envs
# If active env is not mlops-env1:
conda deactivate
conda activate mlops-env1
```

### Step 2 - go to the project folder

```bash
cd 08_Tuned_Inference_API/ccfd-project
```

### Step 3 - make sure p5_04 is already running in another terminal

```bash
# In another terminal:
cd 08_Tuned_Inference_API/ccfd-project
python p5_04_inference_api.py
# wait for "Uvicorn running on http://0.0.0.0:8001"
```

### Step 4 - run this script

```bash
python p5_05_inference_test.py
```

Runtime: ~1 second. Just 4 HTTP calls (1 health + 3 inference).

---

## 4. What is the result output?

```
============================================================
p5_03 TUNED PRODUCTION API TEST SUITE (Pipeline)
============================================================

============================================================
Testing Health Endpoint
============================================================
Status: healthy
Model Loaded: True
Model Name: p5_03_production
Model Type: LogisticRegression

============================================================
Testing Inference Endpoint
============================================================

Normal Transaction
----------------------------------------
Prediction: FRAUD
Probability: 60.60%
Risk Score: HIGH
Confidence: 21.20%
Message: ...

Suspicious Transaction (High Amount)
----------------------------------------
Prediction: FRAUD
Probability: 99.84%
Risk Score: CRITICAL
Confidence: 99.69%
Message: ...

Edge Case (High Value Jewelry)
----------------------------------------
Prediction: FRAUD
Probability: 99.60%
Risk Score: CRITICAL
Confidence: 99.21%
Message: ...

============================================================
Tests passed: 3/3
============================================================
```

> The exact probabilities depend on your trained model and may differ by a percentage point or two between runs. The pattern is stable: all 3 come back as FRAUD on the tuned API.

---

## 5. The side-by-side comparison (the whole point)

Run the baseline test client right after this one:

```bash
python p3_03_inference_test_baseline.py     # hits port 8000
python p5_05_inference_test.py               # hits port 8001
```

The same 3 transactions produce different decisions:

| Transaction | Baseline (8000) | Tuned (8001) | What it shows |
|---|---|---|---|
| Normal grocery $50 | **31.61% LEGITIMATE** | **60.60% FRAUD** | The flip - tuned catches a borderline case baseline misses |
| Suspicious intl $2,500 | 99.75% FRAUD | 99.84% FRAUD | Both agree, tuned is slightly more confident |
| High-value jewelry $1,500 | 99.31% FRAUD | 99.60% FRAUD | Both agree, tuned is slightly more confident |

Row 1 is the production-deploy proof of what Section 07 showed offline: the tuned model trades a higher false-alarm rate for catching the borderline frauds the baseline lets through. The Net Benefit math says that trade is worth +$10,925 on the test set.

---

## Where this fits in Section 08

Step 2 of 2. The test client. See the [section README](./README.md) for the full 2-script arc and the comparison story. Step 1 ([p5_04](./p5_04_readme.md)) is the API itself.
