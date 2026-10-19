# p5_04 - Tuned Inference API (FastAPI, port 8001)

> **Step 1 of 2.** A FastAPI service that wraps the tuned production model from Section 07 (`models/p5_03_production/`). Listens on port **8001** so it can stand next to the Section 04 baseline API on port 8000. Same 8-field request schema. Different decisions. Computes `amount_ratio` server-side.

---

## 1. What we are going to do?

Section 07 produced a tuned production model with `class_weight={0:1, 1:100}` saved at `models/p5_03_production/`. This script wraps that model in a FastAPI service so you can call it over HTTP.

Three things make this API special compared to the baseline API in Section 04:

1. **Different model.** Loads `p5_03_production` (the tuned production pickle), not `p3_01_baseline`.
2. **Different port.** Listens on **8001** so both APIs can run at the same time for side-by-side comparison.
3. **Computes `amount_ratio` server-side.** The tuned model was trained on 9 features including FE1's `amount_ratio`. The API receives only 8 raw fields from the client, so it computes the 9th field inside `prepare_input()` before sending the DataFrame through the Pipeline.

Everything else - request schema, response schema, endpoints (`/`, `/health`, `/ui`, `/inference`, `/docs`), risk score buckets, confidence math - is identical to the baseline API. That is intentional: students see two APIs with the same surface but different decisions, and the only thing that changed is the model behind the door.

---

## 2. What exactly is it?

A standard FastAPI service. Notable code blocks:

- `MODEL_NAME = 'p5_03_production'` - the production model from Section 07
- `API_PORT = 8001` - lets it co-exist with the baseline on 8000
- `DEFAULT_THRESHOLD = 0.5` - the tuned model uses `class_weight` (not threshold) to shift its decision boundary, so 0.5 is the right call
- `load_pipeline(MODEL_NAME)` - the same util used everywhere else in the course; loads the pickle + metadata
- `prepare_input(data)` - converts the request to a 1-row DataFrame **and adds `amount_ratio`**:

```python
df['amount_ratio'] = df['transaction_amount'] / df['avg_transaction_amount'].replace(0, 1)
```

  `.replace(0, 1)` avoids divide-by-zero for any edge case where `avg_transaction_amount` could be 0.

- `PIPELINE.predict_proba(df)[0]` - run inference. The Pipeline handles all preprocessing (encoding categoricals, scaling numerics) internally
- `fraud_prob >= DEFAULT_THRESHOLD` - default-threshold check
- `calculate_risk_score(fraud_prob)` - bucket the probability into LOW / MEDIUM / HIGH / CRITICAL
- `uvicorn.run(app, host="0.0.0.0", port=API_PORT, ...)` - start the server

The web UI at `/ui` reuses the same `templates/index.html` from Section 04 - no separate `templates_tuned/` folder. The UI is just a form that POSTs to `/inference`, so it does not care which model is behind it.

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

### Step 3 - make sure the production model exists

```bash
# Only if models/p5_03_production/ does not exist:
python p1_01_generate_initial_dataset.py     # generate data
python p4_01_feature_engineering_fe1.py      # add amount_ratio feature
python p5_02_class_weight_gridsearch.py      # creates artifacts/p5_02_best_config.yaml
python p5_03_train_production_model.py       # trains the production model
```

### Step 4 - start the API

```bash
python p5_04_inference_api.py
```

You should see:

```
Starting Fraud Detection API (p5_03 Tuned Production)...
   Model: p5_03_production
   Port: 8001
Model loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8001
```

The API is now serving on **port 8001**. Leave this terminal open.

---

## 4. What is the result output?

### Health check

```bash
curl http://127.0.0.1:8001/health
```

**Windows (PowerShell):**

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "p5_03_production",
  "model_type": "LogisticRegression",
  "version": "1.0.0"
}
```

### Inference (the borderline grocery case)

```bash
curl -X POST http://127.0.0.1:8001/inference \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_amount": 50.00,
    "transaction_hour": 14,
    "days_since_last_txn": 1,
    "avg_transaction_amount": 75.00,
    "merchant_category": "grocery",
    "card_present": "yes",
    "international": "no",
    "transaction_count_24h": 2
  }'
```

**Windows (PowerShell):**

```powershell
$body = @{
  transaction_amount     = 50.00
  transaction_hour       = 14
  days_since_last_txn    = 1
  avg_transaction_amount = 75.00
  merchant_category      = "grocery"
  card_present           = "yes"
  international          = "no"
  transaction_count_24h  = 2
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8001/inference -ContentType "application/json" -Body $body
```

PowerShell shows the same fields as a list (`prediction : ...`). The easiest way on any computer is the API's own page: http://127.0.0.1:8001/docs, then "Try it out".

```json
{
  "prediction": "FRAUD",
  "probability": 0.6060,
  "threshold": 0.5,
  "risk_score": "HIGH",
  "confidence": 0.2120,
  "model_name": "p5_03_production",
  "message": "..."
}
```

The same request to the baseline API on port 8000 returns `LEGITIMATE` at about 31.6% probability. That flip - same input, different decision - is the point of this section.

### Web UI

Open http://127.0.0.1:8001/ui in a browser. Fill the form. Click submit. Same UI as the baseline API, different model under the hood.

### Swagger docs

Open http://127.0.0.1:8001/docs for the auto-generated Swagger UI.

---

## 5. The 3 test transactions (sent by p5_05)

The companion test client (`p5_05_inference_test.py`) sends these 3 transactions. Here is what the tuned API returns vs the baseline API:

| Transaction | Baseline (8000) | Tuned (8001) |
|---|---|---|
| Normal grocery $50 | 31.61% LEGITIMATE | **60.60% FRAUD** |
| Suspicious intl $2,500 | 99.75% FRAUD | 99.84% FRAUD |
| High-value jewelry $1,500 | 99.31% FRAUD | 99.60% FRAUD |

Row 1 is the headline - the borderline case that flips. Rows 2 and 3 are the easy cases where both agree but the tuned model is more confident.

---

## Where this fits in Section 08

Step 1 of 2. The API itself. See the [section README](./README.md) for the full 2-script arc and the side-by-side comparison story. Step 2 ([p5_05](./p5_05_readme.md)) is the test client that hits this API.
