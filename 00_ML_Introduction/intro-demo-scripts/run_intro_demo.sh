#!/usr/bin/env bash
# =============================================================================
# Section 00 intro demo: the baseline model and the tuned model, side by side
# =============================================================================
# One script, nine stages (one action each):
#   Stage 1  Generate the dataset                 (p1_01, 10,000 card transactions)  Section 03
#   Stage 2  Train the baseline model             (p3_01)                            Section 04 A
#   Stage 3  Start the baseline API               (demo v2 of p3_02, port 8000)      Section 04 B
#   Stage 4  Add the new feature amount_ratio     (p4_01)                            Section 06
#   Stage 5  Find the best settings (grid search) (p5_02)                            Section 07
#   Stage 6  Train the final tuned model          (p5_03)                            Section 07
#   Stage 7  Start the tuned API                  (demo v2 of p5_04, port 8001)      Section 08
#   Stage 8  Open both APIs side by side          (side-by-side.html, port 8300)     this intro demo only
#   Stage 9  Send the final test data to both     (see the difference)               Sections 04 B and 08
#
# How to run (macOS or Linux; on Windows use Git Bash):
#   conda activate mlops-env1
#   bash run_intro_demo.sh
# Press Ctrl+C (or close the terminal) to stop both APIs and the page server.
# First time? See "Before you run it" in README.md (Miniconda, conda init, mlops-env1).
#
# ccfd-project/ is an unchanged copy of the Section 08 project. This script
# only runs it. The two APIs start through small "v2 demo" files in this folder
# (p3_02_inference_api_baseline_v2_demo.py, p5_04_inference_api_v2_demo.py): they
# run the unchanged course APIs and only add one button to the web page,
# "Real Fraud from Test Data (row 7382)" (see demo_ui.py), so nothing is typed. Press Ctrl+C to stop both APIs and the page server.
# =============================================================================
set -euo pipefail

echo "============================================================================="
echo " Starting the Section 00 intro demo: the baseline ML model vs the tuned ML model"
echo " Nine stages, about 2 minutes. First, checking your Python environment..."
echo "============================================================================="

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$HERE/ccfd-project"
LOGS="$HERE/logs"
PY="${PYTHON:-python}"
BASE_PORT=8000
TUNED_PORT=8001
PAGE_PORT=8300
PIDS=()

stage() {              # stage <number> <what it does> <where it is in the course>
  echo
  echo "============================================================================="
  echo " STAGE $1 of 9: $2"
  echo " In the course: $3"
  echo "============================================================================="
}

run_step() {            # run_step <script.py>: run one course script, keep its output in logs/
  local script="$1"
  echo "  -> python $script"
  if ! (cd "$PROJECT" && "$PY" "$script" > "$LOGS/${script%.py}.log" 2>&1); then
    echo "  FAILED: $script. Last lines of logs/${script%.py}.log:"
    tail -20 "$LOGS/${script%.py}.log"
    exit 1
  fi
}

port_in_use() {         # port_in_use <port>: true if a server is answering there
  # (connect, not bind: right after a stop, macOS keeps a closed port in TIME_WAIT for about
  #  30 seconds; a bind test would wrongly call it busy, but the servers can still start)
  "$PY" - "$1" <<'EOF'
import socket, sys
s = socket.socket()
s.settimeout(1)
busy = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
s.close()
sys.exit(0 if busy else 1)
EOF
}

wait_for() {            # wait_for <url> <name>: wait until the URL answers (up to 60 seconds)
  "$PY" - "$1" "$2" <<'EOF'
import sys, time, urllib.request
url, name = sys.argv[1], sys.argv[2]
for _ in range(120):
    try:
        urllib.request.urlopen(url, timeout=1)
        sys.exit(0)
    except Exception:
        time.sleep(0.5)
print(f"  {name} did not start in 60 seconds. See the logs/ folder.")
sys.exit(1)
EOF
}

cleanup() {
  if [ ${#PIDS[@]} -gt 0 ]; then
    echo
    echo "Stopping the APIs and the page server..."
    kill "${PIDS[@]}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do          # give them up to 5 seconds to stop cleanly
      alive=0
      for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
      [ "$alive" -eq 0 ] && break
      sleep 0.5
    done
    kill -9 "${PIDS[@]}" 2>/dev/null || true   # anything still running is stopped now
    echo "Stopped. Ports $BASE_PORT, $TUNED_PORT and $PAGE_PORT are free again."
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
trap 'exit 129' HUP                            # closing the terminal window stops everything too

# ---------------------------------------------------------------- checks
has_libs() { "$1" -c "import fastapi, uvicorn, sklearn, pandas, yaml, requests" > /dev/null 2>&1; }
if ! has_libs "$PY"; then
  # "conda activate mlops-env1" not done (or conda not set up in this terminal yet):
  # look for the course environment in the usual Miniconda places and use it.
  for candidate in \
      "$HOME/miniconda3/envs/mlops-env1/bin/python" \
      "/opt/homebrew/Caskroom/miniconda/base/envs/mlops-env1/bin/python" \
      "/usr/local/Caskroom/miniconda/base/envs/mlops-env1/bin/python" \
      "$HOME/anaconda3/envs/mlops-env1/bin/python" \
      "$HOME/miniconda3/envs/mlops-env1/python.exe"; do
    if [ -x "$candidate" ] && has_libs "$candidate"; then
      PY="$candidate"
      echo "Using the course environment mlops-env1: $PY"
      break
    fi
  done
fi
if ! has_libs "$PY"; then
  echo "The course environment mlops-env1 was not found."
  echo "  - If you see 'zsh: command not found: conda', run once:  ~/miniconda3/bin/conda init zsh"
  echo "    then close this terminal, open a new one, and run:     conda activate mlops-env1"
  echo "  - If mlops-env1 does not exist yet, create it (see 'Before you run it' in README.md,"
  echo "    or Section 02: Setup Python Environment)."
  exit 1
fi
for port in $BASE_PORT $TUNED_PORT $PAGE_PORT; do
  if port_in_use "$port"; then
    echo "Port $port is already in use. Stop the program using it (for example an API"
    echo "from Section 04 B or 08 that is still running), then run this script again."
    exit 1
  fi
done
mkdir -p "$LOGS"
echo "  Python environment OK: $("$PY" -c 'import sys; print(sys.executable)')"

# ---------------------------------------------------------------- stage 1
stage 1 "Generate the dataset (10,000 card transactions)" "Section 03, Synthetic Dataset Generation"
run_step p1_01_generate_initial_dataset.py
echo "  Dataset saved in ccfd-project/data/ (the training step splits it 80/20 into train and test)"

# ---------------------------------------------------------------- stage 2
stage 2 "Train the baseline model" "Section 04 A, Model Training and Evaluation"
run_step p3_01_train_model_baseline.py
echo "  Baseline model saved in ccfd-project/models/p3_01_baseline/"

# ---------------------------------------------------------------- stage 3
stage 3 "Start the FastAPI app for the baseline model" "Section 04 B, Model Serving with an Inference API"
(cd "$PROJECT" && exec "$PY" "$HERE/p3_02_inference_api_baseline_v2_demo.py" > "$LOGS/api_baseline.log" 2>&1) &
PIDS+=($!)
wait_for "http://localhost:$BASE_PORT/health" "The baseline API"
echo "  Baseline API is running:"
echo "    Web page : http://localhost:$BASE_PORT/ui"
echo "    API docs : http://localhost:$BASE_PORT/docs"

# ---------------------------------------------------------------- stage 4
stage 4 "Add the new feature amount_ratio" "Section 06, Feature Engineering"
run_step p4_01_feature_engineering_fe1.py
echo "  Dataset with amount_ratio saved in ccfd-project/data/"

# ---------------------------------------------------------------- stage 5
stage 5 "Find the best settings with a grid search (about a minute)" "Section 07, Hyperparameter Tuning"
run_step p5_02_class_weight_gridsearch.py
echo "  Best settings saved in ccfd-project/artifacts/p5_02_best_config.yaml"

# ---------------------------------------------------------------- stage 6
stage 6 "Train the final tuned model" "Section 07, Hyperparameter Tuning"
run_step p5_03_train_production_model.py
echo "  Tuned model saved in ccfd-project/models/p5_03_production/"

# ---------------------------------------------------------------- stage 7
stage 7 "Start the FastAPI app for the tuned model" "Section 08, Model Serving with the Tuned Inference API"
(cd "$PROJECT" && exec "$PY" "$HERE/p5_04_inference_api_v2_demo.py" > "$LOGS/api_tuned.log" 2>&1) &
PIDS+=($!)
wait_for "http://localhost:$TUNED_PORT/health" "The tuned API"
echo "  Tuned API is running:"
echo "    Web page : http://localhost:$TUNED_PORT/ui"
echo "    API docs : http://localhost:$TUNED_PORT/docs"

# ---------------------------------------------------------------- stage 8
stage 8 "Open both APIs side by side" "this intro demo only"
"$PY" -m http.server "$PAGE_PORT" --bind 127.0.0.1 --directory "$HERE" > "$LOGS/page_server.log" 2>&1 &
PIDS+=($!)
wait_for "http://localhost:$PAGE_PORT/side-by-side.html" "The page server"
PAGE_URL="http://localhost:$PAGE_PORT/side-by-side.html"
echo "  Open this page in your browser:"
echo "    $PAGE_URL"
echo "  Left: the baseline ML model (port $BASE_PORT). Right: the final tuned ML model (port $TUNED_PORT)."
if [ -z "${INTRO_DEMO_NO_BROWSER:-}" ]; then
  if command -v open > /dev/null 2>&1; then open "$PAGE_URL" || true
  elif command -v xdg-open > /dev/null 2>&1; then xdg-open "$PAGE_URL" > /dev/null 2>&1 || true
  elif command -v start > /dev/null 2>&1; then start "$PAGE_URL" || true
  fi
fi

# ---------------------------------------------------------------- stage 9
stage 9 "Send the final test data to both models and see the difference" "Sections 04 B and 08 (the API test scripts)"
"$PY" - "$BASE_PORT" "$TUNED_PORT" <<'EOF'
import json, sys, urllib.request

base, tuned = sys.argv[1], sys.argv[2]
cases = [
    ("Test row 7382, a REAL fraud from our test data",
     "$363.26 at a gas station, 10 AM, card not present, more than twice the usual amount",
     dict(transaction_amount=363.26, transaction_hour=10, days_since_last_txn=4,
          avg_transaction_amount=161.43, merchant_category="gas_station",
          card_present="no", international="no", transaction_count_24h=1)),
    ("The page's Fraud Example (made up, an obvious fraud)",
     "$2,500 of electronics at 2 AM, from another country, 7 purchases today",
     dict(transaction_amount=2500.0, transaction_hour=2, days_since_last_txn=0,
          avg_transaction_amount=120.0, merchant_category="electronics",
          card_present="no", international="yes", transaction_count_24h=7)),
]

def ask(port, body):
    req = urllib.request.Request(f"http://localhost:{port}/inference", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=10))
    return f"{r['prediction']:<11} {r['probability'] * 100:5.1f}% fraud probability, {r['risk_score']} risk"

for title, about, body in cases:
    print(f"\n  {title}")
    print(f"  {about}")
    print(f"    Baseline model (port {base}) : {ask(base, body)}")
    print(f"    Tuned model    (port {tuned}) : {ask(tuned, body)}")

print("""
  To show the real fraud on the page: on BOTH sides, click the purple button
  "Real Fraud from Test Data (row 7382)", then click CHECK FRAUD on each side.
  (It fills in $363.26, Gas Station, 10 AM, 4 days since the last purchase,
  average $161.43, card not present, domestic, 1 purchase in the last 24h.)
  Your numbers may differ a little from the video, depending on your machine.""")
EOF

echo
echo "============================================================================="
echo " The demo is ready. Side-by-side page: $PAGE_URL"
echo " Press Ctrl+C to stop both APIs and the page server."
echo "============================================================================="
wait
