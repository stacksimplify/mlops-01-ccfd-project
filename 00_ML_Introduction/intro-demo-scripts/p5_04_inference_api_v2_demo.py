"""DEMO VERSION (v2) of the tuned model API, for the Section 00 intro demo only.

It runs the unchanged course API ccfd-project/p5_04_inference_api.py (same model, same
/inference, /health and /docs) and only swaps its /ui web page for one with an extra
"Real Fraud from Test Data (row 7382)" button (see demo_ui.py).

Started by run_intro_demo.sh (stage 7). To go back to the plain course API, start
ccfd-project/p5_04_inference_api.py there instead.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE / "ccfd-project"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)          # the course API loads its saved model from ccfd-project/models/

import uvicorn                                  # noqa: E402
import p5_04_inference_api as course_api        # noqa: E402  (unchanged course code)
from demo_ui import use_demo_page               # noqa: E402

use_demo_page(course_api.app)

if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", course_api.API_PORT))
    uvicorn.run(course_api.app, host="0.0.0.0", port=port, log_level="info")
