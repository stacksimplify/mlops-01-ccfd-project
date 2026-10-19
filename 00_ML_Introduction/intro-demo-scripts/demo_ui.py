"""Section 00 intro demo only: the course web page plus one extra button.

Both demo APIs (p3_02_inference_api_baseline_v2_demo.py and p5_04_inference_api_v2_demo.py)
use this. It reads the unchanged course page (ccfd-project/templates/index.html) and adds a
"Real Fraud from Test Data (row 7382)" button that fills in test row 7382, a real fraud from
our test data, so nothing has to be typed on camera. The course page itself is never changed.
"""
from pathlib import Path

from fastapi.responses import HTMLResponse

PAGE = Path(__file__).resolve().parent / "ccfd-project" / "templates" / "index.html"

# Inserted right after the row of the three course example buttons
BUTTON_ROW = """
                <div class="example-buttons">
                    <button class="btn-example btn-real" onclick="loadRealFraudExample()">
                        🎯 Real Fraud from Test Data (row 7382)
                    </button>
                </div>"""

STYLE = """    <style>
        .btn-real { border-color: #805ad5; color: #805ad5; }
        .btn-real:hover { background: #805ad5; color: white; }
    </style>
</head>"""

SCRIPT = """        // Load test row 7382: a real fraud from our test data (Section 00 intro demo)
        function loadRealFraudExample() {
            document.getElementById('transaction_amount').value = 363.26;
            document.getElementById('transaction_hour').value = 10;
            document.getElementById('days_since_last_txn').value = 4;
            document.getElementById('avg_transaction_amount').value = 161.43;
            document.getElementById('merchant_category').value = 'gas_station';
            document.getElementById('card_present').value = 'no';
            document.getElementById('international').value = 'no';
            document.getElementById('transaction_count_24h').value = 1;
        }

        // Load Fraud Example"""


def _replace_once(html, anchor, new_text):
    """Replace the one place where anchor appears; stop if the course page no longer has it exactly once."""
    if html.count(anchor) != 1:
        raise RuntimeError(f"course page changed: expected exactly one {anchor!r}")
    return html.replace(anchor, new_text, 1)


def demo_page():
    html = PAGE.read_text(encoding="utf-8")
    medium = '<button class="btn-example" onclick="loadMediumExample()">'
    if html.count(medium) != 1:
        raise RuntimeError("course page changed: the Medium Risk button was not found exactly once")
    row_end = html.index("</div>", html.index(medium)) + len("</div>")     # end of the example-button row
    html = html[:row_end] + BUTTON_ROW + html[row_end:]
    html = _replace_once(html, "</head>", STYLE)
    html = _replace_once(html, "        // Load Fraud Example", SCRIPT)
    return html


def use_demo_page(app):
    """Serve the demo page at /ui instead of the course page. Every other route stays as it is."""
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/ui"]

    @app.get("/ui", response_class=HTMLResponse, tags=["Web Interface"])
    async def ui():
        return HTMLResponse(content=demo_page())
