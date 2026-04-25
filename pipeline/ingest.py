"""
ingest.py
---------
Automated data ingestion from passagertal.dk (TARGIT Anywhere v26.3).

Confirmed endpoint (Chrome DevTools):
    POST https://passagertal.dk/Visual/GetModel
         ?ObjectId=%7BD87FA879-800A-498F-A9F7-0BFFE899D24E%7D

The response JSON has this structure:
    {
        "State": { ... },          # chart metadata, criteria, row/col counts
        "Model": {
            "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "Content": "<base64>"  # the complete .xlsx file, base64-encoded
        }
    }

We decode Model.Content and write it directly to Data(update).xlsx.
No browser, no Playwright, no export button — one HTTP call does everything.

Usage:
    python pipeline/ingest.py

Requirements:
    requests>=2.31
"""

import base64
import logging
import requests
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_URL     = "https://passagertal.dk"
GETMODEL_URL = f"{BASE_URL}/Visual/GetModel"
OBJECT_ID    = "%7BD87FA879-800A-498F-A9F7-0BFFE899D24E%7D"   # {D87FA879-...}
OUTPUT_PATH  = Path("Data(update).xlsx")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ingest")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer":      f"{BASE_URL}/Embed",
    "Origin":       BASE_URL,
    "Content-Type": "application/json",
    "Accept":       "application/json, text/plain, */*",
})


def prime_session() -> None:
    """
    Load the dashboard page once so TARGIT sets any required session cookies
    before we call GetModel directly.
    """
    url = (
        f"{BASE_URL}/Embed"
        "#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview"
    )
    log.info("Priming session …")
    SESSION.get(url, timeout=30).raise_for_status()
    log.info("Cookies: %s", list(SESSION.cookies.keys()))


def fetch_and_save() -> None:
    """POST to GetModel, decode the base64 Excel payload, write to disk."""
    log.info("Calling GetModel …")
    resp = SESSION.post(
        GETMODEL_URL,
        params={"ObjectId": OBJECT_ID},
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()

    model = body.get("Model", {})
    content_type = model.get("ContentType", "")
    content_b64  = model.get("Content", "")

    if not content_b64:
        raise ValueError(
            "GetModel response contained no 'Model.Content' field.\n"
            f"Top-level keys returned: {list(body.keys())}\n"
            f"Model keys: {list(model.keys())}"
        )

    log.info("Content-Type: %s", content_type)
    log.info("Base64 payload length: %d chars", len(content_b64))

    xlsx_bytes = base64.b64decode(content_b64)

    # Sanity check: xlsx files are ZIP archives and start with PK\x03\x04
    if not xlsx_bytes[:4] == b"PK\x03\x04":
        raise ValueError(
            f"Decoded content does not look like an xlsx file "
            f"(magic bytes: {xlsx_bytes[:4]!r}). "
            "Check data_raw.json for the raw response."
        )

    OUTPUT_PATH.write_bytes(xlsx_bytes)
    log.info("Saved %d bytes → %s", len(xlsx_bytes), OUTPUT_PATH)

    # Log what TARGIT tells us about the data (from State metadata)
    state = body.get("State", {})
    log.info(
        "Dataset: %d rows × %d cols  |  Criteria: %s",
        state.get("CrosstabRowCount", "?"),
        state.get("CrosstabColCount", "?"),
        state.get("CriteriaText", "?"),
    )


def main() -> None:
    prime_session()
    fetch_and_save()


if __name__ == "__main__":
    main()
