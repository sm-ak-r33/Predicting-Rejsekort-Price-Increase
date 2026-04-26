"""
ingest.py
---------
Automated data ingestion from passagertal.dk (TARGIT Anywhere v26.3).

Root-cause of the original bug
───────────────────────────────
context.on("response", async_handler) has a race condition in Playwright Python:
the async handler is *scheduled* but the response body buffer may already be
released by the time `await response.json()` runs → silent empty capture.

Fix: use page.route() which intercepts the request BEFORE the response is
forwarded to the page, giving guaranteed access to the full body.

What this script does
──────────────────────
1. Opens the dashboard in headless Chromium
2. Intercepts all /Visual/GetModel POST responses via page.route()
3. Waits for the page to fully render (TARGIT WASM bootstrap ~10-15 s)
4. Picks the GetModel response with the most rows
5. Parses Model.Rows → pandas DataFrame → Data(update).xlsx

Usage:
    python pipeline/ingest.py
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from playwright.async_api import async_playwright, Route, Request

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DASHBOARD_URL = (
    "https://passagertal.dk/Embed"
    "#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview"
)
GETMODEL_PATH  = "/Visual/GetModel"
OUTPUT_XLSX    = Path("Data(update).xlsx")
PAGE_LOAD_WAIT = 20   # seconds to wait after domcontentloaded for WASM boot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("ingest")

# Shared store: objectId -> parsed response body
all_responses: Dict[str, dict] = {}


def object_id_from_url(url: str) -> str:
    m = re.search(r"ObjectId=%7B([^%&]+)%7D", url, re.IGNORECASE)
    return m.group(1) if m else url


# ---------------------------------------------------------------------------
# Route handler - called for every /Visual/GetModel request
# page.route() guarantees the body is available before we proceed
# ---------------------------------------------------------------------------
async def intercept_getmodel(route: Route, request: Request) -> None:
    # Let the request go through and get the real response
    response = await route.fetch()

    if response.status == 200:
        try:
            text = await response.text()
            body = json.loads(text)

            oid       = object_id_from_url(request.url)
            state     = body.get("State", {})
            row_count = state.get("CrosstabRowCount", 0)
            title     = body.get("Model", {}).get("Title", "")[:50]

            log.info(
                "GetModel captured  rows=%-5s  title='%s'  id=%s",
                row_count, title, oid[:8] + "..."
            )
            all_responses[oid] = body

        except Exception as e:
            log.warning("Failed to parse GetModel response: %s", e)

    # Forward the response to the page unchanged
    await route.fulfill(response=response)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def ingest() -> None:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = await browser.new_context(
            locale="da-DK",
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        # Intercept GetModel BEFORE navigating so we catch every request
        # from the very first paint, with no body-buffer race condition.
        await page.route(f"**{GETMODEL_PATH}**", intercept_getmodel)

        # Navigate
        log.info("Opening dashboard ...")
        try:
            await page.goto(
                DASHBOARD_URL,
                wait_until="domcontentloaded",
                timeout=60_000,
            )
        except Exception as e:
            log.warning("goto raised (non-fatal): %s", e)

        # Give TARGIT's WASM time to initialise and fire all GetModel requests
        log.info("Waiting %ds for TARGIT WASM to boot ...", PAGE_LOAD_WAIT)
        await asyncio.sleep(PAGE_LOAD_WAIT)

        log.info("Total GetModel responses captured: %d", len(all_responses))
        await browser.close()

    # Pick the response with the most rows
    if not all_responses:
        raise RuntimeError(
            "No GetModel responses captured. "
            "Check your network connection and that passagertal.dk is reachable."
        )

    best_oid, best_body = max(
        all_responses.items(),
        key=lambda kv: kv[1].get("State", {}).get("CrosstabRowCount", 0),
    )
    best_rows  = best_body.get("State", {}).get("CrosstabRowCount", 0)
    best_title = best_body.get("Model", {}).get("Title", "")
    log.info(
        "Using object %s - '%s' (%d rows)",
        best_oid[:8] + "...", best_title, best_rows
    )

    # Save raw JSON for debugging
    Path("data_raw.json").write_text(
        json.dumps(best_body, indent=2, ensure_ascii=False)
    )
    log.info("Raw JSON saved -> data_raw.json")

    # Parse and save
    df = parse_rows(best_body)
    df.to_excel(OUTPUT_XLSX, index=False)
    log.info("Saved %d rows -> %s", len(df), OUTPUT_XLSX)


# ---------------------------------------------------------------------------
# Parse Model.Rows -> tidy DataFrame
# ---------------------------------------------------------------------------
def parse_rows(body: dict) -> pd.DataFrame:
    model   = body.get("Model", {})
    rows    = model.get("Rows", [])
    columns = model.get("Columns", [])

    if not rows:
        raise ValueError(
            "Model.Rows is empty - check data_raw.json for response structure."
        )

    # Column headers
    col_names = [
        c.get("Member", {}).get("ClickableLabel", {}).get("LabelText", f"Col_{i}")
        for i, c in enumerate(columns)
    ] or ["Antal Personrejser"]

    records = []
    for row in rows:
        label = (
            row.get("Member", {})
               .get("ClickableLabel", {})
               .get("LabelText", "")
        )
        record = {"Dato": label}
        for j, val_obj in enumerate(row.get("Values", [])):
            numeric = val_obj.get("NumericValue", {})
            hint    = (
                val_obj.get("ClickableLabel", {})
                       .get("Clickable", {})
                       .get("HintText", "")
            )
            measure = col_names[j] if j < len(col_names) else f"Col_{j}"
            record[measure] = numeric.get("Value") if numeric else hint
        records.append(record)

    df = pd.DataFrame(records)
    log.info(
        "Parsed %d rows x %d cols: %s",
        len(df), len(df.columns), df.columns.tolist()
    )
    log.info("Sample:\n%s", df.head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    asyncio.run(ingest())
