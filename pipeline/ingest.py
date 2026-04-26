"""
ingest.py
---------
Automated data ingestion from passagertal.dk (TARGIT Anywhere v26.3).

The dashboard contains a crosstab table with a Year → Month → Day hierarchy.
Clicking the "+" next to each year/month expands it to daily data.
This script:
  1. Opens the dashboard in headless Chromium
  2. Clicks ALL "+" expand buttons repeatedly until no more appear
     (Year → Month → Day, fully expanded)
  3. Captures the resulting GetModel response which contains daily data
  4. Parses Model.Rows → pandas DataFrame → Data(update).xlsx

Usage:
    python pipeline/ingest.py

Requirements:
    playwright>=1.44   (pip install playwright && playwright install chromium)
    pandas>=2.0        (pip install pandas)
    openpyxl>=3.1      (pip install openpyxl)
"""

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd
from typing import Dict
from playwright.async_api import async_playwright, Response

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DASHBOARD_URL = (
    "https://passagertal.dk/Embed"
    "#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview"
)
GETMODEL_PATH = "/Visual/GetModel"
OUTPUT_XLSX   = Path("Data(update).xlsx")
TIMEOUT_S     = 180

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Track ALL GetModel responses, keyed by ObjectId
# We want the one with the most rows (= most drilled-down = daily data)
# ---------------------------------------------------------------------------
all_responses: Dict[str, dict] = {}   # objectId → latest response body


def object_id_from_url(url: str) -> str:
    """Extract the raw GUID from a GetModel URL."""
    import re
    m = re.search(r"ObjectId=%7B([^%]+)%7D", url, re.IGNORECASE)
    return m.group(1) if m else url


async def handle_response(response: Response) -> None:
    if GETMODEL_PATH not in response.url:
        return
    if response.status != 200:
        return
    try:
        body = await response.json()
    except Exception:
        return

    oid       = object_id_from_url(response.url)
    state     = body.get("State", {})
    row_count = state.get("CrosstabRowCount", 0)
    title     = body.get("Model", {}).get("Title", "")

    log.info(
        "GetModel  rows=%-5s  title='%s'  id=%s",
        row_count, title[:50], oid[:8] + "…"
    )
    all_responses[oid] = body


# ---------------------------------------------------------------------------
# Click all expand ("+" drill-down) buttons visible on the page
# Returns the number of buttons clicked
# ---------------------------------------------------------------------------
async def click_all_expand_buttons(page) -> int:
    """
    TARGIT renders expand/collapse controls as elements that:
      - contain the text "+" or "−"
      - OR have a CSS class containing 'expand', 'drill', 'toggle', 'open'
      - OR are <td>/<span> with role="button" or a click handler

    We try several selectors and click every visible one.
    """
    selectors = [
        # Text content "+"
        "text=+",
        # Common TARGIT expand class patterns
        "[class*='expand']",
        "[class*='drill']",
        "[class*='toggle']",
        "[class*='lsExpand']",
        "[class*='lsDrill']",
        "[class*='tg-expand']",
        # SVG/icon expand arrows
        "svg[class*='expand']",
        "span[class*='expand']",
        "div[class*='expand']",
        "td[class*='expand']",
    ]

    clicked = 0
    for sel in selectors:
        try:
            elements = await page.locator(sel).all()
            for el in elements:
                try:
                    if await el.is_visible():
                        await el.click(timeout=2000)
                        clicked += 1
                        await asyncio.sleep(0.3)
                except Exception:
                    pass
        except Exception:
            pass

    return clicked


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
        page.on("response", handle_response)

        # ── 1. Load dashboard ────────────────────────────────────────────────
        log.info("Opening dashboard …")
        await page.goto(DASHBOARD_URL, wait_until="networkidle", timeout=TIMEOUT_S * 1000)
        log.info("Page loaded. Objects captured so far: %d", len(all_responses))
        await asyncio.sleep(3)

        # ── 2. Drill down: click "+" buttons up to 3 levels deep ────────────
        #    Level 1: expand years  → months appear
        #    Level 2: expand months → days appear
        #    Level 3: expand any remaining collapsed nodes
        for level in range(1, 4):
            log.info("Drill level %d — looking for expand buttons …", level)
            n = await click_all_expand_buttons(page)
            log.info("Clicked %d expand button(s) at level %d", n, level)
            if n == 0:
                log.info("No more expand buttons found — drill complete.")
                break
            # Wait for TARGIT to fire new GetModel requests after expansion
            await page.wait_for_load_state("networkidle", timeout=30_000)
            await asyncio.sleep(3)

        log.info("Total GetModel responses captured: %d", len(all_responses))
        await browser.close()

    # ── 3. Pick the response with the most rows (= daily data) ──────────────
    if not all_responses:
        raise RuntimeError("No GetModel responses captured at all.")

    best_oid, best_body = max(
        all_responses.items(),
        key=lambda kv: kv[1].get("State", {}).get("CrosstabRowCount", 0),
    )
    best_rows = best_body.get("State", {}).get("CrosstabRowCount", 0)
    best_title = best_body.get("Model", {}).get("Title", "")
    log.info(
        "Using object %s — '%s' (%d rows)",
        best_oid[:8] + "…", best_title, best_rows
    )

    # Save raw JSON for debugging
    Path("data_raw.json").write_text(
        json.dumps(best_body, indent=2, ensure_ascii=False)
    )

    # ── 4. Parse Model.Rows → DataFrame ─────────────────────────────────────
    df = parse_rows(best_body)
    df.to_excel(OUTPUT_XLSX, index=False)
    log.info("Saved %d rows → %s", len(df), OUTPUT_XLSX)


# ---------------------------------------------------------------------------
# Parse Model.Rows into a tidy DataFrame
# ---------------------------------------------------------------------------
def parse_rows(body: dict) -> pd.DataFrame:
    model   = body.get("Model", {})
    rows    = model.get("Rows", [])
    columns = model.get("Columns", [])

    if not rows:
        raise ValueError(
            "Model.Rows is empty. Check data_raw.json for the response structure."
        )

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
        values = row.get("Values", [])
        record = {"Dato": label}
        for j, val_obj in enumerate(values):
            numeric = val_obj.get("NumericValue", {})
            hint    = val_obj.get("ClickableLabel", {}).get("Clickable", {}).get("HintText", "")
            measure = col_names[j] if j < len(col_names) else f"Col_{j}"
            # NumericValue.Value is the raw number; HintText has the formatted string
            record[measure] = numeric.get("Value") if numeric else hint
        records.append(record)

    df = pd.DataFrame(records)
    log.info("Parsed %d rows × %d cols: %s", *df.shape, df.columns.tolist())
    log.info("Sample:\n%s", df.head(5).to_string(index=False))
    return df


if __name__ == "__main__":
    asyncio.run(ingest())