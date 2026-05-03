"""
ingest_playwright_click.py

Browser-driven scraper for passagertal.dk Rejsekort daily passenger data.

This uses Playwright to open the TARGIT dashboard, click/drill through the visual UI,
read the rendered VirtualGrid text, parse date + Antal Personrejser pairs, and write
Data(update).xlsx.

Install:
    pip install playwright pandas openpyxl
    playwright install chromium

Run:
    python ingest_playwright_click.py --year 2025 --output "Data(update).xlsx"

Debug mode:
    python ingest_playwright_click.py --year 2025 --headful --output "Data(update).xlsx"
"""

import argparse
import asyncio
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError, async_playwright

URL = "https://passagertal.dk/Embed#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview"

DANISH_MONTHS = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "Maj",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Okt",
    11: "Nov",
    12: "Dec",
}

DATE_RE = re.compile(r"\b(\d{2}-\d{2}-\d{4})\b")
NUM_RE = re.compile(r"^[0-9][0-9.\s,]*$")


@dataclass(frozen=True)
class Row:
    date: str
    antal_personrejser: int


def clean_number(value: str) -> Optional[int]:
    value = value.strip().replace(".", "").replace(" ", "").replace(",", "")
    if not value.isdigit():
        return None
    return int(value)


def parse_grid_text(text: str) -> List[Row]:
    """Parse rendered grid text containing alternating date/value rows."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows: List[Row] = []

    i = 0
    while i < len(lines):
        m = DATE_RE.search(lines[i])
        if not m:
            i += 1
            continue

        date_s = m.group(1)
        value: Optional[int] = None

        # Usually value is next line, but scan a few lines to be robust.
        for j in range(i + 1, min(i + 5, len(lines))):
            if DATE_RE.search(lines[j]):
                break
            if NUM_RE.match(lines[j]):
                value = clean_number(lines[j])
                if value is not None:
                    break

        if value is not None:
            rows.append(Row(date=date_s, antal_personrejser=value))
        i += 1

    return rows


async def wait_for_dashboard(page: Page) -> None:
    """Wait until the TARGIT shell and visible dashboard have loaded."""
    await page.goto(URL, wait_until="domcontentloaded")

    # Wait for auth/doc/cube bootstrap endpoints. These are reliable signs of real dashboard init.
    try:
        await page.wait_for_response(lambda r: "Documents/GetDocument" in r.url, timeout=60_000)
    except PlaywrightTimeoutError:
        pass

    try:
        await page.wait_for_response(lambda r: "server/connections" in r.url, timeout=60_000)
    except PlaywrightTimeoutError:
        pass

    # Allow Blazor/canvas rendering to finish.
    await page.wait_for_timeout(10_000)


async def get_grid_text(page: Page) -> str:
    """Return the largest visible VirtualGrid text block."""
    grids = page.locator("div.VirtualGrid")
    count = await grids.count()
    texts: List[str] = []
    for i in range(count):
        try:
            txt = await grids.nth(i).inner_text(timeout=2_000)
            if "Antal Personrejser" in txt or DATE_RE.search(txt):
                texts.append(txt)
        except Exception:
            continue

    if texts:
        return max(texts, key=len)

    # Fallback: whole page text. Less clean, but often still parseable.
    try:
        return await page.locator("body").inner_text(timeout=5_000)
    except Exception:
        return ""


async def click_text_if_available(page: Page, text: str, timeout: int = 2_000) -> bool:
    candidates = [
        page.get_by_text(text, exact=True),
        page.locator(f"text={text}"),
    ]
    for loc in candidates:
        try:
            if await loc.count() > 0:
                await loc.first.click(timeout=timeout)
                return True
        except Exception:
            continue
    return False


async def click_year(page: Page, year: int) -> bool:
    # Try simple text click first.
    if await click_text_if_available(page, str(year), timeout=3_000):
        await page.wait_for_timeout(2_000)
        return True

    # Fallback: click approximate year labels in crosstab cells.
    cells = page.locator("div.CrosstabCellInner")
    n = await cells.count()
    for i in range(n):
        try:
            txt = (await cells.nth(i).inner_text(timeout=500)).strip()
            if txt == str(year):
                await cells.nth(i).click(timeout=2_000)
                await page.wait_for_timeout(2_000)
                return True
        except Exception:
            continue
    return False


async def click_month(page: Page, month: int) -> bool:
    label = DANISH_MONTHS[month]
    if await click_text_if_available(page, label, timeout=3_000):
        await page.wait_for_timeout(2_000)
        return True

    cells = page.locator("div.CrosstabCellInner")
    n = await cells.count()
    for i in range(n):
        try:
            txt = (await cells.nth(i).inner_text(timeout=500)).strip()
            if txt == label:
                await cells.nth(i).click(timeout=2_000)
                await page.wait_for_timeout(2_000)
                return True
        except Exception:
            continue
    return False


async def nudge_grid_scroll(page: Page, steps: int = 6) -> None:
    """Scroll inside the grid so lazy-rendered rows become visible."""
    grid = page.locator("div.VirtualGrid").first
    try:
        box = await grid.bounding_box(timeout=2_000)
        if not box:
            return
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        await page.mouse.move(x, y)
        for _ in range(steps):
            await page.mouse.wheel(0, 700)
            await page.wait_for_timeout(300)
    except Exception:
        return


async def scrape_month(page: Page, year: int, month: int) -> List[Row]:
    before = await get_grid_text(page)

    clicked = await click_month(page, month)
    if not clicked:
        print(f"WARN: Could not click month {month:02d}; trying to parse current grid anyway.")

    try:
        await page.wait_for_response(lambda r: "server/connections" in r.url or "Visual/GetModel" in r.url, timeout=8_000)
    except PlaywrightTimeoutError:
        pass

    await page.wait_for_timeout(2_000)

    # Capture top + scroll-rendered rows.
    texts = [await get_grid_text(page)]
    await nudge_grid_scroll(page, steps=10)
    texts.append(await get_grid_text(page))

    rows_by_date: Dict[str, Row] = {}
    for txt in texts:
        for row in parse_grid_text(txt):
            try:
                dt = datetime.strptime(row.date, "%d-%m-%Y")
            except ValueError:
                continue
            if dt.year == year and dt.month == month:
                rows_by_date[row.date] = row

    # If we got nothing but page changed, include debug print to help troubleshoot.
    if not rows_by_date:
        after = texts[-1]
        Path("debug_passagertal").mkdir(exist_ok=True)
        Path(f"debug_passagertal/grid_{year}_{month:02d}.txt").write_text(after, encoding="utf-8")
        if after == before:
            print(f"WARN: Month {month:02d} click did not change grid text.")
        else:
            print(f"WARN: Month {month:02d} changed grid but no rows parsed. Saved debug grid text.")

    return [rows_by_date[k] for k in sorted(rows_by_date.keys(), key=lambda s: datetime.strptime(s, "%d-%m-%Y"))]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=datetime.now().year - 1)
    parser.add_argument("--output", default="Data(update).xlsx")
    parser.add_argument("--headful", action="store_true", help="Show browser window")
    parser.add_argument("--months", default="1-12", help="Month range, e.g. 1-12 or 2,3,4")
    args = parser.parse_args()

    if "-" in args.months:
        a, b = args.months.split("-", 1)
        months = list(range(int(a), int(b) + 1))
    else:
        months = [int(x) for x in args.months.split(",") if x.strip()]

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.headful,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 1000},
            locale="da-DK",
            timezone_id="Europe/Copenhagen",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
            ),
        )
        page = await context.new_page()

        print("Opening dashboard...")
        await wait_for_dashboard(page)

        print(f"Selecting year {args.year}...")
        if not await click_year(page, args.year):
            print(f"WARN: Could not click year {args.year}. Continuing with current dashboard state.")

        all_rows: Dict[str, Row] = {}
        for month in months:
            print(f"Scraping {args.year}-{month:02d}...")
            rows = await scrape_month(page, args.year, month)
            print(f"  rows: {len(rows)}")
            for row in rows:
                all_rows[row.date] = row

        await browser.close()

    df = pd.DataFrame(
        [
            {
                "Afgangsdato": datetime.strptime(r.date, "%d-%m-%Y").date(),
                "Antal Personrejser": r.antal_personrejser,
            }
            for r in sorted(all_rows.values(), key=lambda x: datetime.strptime(x.date, "%d-%m-%Y"))
        ]
    )

    output = Path(args.output)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")

    print(f"Saved {output} with {len(df)} rows")
    if len(df) == 0:
        print("No rows parsed. Re-run with --headful and inspect debug_passagertal/grid_*.txt")


if __name__ == "__main__":
    asyncio.run(main())
