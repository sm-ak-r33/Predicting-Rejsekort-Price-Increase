"""
diagnose.py
-----------
Run this BEFORE ingest.py to understand what Playwright actually sees on
the passagertal.dk dashboard. It will produce:

  - screenshot.png        → visual snapshot of what was rendered
  - page_source.html      → raw HTML of the top-level page
  - all_requests.txt      → every network request URL (main page + iframes)
  - frames.txt            → list of all frames/iframes and their URLs

Usage:
    python pipeline/diagnose.py
"""

import asyncio
import logging
from pathlib import Path
from playwright.async_api import async_playwright

DASHBOARD_URL = (
    "https://passagertal.dk/Embed"
    "#vfs://Global/passagertal.dk/Rejsekort/Rejsekortrejser.xview"
)
TIMEOUT_S = 120

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("diagnose")

all_requests = []

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )
        context = await browser.new_context(
            locale="da-DK",
            viewport={"width": 1920, "height": 1080},
        )

        # Capture ALL network requests across context (incl. iframes)
        context.on("request",  lambda req: all_requests.append(("REQ", req.url, req.method)))
        context.on("response", lambda res: all_requests.append(("RES", res.url, str(res.status))))

        page = await context.new_page()

        # ── 1. Navigate ──────────────────────────────────────────────────────
        log.info("Navigating to dashboard …")
        try:
            await page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=TIMEOUT_S * 1000)
            log.info("domcontentloaded fired")
        except Exception as e:
            log.warning("goto raised: %s", e)

        # Give TARGIT extra time to bootstrap
        log.info("Waiting 15 s for TARGIT to load …")
        await asyncio.sleep(15)

        # ── 2. Screenshot ────────────────────────────────────────────────────
        await page.screenshot(path="screenshot.png", full_page=True)
        log.info("Screenshot saved → screenshot.png")

        # ── 3. Page HTML ─────────────────────────────────────────────────────
        html = await page.content()
        Path("page_source.html").write_text(html, encoding="utf-8")
        log.info("Page HTML saved → page_source.html  (%d chars)", len(html))

        # ── 4. All frames ────────────────────────────────────────────────────
        frame_lines = []
        for i, frame in enumerate(page.frames):
            url = frame.url
            name = frame.name
            frame_lines.append(f"Frame {i}: name={name!r}  url={url}")
            log.info("  %s", frame_lines[-1])

            # Also dump inner HTML of each frame for inspection
            try:
                fhtml = await frame.content()
                Path(f"frame_{i}_source.html").write_text(fhtml, encoding="utf-8")
                log.info("    → frame_%d_source.html (%d chars)", i, len(fhtml))
            except Exception as e:
                log.warning("    Could not get frame %d content: %s", i, e)

        Path("frames.txt").write_text("\n".join(frame_lines), encoding="utf-8")
        log.info("Frame list saved → frames.txt")

        # ── 5. Network log ───────────────────────────────────────────────────
        lines = [f"{kind}  {method_or_status}  {url}" for kind, url, method_or_status in all_requests]
        Path("all_requests.txt").write_text("\n".join(lines), encoding="utf-8")
        log.info("Network log saved → all_requests.txt  (%d entries)", len(lines))

        # ── 6. Quick element probe ───────────────────────────────────────────
        log.info("Probing for expand-like elements across all frames …")
        selectors_to_probe = ["text=+", "[class*='expand']", "[class*='drill']", "iframe", "button", "td"]
        for frame in page.frames:
            for sel in selectors_to_probe:
                try:
                    count = await frame.locator(sel).count()
                    if count:
                        log.info("  frame=%r  selector=%r  → %d element(s)", frame.url[:60], sel, count)
                except Exception:
                    pass

        await browser.close()
        log.info("Done. Check screenshot.png, page_source.html, frames.txt, all_requests.txt")

if __name__ == "__main__":
    asyncio.run(main())
