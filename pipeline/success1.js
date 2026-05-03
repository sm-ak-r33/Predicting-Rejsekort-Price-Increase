const { chromium } = require("playwright");
const path = require("path");

const URL =
  "https://passagertal.dk/Embed#vfs://global/passagertal.dk/Rejsekort/Rejsekortrejser.xview";

async function getFrame(page) {
  for (const frame of page.frames()) {
    try {
      const count = await frame.evaluate(() =>
        document.querySelectorAll(".ObjectToolbarButton").length
      );
      if (count > 0) return frame;
    } catch {}
  }
  return null;
}

async function snap(page, label) {
  const p = path.join(__dirname, `snap_${label}.png`);
  await page.screenshot({ path: p });
  console.log(`📸 ${p}`);
}

/**
 * Get all currently visible expand icons that look "collapsed"
 * (i.e. their inner rect/path suggests a + not a -)
 * We track already-clicked positions to avoid re-clicking.
 */
async function getCollapsedIcons(frame, clickedSet) {
  return await frame.evaluate((alreadyClicked) => {
    return [...document.querySelectorAll(".expandIcon")]
      .filter(el => {
        const r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0 || r.top < 0 || r.top > window.innerHeight) return false;

        // Check if this icon looks "collapsed" by inspecting its child rects/paths
        // A collapsed (+) icon typically has 2 visible rects (horizontal + vertical bar)
        // An expanded (-) icon typically has 1 visible rect (horizontal bar only)
        const rects = [...el.querySelectorAll("rect")];
        if (rects.length >= 2) {
          // Check if the vertical bar rect is visible (height > width means vertical)
          const vertBar = rects.find(r2 => {
            const h = parseFloat(r2.getAttribute("height") || "0");
            const w = parseFloat(r2.getAttribute("width") || "0");
            return h > w; // vertical bar = collapsed state
          });
          if (!vertBar) return false; // already expanded (only horizontal bar)
        }

        const cx = Math.round(r.left + r.width / 2);
        const cy = Math.round(r.top + r.height / 2);
        const key = `${cx},${cy}`;
        return !alreadyClicked.includes(key);
      })
      .map(el => {
        const r = el.getBoundingClientRect();
        return {
          x: r.left + r.width / 2,
          y: r.top + r.height / 2,
          key: `${Math.round(r.left + r.width / 2)},${Math.round(r.top + r.height / 2)}`,
        };
      });
  }, [...clickedSet]);
}

(async () => {
  const browser = await chromium.launch({ headless: false, slowMo: 80 });
  const context = await browser.newContext({
    acceptDownloads: true,
    viewport: { width: 1600, height: 1000 },
  });
  const page = await context.newPage();

  console.log("Loading...");
  await page.goto(URL, { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(18000);

  let frame = await getFrame(page);
  if (!frame) { console.log("❌ Frame not found"); await browser.close(); return; }

  // ── Switch top-left chart to crosstab view ────────────────────────────────────
  console.log("Switching to crosstab view...");
  await frame.evaluate(() => {
    const specific = document.querySelector("div:nth-child(4) > .ObjectToolbarButton");
    if (specific) { specific.click(); return; }
    const btn = [...document.querySelectorAll(".ObjectToolbarButton")]
      .find(b => b.className.includes("crosstab"));
    if (btn) btn.click();
  });
  await page.waitForTimeout(6000);
  await snap(page, "01_crosstab");

  frame = await getFrame(page) || frame;

  // ── Drill down ONE icon at a time ─────────────────────────────────────────────
  console.log("\nDrilling down (one icon at a time)...");
  const clickedKeys = new Set();
  let noNewIconStreak = 0;

  for (let i = 0; i < 200; i++) { // generous upper bound
    // Scroll to reveal more rows before checking
    await frame.evaluate(() => {
      document.querySelectorAll(".VirtualGrid").forEach(g => {
        g.scrollTop += 200;
      });
    });
    await page.waitForTimeout(300);

    const icons = await getCollapsedIcons(frame, [...clickedKeys]);

    if (icons.length === 0) {
      noNewIconStreak++;
      console.log(`  [${i}] No new collapsed icons (streak: ${noNewIconStreak})`);
      if (noNewIconStreak >= 3) {
        // Scroll back to top and do one final check
        await frame.evaluate(() => {
          document.querySelectorAll(".VirtualGrid").forEach(g => g.scrollTop = 0);
        });
        await page.waitForTimeout(1000);
        const finalCheck = await getCollapsedIcons(frame, [...clickedKeys]);
        if (finalCheck.length === 0) {
          console.log("  ✅ All icons expanded — done!");
          break;
        }
        noNewIconStreak = 0;
      }
      continue;
    }

    noNewIconStreak = 0;
    const { x, y, key } = icons[0]; // click ONE at a time
    clickedKeys.add(key);

    console.log(`  [${i}] Clicking icon at (${Math.round(x)}, ${Math.round(y)}) — ${icons.length} collapsed remaining`);
    await page.mouse.click(x, y);
    await page.waitForTimeout(1200); // wait for children to render
  }

  // Scroll back to top before export
  await frame.evaluate(() => {
    document.querySelectorAll(".VirtualGrid").forEach(g => g.scrollTop = 0);
  });
  await page.waitForTimeout(1000);
  await snap(page, "02_fully_expanded");

  // ── Right-click the grid and export ──────────────────────────────────────────
  console.log("\nExporting...");
  const gridPos = await frame.evaluate(() => {
    const g = document.querySelector(".VirtualGrid");
    if (!g) return null;
    const r = g.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  });

  if (!gridPos) { console.log("❌ VirtualGrid not found"); await browser.close(); return; }

  await page.mouse.click(gridPos.x, gridPos.y, { button: "right" });
  await page.waitForTimeout(2000);
  await snap(page, "03_context_menu");

  const downloadPromise = page.waitForEvent("download", { timeout: 60000 });

  // Click the Export to Excel menu item
  const clicked = await frame.evaluate(() => {
    const btn = [...document.querySelectorAll("button, [role='menuitem'], li, div, span")]
      .filter(el => {
        const r = el.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      })
      .find(el => el.textContent.trim() === "Export to Excel");
    if (btn) { btn.click(); return true; }
    return false;
  });

  if (!clicked) {
    console.log("❌ Export to Excel button not found in menu");
    await browser.close();
    return;
  }

  const download = await downloadPromise;
  await download.saveAs(path.join(__dirname, "rejsekort_daily_export.xlsx"));
  console.log("✅ Saved!");

  await browser.close();
})();