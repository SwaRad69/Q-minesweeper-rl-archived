"""Dump the actual CSS classes of the first few minesweeper cells.

This tells us exactly what the website is rendering, so we can see
if the class names changed from hd_closed/hd_opened to something else.
"""

import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://minesweeper.online/start/3"
PROFILE = "browser_profile"

with sync_playwright() as pw:
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=str(Path(PROFILE).resolve()),
        headless=False,
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded")

    # Wait a generous amount for the board to load
    print("Waiting for cells to appear...")
    try:
        page.locator(".cell[data-x][data-y]").first.wait_for(
            state="attached", timeout=15_000
        )
    except Exception as e:
        print(f"No .cell[data-x][data-y] elements found: {e}")
        # Try broader selectors
        print("\nTrying broader selector: any element with data-x and data-y...")
        count = page.locator("[data-x][data-y]").count()
        print(f"  Found {count} elements with [data-x][data-y]")
        if count > 0:
            for i in range(min(5, count)):
                el = page.locator("[data-x][data-y]").nth(i)
                tag = el.evaluate("e => e.tagName")
                classes = el.evaluate("e => [...e.classList]")
                x = el.get_attribute("data-x")
                y = el.get_attribute("data-y")
                print(f"  [{i}] <{tag}> data-x={x} data-y={y} classes={classes}")

    # Give extra time for JS to set classes
    print("Waiting 5 more seconds for JS to finish...")
    time.sleep(5)

    # Dump first 10 cells
    cells = page.locator(".cell[data-x][data-y]")
    total = cells.count()
    print(f"\nTotal .cell[data-x][data-y] elements: {total}")

    if total == 0:
        # Try without .cell class
        print("\nNo elements with .cell class. Trying div[data-x][data-y]...")
        cells = page.locator("div[data-x][data-y]")
        total = cells.count()
        print(f"Total div[data-x][data-y] elements: {total}")

    if total == 0:
        print("\nDumping all elements in #game_board or similar container...")
        # Try to find the board container
        for selector in ["#game_board", "#game", "#field", ".board", "#board"]:
            count = page.locator(selector).count()
            if count > 0:
                print(f"  Found {selector} ({count} elements)")
                inner = page.locator(selector).first.evaluate(
                    "e => e.innerHTML.substring(0, 2000)"
                )
                print(f"  innerHTML (first 2000 chars):\n{inner}")
                break

    for i in range(min(10, total)):
        el = cells.nth(i)
        classes = el.evaluate("e => [...e.classList]")
        x = el.get_attribute("data-x")
        y = el.get_attribute("data-y")
        print(f"  cell ({x}, {y}): classes = {classes}")

    # Also dump the full class list of cell (0,0) specifically
    if total > 0:
        print("\n--- Full detail for cell (0, 0) ---")
        cell_00 = page.locator('.cell[data-x="0"][data-y="0"]')
        if cell_00.count() > 0:
            full = cell_00.first.evaluate("""e => ({
                tagName: e.tagName,
                id: e.id,
                className: e.className,
                classList: [...e.classList],
                outerHTML: e.outerHTML
            })""")
            print(json.dumps(full, indent=2))
        else:
            print("  No cell with data-x=0 data-y=0 found")

    print("\nDone. Closing browser...")
    ctx.close()
