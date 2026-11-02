"""Exercise the local verification UI in an isolated headless Chrome session."""

import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

OUTPUT = Path(__file__).resolve().parents[2] / "artifacts" / "demo_ui_checks"
OUTPUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        headless=True,
    )
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:7864", wait_until="networkidle")
    page.get_by_role("textbox", name="Customer message", exact=True).fill(
        "There is a payment problem with my card."
    )
    page.get_by_role("button", name="Analyze query", exact=True).click()
    expect(page.locator(".aura-summary h2")).to_have_text("declined card payment", timeout=30000)
    expect(page.locator(".aura-decision")).to_have_text(["DEFER", "DEFER", "DEFER"])
    page.get_by_text("Why did AURA make this decision?", exact=True).click()
    expect(page.get_by_text("AURA chose DEFER", exact=False)).to_be_visible()
    expect(page.locator(".katex-display")).to_be_visible()
    page.screenshot(path=OUTPUT / "live_routing.png", full_page=True)
    page.get_by_role("combobox", name="Target automation", exact=True).first.click()
    page.get_by_role("option", name="100%", exact=True).click()
    expect(page.locator(".aura-decision")).to_have_count(0)
    page.get_by_role("button", name="Analyze query", exact=True).click()
    expect(page.locator(".aura-decision")).to_have_text(["AUTO", "AUTO", "AUTO"], timeout=30000)
    page.get_by_role("tab", name="Experimental results", exact=True).click()
    expect(page.get_by_text("AURA minus confidence cost:", exact=False)).to_be_visible()
    page.screenshot(path=OUTPUT / "experimental_results.png", full_page=True)
    page.get_by_role("combobox", name="Evaluation cost scenario", exact=True).click()
    page.get_by_role("option", name="high_protection", exact=True).click()
    expect(page.get_by_text("high_protection evaluation scenario", exact=False)).to_be_visible()
    page.get_by_role("tab", name="Live routing", exact=True).click()
    page.get_by_role("textbox", name="Customer message", exact=True).fill("")
    page.get_by_role("button", name="Analyze query", exact=True).click()
    expect(
        page.get_by_text("Enter a banking-support message before analyzing it.", exact=False)
    ).to_be_visible(timeout=10000)
    assert not errors, errors
    print(
        json.dumps(
            {
                "page_errors": errors,
                "live_inference": "passed",
                "full_automation_and_input_reset": "passed",
                "saved_charts": "passed",
                "scenario_change": "passed",
                "empty_input": "passed",
            }
        )
    )
    browser.close()
