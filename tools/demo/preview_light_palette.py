"""Inspect the light presentation theme under both browser color preferences."""

from pathlib import Path

from playwright.sync_api import expect, sync_playwright
from verify_saved_distilbert import prepared_backend, ui

output = Path(__file__).resolve().parents[2] / "artifacts" / "demo_ui_checks"
output.mkdir(parents=True, exist_ok=True)
app = ui.build_app(prepared_backend())
try:
    app.launch(
        server_name="127.0.0.1",
        server_port=7866,
        share=False,
        prevent_thread_lock=True,
        css=ui.CSS,
        theme=ui.light_theme(),
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            headless=True,
        )
        for preference in ("light", "dark"):
            page = browser.new_page(
                viewport={"width": 1440, "height": 1100}, color_scheme=preference
            )
            page.goto(f"http://127.0.0.1:7866/?__theme={preference}", wait_until="networkidle")
            expect(page.get_by_role("button", name="Analyze query", exact=True)).to_be_visible()
            page.get_by_role("textbox", name="Customer message", exact=True).fill(
                "I topped up but the app declined it"
            )
            page.get_by_role("button", name="Analyze query", exact=True).click()
            expect(page.locator(".aura-decision")).to_have_text(
                ["AUTO", "AUTO", "DEFER"], timeout=30000
            )
            gradient = page.locator(".aura-hero").evaluate(
                "el => getComputedStyle(el).backgroundImage"
            )
            body_color = page.locator(".gradio-container").evaluate(
                "el => getComputedStyle(el).getPropertyValue('--body-background-fill').trim()"
            )
            assert "237, 248, 255" in gradient and "232, 248, 241" in gradient
            assert body_color.lower() == "#f7fafc", body_color
            page.screenshot(path=output / f"light_palette_{preference}_browser.png", full_page=True)
            print(f"{preference} browser: light body and header verified; routing unchanged.")
            page.close()
        browser.close()
finally:
    app.close()
