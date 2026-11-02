"""Verify the configured portable demo with real local models and optional Chrome."""

import importlib
import json
import sys
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
package = importlib.import_module("aura_demo_package")
ui = importlib.import_module("aura_demo_ui")
MESSAGES = [
    "How do I activate my new card?",
    "I topped up but the app declined it",
    "There is a payment problem with my card.",
]


def check_backend(backend: Any, folder: Path) -> dict[str, Any]:
    predictions = []
    for encoder in backend.encoders:
        for message in MESSAGES:
            for coverage in [80, 100]:
                result = backend.analyze(message, encoder, coverage)
                p = result["probabilities"]
                assert np.isfinite(p).all() and np.isclose(p.sum(), 1)
                np.testing.assert_allclose(
                    result["decisions"][-1]["score"],
                    result["q"] * result["severity"],
                    rtol=1e-12,
                    atol=1e-12,
                )
                if coverage == 100:
                    assert all(row["decision"] == "AUTO" for row in result["decisions"])
                predictions.append(
                    {
                        "encoder": encoder,
                        "message": message,
                        "coverage": coverage,
                        "intent": result["intent"],
                        "confidence": result["confidence"],
                        "decisions": result["decisions"],
                    }
                )
    for bad in ["", "   ", "x" * 4001]:
        try:
            backend.analyze(bad, next(iter(backend.encoders)), 80)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    selections = 0
    for partition in backend.routing_reports:
        for encoder in backend.encoders:
            for scenario in ["flat", "moderate", "high_protection"]:
                for coverage in ui.COVERAGES:
                    rows = backend.selected_results(partition, encoder, scenario, coverage)
                    assert rows.accepted_count.nunique() == rows.deferred_count.nunique() == 1
                    if coverage == 100:
                        assert rows.accepted_total_cost.nunique() == 1
                        assert rows.accepted_errors.nunique() == 1
                        assert (rows.deferred_count == 0).all()
                    selections += 1
    classical = {}
    for path in sorted((folder / "models/classical").glob("*.joblib")):
        fitted = joblib.load(path)
        predicted = np.asarray(fitted.predict(MESSAGES), dtype=int)
        assert len(predicted) == len(MESSAGES)
        assert ((predicted >= 0) & (predicted < len(backend.intents))).all()
        classical[path.stem] = [backend.intents[label] for label in predicted]
    assert set(classical) == {"logistic_regression", "random_forest"}
    return {
        "live_predictions": predictions,
        "result_selections_checked": selections,
        "classical_predictions": classical,
        "invalid_input": "passed",
    }


def check_browser(backend: Any, app: Any, port: int, output: Path) -> dict[str, Any]:
    from playwright.sync_api import expect, sync_playwright

    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        prevent_thread_lock=True,
        css=ui.CSS,
        theme=ui.light_theme(),
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                headless=True,
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}", wait_until="networkidle")
            expect(page.get_by_text("Inspect exact saved values", exact=True)).to_have_count(0)
            expect(page.get_by_text("Main policy: Cost-aware gate", exact=False)).to_be_visible()

            def choose(label: str, value: str) -> None:
                control = page.get_by_role("combobox", name=label, exact=True)
                if control.input_value() != value:
                    control.click()
                    page.get_by_role("option", name=value, exact=True).click()

            for encoder in backend.encoders:
                choose("Encoder", encoder)
                choose("Target automation", "80%")
                message = MESSAGES[1]
                expected = backend.analyze(message, encoder, 80)
                page.get_by_role("textbox", name="Customer message", exact=True).fill(message)
                page.get_by_role("button", name="Analyze query", exact=True).click()
                expect(page.locator(".aura-summary h2")).to_have_text(
                    expected["intent"].replace("_", " "),
                    timeout=60000,
                )
                expect(page.locator(".aura-decision")).to_have_text(
                    [row["decision"] for row in expected["decisions"]],
                    timeout=60000,
                )
                expect(page.locator(".aura-card h3")).to_have_text(list(ui.POLICIES.values()))
                accordion = page.get_by_text("Why did AURA make this decision?", exact=True)
                if not page.locator(".katex-display").is_visible():
                    accordion.click()
                expect(page.locator(".katex-display")).to_be_visible()
                page.screenshot(path=output / f"live_{encoder}.png", full_page=True)
                choose("Target automation", "100%")
                expect(page.locator(".aura-decision")).to_have_count(0)
                page.get_by_role("button", name="Analyze query", exact=True).click()
                expect(page.locator(".aura-decision")).to_have_text(
                    ["AUTO", "AUTO", "AUTO"],
                    timeout=60000,
                )

            page.get_by_role("tab", name="Experimental results", exact=True).click()
            for partition in backend.routing_reports:
                choose("Result partition", partition.removesuffix(" (already exposed)"))
                for encoder in backend.encoders:
                    choose("Encoder", encoder)
                    for scenario in ["moderate", "high_protection", "flat"]:
                        choose("Evaluation cost scenario", scenario)
                        choose("Target automation", "80%")
                        selected = backend.selected_results(partition, encoder, scenario, 80)
                        selected = selected.set_index("policy")
                        difference = (
                            selected.loc[ui.MAIN_POLICY_ID, "accepted_total_cost"]
                            - selected.loc["confidence", "accepted_total_cost"]
                        )
                        expected_note = (
                            f"{encoder} · {partition.removesuffix(' (already exposed)')}"
                            f" · {scenario} evaluation scenario · 80% target"
                        )
                        expect(page.get_by_text(expected_note, exact=True)).to_be_visible()
                        note = page.get_by_text("AURA minus confidence cost:", exact=False)
                        expect(note).to_contain_text(f"AURA minus confidence cost: {difference:+g}")
            for partition in backend.classifier_reports:
                choose(
                    "Classifier comparison partition", partition.removesuffix(" (already exposed)")
                )
            page.screenshot(path=output / "experimental_results.png", full_page=True)
            page.get_by_role("tab", name="Live routing", exact=True).click()
            page.get_by_role("textbox", name="Customer message", exact=True).fill("")
            page.get_by_role("button", name="Analyze query", exact=True).click()
            expect(
                page.get_by_text(
                    "Enter a banking-support message before analyzing it.", exact=False
                )
            ).to_be_visible(timeout=10000)
            assert not errors, errors
            browser.close()
            return {
                "status": "passed",
                "javascript_errors": errors,
                "encoders_checked": list(backend.encoders),
                "table_removed": True,
                "policy_name": ui.MAIN_POLICY_LABEL,
            }
    finally:
        app.close()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true")
    parser.add_argument("--port", type=int, default=7867)
    args = parser.parse_args()
    output = ROOT / "artifacts/demo_ui_checks" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    folder = package.configured_package_directory()
    assert folder is not None
    torch.set_num_threads(2)
    report = {"package": package.inspect_demo_directory(folder), "status": "running"}
    try:
        assert report["package"]["ready"]
        print("Loading both encoders from the verified local package…", flush=True)
        backend = package.load_demo_package(folder)
        report["backend"] = check_backend(backend, folder)
        print(
            "Real-model predictions, invalid input, and saved result selections passed.", flush=True
        )
        app = ui.build_app(backend)
        if args.browser:
            report["browser"] = check_browser(backend, app, args.port, output)
        else:
            report["browser"] = "not requested; interface construction passed"
            app.close()
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Verification report:", output / "verification.json", flush=True)


if __name__ == "__main__":
    main()
