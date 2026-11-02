"""UI routing correctness and runtime wiring, using clearly synthetic inputs."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Exercise the same single-file upload mechanism as Colab. This UI intentionally
# lives outside the installable src package and must not require PYTHONPATH edits.
UI_PATH = Path(__file__).parents[2] / "aura_demo_ui.py"
SPEC = importlib.util.spec_from_file_location("aura_demo_ui", UI_PATH)
assert SPEC is not None and SPEC.loader is not None
UI = importlib.util.module_from_spec(SPEC)
sys.modules["aura_demo_ui"] = UI
SPEC.loader.exec_module(UI)
if TYPE_CHECKING:
    from aura_demo_ui import NotebookDemo
else:
    NotebookDemo = UI.NotebookDemo
POLICIES = UI.POLICIES
build_app = UI.build_app
render_prediction = UI.render_prediction
results_figures = UI.results_figures
validation_cutoff = UI.validation_cutoff


@pytest.fixture
def backend() -> NotebookDemo:
    """Use the actual notebook scoring functions, without loading GPU checkpoints."""
    notebook = json.loads(
        (Path(__file__).parents[2] / "notebooks/AURA_Colab_Demo.ipynb").read_text()
    )
    costs = np.array([[0, 5, 2], [2, 0, 3], [1, 2, 0]], dtype=float)
    namespace: dict[str, Any] = {
        "np": np,
        "NUM_CLASSES": 3,
        "C_MODERATE": costs,
        "COST_MATRICES": {"flat": np.ones((3, 3)) - np.eye(3), "high_protection": costs * 2},
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.startswith(("SCENARIOS =", "from sklearn.pipeline", "POLICY_LABELS =")):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in {
                "expected_risk",
                "lexical_features",
                "routing_variant_scores",
            }:
                exec(
                    compile(ast.Module(body=[node], type_ignores=[]), "notebook", "exec"), namespace
                )
    gate = Mock()
    gate.predict_proba.return_value = np.array([[0.2, 0.8]])
    witness = Mock(classes_=np.arange(3))
    witness.decision_function.return_value = np.array([[0.8, 0.4, 0.2]])
    predict = Mock(return_value=np.array([[2.0, 1.0, 0.0]]))
    states = {}
    for name, temperature in [("DistilBERT", 0.8), ("Ettin-150M", 1.2)]:
        states[name] = {
            "temperature": temperature,
            "model": object(),
            "tokenizer": object(),
            "config": {"maximum_length": 128},
            "gates": {"semantic_gate": gate, "signed_gate": gate},
            "val_scores": {key: np.linspace(-1, 1, 10) for key in POLICIES},
        }
    records = [
        {
            "encoder": encoder,
            "evaluation_scenario": scenario,
            "policy": policy,
            "target_coverage": coverage,
            "accepted_count": int(coverage * 100),
            "deferred_count": 100 - int(coverage * 100),
            "accepted_errors": 3,
            "accepted_total_cost": 5,
            "accepted_accuracy": 0.95,
        }
        for encoder in states
        for scenario in ["flat", "moderate", "high_protection"]
        for policy in POLICIES
        for coverage in [0.5, 0.7, 0.8, 0.9, 1.0]
    ]
    return NotebookDemo(
        states,
        ["first", "second", "third"],
        [{"label": i, "required_action": "provide_information"} for i in range(3)],
        costs,
        witness,
        predict,
        namespace["lexical_features"],
        namespace["routing_variant_scores"],
        {"Validation": pd.DataFrame(records)},
        {"Validation": pd.DataFrame({"model": ["Synthetic fixture"], "accuracy": [0.8]})},
        "SYNTHETIC-UI-TEST-NOT-RESEARCH-RESULTS",
    )


def test_one_inference_uses_selected_encoder_and_existing_cost_formula(
    backend: NotebookDemo,
) -> None:
    result = backend.analyze("Where is my card?", "Ettin-150M", 80)
    assert backend.predict_logits.call_count == 1  # type: ignore[attr-defined]
    arguments = backend.predict_logits.call_args.kwargs  # type: ignore[attr-defined]
    assert arguments["trained_model"] is backend.encoders["Ettin-150M"]["model"]
    assert result["temperature"] == 1.2
    assert result["decisions"][-1]["score"] == pytest.approx(
        result["q"] * result["risk"] / result["mass"]
    )
    assert result["intent"] == "first"
    assert all(not state["gates"]["signed_gate"].fit.called for state in backend.encoders.values())


@pytest.mark.parametrize("bad", ["", "  ", "x" * 4001])
def test_bad_input_never_runs_inference(backend: NotebookDemo, bad: str) -> None:
    with pytest.raises(ValueError):
        backend.analyze(bad, "Ettin-150M", 80)
    backend.predict_logits.assert_not_called()  # type: ignore[attr-defined]


def test_validation_cutoff_exact_floor_and_full_automation(backend: NotebookDemo) -> None:
    assert validation_cutoff(np.arange(1501), 80) == 1199
    assert validation_cutoff(np.arange(5), 100) == np.inf
    result = backend.analyze("A query", "DistilBERT", 100)
    assert all(item["decision"] == "AUTO" for item in result["decisions"])
    with pytest.raises(ValueError, match="automation target"):
        validation_cutoff(np.arange(10), 75)
    with pytest.raises(ValueError, match="finite vector"):
        validation_cutoff([1, np.nan], 80)


def test_live_result_is_escaped_and_has_no_claim_of_actual_harm(backend: NotebookDemo) -> None:
    result = backend.analyze("A query", "DistilBERT", 80)
    result["intent"] = "<script>alert(1)</script>"
    summary, cards, explanation = render_prediction(result)
    assert "<script>" not in summary
    assert "&lt;script&gt;" in summary
    assert cards.count('class="aura-card') == 4  # The container plus three cards.
    assert "not known true labels" in explanation


def test_result_scenarios_only_select_saved_rows(backend: NotebookDemo) -> None:
    selected = backend.selected_results("Validation", "DistilBERT", "high_protection", 90)
    assert set(selected.evaluation_scenario) == {"high_protection"}
    assert set(selected.target_coverage) == {0.9}
    backend.predict_logits.assert_not_called()  # type: ignore[attr-defined]
    report = backend.routing_reports["Validation"]
    report.loc[report.policy == "confidence", "accepted_count"] = 1
    with pytest.raises(ValueError, match="matched acceptance"):
        backend.selected_results("Validation", "DistilBERT", "moderate", 80)


def test_missing_preparation_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="through section 11B"):
        NotebookDemo.from_namespace({})


def test_gradio_constructs_and_results_render(backend: NotebookDemo) -> None:
    pytest.importorskip("gradio")
    pytest.importorskip("matplotlib")
    figure, curve, note = results_figures(backend, "Validation", "Ettin-150M", "moderate", 80)
    assert len(figure.axes) == 2
    assert len(curve.axes[0].lines) == 4
    assert "not establish statistical significance" in note
    app = build_app(backend)
    assert app.config["title"] == "AURA | Banking Intent Routing"
    assert len(app.config["dependencies"]) >= 9
    app.close()
