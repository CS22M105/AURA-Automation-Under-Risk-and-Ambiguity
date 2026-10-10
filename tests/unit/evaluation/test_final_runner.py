"""Exercise runner safeguards and an end-to-end synthetic run, never real test rows."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from aura.data.intents import BANKING77_LABELS
from aura.data.manifests import calculate_sha256
from aura.evaluation.routing_uncertainty import RoutingCostInterval
from aura.models.predictions import create_prediction_batch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/evaluate_final_routing.py"


def test_default_mode_only_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    globals_ = namespace["main"].__globals__
    monkeypatch.setitem(globals_, "preflight", lambda: {})

    def forbidden(*args: object) -> None:
        raise AssertionError("Final run must require explicit opt-in")

    monkeypatch.setitem(globals_, "execute_final", forbidden)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    namespace["main"]()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    marker = tmp_path / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        namespace["execute_final"]({"output_directory": str(tmp_path)}, "cpu", "synthetic")
    assert marker.read_text() == "keep"


def test_runner_lock_drift_stops_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    lock = tmp_path / "runner.json"
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    lock.write_text(json.dumps({"protocol_lock_sha256": "wrong"}))
    monkeypatch.setitem(namespace["preflight"].__globals__, "RUNNER_LOCK", lock)
    monkeypatch.setitem(namespace["preflight"].__globals__, "PROTOCOL_LOCK", protocol)
    with pytest.raises(ValueError, match="protocol lock changed"):
        namespace["preflight"]()


def test_end_to_end_synthetic_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import joblib

    from aura.calibration import artifacts as calibration_artifacts
    from aura.calibration.temperature import TemperatureScaler
    from aura.models import distilbert, transformer_artifacts
    from aura.risk import scenarios, workflows

    namespace = runpy.run_path(str(SCRIPT))
    globals_ = namespace["execute_final"].__globals__
    config = json.loads((ROOT / "config/final_evaluation.json").read_text())
    monkeypatch.chdir(tmp_path)
    labels = np.repeat(np.arange(77), 40)
    test = pd.DataFrame(
        {
            "text": ["synthetic example"] * len(labels),
            "label": labels,
            "label_text": [BANKING77_LABELS[int(i)] for i in labels],
        }
    )
    Path("datasets").mkdir()
    test.to_csv("datasets/test.csv", index=False)
    test.iloc[:1].to_csv("datasets/train.csv", index=False)
    source = Path("artifacts/risk/distilbert/summary.json")
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps({"provenance": {"dataset_sha256": calculate_sha256(Path("datasets/train.csv"))}})
    )
    Path("reports/protocols").mkdir(parents=True)
    for filename in ("aura-final-runner-v1.json", "aura-final-evaluation-v1.json"):
        (Path("reports/protocols") / filename).write_text("{}")
    p = np.full((len(labels), 77), 0.1 / 76)
    p[np.arange(len(labels)), labels] = 0.9
    raw = create_prediction_batch(list(range(len(labels))), list(range(77)), p, np.log(p))
    model = SimpleNamespace(
        model=None,
        tokenizer=None,
        metadata={"configuration": {"maximum_length": 128, "batch_size": 16}},
    )
    monkeypatch.setattr(transformer_artifacts, "load_transformer_artifact", lambda _: model)
    monkeypatch.setattr(distilbert, "predict_with_transformer", lambda *args, **kwargs: raw)
    monkeypatch.setattr(
        calibration_artifacts,
        "load_calibration_artifact",
        lambda _: SimpleNamespace(scaler=TemperatureScaler(1.0, tuple(range(77)))),
    )
    monkeypatch.setattr(workflows, "load_workflow_specification", lambda _: None)
    monkeypatch.setattr(scenarios, "load_consequence_scenario_set", lambda _: None)
    monkeypatch.setattr(
        scenarios, "build_cost_matrix", lambda *args: np.ones((77, 77)) - np.eye(77)
    )

    class Witness:
        classes_ = np.arange(77)

        def decision_function(self, messages: list[str]) -> np.ndarray:
            return np.log(p)

    class Gate:
        classes_ = np.array([0, 1])

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            return np.tile([0.1, 0.9], (len(features), 1))

    monkeypatch.setattr(
        joblib,
        "load",
        lambda _: {"witness": Witness(), "semantic_gate": Gate(), "signed_gate": Gate()},
    )
    monkeypatch.setitem(globals_, "preflight", lambda: config)

    def bootstrap(*args: object, **kwargs: object) -> list[RoutingCostInterval]:
        assert kwargs["n_resamples"] == 10000
        assert kwargs["seed"] == 42
        return [RoutingCostInterval(0.8, 2464, 0.0, 0.0, 0.0)]

    monkeypatch.setitem(globals_, "bootstrap_routing_cost", bootstrap)
    original_evaluate = globals_["evaluate_final_decisions"]

    def evaluate(*args: object, **kwargs: object) -> pd.DataFrame:
        assert (Path(config["output_directory"]) / "decisions.csv").is_file()
        assert (Path(config["output_directory"]) / "scores.csv").is_file()
        return cast(pd.DataFrame, original_evaluate(*args, **kwargs))

    monkeypatch.setitem(globals_, "evaluate_final_decisions", evaluate)
    namespace["execute_final"](config, "cpu", "synthetic test only")
    output = Path(config["output_directory"])
    assert json.loads((output / "run_state.json").read_text())["status"] == "completed"
    primary = json.loads((output / "primary_result.json").read_text())
    assert primary["reference"] == "signed_gate"
    assert primary["interpretation"] == "inconclusive_not_evidence_of_equivalence"
    comparison = pd.read_csv(output / "comparison.csv")
    assert (
        len(comparison) == 165
    )  # Nine policies plus two matched controls, five budgets, three costs.
    assert comparison.accepted_total_cost.eq(0).all()
    overlap = json.loads((output / "overlap_summary.json").read_text())
    assert overlap["overlaps_training_pool"] == 3080
    assert overlap["primary_rows_excluded"] == 0
    summary = json.loads((output / "summary.json").read_text())
    for filename, expected in summary["output_hashes"].items():
        assert calculate_sha256(output / filename) == expected


def test_failed_run_is_retained_and_cannot_be_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    monkeypatch.chdir(tmp_path)
    config = {"output_directory": "failed-run", "protocol_id": "synthetic"}
    with pytest.raises(FileNotFoundError):
        namespace["execute_final"](config, "cpu", "synthetic failure")
    state = json.loads(Path("failed-run/run_state.json").read_text())
    assert state["status"] == "failed"
    assert state["test_access_started"] is False
    with pytest.raises(FileExistsError):
        namespace["execute_final"](config, "cpu", "synthetic retry")
