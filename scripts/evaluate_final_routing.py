"""Explicit, write-once final-test evaluation under the frozen AURA protocol."""

import json
import platform
import runpy
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aura.data.manifests import calculate_sha256
from aura.evaluation.final_routing import (
    FINAL_POLICIES,
    build_final_scores,
    evaluate_final_decisions,
    normalized_overlap_audit,
    primary_interpretation,
    select_final_decisions,
    validate_final_frame,
)
from aura.evaluation.routing_uncertainty import bootstrap_routing_cost

RUNNER_LOCK = Path("reports/protocols/aura-final-runner-v1.json")
PROTOCOL_LOCK = Path("reports/protocols/aura-final-evaluation-v1.json")


def write_json(path: Path, value: object) -> None:
    """Write finite, auditable JSON; do not silently serialize NaN results."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def preflight() -> dict[str, Any]:
    """Check both frozen input locks before touching any test data."""
    runner = json.loads(RUNNER_LOCK.read_text())
    if calculate_sha256(PROTOCOL_LOCK) != runner["protocol_lock_sha256"]:
        raise ValueError("Original protocol lock changed")
    for name, expected in runner["file_sha256"].items():
        if not Path(name).is_file() or calculate_sha256(Path(name)) != expected:
            raise ValueError(f"Frozen runner code changed: {name}")
    runpy.run_path("scripts/check_final_protocol.py", run_name="__main__")
    config: dict[str, Any] = json.loads(Path("config/final_evaluation.json").read_text())
    if (
        config["protocol_id"] != runner["protocol_id"]
        or tuple(config["policies"]) != FINAL_POLICIES
    ):
        raise ValueError("Runner/protocol policy mismatch")
    fitted = json.loads(Path("artifacts/policies/lexical_distilbert/summary.json").read_text())
    if version("scikit-learn") != fitted["sklearn_version"]:
        raise ValueError("Use the fitted gates' scikit-learn version before loading joblib")
    return config


def execute_final(config: dict[str, Any], device: str, exposure_note: str) -> None:
    """Run once; retain failures and refuse to overwrite any prior run directory."""
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": config["protocol_id"],
        "test_access_started": False,
        "test_exposure_note": exposure_note,
    }
    write_json(output / "run_state.json", state)
    try:
        # Heavy imports and trusted deserialization occur only after explicit --run-final.
        import joblib

        from aura.calibration.artifacts import load_calibration_artifact
        from aura.data.loading import load_banking77_csv
        from aura.models.distilbert import (
            predict_with_transformer,
            select_device,
            set_reproducible_seed,
        )
        from aura.models.predictions import create_prediction_batch
        from aura.models.transformer_artifacts import load_transformer_artifact
        from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
        from aura.risk.workflows import load_workflow_specification

        source = json.loads(Path("artifacts/risk/distilbert/summary.json").read_text())
        training_path = Path("datasets/train.csv")
        if calculate_sha256(training_path) != source["provenance"]["dataset_sha256"]:
            raise ValueError("Training pool changed; cannot perform the frozen overlap audit")
        model = load_transformer_artifact(Path("artifacts/models/distilbert"))
        calibration = load_calibration_artifact(
            Path("artifacts/calibration/distilbert_temperature.json")
        )
        gates = joblib.load("artifacts/policies/lexical_distilbert/gates.joblib")
        workflows = load_workflow_specification(Path("config/intent_workflows.yaml"))
        scenarios = load_consequence_scenario_set(Path("config/consequence_scenarios.yaml"))
        matrices = {
            name: build_cost_matrix(workflows, scenarios, name)
            for name in config["evaluation_scenarios"]
        }
        resolved_device = str(select_device(device))
        environment = {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "transformers")
        }
        environment.update(
            {"python": sys.version, "platform": platform.platform(), "device": resolved_device}
        )
        write_json(output / "protocol.json", config)
        write_json(output / "environment.json", environment)
        write_json(output / "protocol_lock.json", json.loads(PROTOCOL_LOCK.read_text()))
        write_json(output / "runner_lock.json", json.loads(RUNNER_LOCK.read_text()))
        state["test_access_started"] = True
        write_json(output / "run_state.json", state)
        test_path = Path(config["dataset"])
        test_hash = calculate_sha256(test_path)
        state["test_sha256"] = test_hash
        write_json(output / "run_state.json", state)
        test = load_banking77_csv(test_path)
        validate_final_frame(test)
        set_reproducible_seed(config["uncertainty"]["seed"])
        model_config = model.metadata["configuration"]
        print(f"Scoring {len(test)} official test messages with frozen models", flush=True)
        raw = predict_with_transformer(
            model.model,
            model.tokenizer,
            test.text.tolist(),
            test.index.tolist(),
            maximum_length=int(model_config["maximum_length"]),
            batch_size=int(model_config["batch_size"]),
            requested_device=resolved_device,
        )
        if (
            raw.decision_scores is None
            or raw.classes != calibration.scaler.classes
            or raw.row_indices != tuple(test.index)
            or set(raw.classes) != set(range(77))
        ):
            raise ValueError("Final classifier/calibration alignment mismatch")
        calibrated = create_prediction_batch(
            raw.row_indices, raw.classes, calibration.scaler.transform(raw.decision_scores)
        )
        scores, features = build_final_scores(
            raw, calibrated, test.text.tolist(), gates, matrices, range(77)
        )
        masks = select_final_decisions(scores, calibrated.row_indices, config["target_coverages"])
        decisions = pd.concat(
            [
                pd.DataFrame(
                    {
                        "row_id": calibrated.row_indices,
                        "target_coverage": coverage,
                        "policy": policy,
                        "accepted": accepted,
                    }
                )
                for (coverage, policy), accepted in masks.items()
            ],
            ignore_index=True,
        )
        # Persist label-free scores and decisions before evaluating any labeled outcome.
        decisions.to_csv(output / "decisions.csv", index=False)
        score_frame = pd.DataFrame(
            {
                "row_id": calibrated.row_indices,
                "predicted_label": calibrated.predicted_labels,
                **scores,
            }
        )
        score_frame["semantic_margin"] = features[:, 0]
        score_frame["signed_lexical_support"] = features[:, 1]
        score_frame.to_csv(output / "scores.csv", index=False)
        np.savez_compressed(
            output / "predictions.npz",
            row_indices=raw.row_indices,
            classes=raw.classes,
            logits=raw.decision_scores,
            raw_probabilities=raw.probabilities,
            calibrated_probabilities=calibrated.probabilities,
        )
        report = evaluate_final_decisions(
            calibrated, test.label.tolist(), masks, matrices, range(77)
        )
        report.to_csv(output / "comparison.csv", index=False)
        predictions = test.copy()
        predictions.index.name = "row_id"
        predictions["predicted_label"] = calibrated.predicted_labels
        predictions.to_csv(output / "labeled_predictions.csv")
        primary = config["primary"]
        candidate, reference = primary["candidate"], primary["reference"]
        realized = matrices[primary["evaluation_scenario"]][
            test.label.to_numpy(), calibrated.predicted_labels
        ]
        uncertainty = config["uncertainty"]
        interval = bootstrap_routing_cost(
            scores[reference],
            scores[candidate],
            realized,
            calibrated.row_indices,
            [primary["target_coverage"]],
            n_resamples=uncertainty["n_resamples"],
            seed=uncertainty["seed"],
            confidence_level=uncertainty["confidence_level"],
        )[0]
        pair = report[
            (report.target_coverage == primary["target_coverage"])
            & (report.evaluation_scenario == primary["evaluation_scenario"])
            & report.policy.isin([candidate, reference])
        ].set_index("policy")
        if len(pair) != 2 or not pair.accepted_count.eq(interval.accepted_count).all():
            raise ValueError("Primary comparison is not matched coverage")
        observed = float(
            pair.loc[candidate, "accepted_cost_per_input"]
            - pair.loc[reference, "accepted_cost_per_input"]
        )
        if not np.isclose(observed, interval.difference, rtol=0, atol=1e-12):
            raise ValueError("Primary bootstrap point estimate differs from comparison")
        result = {
            **primary,
            **asdict(interval),
            "confidence_level": uncertainty["confidence_level"],
            "n_resamples": uncertainty["n_resamples"],
            "seed": uncertainty["seed"],
            "interpretation": primary_interpretation(interval.lower, interval.upper),
        }
        write_json(output / "primary_result.json", result)
        pd.DataFrame([result]).to_csv(output / "primary_interval.csv", index=False)
        training = load_banking77_csv(training_path)
        overlap = normalized_overlap_audit(test.text.tolist(), training.text.tolist())
        overlap.to_csv(output / "overlap_audit.csv", index=False)
        overlap_summary = {
            name: int(overlap[name].sum()) for name in overlap.columns if name != "row_id"
        }
        overlap_summary["primary_rows_excluded"] = 0
        write_json(output / "overlap_summary.json", overlap_summary)
        if calculate_sha256(test_path) != test_hash:
            raise ValueError("Test dataset changed during evaluation")
        if calculate_sha256(training_path) != source["provenance"]["dataset_sha256"]:
            raise ValueError("Training pool changed during overlap audit")
        preflight()
        summary = {
            "protocol_id": config["protocol_id"],
            "evaluation_stage": "final_test",
            "sample_count": len(test),
            "test_sha256": test_hash,
            "protocol_lock_sha256": calculate_sha256(PROTOCOL_LOCK),
            "runner_lock_sha256": calculate_sha256(RUNNER_LOCK),
            "test_exposure_note": exposure_note,
            "primary": result,
            "secondary_intervals": "not computed; secondary tables are descriptive",
            "limitations": [
                "offline ranking, not fixed-threshold deployment",
                "research-scenario costs, not monetary or measured banking harm",
                "bootstrap conditions on fitted models and independent-message assumption",
                "no review costs/errors or uncertainty in cost definitions included",
            ],
            "output_hashes": {
                path.name: calculate_sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "run_state.json"
            },
        }
        write_json(output / "summary.json", summary)
        state.update(status="completed", completed_at_utc=datetime.now(UTC).isoformat())
        write_json(output / "run_state.json", state)
        print(json.dumps(result, indent=2))
        print(f"Final results saved: {output}")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "run_state.json", state)
        raise


def main() -> None:
    """Default to preflight; only explicit --run-final permits reading test data."""
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run-final", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--test-exposure-note", default="Historical test exposure not independently verified"
    )
    args = parser.parse_args()
    config = preflight()
    if not args.run_final:
        print("Runner preflight passed. Use --run-final explicitly to access the official test.")
        return
    if not args.test_exposure_note.strip():
        raise ValueError("Test exposure note must not be blank")
    execute_final(config, args.device, args.test_exposure_note)


if __name__ == "__main__":
    main()
