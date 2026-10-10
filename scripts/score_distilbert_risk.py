"""Score calibrated DistilBERT routing risk on the policy-validation partition."""

import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from aura.calibration.artifacts import load_calibration_artifact
from aura.data.intents import BANKING77_LABELS
from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.models.distilbert import predict_with_transformer
from aura.models.predictions import create_prediction_batch
from aura.models.transformer_artifacts import (
    calculate_transformer_artifact_sha256,
    load_transformer_artifact,
)
from aura.risk.expected_risk import compute_expected_routing_risk
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def build_parser() -> ArgumentParser:
    """Create arguments for a reproducible development scoring run."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/splits/development_seed_42.json")
    )
    parser.add_argument("--model-directory", type=Path, default=Path("artifacts/models/distilbert"))
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/calibration/distilbert_temperature.json"),
    )
    parser.add_argument("--workflows", type=Path, default=Path("config/intent_workflows.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("config/consequence_scenarios.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/risk/distilbert"))
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    """Save row-level scores, reusable probabilities, and an input-hashed summary."""
    args = build_parser().parse_args()
    frame = load_banking77_csv(args.dataset)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(args.manifest)
    validate_manifest_source(manifest, args.dataset, len(frame))
    calibration = load_calibration_artifact(args.calibration)
    specification = load_workflow_specification(args.workflows)
    scenarios = load_consequence_scenario_set(args.scenarios)
    matrices = {
        name: build_cost_matrix(specification, scenarios, name) for name in scenarios.scenarios
    }
    model_hash = calculate_transformer_artifact_sha256(args.model_directory)
    manifest_hash = calculate_sha256(args.manifest)
    for key, expected in {
        "dataset_sha256": manifest.source_sha256,
        "manifest_sha256": manifest_hash,
        "model_artifact_sha256": model_hash,
    }.items():
        if calibration.metadata.get(key) != expected:
            raise ValueError(f"Calibration provenance mismatch: {key}")

    artifact = load_transformer_artifact(args.model_directory)
    config = artifact.metadata.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Transformer artifact configuration is missing")
    evaluation = frame.loc[list(manifest.policy_validation)]
    print(f"Scoring policy_validation: {len(evaluation)} messages", flush=True)
    raw = predict_with_transformer(
        artifact.model,
        artifact.tokenizer,
        evaluation["text"].tolist(),
        evaluation.index.tolist(),
        maximum_length=int(config["maximum_length"]),
        batch_size=int(config["batch_size"]),
        requested_device=args.device,
    )
    if raw.classes != calibration.scaler.classes or set(raw.classes) != set(BANKING77_LABELS):
        raise ValueError("Transformer, calibration, and BANKING77 classes must match")
    if raw.decision_scores is None:
        raise ValueError("Temperature scaling requires logits")
    calibrated = create_prediction_batch(
        raw.row_indices, raw.classes, calibration.scaler.transform(raw.decision_scores)
    )
    if not np.array_equal(calibrated.predicted_labels, raw.predicted_labels):
        raise ValueError("Temperature scaling unexpectedly changed predicted intents")

    # Ground truth is copied for later evaluation; it is never passed to the risk scorer.
    rows = evaluation[["text", "label", "label_text"]].copy()
    rows.index.name = "row_id"
    rows = rows.rename(columns={"label": "true_label", "label_text": "true_intent"})
    rows["predicted_label"] = calibrated.predicted_labels
    rows["predicted_intent"] = [
        BANKING77_LABELS[int(label)] for label in calibrated.predicted_labels
    ]
    confidence = calibrated.probabilities.max(axis=1)
    rows["confidence"] = confidence
    summary: dict[str, object] = {}
    for name, matrix in matrices.items():
        risk = compute_expected_routing_risk(calibrated, matrix, range(77))
        values = risk.expected_costs
        rows[f"{name}_risk"] = values
        # The largest contribution identifies which alternative drives the score.
        drivers = np.argmax(risk.contributions, axis=1)
        rows[f"{name}_driver_intent"] = [
            BANKING77_LABELS[calibrated.classes[int(column)]] for column in drivers
        ]
        rows[f"{name}_driver_contribution"] = risk.contributions[np.arange(len(rows)), drivers]
        summary[name] = {
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values.max()),
        }
    flat_unit = scenarios.scenarios["flat"].base_incorrect_cost
    flat_error = float(np.max(np.abs(rows["flat_risk"].to_numpy() - flat_unit * (1 - confidence))))
    if flat_error > 1e-6:
        raise ValueError("Flat risk does not equal base cost times one minus confidence")

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / "policy_validation_scores.csv"
    probabilities_path = output / "policy_validation_probabilities.npz"
    rows.to_csv(scores_path)
    np.savez_compressed(
        probabilities_path,
        row_indices=np.asarray(calibrated.row_indices),
        classes=np.asarray(calibrated.classes),
        probabilities=calibrated.probabilities,
    )
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "sample_count": len(rows),
        "temperature": calibration.scaler.temperature,
        "workflow_specification_id": specification.specification_id,
        "scenario_set_id": scenarios.scenario_set_id,
        "unit": scenarios.unit,
        "requested_device": args.device,
        "flat_identity_max_error": flat_error,
        "scenario_summaries": summary,
        "provenance": {
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": manifest_hash,
            "model_artifact_sha256": model_hash,
            "calibration_sha256": calculate_sha256(args.calibration),
            "workflows_sha256": calculate_sha256(args.workflows),
            "scenarios_sha256": calculate_sha256(args.scenarios),
            "scores_sha256": calculate_sha256(scores_path),
            "probabilities_sha256": calculate_sha256(probabilities_path),
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Flat identity maximum error: {flat_error:.3g}")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
