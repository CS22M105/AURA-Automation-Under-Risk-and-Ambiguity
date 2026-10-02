"""Compare raw and calibrated baseline probabilities on development data."""

import json
from argparse import ArgumentParser
from pathlib import Path

from aura.calibration.artifacts import load_calibration_artifact
from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.evaluation.calibration import compute_calibration_metrics
from aura.evaluation.classification import compute_classification_metrics
from aura.models.artifacts import load_model_artifact
from aura.models.predictions import predict_with_sklearn_pipeline


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Evaluate baseline probability calibration.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/tfidf_logistic_regression.joblib"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/calibration/baseline_temperature.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/baseline_calibration_comparison.json"),
    )
    return parser


def main() -> None:
    """Evaluate raw and calibrated probabilities without refitting."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    model_path: Path = args.model
    calibration_path: Path = args.calibration
    output_path: Path = args.output

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    model_artifact = load_model_artifact(model_path)
    calibration_artifact = load_calibration_artifact(calibration_path)

    manifest_hash = calculate_sha256(manifest_path)
    model_hash = calculate_sha256(model_path)
    if calibration_artifact.metadata.get("dataset_sha256") != manifest.source_sha256:
        raise ValueError("Calibration artifact belongs to a different dataset")
    if calibration_artifact.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Calibration artifact belongs to a different split manifest")
    if calibration_artifact.metadata.get("model_sha256") != model_hash:
        raise ValueError("Calibration artifact belongs to a different model")

    evaluation_frame = frame.loc[list(manifest.policy_validation)]
    pipeline = model_artifact.pipeline
    predictions = predict_with_sklearn_pipeline(
        pipeline,
        evaluation_frame["text"].tolist(),
        evaluation_frame.index.tolist(),
    )
    if predictions.classes != calibration_artifact.scaler.classes:
        raise ValueError("Calibration classes do not match the model classes")
    if predictions.decision_scores is None:
        raise ValueError("Temperature scaling requires classifier decision scores")

    classes = predictions.classes
    decision_scores = predictions.decision_scores
    raw_probabilities = predictions.probabilities
    calibrated_probabilities = calibration_artifact.scaler.transform(decision_scores)
    true_labels = evaluation_frame["label"].tolist()

    raw_calibration = compute_calibration_metrics(true_labels, raw_probabilities, classes)
    calibrated_calibration = compute_calibration_metrics(
        true_labels,
        calibrated_probabilities,
        classes,
    )
    raw_classification = compute_classification_metrics(true_labels, raw_probabilities, classes)
    calibrated_classification = compute_classification_metrics(
        true_labels,
        calibrated_probabilities,
        classes,
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "temperature": calibration_artifact.scaler.temperature,
        "raw": {
            "classification": raw_classification.to_dict(),
            "calibration": raw_calibration.to_dict(),
        },
        "calibrated": {
            "classification": calibrated_classification.to_dict(),
            "calibration": calibrated_calibration.to_dict(),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Partition: policy_validation ({len(evaluation_frame)} rows)")
    print(f"Temperature: {calibration_artifact.scaler.temperature:.4f}")
    print(
        "ECE: "
        f"{raw_calibration.expected_calibration_error:.4f} -> "
        f"{calibrated_calibration.expected_calibration_error:.4f}"
    )
    print(f"Log loss: {raw_calibration.log_loss:.4f} -> {calibrated_calibration.log_loss:.4f}")
    print(
        "Brier score: "
        f"{raw_calibration.multiclass_brier_score:.4f} -> "
        f"{calibrated_calibration.multiclass_brier_score:.4f}"
    )
    print(
        f"Accuracy: {raw_classification.accuracy:.4f} -> {calibrated_classification.accuracy:.4f}"
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
