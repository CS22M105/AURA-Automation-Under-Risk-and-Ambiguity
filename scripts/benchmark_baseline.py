"""Generate the standardized development benchmark for the baseline model."""

import csv
import json
import platform
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from pathlib import Path

import sklearn

from aura.calibration.artifacts import load_calibration_artifact
from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.evaluation.benchmarking import measure_latency
from aura.evaluation.calibration import compute_calibration_metrics
from aura.evaluation.classification import compute_classification_metrics
from aura.evaluation.diagnostics import compute_classification_diagnostics
from aura.models.artifacts import load_model_artifact
from aura.models.predictions import predict_with_sklearn_pipeline


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Benchmark the TF-IDF Logistic Regression baseline.")
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
        "--output-directory",
        type=Path,
        default=Path("artifacts/benchmarks/tfidf_logistic_regression"),
    )
    parser.add_argument("--latency-runs", type=int, default=20)
    return parser


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write dictionaries to a CSV file with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Create aggregate, per-intent, confusion, and runtime benchmark outputs."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    model_path: Path = args.model
    calibration_path: Path = args.calibration
    output_directory: Path = args.output_directory
    latency_runs: int = args.latency_runs

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    model_artifact = load_model_artifact(model_path)
    calibration_artifact = load_calibration_artifact(calibration_path)

    manifest_hash = calculate_sha256(manifest_path)
    model_hash = calculate_sha256(model_path)
    if model_artifact.metadata.get("dataset_sha256") != manifest.source_sha256:
        raise ValueError("Model artifact was trained from a different dataset")
    if model_artifact.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Model artifact was trained from a different split manifest")
    if calibration_artifact.metadata.get("model_sha256") != model_hash:
        raise ValueError("Calibration artifact belongs to a different model")

    evaluation_frame = frame.loc[list(manifest.policy_validation)]
    messages = evaluation_frame["text"].tolist()
    row_indices = evaluation_frame.index.tolist()
    true_labels = evaluation_frame["label"].tolist()
    predictions = predict_with_sklearn_pipeline(
        model_artifact.pipeline,
        messages,
        row_indices,
    )
    if predictions.decision_scores is None:
        raise ValueError("Baseline benchmark requires classifier decision scores")
    if predictions.classes != calibration_artifact.scaler.classes:
        raise ValueError("Calibration classes do not match the model classes")

    calibrated_probabilities = calibration_artifact.scaler.transform(predictions.decision_scores)
    raw_classification = compute_classification_metrics(
        true_labels,
        predictions.probabilities,
        predictions.classes,
    )
    calibrated_classification = compute_classification_metrics(
        true_labels,
        calibrated_probabilities,
        predictions.classes,
    )
    raw_calibration = compute_calibration_metrics(
        true_labels,
        predictions.probabilities,
        predictions.classes,
    )
    calibrated_calibration = compute_calibration_metrics(
        true_labels,
        calibrated_probabilities,
        predictions.classes,
    )

    intent_names = {
        int(label): str(intent)
        for label, intent in frame[["label", "label_text"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    diagnostics = compute_classification_diagnostics(
        true_labels,
        predictions.predicted_labels.tolist(),
        predictions.classes,
        intent_names,
    )

    def predict_raw() -> object:
        return predict_with_sklearn_pipeline(
            model_artifact.pipeline,
            messages,
            row_indices,
        )

    def predict_calibrated() -> object:
        batch = predict_with_sklearn_pipeline(
            model_artifact.pipeline,
            messages,
            row_indices,
        )
        if batch.decision_scores is None:
            raise ValueError("Baseline benchmark requires classifier decision scores")
        return calibration_artifact.scaler.transform(batch.decision_scores)

    raw_latency = measure_latency(
        predict_raw,
        batch_size=len(evaluation_frame),
        measured_runs=latency_runs,
    )
    calibrated_latency = measure_latency(
        predict_calibrated,
        batch_size=len(evaluation_frame),
        measured_runs=latency_runs,
    )

    vectorizer = model_artifact.pipeline.named_steps["tfidf"]
    measured_feature_count = len(vectorizer.get_feature_names_out())
    recorded_feature_count = model_artifact.metadata.get("feature_count")
    if recorded_feature_count != measured_feature_count:
        raise ValueError("Recorded feature count does not match the fitted vectorizer")

    output_directory.mkdir(parents=True, exist_ok=True)
    per_intent_path = output_directory / "per_intent.csv"
    top_confusions_path = output_directory / "top_confusions.csv"
    confusion_matrix_path = output_directory / "confusion_matrix.csv"
    summary_path = output_directory / "summary.json"

    per_intent_rows = [metric.to_dict() for metric in diagnostics.per_intent]
    write_csv(
        per_intent_path,
        ["label", "intent", "precision", "recall", "f1", "support"],
        per_intent_rows,
    )
    confusion_rows = [pair.to_dict() for pair in diagnostics.top_confusions]
    write_csv(
        top_confusions_path,
        [
            "true_label",
            "true_intent",
            "predicted_label",
            "predicted_intent",
            "count",
            "true_intent_support",
            "rate_within_true_intent",
        ],
        confusion_rows,
    )

    matrix_fields = ["true_label", "true_intent"] + [
        f"predicted_{label}" for label in predictions.classes
    ]
    matrix_rows: list[dict[str, object]] = []
    for position, true_label in enumerate(predictions.classes):
        row: dict[str, object] = {
            "true_label": true_label,
            "true_intent": intent_names[true_label],
        }
        row.update(
            {
                f"predicted_{predicted_label}": int(
                    diagnostics.confusion_matrix[position, predicted_position]
                )
                for predicted_position, predicted_label in enumerate(predictions.classes)
            }
        )
        matrix_rows.append(row)
    write_csv(confusion_matrix_path, matrix_fields, matrix_rows)

    summary: dict[str, object] = {
        "schema_version": 1,
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "model": {
            "type": model_artifact.metadata.get("model_type"),
            "configuration": model_artifact.metadata.get("configuration"),
            "training_rows": model_artifact.metadata.get("training_rows"),
            "training_seconds": model_artifact.metadata.get("training_seconds"),
            "feature_count": measured_feature_count,
            "class_count": len(predictions.classes),
            "artifact_size_bytes": model_path.stat().st_size,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn_version": sklearn.__version__,
            "raw_prediction": raw_latency.to_dict(),
            "calibrated_prediction": calibrated_latency.to_dict(),
        },
        "provenance": {
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": manifest_hash,
            "model_sha256": model_hash,
            "calibration_sha256": calculate_sha256(calibration_path),
        },
        "raw": {
            "classification": raw_classification.to_dict(),
            "calibration": raw_calibration.to_dict(),
        },
        "calibrated": {
            "classification": calibrated_classification.to_dict(),
            "calibration": calibrated_calibration.to_dict(),
        },
        "diagnostics": {
            "top_confusion_count": len(diagnostics.top_confusions),
            "per_intent_file": per_intent_path.name,
            "top_confusions_file": top_confusions_path.name,
            "confusion_matrix_file": confusion_matrix_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Model: {model_artifact.metadata.get('model_type')}")
    print(f"Features: {measured_feature_count}")
    print(f"Artifact size: {model_path.stat().st_size} bytes")
    print(f"Accuracy: {calibrated_classification.accuracy:.4f}")
    print(f"Macro-F1: {calibrated_classification.macro_f1:.4f}")
    print(
        "Median calibrated latency: "
        f"{calibrated_latency.median_batch_milliseconds:.3f} ms per batch"
    )
    print(f"Benchmark: {summary_path}")


if __name__ == "__main__":
    main()
