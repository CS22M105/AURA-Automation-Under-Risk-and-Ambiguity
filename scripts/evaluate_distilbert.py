"""Evaluate DistilBERT on the locked policy-validation partition."""

import json
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.evaluation.calibration import compute_calibration_metrics
from aura.evaluation.classification import compute_classification_metrics
from aura.models.distilbert import predict_with_transformer
from aura.models.transformer_artifacts import load_transformer_artifact


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Evaluate DistilBERT on BANKING77.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=Path("artifacts/models/distilbert"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/metrics/distilbert_policy_validation.json"),
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    """Evaluate raw DistilBERT probabilities and logits."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    model_directory: Path = args.model_directory
    output_path: Path = args.output
    device: str = args.device

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    artifact = load_transformer_artifact(model_directory)
    manifest_hash = calculate_sha256(manifest_path)
    if artifact.metadata.get("dataset_sha256") != manifest.source_sha256:
        raise ValueError("Transformer artifact was trained from a different dataset")
    if artifact.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Transformer artifact was trained from a different split manifest")

    config = artifact.metadata.get("configuration")
    if not isinstance(config, dict):
        raise ValueError("Transformer artifact configuration is missing")
    maximum_length = int(config["maximum_length"])
    batch_size = int(config["batch_size"])
    evaluation_frame = frame.loc[list(manifest.policy_validation)]
    started_at = perf_counter()
    predictions = predict_with_transformer(
        artifact.model,
        artifact.tokenizer,
        evaluation_frame["text"].tolist(),
        evaluation_frame.index.tolist(),
        maximum_length,
        batch_size,
        requested_device=device,
    )
    prediction_seconds = perf_counter() - started_at
    true_labels = evaluation_frame["label"].tolist()
    classification = compute_classification_metrics(
        true_labels,
        predictions.probabilities,
        predictions.classes,
    )
    calibration = compute_calibration_metrics(
        true_labels,
        predictions.probabilities,
        predictions.classes,
    )
    artifact_size = sum(
        path.stat().st_size for path in model_directory.rglob("*") if path.is_file()
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "model": {
            "type": artifact.metadata.get("model_type"),
            "configuration": config,
            "training_rows": artifact.metadata.get("training_rows"),
            "training_seconds": artifact.metadata.get("training_seconds"),
            "parameter_count": artifact.metadata.get("parameter_count"),
            "artifact_size_bytes": artifact_size,
        },
        "classification": classification.to_dict(),
        "calibration": calibration.to_dict(),
        "runtime": {
            "prediction_seconds": prediction_seconds,
            "milliseconds_per_message": 1000 * prediction_seconds / len(evaluation_frame),
            "device": device,
        },
        "provenance": {
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": manifest_hash,
            "model_metadata_sha256": calculate_sha256(model_directory / "aura_metadata.json"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Partition: policy_validation ({classification.sample_count} rows)")
    print(f"Accuracy: {classification.accuracy:.4f}")
    print(f"Macro-F1: {classification.macro_f1:.4f}")
    print(f"Top-3 accuracy: {classification.top_3_accuracy:.4f}")
    print(f"Log loss: {classification.log_loss:.4f}")
    print(f"ECE: {calibration.expected_calibration_error:.4f}")
    print(f"Prediction seconds: {prediction_seconds:.4f}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
