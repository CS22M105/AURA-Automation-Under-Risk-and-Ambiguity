"""Evaluate the uncalibrated baseline on a development partition."""

import json
from argparse import ArgumentParser
from pathlib import Path

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.evaluation.classification import compute_classification_metrics
from aura.models.artifacts import load_model_artifact


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Evaluate the uncalibrated baseline.")
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
        "--output",
        type=Path,
        default=Path("artifacts/metrics/baseline_policy_validation.json"),
    )
    return parser


def main() -> None:
    """Evaluate and save development metrics for the baseline."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    model_path: Path = args.model
    output_path: Path = args.output

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    artifact = load_model_artifact(model_path)

    expected_manifest_hash = calculate_sha256(manifest_path)
    if artifact.metadata.get("dataset_sha256") != manifest.source_sha256:
        raise ValueError("Model artifact was trained from a different dataset")
    if artifact.metadata.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("Model artifact was trained from a different split manifest")

    evaluation_frame = frame.loc[list(manifest.policy_validation)]
    probabilities = artifact.pipeline.predict_proba(evaluation_frame["text"])
    classifier = artifact.pipeline.named_steps["classifier"]
    metrics = compute_classification_metrics(
        evaluation_frame["label"].tolist(),
        probabilities,
        classifier.classes_,
    )

    report: dict[str, object] = {
        "schema_version": 1,
        "evaluation_stage": "development",
        "model_type": artifact.metadata.get("model_type"),
        "partition": "policy_validation",
        "dataset_sha256": manifest.source_sha256,
        "manifest_sha256": expected_manifest_hash,
        "metrics": metrics.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Partition: policy_validation ({metrics.sample_count} rows)")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Macro-F1: {metrics.macro_f1:.4f}")
    print(f"Weighted-F1: {metrics.weighted_f1:.4f}")
    print(f"Top-3 accuracy: {metrics.top_3_accuracy:.4f}")
    print(f"Log loss: {metrics.log_loss:.4f}")
    print(f"Mean confidence: {metrics.mean_confidence:.4f}")
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
