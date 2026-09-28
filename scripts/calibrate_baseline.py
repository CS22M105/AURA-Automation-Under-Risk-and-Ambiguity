"""Fit temperature scaling for the trained baseline classifier."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from aura.calibration.artifacts import save_calibration_artifact
from aura.calibration.temperature import fit_temperature_scaler
from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.evaluation.calibration import compute_calibration_metrics
from aura.models.artifacts import load_model_artifact


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Calibrate the baseline with temperature scaling.")
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
        default=Path("artifacts/calibration/baseline_temperature.json"),
    )
    return parser


def main() -> None:
    """Fit and save the baseline temperature scaler."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    model_path: Path = args.model
    output_path: Path = args.output

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    model_artifact = load_model_artifact(model_path)

    manifest_hash = calculate_sha256(manifest_path)
    if model_artifact.metadata.get("dataset_sha256") != manifest.source_sha256:
        raise ValueError("Model artifact was trained from a different dataset")
    if model_artifact.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Model artifact was trained from a different split manifest")

    calibration_frame = frame.loc[list(manifest.calibration)]
    pipeline = model_artifact.pipeline
    classifier = pipeline.named_steps["classifier"]
    classes = classifier.classes_
    decision_scores = np.asarray(
        pipeline.decision_function(calibration_frame["text"]),
        dtype=np.float64,
    )
    raw_probabilities = np.asarray(
        pipeline.predict_proba(calibration_frame["text"]),
        dtype=np.float64,
    )
    true_labels = calibration_frame["label"].tolist()

    scaler = fit_temperature_scaler(decision_scores, true_labels, classes)
    calibrated_probabilities = scaler.transform(decision_scores)
    raw_metrics = compute_calibration_metrics(true_labels, raw_probabilities, classes)
    calibrated_metrics = compute_calibration_metrics(
        true_labels,
        calibrated_probabilities,
        classes,
    )

    save_calibration_artifact(
        output_path,
        scaler,
        {
            "fitted_partition": "calibration",
            "fitting_rows": len(calibration_frame),
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": manifest_hash,
            "model_sha256": calculate_sha256(model_path),
            "fit_partition_raw_metrics": raw_metrics.to_dict(),
            "fit_partition_calibrated_metrics": calibrated_metrics.to_dict(),
        },
    )

    print(f"Calibration rows: {len(calibration_frame)}")
    print(f"Temperature: {scaler.temperature:.4f}")
    print(
        f"Fit-partition log loss: {raw_metrics.log_loss:.4f} -> {calibrated_metrics.log_loss:.4f}"
    )
    print(
        "Fit-partition ECE: "
        f"{raw_metrics.expected_calibration_error:.4f} -> "
        f"{calibrated_metrics.expected_calibration_error:.4f}"
    )
    print(f"Artifact: {output_path}")


if __name__ == "__main__":
    main()
