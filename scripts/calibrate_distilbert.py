"""Fit temperature scaling for the frozen DistilBERT classifier."""

from argparse import ArgumentParser
from pathlib import Path

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
from aura.models.distilbert import predict_with_transformer
from aura.models.transformer_artifacts import (
    calculate_transformer_artifact_sha256,
    load_transformer_artifact,
)


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Calibrate DistilBERT with temperature scaling.")
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
        default=Path("artifacts/calibration/distilbert_temperature.json"),
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    """Fit and save temperature scaling using only the calibration partition."""
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

    calibration_frame = frame.loc[list(manifest.calibration)]
    predictions = predict_with_transformer(
        artifact.model,
        artifact.tokenizer,
        calibration_frame["text"].tolist(),
        calibration_frame.index.tolist(),
        maximum_length=int(config["maximum_length"]),
        batch_size=int(config["batch_size"]),
        requested_device=device,
    )
    if predictions.decision_scores is None:
        raise ValueError("Temperature scaling requires transformer logits")

    true_labels = calibration_frame["label"].tolist()
    scaler = fit_temperature_scaler(
        predictions.decision_scores,
        true_labels,
        predictions.classes,
    )
    calibrated_probabilities = scaler.transform(predictions.decision_scores)
    raw_metrics = compute_calibration_metrics(
        true_labels,
        predictions.probabilities,
        predictions.classes,
    )
    calibrated_metrics = compute_calibration_metrics(
        true_labels,
        calibrated_probabilities,
        predictions.classes,
    )
    save_calibration_artifact(
        output_path,
        scaler,
        {
            "model_type": artifact.metadata.get("model_type"),
            "fitted_partition": "calibration",
            "fitting_rows": len(calibration_frame),
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": manifest_hash,
            "model_artifact_sha256": calculate_transformer_artifact_sha256(model_directory),
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
