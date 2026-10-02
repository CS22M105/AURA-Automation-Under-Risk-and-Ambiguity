"""Train and save the reproducible AURA baseline classifier."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.models.artifacts import save_model_artifact
from aura.models.baseline import BaselineConfig
from aura.models.training import train_baseline


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Train the TF-IDF Logistic Regression baseline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/train.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/tfidf_logistic_regression.joblib"),
    )
    return parser


def main() -> None:
    """Train the baseline on the manifest's model-training partition."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    output_path: Path = args.output

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))

    config = BaselineConfig()
    started_at = perf_counter()
    pipeline = train_baseline(frame, manifest.model_training, config)
    training_seconds = perf_counter() - started_at
    vectorizer = pipeline.named_steps["tfidf"]
    feature_count = len(vectorizer.get_feature_names_out())
    save_model_artifact(
        output_path,
        pipeline,
        {
            "model_type": "tfidf_logistic_regression",
            "training_rows": len(manifest.model_training),
            "training_seconds": training_seconds,
            "feature_count": feature_count,
            "configuration": asdict(config),
            "random_state": manifest.random_state,
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": calculate_sha256(manifest_path),
        },
    )

    classifier = pipeline.named_steps["classifier"]
    print(f"Artifact: {output_path}")
    print(f"Training rows: {len(manifest.model_training)}")
    print(f"Classes learned: {len(classifier.classes_)}")
    print(f"Features: {feature_count}")
    print(f"Training seconds: {training_seconds:.4f}")
    print("Baseline training: PASSED")


if __name__ == "__main__":
    main()
