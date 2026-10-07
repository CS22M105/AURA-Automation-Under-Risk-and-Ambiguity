"""Train and save the reproducible AURA Random Forest comparison model."""

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
from aura.models.random_forest import RandomForestConfig
from aura.models.training import train_random_forest


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Train the TF-IDF SVD Random Forest model.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/tfidf_svd_random_forest.joblib"),
    )
    return parser


def main() -> None:
    """Train the comparison model on the manifest's model-training partition."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    output_path: Path = args.output

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))

    config = RandomForestConfig(random_state=manifest.random_state)
    started_at = perf_counter()
    pipeline = train_random_forest(frame, manifest.model_training, config)
    training_seconds = perf_counter() - started_at

    vectorizer = pipeline.named_steps["tfidf"]
    svd = pipeline.named_steps["svd"]
    classifier = pipeline.named_steps["classifier"]
    tfidf_feature_count = len(vectorizer.get_feature_names_out())
    save_model_artifact(
        output_path,
        pipeline,
        {
            "model_type": "tfidf_svd_random_forest",
            "training_rows": len(manifest.model_training),
            "training_seconds": training_seconds,
            "tfidf_feature_count": tfidf_feature_count,
            "model_feature_count": int(svd.components_.shape[0]),
            "configuration": asdict(config),
            "random_state": manifest.random_state,
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": calculate_sha256(manifest_path),
        },
    )

    print(f"Artifact: {output_path}")
    print(f"Training rows: {len(manifest.model_training)}")
    print(f"Classes learned: {len(classifier.classes_)}")
    print(f"TF-IDF features: {tfidf_feature_count}")
    print(f"SVD features: {svd.components_.shape[0]}")
    print(f"Training seconds: {training_seconds:.4f}")
    print("Random Forest training: PASSED")


if __name__ == "__main__":
    main()
