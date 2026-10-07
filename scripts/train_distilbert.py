"""Fine-tune and save the reproducible AURA DistilBERT classifier."""

import platform
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
import transformers

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.models.distilbert import DistilBertConfig, train_distilbert
from aura.models.transformer_artifacts import save_transformer_artifact


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Fine-tune DistilBERT on BANKING77.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/models/distilbert"),
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    """Train DistilBERT on the locked model-training partition."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    output_directory: Path = args.output_directory
    device: str = args.device

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    training_frame = frame.loc[list(manifest.model_training)]
    intent_names = {
        int(label): str(intent)
        for label, intent in frame[["label", "label_text"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    config = DistilBertConfig(random_state=manifest.random_state)

    started_at = perf_counter()
    result = train_distilbert(
        training_frame["text"].tolist(),
        training_frame["label"].tolist(),
        intent_names,
        config,
        requested_device=device,
    )
    training_seconds = perf_counter() - started_at
    parameter_count = sum(parameter.numel() for parameter in result.model.parameters())
    save_transformer_artifact(
        output_directory,
        result,
        {
            "schema_version": 1,
            "model_type": "distilbert_sequence_classifier",
            "training_rows": len(training_frame),
            "training_seconds": training_seconds,
            "parameter_count": parameter_count,
            "configuration": asdict(config),
            "checkpoint_selection": "final_epoch_fixed_budget",
            "epoch_losses": result.epoch_losses,
            "training_device": result.device,
            "random_state": manifest.random_state,
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": calculate_sha256(manifest_path),
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            },
        },
    )

    print(f"Artifact: {output_directory}")
    print(f"Training rows: {len(training_frame)}")
    print(f"Classes learned: {result.model.config.num_labels}")
    print(f"Parameters: {parameter_count}")
    print(f"Device: {result.device}")
    print(f"Epoch losses: {[round(loss, 4) for loss in result.epoch_losses]}")
    print(f"Training seconds: {training_seconds:.4f}")
    print("DistilBERT training: PASSED")


if __name__ == "__main__":
    main()
