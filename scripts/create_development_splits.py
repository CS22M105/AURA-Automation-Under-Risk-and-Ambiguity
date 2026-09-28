"""Create reproducible BANKING77 development-split indices."""

import json
from argparse import ArgumentParser
from pathlib import Path

from aura.data.loading import load_banking77_csv
from aura.data.manifests import calculate_sha256
from aura.data.splitting import (
    DEFAULT_RANDOM_STATE,
    create_development_splits,
)
from aura.data.validation import (
    BANKING77_TRAIN_PROFILE,
    validate_dataset_profile,
)


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Create deterministic BANKING77 development splits.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/train.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )
    return parser


def main() -> None:
    """Create and save the development split manifest."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    output_path: Path = args.output
    random_state: int = args.seed

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    splits = create_development_splits(frame, random_state=random_state)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_file": dataset_path.name,
        "source_sha256": calculate_sha256(dataset_path),
        "random_state": random_state,
        "strategy": "stratified_70_15_15",
        "total_rows": len(frame),
        "partitions": {
            "model_training": [int(index) for index in splits.model_training.index],
            "calibration": [int(index) for index in splits.calibration.index],
            "policy_validation": [int(index) for index in splits.policy_validation.index],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest: {output_path}")
    print(f"Model training: {len(splits.model_training)}")
    print(f"Calibration: {len(splits.calibration)}")
    print(f"Policy validation: {len(splits.policy_validation)}")
    print("Split creation: PASSED")


if __name__ == "__main__":
    main()
