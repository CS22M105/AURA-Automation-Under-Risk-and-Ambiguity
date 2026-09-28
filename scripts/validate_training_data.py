"""Validate and summarize the official BANKING77 training split."""

from argparse import ArgumentParser
from pathlib import Path

from aura.data.loading import load_banking77_csv
from aura.data.validation import (
    BANKING77_TRAIN_PROFILE,
    validate_dataset_profile,
)


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Validate the BANKING77 training split.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        type=Path,
        default=Path("datasets/train.csv"),
        help="Path to the BANKING77 training CSV.",
    )
    return parser


def main() -> None:
    """Validate the training data and print its structural summary."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset_path

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)

    class_counts = frame["label"].value_counts()

    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(frame)}")
    print(f"Intents: {frame['label'].nunique()}")
    print(f"Label range: {frame['label'].min()}-{frame['label'].max()}")
    print(f"Duplicate messages: {frame['text'].duplicated().sum()}")
    print(f"Smallest class: {class_counts.min()}")
    print(f"Largest class: {class_counts.max()}")
    print("Validation: PASSED")


if __name__ == "__main__":
    main()
