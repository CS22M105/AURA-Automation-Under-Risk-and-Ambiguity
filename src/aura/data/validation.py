"""Validation utilities for BANKING77 datasets."""

import pandas as pd

REQUIRED_COLUMNS: frozenset[str] = frozenset({"text", "label", "label_text"})


class DatasetValidationError(ValueError):
    """Raised when a dataset does not satisfy the expected schema."""


def validate_required_columns(frame: pd.DataFrame) -> None:
    """Verify that a dataset contains every required BANKING77 column."""
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)

    if missing_columns:
        formatted_columns = ", ".join(sorted(missing_columns))
        raise DatasetValidationError(f"Dataset is missing required columns: {formatted_columns}")


def validate_text_values(frame: pd.DataFrame) -> None:
    """Reject missing, empty, or whitespace-only customer messages."""
    invalid_text = frame["text"].isna() | (frame["text"].astype("string").str.strip() == "")
    invalid_count = int(invalid_text.sum())

    if invalid_count:
        raise DatasetValidationError(
            f"Dataset contains {invalid_count} missing or blank text value(s)"
        )


def validate_banking77_frame(frame: pd.DataFrame) -> None:
    """Run all required BANKING77 validation checks."""
    validate_required_columns(frame)
    validate_text_values(frame)
