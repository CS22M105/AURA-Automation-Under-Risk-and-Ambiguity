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
