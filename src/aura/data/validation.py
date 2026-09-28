"""Validation utilities for BANKING77 datasets."""

import pandas as pd
from pandas.api.types import is_integer_dtype

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


def validate_label_values(frame: pd.DataFrame) -> None:
    """Reject missing, blank, or non-integer intent labels."""
    if frame["label"].isna().any():
        raise DatasetValidationError("Dataset contains missing numeric label values")

    if not is_integer_dtype(frame["label"].dtype):
        raise DatasetValidationError("Dataset labels must use integer IDs")

    invalid_label_text = frame["label_text"].isna() | (
        frame["label_text"].astype("string").str.strip() == ""
    )
    invalid_count = int(invalid_label_text.sum())

    if invalid_count:
        raise DatasetValidationError(
            f"Dataset contains {invalid_count} missing or blank label name(s)"
        )


def validate_label_mapping(frame: pd.DataFrame) -> None:
    """Verify a one-to-one mapping between label IDs and intent names."""
    names_per_id = frame.groupby("label")["label_text"].nunique()
    inconsistent_ids = sorted(int(label) for label in names_per_id[names_per_id > 1].index)

    if inconsistent_ids:
        raise DatasetValidationError(
            f"Numeric labels map to multiple intent names: {inconsistent_ids}"
        )

    ids_per_name = frame.groupby("label_text")["label"].nunique()
    inconsistent_names = sorted(str(name) for name in ids_per_name[ids_per_name > 1].index)

    if inconsistent_names:
        raise DatasetValidationError(
            f"Intent names map to multiple numeric labels: {inconsistent_names}"
        )


def validate_banking77_frame(frame: pd.DataFrame) -> None:
    """Run all required BANKING77 validation checks."""
    validate_required_columns(frame)
    validate_text_values(frame)
    validate_label_values(frame)
    validate_label_mapping(frame)
