"""Training operations for AURA classifiers."""

from collections.abc import Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from aura.data.validation import DatasetValidationError, validate_banking77_frame
from aura.models.baseline import BaselineConfig, build_baseline_pipeline


def train_baseline(
    frame: pd.DataFrame,
    training_indices: Sequence[int],
    config: BaselineConfig | None = None,
) -> Pipeline:
    """Fit the baseline using only the designated training rows."""
    validate_banking77_frame(frame)

    if not training_indices:
        raise DatasetValidationError("Training indices must not be empty")
    if len(set(training_indices)) != len(training_indices):
        raise DatasetValidationError("Training indices must be unique")

    missing_indices = set(training_indices) - set(frame.index)
    if missing_indices:
        raise DatasetValidationError(
            f"Training indices are absent from the dataset: {sorted(missing_indices)[:5]}"
        )

    training_frame = frame.loc[list(training_indices)]
    pipeline = build_baseline_pipeline(config)
    pipeline.fit(training_frame["text"], training_frame["label"])
    return pipeline
