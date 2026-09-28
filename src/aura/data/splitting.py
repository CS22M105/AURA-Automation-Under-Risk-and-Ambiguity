"""Deterministic development splits for BANKING77."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from aura.data.validation import (
    DatasetValidationError,
    validate_banking77_frame,
)

DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class DevelopmentSplits:
    """Training, calibration, and policy-validation partitions."""

    model_training: pd.DataFrame
    calibration: pd.DataFrame
    policy_validation: pd.DataFrame


def create_development_splits(
    frame: pd.DataFrame,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DevelopmentSplits:
    """Create stratified 70/15/15 partitions from training data."""
    validate_banking77_frame(frame)

    if not frame.index.is_unique:
        raise DatasetValidationError("Dataset index must be unique before splitting")

    model_training, holdout = train_test_split(
        frame,
        test_size=0.30,
        random_state=random_state,
        shuffle=True,
        stratify=frame["label"],
    )

    calibration, policy_validation = train_test_split(
        holdout,
        test_size=0.50,
        random_state=random_state,
        shuffle=True,
        stratify=holdout["label"],
    )

    return DevelopmentSplits(
        model_training=model_training.sort_index(),
        calibration=calibration.sort_index(),
        policy_validation=policy_validation.sort_index(),
    )
