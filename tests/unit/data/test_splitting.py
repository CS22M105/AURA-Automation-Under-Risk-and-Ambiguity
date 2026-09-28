import pandas as pd
import pytest

from aura.data.splitting import create_development_splits
from aura.data.validation import DatasetValidationError


def create_test_frame() -> pd.DataFrame:
    """Create balanced synthetic data for stratified split tests."""
    labels = [label for label in range(4) for _ in range(10)]

    return pd.DataFrame(
        {
            "text": [f"message-{index}" for index in range(len(labels))],
            "label": labels,
            "label_text": [f"intent_{label}" for label in labels],
        }
    )


def test_creates_expected_split_sizes() -> None:
    splits = create_development_splits(create_test_frame())

    assert len(splits.model_training) == 28
    assert len(splits.calibration) == 6
    assert len(splits.policy_validation) == 6


def test_splits_are_complete_and_non_overlapping() -> None:
    frame = create_test_frame()
    splits = create_development_splits(frame)

    training_indices = set(splits.model_training.index)
    calibration_indices = set(splits.calibration.index)
    policy_indices = set(splits.policy_validation.index)

    assert training_indices.isdisjoint(calibration_indices)
    assert training_indices.isdisjoint(policy_indices)
    assert calibration_indices.isdisjoint(policy_indices)

    combined_indices = training_indices | calibration_indices | policy_indices
    assert combined_indices == set(frame.index)


def test_every_split_contains_every_intent() -> None:
    splits = create_development_splits(create_test_frame())
    expected_labels = {0, 1, 2, 3}

    assert set(splits.model_training["label"]) == expected_labels
    assert set(splits.calibration["label"]) == expected_labels
    assert set(splits.policy_validation["label"]) == expected_labels


def test_splitting_is_deterministic() -> None:
    frame = create_test_frame()

    first = create_development_splits(frame, random_state=42)
    second = create_development_splits(frame, random_state=42)

    assert first.model_training.index.equals(second.model_training.index)
    assert first.calibration.index.equals(second.calibration.index)
    assert first.policy_validation.index.equals(second.policy_validation.index)


def test_rejects_non_unique_source_index() -> None:
    frame = create_test_frame()
    frame.index = [0] * len(frame)

    with pytest.raises(
        DatasetValidationError,
        match="index must be unique",
    ):
        create_development_splits(frame)
