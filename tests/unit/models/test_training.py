import pandas as pd
import pytest

from aura.data.validation import DatasetValidationError
from aura.models.baseline import BaselineConfig
from aura.models.random_forest import RandomForestConfig
from aura.models.training import train_baseline, train_random_forest


def create_training_frame() -> pd.DataFrame:
    """Create valid synthetic intent data for training tests."""
    return pd.DataFrame(
        {
            "text": [
                "cash withdrawal missing",
                "cash withdrawal declined",
                "bank transfer pending",
                "bank transfer declined",
                "card delivery late",
                "card delivery tracking",
            ],
            "label": [0, 0, 1, 1, 2, 2],
            "label_text": [
                "cash_withdrawal",
                "cash_withdrawal",
                "bank_transfer",
                "bank_transfer",
                "card_delivery",
                "card_delivery",
            ],
        }
    )


def test_trains_on_only_designated_indices() -> None:
    frame = create_training_frame()
    config = BaselineConfig(min_document_frequency=1)

    pipeline = train_baseline(frame, [0, 1, 2, 3], config)
    classifier = pipeline.named_steps["classifier"]

    assert classifier.classes_.tolist() == [0, 1]


def test_rejects_missing_training_index() -> None:
    with pytest.raises(DatasetValidationError, match="absent from the dataset"):
        train_baseline(create_training_frame(), [0, 99])


def test_rejects_duplicate_training_indices() -> None:
    with pytest.raises(DatasetValidationError, match="must be unique"):
        train_baseline(create_training_frame(), [0, 0])


def test_trains_random_forest_on_only_designated_indices() -> None:
    frame = create_training_frame()
    config = RandomForestConfig(
        min_document_frequency=1,
        svd_components=2,
        tree_count=10,
        minimum_samples_per_leaf=1,
        parallel_jobs=1,
    )

    pipeline = train_random_forest(frame, [0, 1, 2, 3], config)
    classifier = pipeline.named_steps["classifier"]

    assert classifier.classes_.tolist() == [0, 1]
