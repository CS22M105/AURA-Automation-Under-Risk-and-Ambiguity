import numpy as np
import pytest

from aura.models.baseline import BaselineConfig, build_baseline_pipeline
from aura.models.predictions import (
    PredictionValidationError,
    create_prediction_batch,
    predict_with_sklearn_pipeline,
)


def test_generates_aligned_predictions_from_baseline() -> None:
    messages = [
        "cash withdrawal missing",
        "cash withdrawal declined",
        "bank transfer pending",
        "bank transfer declined",
        "card delivery late",
        "card delivery tracking",
    ]
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = build_baseline_pipeline(BaselineConfig(min_document_frequency=1))
    pipeline.fit(messages, labels)

    batch = predict_with_sklearn_pipeline(pipeline, messages, [10, 11, 12, 13, 14, 15])

    assert batch.row_indices == (10, 11, 12, 13, 14, 15)
    assert batch.classes == (0, 1, 2)
    assert batch.sample_count == 6
    assert batch.predicted_labels.shape == (6,)
    assert batch.probabilities.shape == (6, 3)
    assert batch.decision_scores is not None
    assert batch.decision_scores.shape == (6, 3)
    np.testing.assert_allclose(batch.probabilities.sum(axis=1), np.ones(6))


def test_rejects_duplicate_prediction_indices() -> None:
    with pytest.raises(PredictionValidationError, match="indices must be unique"):
        create_prediction_batch(
            row_indices=[1, 1],
            classes=[0, 1],
            probabilities=np.asarray([[0.8, 0.2], [0.3, 0.7]]),
        )


def test_rejects_probability_shape_mismatch() -> None:
    with pytest.raises(PredictionValidationError, match="dimensions must match"):
        create_prediction_batch(
            row_indices=[1, 2],
            classes=[0, 1],
            probabilities=np.asarray([[0.8, 0.2]]),
        )


def test_rejects_messages_without_matching_indices() -> None:
    pipeline = build_baseline_pipeline(BaselineConfig(min_document_frequency=1))
    pipeline.fit(["cash missing", "cash pending", "card late", "card missing"], [0, 0, 1, 1])

    with pytest.raises(PredictionValidationError, match="equal length"):
        predict_with_sklearn_pipeline(pipeline, ["cash missing"], [1, 2])
