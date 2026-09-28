import numpy as np
import pytest

from aura.evaluation.classification import (
    ClassificationEvaluationError,
    compute_classification_metrics,
)


def test_computes_multiclass_metrics() -> None:
    y_true = [0, 1, 2, 1]
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1, 0.0],
            [0.2, 0.7, 0.1, 0.0],
            [0.1, 0.1, 0.8, 0.0],
            [0.6, 0.3, 0.1, 0.0],
        ]
    )

    metrics = compute_classification_metrics(y_true, probabilities, [0, 1, 2, 3])

    assert metrics.sample_count == 4
    assert metrics.accuracy == pytest.approx(0.75)
    assert metrics.top_3_accuracy == pytest.approx(1.0)
    assert metrics.mean_confidence == pytest.approx(0.725)
    assert metrics.log_loss > 0


def test_rejects_incompatible_probability_shape() -> None:
    with pytest.raises(ClassificationEvaluationError, match="dimensions must match"):
        compute_classification_metrics([0, 1], np.asarray([[0.8, 0.2]]), [0, 1])


def test_rejects_probabilities_that_do_not_sum_to_one() -> None:
    probabilities = np.asarray([[0.8, 0.8], [0.2, 0.2]])

    with pytest.raises(ClassificationEvaluationError, match="sum to one"):
        compute_classification_metrics([0, 1], probabilities, [0, 1])
