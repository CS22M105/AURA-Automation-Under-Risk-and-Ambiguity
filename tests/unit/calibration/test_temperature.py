import numpy as np
import pytest
from scipy.special import softmax

from aura.calibration.temperature import (
    TemperatureScaler,
    TemperatureScalingError,
    fit_temperature_scaler,
)


def test_fitted_temperature_reduces_log_loss() -> None:
    scores = np.asarray(
        [
            [4.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
            [0.0, 3.0],
        ]
    )
    labels = [0, 1, 1, 0]
    scaler = fit_temperature_scaler(scores, labels, [0, 1])
    raw_probabilities = softmax(scores, axis=1)
    calibrated_probabilities = scaler.transform(scores)
    targets = np.asarray(labels)
    rows = np.arange(len(labels))

    raw_loss = -np.log(raw_probabilities[rows, targets]).mean()
    calibrated_loss = -np.log(calibrated_probabilities[rows, targets]).mean()

    assert scaler.temperature > 1.0
    assert calibrated_loss < raw_loss


def test_transform_preserves_ranking_and_normalizes_probabilities() -> None:
    scores = np.asarray([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    scaler = TemperatureScaler(temperature=0.5, classes=(0, 1, 2))

    probabilities = scaler.transform(scores)

    assert np.array_equal(np.argmax(probabilities, axis=1), np.argmax(scores, axis=1))
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))


def test_rejects_score_columns_that_do_not_match_classes() -> None:
    scaler = TemperatureScaler(temperature=1.0, classes=(0, 1))

    with pytest.raises(TemperatureScalingError, match="columns must match"):
        scaler.transform(np.asarray([[1.0, 2.0, 3.0]]))
