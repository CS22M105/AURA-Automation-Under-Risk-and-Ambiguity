import numpy as np
import pytest

from aura.evaluation.calibration import compute_calibration_metrics


def test_computes_calibration_metrics() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]])

    metrics = compute_calibration_metrics([0, 1], probabilities, [0, 1], n_bins=10)

    assert metrics.sample_count == 2
    assert metrics.expected_calibration_error == pytest.approx(0.1)
    assert metrics.maximum_calibration_error == pytest.approx(0.1)
    assert metrics.multiclass_brier_score == pytest.approx(0.02)
    assert metrics.log_loss == pytest.approx(-np.log(0.9))


def test_rejects_too_few_bins() -> None:
    with pytest.raises(ValueError, match="at least two bins"):
        compute_calibration_metrics([0], np.asarray([[1.0]]), [0], n_bins=1)
