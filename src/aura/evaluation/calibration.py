"""Probability-calibration metrics for multiclass classifiers."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import log_loss

from aura.evaluation.classification import ClassificationEvaluationError


@dataclass(frozen=True)
class CalibrationMetrics:
    """Top-label and distribution-level calibration measurements."""

    sample_count: int
    expected_calibration_error: float
    maximum_calibration_error: float
    multiclass_brier_score: float
    log_loss: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable metric values."""
        return asdict(self)


def compute_calibration_metrics(
    y_true: Sequence[int],
    probabilities: NDArray[np.float64],
    classes: Sequence[int],
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Compute fixed-width top-label ECE, MCE, Brier score, and log loss."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    class_labels = np.asarray(classes, dtype=np.int64)
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)

    if n_bins < 2:
        raise ClassificationEvaluationError("Calibration requires at least two bins")
    if true_labels.ndim != 1 or len(true_labels) == 0:
        raise ClassificationEvaluationError("True labels must be a non-empty vector")
    if predicted_probabilities.shape != (len(true_labels), len(class_labels)):
        raise ClassificationEvaluationError(
            "Probability dimensions must match the samples and classifier classes"
        )
    if not np.isfinite(predicted_probabilities).all():
        raise ClassificationEvaluationError("Probabilities must contain only finite values")
    if not np.allclose(predicted_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ClassificationEvaluationError("Each probability row must sum to one")

    class_positions = {int(label): position for position, label in enumerate(class_labels)}
    if not set(true_labels).issubset(class_positions):
        raise ClassificationEvaluationError("True labels contain a class unknown to the classifier")
    target_positions = np.asarray(
        [class_positions[int(label)] for label in true_labels],
        dtype=np.int64,
    )

    predicted_positions = np.argmax(predicted_probabilities, axis=1)
    confidences = np.max(predicted_probabilities, axis=1)
    correctness = (predicted_positions == target_positions).astype(np.float64)
    bin_indices = np.minimum((confidences * n_bins).astype(np.int64), n_bins - 1)

    expected_error = 0.0
    maximum_error = 0.0
    for bin_index in range(n_bins):
        members = bin_indices == bin_index
        if not np.any(members):
            continue
        gap = abs(float(correctness[members].mean() - confidences[members].mean()))
        expected_error += float(members.mean()) * gap
        maximum_error = max(maximum_error, gap)

    one_hot_targets = np.zeros_like(predicted_probabilities)
    one_hot_targets[np.arange(len(target_positions)), target_positions] = 1.0
    brier_score = np.mean(np.sum((predicted_probabilities - one_hot_targets) ** 2, axis=1))

    return CalibrationMetrics(
        sample_count=len(true_labels),
        expected_calibration_error=expected_error,
        maximum_calibration_error=maximum_error,
        multiclass_brier_score=float(brier_score),
        log_loss=float(log_loss(true_labels, predicted_probabilities, labels=class_labels)),
    )
