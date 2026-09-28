"""Multiclass temperature scaling for classifier decision scores."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


class TemperatureScalingError(ValueError):
    """Raised when temperature scaling inputs or optimization are invalid."""


@dataclass(frozen=True)
class TemperatureScaler:
    """A fitted scalar temperature for multiclass decision scores."""

    temperature: float
    classes: tuple[int, ...]

    def transform(self, decision_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert decision scores into temperature-scaled probabilities."""
        scores = _validate_scores(decision_scores, len(self.classes))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise TemperatureScalingError("Temperature must be finite and positive")
        return np.asarray(softmax(scores / self.temperature, axis=1), dtype=np.float64)


def _validate_scores(
    decision_scores: NDArray[np.float64],
    class_count: int,
) -> NDArray[np.float64]:
    scores = np.asarray(decision_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0:
        raise TemperatureScalingError("Decision scores must be a non-empty matrix")
    if scores.shape[1] != class_count:
        raise TemperatureScalingError("Decision-score columns must match classifier classes")
    if not np.isfinite(scores).all():
        raise TemperatureScalingError("Decision scores must contain only finite values")
    return scores


def fit_temperature_scaler(
    decision_scores: NDArray[np.float64],
    y_true: Sequence[int],
    classes: Sequence[int],
) -> TemperatureScaler:
    """Fit one positive temperature by minimizing multiclass log loss."""
    class_labels = tuple(int(label) for label in classes)
    scores = _validate_scores(decision_scores, len(class_labels))
    true_labels = np.asarray(y_true, dtype=np.int64)

    if true_labels.ndim != 1 or len(true_labels) != scores.shape[0]:
        raise TemperatureScalingError("True labels must match the decision-score rows")
    class_positions = {label: position for position, label in enumerate(class_labels)}
    if not set(true_labels).issubset(class_positions):
        raise TemperatureScalingError("True labels contain a class unknown to the classifier")
    target_positions = np.asarray(
        [class_positions[int(label)] for label in true_labels],
        dtype=np.int64,
    )

    def negative_log_likelihood(log_temperature: float) -> float:
        scaled_scores = scores / np.exp(log_temperature)
        losses = (
            logsumexp(scaled_scores, axis=1)
            - scaled_scores[np.arange(len(target_positions)), target_positions]
        )
        return float(np.mean(losses))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):
        raise TemperatureScalingError("Temperature optimization did not converge")

    return TemperatureScaler(
        temperature=float(np.exp(result.x)),
        classes=class_labels,
    )
