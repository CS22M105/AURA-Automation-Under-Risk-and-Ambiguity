"""Validated prediction outputs shared by AURA classifier families."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline


class PredictionValidationError(ValueError):
    """Raised when classifier prediction outputs are inconsistent."""


@dataclass(frozen=True)
class PredictionBatch:
    """Aligned predictions and scores for a fixed set of dataset rows."""

    row_indices: tuple[int, ...]
    classes: tuple[int, ...]
    predicted_labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    decision_scores: NDArray[np.float64] | None

    @property
    def sample_count(self) -> int:
        """Return the number of predicted examples."""
        return len(self.row_indices)


def create_prediction_batch(
    row_indices: Sequence[int],
    classes: Sequence[int],
    probabilities: NDArray[np.float64],
    decision_scores: NDArray[np.float64] | None = None,
) -> PredictionBatch:
    """Validate model outputs and create an aligned prediction batch."""
    resolved_indices = tuple(int(index) for index in row_indices)
    resolved_classes = tuple(int(label) for label in classes)
    resolved_probabilities = np.asarray(probabilities, dtype=np.float64)

    if not resolved_indices:
        raise PredictionValidationError("Prediction rows must not be empty")
    if len(set(resolved_indices)) != len(resolved_indices):
        raise PredictionValidationError("Prediction row indices must be unique")
    if not resolved_classes or len(set(resolved_classes)) != len(resolved_classes):
        raise PredictionValidationError("Classifier classes must be non-empty and unique")
    if resolved_probabilities.shape != (len(resolved_indices), len(resolved_classes)):
        raise PredictionValidationError(
            "Probability dimensions must match prediction rows and classifier classes"
        )
    if not np.isfinite(resolved_probabilities).all():
        raise PredictionValidationError("Probabilities must contain only finite values")
    if np.any(resolved_probabilities < 0.0) or np.any(resolved_probabilities > 1.0):
        raise PredictionValidationError("Probabilities must be between zero and one")
    if not np.allclose(resolved_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise PredictionValidationError("Each probability row must sum to one")

    resolved_scores: NDArray[np.float64] | None = None
    if decision_scores is not None:
        resolved_scores = np.asarray(decision_scores, dtype=np.float64)
        if resolved_scores.ndim == 1:
            resolved_scores = resolved_scores.reshape(-1, 1)
        if resolved_scores.ndim != 2 or resolved_scores.shape[0] != len(resolved_indices):
            raise PredictionValidationError("Decision scores must align with prediction rows")
        if not np.isfinite(resolved_scores).all():
            raise PredictionValidationError("Decision scores must contain only finite values")

    class_array = np.asarray(resolved_classes, dtype=np.int64)
    predicted_labels = class_array[np.argmax(resolved_probabilities, axis=1)]

    return PredictionBatch(
        row_indices=resolved_indices,
        classes=resolved_classes,
        predicted_labels=predicted_labels,
        probabilities=resolved_probabilities,
        decision_scores=resolved_scores,
    )


def predict_with_sklearn_pipeline(
    pipeline: Pipeline,
    messages: Sequence[str],
    row_indices: Sequence[int],
) -> PredictionBatch:
    """Generate the common prediction contract from a fitted sklearn pipeline."""
    resolved_messages = list(messages)
    if len(resolved_messages) != len(row_indices):
        raise PredictionValidationError("Messages and row indices must have equal length")

    probabilities = np.asarray(pipeline.predict_proba(resolved_messages), dtype=np.float64)
    classes = pipeline.classes_
    decision_scores: NDArray[np.float64] | None = None
    if hasattr(pipeline, "decision_function"):
        decision_scores = np.asarray(
            pipeline.decision_function(resolved_messages),
            dtype=np.float64,
        )

    return create_prediction_batch(
        row_indices=row_indices,
        classes=classes,
        probabilities=probabilities,
        decision_scores=decision_scores,
    )
