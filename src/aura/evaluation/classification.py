"""Classification metrics used in AURA experiments."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, f1_score, log_loss, top_k_accuracy_score


class ClassificationEvaluationError(ValueError):
    """Raised when classifier outputs cannot be evaluated safely."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate metrics for a multiclass classifier."""

    sample_count: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    top_3_accuracy: float
    log_loss: float
    mean_confidence: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable metric values."""
        return asdict(self)


def compute_classification_metrics(
    y_true: Sequence[int],
    probabilities: NDArray[np.float64],
    classes: Sequence[int],
) -> ClassificationMetrics:
    """Compute aggregate multiclass metrics from class probabilities."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    class_labels = np.asarray(classes, dtype=np.int64)
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)

    if true_labels.ndim != 1 or len(true_labels) == 0:
        raise ClassificationEvaluationError("True labels must be a non-empty vector")
    if predicted_probabilities.ndim != 2:
        raise ClassificationEvaluationError("Probabilities must be a two-dimensional matrix")
    if predicted_probabilities.shape != (len(true_labels), len(class_labels)):
        raise ClassificationEvaluationError(
            "Probability dimensions must match the samples and classifier classes"
        )
    if not np.isfinite(predicted_probabilities).all():
        raise ClassificationEvaluationError("Probabilities must contain only finite values")
    if not np.allclose(predicted_probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ClassificationEvaluationError("Each probability row must sum to one")
    if not set(true_labels).issubset(set(class_labels)):
        raise ClassificationEvaluationError("True labels contain a class unknown to the classifier")

    predicted_labels = class_labels[np.argmax(predicted_probabilities, axis=1)]
    top_k = min(3, len(class_labels))

    return ClassificationMetrics(
        sample_count=len(true_labels),
        accuracy=float(accuracy_score(true_labels, predicted_labels)),
        macro_f1=float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        weighted_f1=float(
            f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        ),
        top_3_accuracy=float(
            top_k_accuracy_score(
                true_labels,
                predicted_probabilities,
                k=top_k,
                labels=class_labels,
            )
        ),
        log_loss=float(log_loss(true_labels, predicted_probabilities, labels=class_labels)),
        mean_confidence=float(np.max(predicted_probabilities, axis=1).mean()),
    )
