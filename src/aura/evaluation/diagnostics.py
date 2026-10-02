"""Per-intent and confusion diagnostics for classifier comparison."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from aura.evaluation.classification import ClassificationEvaluationError


@dataclass(frozen=True)
class PerIntentMetrics:
    """Classification metrics for one intent."""

    label: int
    intent: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, int | float | str]:
        """Return CSV- and JSON-serializable values."""
        return asdict(self)


@dataclass(frozen=True)
class ConfusionPair:
    """One directed true-intent to predicted-intent confusion."""

    true_label: int
    true_intent: str
    predicted_label: int
    predicted_intent: str
    count: int
    true_intent_support: int
    rate_within_true_intent: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return CSV- and JSON-serializable values."""
        return asdict(self)


@dataclass(frozen=True)
class ClassificationDiagnostics:
    """Detailed per-intent metrics and confusion information."""

    per_intent: tuple[PerIntentMetrics, ...]
    confusion_matrix: NDArray[np.int64]
    top_confusions: tuple[ConfusionPair, ...]


def compute_classification_diagnostics(
    y_true: Sequence[int],
    y_predicted: Sequence[int],
    classes: Sequence[int],
    intent_names: Mapping[int, str],
    top_confusion_count: int = 25,
) -> ClassificationDiagnostics:
    """Compute per-intent metrics and the largest directed confusions."""
    true_labels = np.asarray(y_true, dtype=np.int64)
    predicted_labels = np.asarray(y_predicted, dtype=np.int64)
    class_labels = np.asarray(classes, dtype=np.int64)

    if true_labels.ndim != 1 or len(true_labels) == 0:
        raise ClassificationEvaluationError("True labels must be a non-empty vector")
    if predicted_labels.shape != true_labels.shape:
        raise ClassificationEvaluationError("Predicted labels must match true-label dimensions")
    if top_confusion_count < 1:
        raise ClassificationEvaluationError("Top confusion count must be positive")
    if not set(true_labels).issubset(set(class_labels)):
        raise ClassificationEvaluationError("True labels contain an unknown class")
    if not set(predicted_labels).issubset(set(class_labels)):
        raise ClassificationEvaluationError("Predictions contain an unknown class")

    missing_names = set(int(label) for label in class_labels) - set(intent_names)
    if missing_names:
        raise ClassificationEvaluationError(
            f"Intent names are missing for labels: {sorted(missing_names)[:5]}"
        )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=class_labels,
        zero_division=0,
    )
    matrix = np.asarray(
        confusion_matrix(true_labels, predicted_labels, labels=class_labels),
        dtype=np.int64,
    )

    per_intent = tuple(
        PerIntentMetrics(
            label=int(label),
            intent=intent_names[int(label)],
            precision=float(precision[position]),
            recall=float(recall[position]),
            f1=float(f1[position]),
            support=int(support[position]),
        )
        for position, label in enumerate(class_labels)
    )

    confusion_pairs: list[ConfusionPair] = []
    for true_position, true_label in enumerate(class_labels):
        true_support = int(matrix[true_position].sum())
        for predicted_position, predicted_label in enumerate(class_labels):
            if true_position == predicted_position:
                continue
            count = int(matrix[true_position, predicted_position])
            if count == 0:
                continue
            confusion_pairs.append(
                ConfusionPair(
                    true_label=int(true_label),
                    true_intent=intent_names[int(true_label)],
                    predicted_label=int(predicted_label),
                    predicted_intent=intent_names[int(predicted_label)],
                    count=count,
                    true_intent_support=true_support,
                    rate_within_true_intent=count / true_support,
                )
            )

    confusion_pairs.sort(
        key=lambda pair: (
            -pair.count,
            -pair.rate_within_true_intent,
            pair.true_label,
            pair.predicted_label,
        )
    )
    return ClassificationDiagnostics(
        per_intent=per_intent,
        confusion_matrix=matrix,
        top_confusions=tuple(confusion_pairs[:top_confusion_count]),
    )
