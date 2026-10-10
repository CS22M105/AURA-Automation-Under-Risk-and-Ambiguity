"""Expected cost of accepting the classifier's highest-probability intent."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aura.models.predictions import PredictionBatch, create_prediction_batch


class ExpectedRiskError(ValueError):
    """Raised when predictions and routing costs cannot be aligned safely."""


@dataclass(frozen=True)
class ExpectedRoutingRisk:
    """Per-message costs and contributions in the supplied probability-column order."""

    row_indices: tuple[int, ...]
    classes: tuple[int, ...]
    predicted_labels: NDArray[np.int64]
    expected_costs: NDArray[np.float64]
    contributions: NDArray[np.float64]


def compute_expected_routing_risk(
    predictions: PredictionBatch,
    cost_matrix: NDArray[np.float64],
    cost_classes: Sequence[int],
) -> ExpectedRoutingRisk:
    """Compute sum_i p(i|x) C[i,j] for the unchanged argmax prediction j.

    Both matrix axes follow cost_classes: rows are true intents, columns are
    predicted routes. Callers supply calibrated probabilities when available.
    No ground-truth labels or deferral threshold are used.
    """
    labels = tuple(cost_classes)
    if any(
        isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer))
        for label in labels
    ):
        raise ExpectedRiskError("Cost classes must be integer labels")
    if not labels or len(set(labels)) != len(labels):
        raise ExpectedRiskError("Cost classes must be non-empty and unique")
    if set(labels) != set(predictions.classes):
        raise ExpectedRiskError("Cost classes must match prediction classes exactly")

    # Revalidate because frozen dataclasses can still contain mutable NumPy arrays.
    validated = create_prediction_batch(
        predictions.row_indices, predictions.classes, predictions.probabilities
    )
    if not np.array_equal(validated.predicted_labels, predictions.predicted_labels):
        raise ExpectedRiskError("Predicted labels must match probability argmax")

    costs = np.asarray(cost_matrix, dtype=np.float64)
    if costs.shape != (len(labels), len(labels)):
        raise ExpectedRiskError("Cost matrix dimensions must match cost classes")
    if not np.isfinite(costs).all() or np.any(costs < 0):
        raise ExpectedRiskError("Costs must be finite and non-negative")
    if np.any(np.diag(costs) != 0):
        raise ExpectedRiskError("Correct-route costs must be zero")

    positions = {label: position for position, label in enumerate(labels)}
    true_positions = [positions[label] for label in validated.classes]
    route_positions = [positions[int(label)] for label in validated.predicted_labels]
    # Select one cost column per message, retaining probability-column order.
    selected_costs = costs[np.ix_(true_positions, route_positions)].T
    contributions = validated.probabilities * selected_costs
    expected_costs = contributions.sum(axis=1)
    if not np.isfinite(expected_costs).all():
        raise ExpectedRiskError("Expected costs must be finite")

    return ExpectedRoutingRisk(
        row_indices=validated.row_indices,
        classes=validated.classes,
        predicted_labels=validated.predicted_labels.copy(),
        expected_costs=expected_costs,
        contributions=contributions,
    )
