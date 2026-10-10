"""Descriptive support counts for a prospective learned-cost selector."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def summarize_cost_support(
    truth: Sequence[int],
    predicted: Sequence[int],
    costs: NDArray[np.float64],
    classes: Sequence[int],
) -> dict[str, object]:
    """Count cost levels and sparse error support without asserting sample adequacy."""
    y, route = np.asarray(truth), np.asarray(predicted)
    labels = tuple(classes)
    if (
        not labels
        or len(set(labels)) != len(labels)
        or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in labels
        )
    ):
        raise ValueError("Classes must be unique integer labels")
    if (
        y.ndim != 1
        or y.size == 0
        or route.shape != y.shape
        or y.dtype.kind not in "iu"
        or route.dtype.kind not in "iu"
        or not set(y).issubset(labels)
        or not set(route).issubset(labels)
    ):
        raise ValueError("Labels must be nonempty aligned integer vectors in the class set")
    matrix = np.asarray(costs, dtype=np.float64)
    if (
        matrix.shape != (len(labels), len(labels))
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or np.any(np.diag(matrix) != 0)
    ):
        raise ValueError("Costs must be finite, nonnegative, square, with a zero diagonal")
    positions = {label: i for i, label in enumerate(labels)}
    realized = matrix[[positions[int(i)] for i in y], [positions[int(i)] for i in route]]
    errors = y != route
    max_cost = float(matrix.max())
    if max_cost <= 0:
        raise ValueError("At least one positive scenario cost is required")
    highest = realized == max_cost
    pairs, counts = np.unique(
        np.column_stack((y[errors], route[errors])), axis=0, return_counts=True
    )
    return {
        "messages": len(y),
        "errors": int(errors.sum()),
        "correct": int((~errors).sum()),
        "scenario_total_cost": float(realized.sum()),
        "cost_levels": [
            {"cost": float(level), "messages": int(np.sum(realized == level))}
            for level in np.unique(matrix)
        ],
        "maximum_scenario_cost": max_cost,
        "maximum_cost_messages": int(highest.sum()),
        "true_intents_with_errors": len(np.unique(y[errors])),
        "true_intents_without_errors": len(labels) - len(np.unique(y[errors])),
        "true_intents_with_maximum_cost": len(np.unique(y[highest])),
        "distinct_error_pairs": len(pairs),
        "singleton_error_pairs": int(np.sum(counts == 1)),
        "per_intent": [
            {
                "true_label": int(label),
                "messages": int(np.sum(y == label)),
                "errors": int(np.sum(errors & (y == label))),
                "maximum_cost_messages": int(np.sum(highest & (y == label))),
            }
            for label in labels
        ],
    }
