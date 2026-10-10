"""Experimental majority-confidence safeguard for offline AURA ranking."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from aura.policies.coverage import select_at_coverage


def select_with_majority_guard(
    expected_cost: NDArray[np.float64],
    confidence: NDArray[np.float64],
    row_ids: Sequence[int] | NDArray[np.int64],
    coverage: float,
) -> NDArray[np.bool_]:
    """Rank eligible messages by cost, accepting up to the requested floor budget.

    Eligibility requires max class probability > 0.5: more mass than all other
    intents combined. This is an experimental constraint, not a safety claim.
    Ineligible messages are never used to fill a coverage shortfall.
    """
    costs = np.asarray(expected_cost, dtype=np.float64)
    ids = np.asarray(row_ids)
    budget = int(select_at_coverage(costs, ids, coverage).sum())
    probabilities = np.asarray(confidence, dtype=np.float64)
    if (
        probabilities.shape != costs.shape
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Confidence must be an aligned finite vector between zero and one")
    if np.any(costs < 0):
        raise ValueError("Expected costs must be nonnegative")
    order = np.lexsort((ids, costs))
    eligible_order = order[probabilities[order] > 0.5]
    accepted = np.zeros(len(costs), dtype=np.bool_)
    accepted[eligible_order[:budget]] = True
    return accepted
