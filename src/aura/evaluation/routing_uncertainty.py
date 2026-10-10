"""Paired bootstrap of offline, equal-coverage routing cost differences."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import bootstrap

from aura.policies.coverage import select_at_coverage


@dataclass(frozen=True)
class RoutingCostInterval:
    """Candidate minus reference accepted cost per input, including deferred inputs."""

    target_coverage: float
    accepted_count: int
    difference: float
    lower: float
    upper: float


def bootstrap_routing_cost(
    reference_scores: NDArray[np.float64],
    candidate_scores: NDArray[np.float64],
    costs: NDArray[np.float64],
    row_ids: Sequence[int] | NDArray[np.int64],
    coverages: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> list[RoutingCostInterval]:
    """Resample messages jointly and rerank both policies in each bootstrap sample.

    Lower scores are preferred. Costs are identical for both policies because
    the classifier is fixed; only acceptance changes. Deferred inputs contribute
    zero accepted-case cost, not an assertion that deferral itself is free.
    Intervals are pointwise percentile intervals conditional on frozen artifacts.
    """
    reference = np.asarray(reference_scores, dtype=np.float64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    observed_costs = np.asarray(costs, dtype=np.float64)
    ids = np.asarray(row_ids)
    levels = tuple(coverages)
    if not levels:
        raise ValueError("At least one coverage is required")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one exclusively")
    # Reuse the production selector's score, ID, coverage, and floor-budget checks.
    budgets = []
    for level in levels:
        budgets.append(int(select_at_coverage(reference, ids, level).sum()))
        select_at_coverage(candidate, ids, level)
    if (
        observed_costs.shape != reference.shape
        or not np.isfinite(observed_costs).all()
        or np.any(observed_costs < 0)
    ):
        raise ValueError("Costs must be an aligned finite nonnegative vector")
    size = len(reference)
    if size < 2:
        raise ValueError("At least two messages are required")

    def statistic(indices: NDArray[np.int64]) -> NDArray[np.float64]:
        # Repeated IDs represent identical resampled messages. Their relative
        # order is immaterial; distinct tied messages retain original ID order.
        reference_order = np.lexsort((ids[indices], reference[indices]))
        candidate_order = np.lexsort((ids[indices], candidate[indices]))
        reference_totals = np.r_[0.0, np.cumsum(observed_costs[indices[reference_order]])]
        candidate_totals = np.r_[0.0, np.cumsum(observed_costs[indices[candidate_order]])]
        return np.asarray(
            (candidate_totals[budgets] - reference_totals[budgets]) / size, dtype=np.float64
        )

    indices = np.arange(size, dtype=np.int64)
    point = statistic(indices)
    result = bootstrap(
        (indices,),
        statistic,
        vectorized=False,
        paired=True,
        n_resamples=n_resamples,
        batch=100,
        confidence_level=confidence_level,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    return [
        RoutingCostInterval(
            target_coverage=float(level),
            accepted_count=budget,
            difference=float(point[i]),
            lower=float(result.confidence_interval.low[i]),
            upper=float(result.confidence_interval.high[i]),
        )
        for i, (level, budget) in enumerate(zip(levels, budgets, strict=True))
    ]
