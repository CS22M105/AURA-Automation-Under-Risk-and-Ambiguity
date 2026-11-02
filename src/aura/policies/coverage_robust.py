"""Calibration-only minimax selection of one convex score blend across coverages."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aura.policies.coverage import select_at_coverage


@dataclass(frozen=True)
class CoverageBlend:
    """Fixed mixture, not a different policy chosen after observing each budget."""

    weights: tuple[float, ...]

    def score(self, component_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mix aligned nonnegative cost-score proxies without query labels."""
        values = np.asarray(component_scores, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if (
            weights.ndim != 1
            or len(weights) < 2
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)
            or values.ndim != 2
            or values.shape[1] != len(weights)
            or len(values) == 0
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("Expected convex weights and aligned finite nonnegative scores")
        return np.asarray(values @ weights, dtype=np.float64)


def select_coverage_blend(
    out_of_fold_scores: NDArray[np.float64],
    observed_cost: NDArray[np.float64],
    row_ids: Sequence[int],
    fold_ids: Sequence[int],
    coverages: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    *,
    grid_denominator: int = 10,
) -> tuple[CoverageBlend, pd.DataFrame]:
    """Minimize worst excess cost over component 0 on calibration folds only.

    The reference mixture is included, so fitted worst excess cannot exceed zero.
    This property says nothing about performance on new or reused evaluation rows.
    """
    values = np.asarray(out_of_fold_scores, dtype=np.float64)
    costs, folds, ids = (
        np.asarray(observed_cost, dtype=np.float64),
        np.asarray(fold_ids),
        np.asarray(row_ids),
    )
    if (
        values.ndim != 2
        or values.shape[1] not in (2, 3)
        or costs.shape != (len(values),)
        or folds.shape != costs.shape
        or folds.dtype.kind not in "iu"
        or len(np.unique(folds)) < 2
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
        or not coverages
        or len(set(coverages)) != len(coverages)
        or isinstance(grid_denominator, bool)
        or not isinstance(grid_denominator, int)
        or grid_denominator < 1
    ):
        raise ValueError("Expected aligned calibration folds, costs, scores and valid grid")
    dimension = values.shape[1]
    reference = CoverageBlend((1.0,) + (0.0,) * (dimension - 1))
    reference.score(values)
    for coverage in coverages:
        select_at_coverage(values[:, 0], ids, coverage)

    def curve(scores: NDArray[np.float64]) -> NDArray[np.float64]:
        totals = np.zeros(len(coverages), dtype=np.float64)
        for fold in np.unique(folds):
            keep = folds == fold
            for index, coverage in enumerate(coverages):
                selected = select_at_coverage(scores[keep], ids[keep], coverage)
                totals[index] += costs[keep][selected].sum()
        return totals

    baseline = curve(values[:, 0])
    records, choices = [], []
    for numerators in product(range(grid_denominator + 1), repeat=dimension):
        if sum(numerators) != grid_denominator:
            continue
        weights = tuple(value / grid_denominator for value in numerators)
        blend = CoverageBlend(weights)
        totals = curve(blend.score(values))
        excess = totals - baseline
        worst, mean = float(excess.max()), float(excess.mean())
        distance = float(np.abs(np.asarray(weights) - np.asarray(reference.weights)).sum())
        records.append(
            {
                **{f"weight_{i}": weight for i, weight in enumerate(weights)},
                **{
                    f"cost_at_{c:g}": float(total)
                    for c, total in zip(coverages, totals, strict=True)
                },
                **{
                    f"excess_at_{c:g}": float(delta)
                    for c, delta in zip(coverages, excess, strict=True)
                },
                "worst_excess": worst,
                "mean_excess": mean,
                "distance_to_reference": distance,
            }
        )
        choices.append(((worst, mean, distance, weights), blend))
    selected = min(choices, key=lambda item: item[0])[1]
    return selected, pd.DataFrame(records)
