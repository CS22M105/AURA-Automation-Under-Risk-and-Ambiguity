"""Joint paired reranking draws for a finite exploratory routing comparison menu."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import bootstrap

from aura.policies.coverage import select_at_coverage


def bootstrap_policy_contrasts(
    scores: Mapping[str, NDArray[np.float64]],
    costs: NDArray[np.float64],
    row_ids: Sequence[int],
    contrasts: Sequence[tuple[str, str]],
    *,
    coverage: float = 0.8,
    resamples: int = 10_000,
    family_size: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute ordinary and Bonferroni percentile intervals from shared SciPy draws.

    Conditions on the supplied fitted scores. Multiplicity adjustment cannot turn
    a repeatedly inspected validation set into independent confirmatory evidence.
    """
    if not contrasts or family_size < len(contrasts) or resamples < 2:
        raise ValueError("Need contrasts, a sufficient family size, and at least two draws")
    names = list(dict.fromkeys(name for pair in contrasts for name in pair))
    ids, loss = np.asarray(row_ids), np.asarray(costs, dtype=np.float64)
    if loss.shape != (len(ids),) or not np.isfinite(loss).all() or np.any(loss < 0):
        raise ValueError("Costs must be aligned, finite and nonnegative")
    count = 0
    for name in names:
        count = int(select_at_coverage(scores[name], ids, coverage).sum())
    arrays = [np.asarray(scores[name], dtype=np.float64) for name in names]

    def statistic(indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.asarray(
            [
                loss[indices[np.lexsort((ids[indices], values[indices]))[:count]]].sum() / len(ids)
                for values in arrays
            ],
            dtype=np.float64,
        )

    point = statistic(np.arange(len(ids), dtype=np.int64))
    result = bootstrap(
        (np.arange(len(ids), dtype=np.int64),),
        statistic,
        paired=True,
        vectorized=False,
        method="percentile",
        n_resamples=resamples,
        batch=100,
        rng=np.random.default_rng(seed),
    )
    draws = result.bootstrap_distribution
    positions = {name: i for i, name in enumerate(names)}
    rows = []
    for candidate, reference in contrasts:
        left, right = positions[candidate], positions[reference]
        differences = draws[left] - draws[right]
        for confidence in [0.95, 1 - 0.05 / family_size]:
            alpha = (1 - confidence) / 2
            lower, upper = np.quantile(differences, [alpha, 1 - alpha])
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "target_coverage": coverage,
                    "accepted_count": count,
                    "difference": float(point[left] - point[right]),
                    "lower": float(lower),
                    "upper": float(upper),
                    "confidence_level": confidence,
                    "family_size": family_size,
                    "resamples": resamples,
                }
            )
    return pd.DataFrame(rows)
