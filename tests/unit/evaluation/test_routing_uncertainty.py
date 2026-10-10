"""Checks for paired, reranked, equal-budget uncertainty estimates."""

import numpy as np
import pytest
from scipy.stats import bootstrap

from aura.evaluation.routing_uncertainty import bootstrap_routing_cost


def test_point_estimate_sign_budget_and_full_coverage() -> None:
    results = bootstrap_routing_cost(
        np.array([0.1, 0.2, 0.3]),
        np.array([0.3, 0.2, 0.1]),
        np.array([5.0, 1.0, 0.0]),
        [0, 1, 2],
        [0.0, 0.7, 1.0],
        n_resamples=100,
    )
    assert results[1].accepted_count == 2
    assert results[1].difference == pytest.approx(-5 / 3)
    for result in (results[0], results[2]):
        assert result.difference == result.lower == result.upper == 0


def test_identical_policies_have_zero_paired_uncertainty() -> None:
    scores = np.array([0.1, 0.1, 0.3])
    results = bootstrap_routing_cost(
        scores, scores, np.array([5.0, 1.0, 0.0]), [2, 0, 1], [0.5, 1.0], n_resamples=100
    )
    assert all(r.difference == r.lower == r.upper == 0 for r in results)


def test_matches_independent_reranking_oracle_with_duplicate_samples() -> None:
    reference = np.array([0.1, 0.1, 0.3])
    candidate = np.array([0.3, 0.1, 0.1])
    costs = np.array([5.0, 2.0, 0.0])
    ids = [20, 10, 30]

    def oracle(indices: np.ndarray) -> float:
        selected_reference = sorted(indices.tolist(), key=lambda i: (reference[i], ids[i]))[:2]
        selected_candidate = sorted(indices.tolist(), key=lambda i: (candidate[i], ids[i]))[:2]
        return float((sum(costs[selected_candidate]) - sum(costs[selected_reference])) / 3)

    expected = bootstrap(
        (np.arange(3),),
        oracle,
        vectorized=False,
        paired=True,
        method="percentile",
        n_resamples=200,
        batch=100,
        rng=np.random.default_rng(7),
    )
    actual = bootstrap_routing_cost(
        reference, candidate, costs, ids, [0.7], n_resamples=200, seed=7
    )[0]
    assert actual.lower == expected.confidence_interval.low
    assert actual.upper == expected.confidence_interval.high
    assert actual.difference == oracle(np.arange(3))


def test_reproducible_seed() -> None:
    args = (np.array([0.1, 0.2]), np.array([0.2, 0.1]), np.array([5.0, 0.0]), [1, 2], [0.5])
    assert bootstrap_routing_cost(*args, n_resamples=100) == bootstrap_routing_cost(
        *args, n_resamples=100
    )


@pytest.mark.parametrize("costs", [[-1.0, 0.0], [np.nan, 0.0], [np.inf, 0.0], [1.0]])
def test_rejects_invalid_costs(costs: list[float]) -> None:
    with pytest.raises(ValueError, match="Costs"):
        bootstrap_routing_cost(
            np.array([0.1, 0.2]), np.array([0.2, 0.1]), np.array(costs), [1, 2], [0.5]
        )


@pytest.mark.parametrize("levels", [[], [-0.1], [1.1], [float("nan")]])
def test_rejects_invalid_coverage(levels: list[float]) -> None:
    with pytest.raises(ValueError):
        bootstrap_routing_cost(
            np.array([0.1, 0.2]), np.array([0.2, 0.1]), np.array([1.0, 0.0]), [1, 2], levels
        )


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        bootstrap_routing_cost(
            np.array([0.1, 0.2]), np.array([0.2, 0.1]), np.array([1.0, 0.0]), [1, 1], [0.5]
        )


@pytest.mark.parametrize("count", [0, 1, -1, True])
def test_rejects_invalid_resample_count(count: int) -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_routing_cost(
            np.array([0.1, 0.2]),
            np.array([0.2, 0.1]),
            np.array([1.0, 0.0]),
            [1, 2],
            [0.5],
            n_resamples=count,
        )


@pytest.mark.parametrize("level", [0.0, 1.0, float("nan")])
def test_rejects_invalid_confidence_level(level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        bootstrap_routing_cost(
            np.array([0.1, 0.2]),
            np.array([0.2, 0.1]),
            np.array([1.0, 0.0]),
            [1, 2],
            [0.5],
            confidence_level=level,
        )
