"""A fitted multi-budget mixture is bounded and must retain the reference option."""

import numpy as np
import pytest

from aura.policies.coverage_robust import CoverageBlend, select_coverage_blend


def test_convex_score_is_fixed_and_has_no_labels_or_coverage_input() -> None:
    mixture = CoverageBlend((0.25, 0.75))
    np.testing.assert_allclose(mixture.score(np.array([[1.0, 3.0], [2.0, 0.0]])), [2.5, 0.5])


def test_selects_weakly_dominating_mixture_on_calibration_folds() -> None:
    baseline = np.array([1.0, 0.0, 1.0, 0.0])
    evidence = 1 - baseline
    chosen, grid = select_coverage_blend(
        np.column_stack([baseline, evidence]),
        np.array([0.0, 5.0, 0.0, 1.0]),
        [0, 1, 2, 3],
        [0, 0, 1, 1],
        coverages=[0.5],
    )
    assert chosen.weights[1] >= 0.5
    assert grid.worst_excess.min() == -6
    assert len(grid) == 11


def test_reference_is_conservative_tie_breaker() -> None:
    baseline = np.array([0.0, 1.0, 0.0, 1.0])
    scores = np.column_stack([baseline, 1 - baseline])
    selected, grid = select_coverage_blend(
        scores, np.array([0.0, 5.0, 0.0, 1.0]), [0, 1, 2, 3], [0, 0, 1, 1], coverages=[0.5]
    )
    assert selected.weights == (1.0, 0.0)
    assert grid.worst_excess.min() == 0


def test_three_component_grid_and_no_training_regression_property() -> None:
    scores = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    chosen, grid = select_coverage_blend(
        scores, np.array([0.0, 5.0, 0.0, 1.0]), [0, 1, 2, 3], [0, 0, 1, 1], coverages=[0.5, 1.0]
    )
    assert len(grid) == 66
    assert sum(chosen.weights) == pytest.approx(1)
    assert grid.worst_excess.min() <= 0


@pytest.mark.parametrize("weights", [(-0.1, 1.1), (0.2, 0.2), (float("nan"), 1.0), (1.0,)])
def test_rejects_invalid_mixture(weights: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="convex"):
        CoverageBlend(weights).score(np.ones((2, len(weights))))


def test_rejects_missing_fold_boundary() -> None:
    with pytest.raises(ValueError, match="folds"):
        select_coverage_blend(np.ones((4, 2)), np.ones(4), [0, 1, 2, 3], [0, 0, 0, 0])


def test_rejects_reused_row_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_coverage_blend(np.ones((4, 2)), np.ones(4), [0, 0, 2, 3], [0, 0, 1, 1])
