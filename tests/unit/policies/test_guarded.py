import numpy as np
import pytest

from aura.policies.coverage import select_at_coverage
from aura.policies.guarded import select_with_majority_guard


def test_guard_excludes_half_probability_and_fills_from_eligible_by_risk() -> None:
    result = select_with_majority_guard(
        np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.9, 0.8, 0.7]), [0, 1, 2, 3], 0.5
    )
    assert result.tolist() == [False, True, True, False]


def test_shortfall_never_backfills_ineligible_messages() -> None:
    result = select_with_majority_guard(
        np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.5, 0.4]), [0, 1, 2], 1.0
    )
    assert result.tolist() == [True, False, False]


def test_ties_use_row_ids_and_budget_is_floored() -> None:
    result = select_with_majority_guard(np.ones(3), np.ones(3), [2, 0, 1], 0.5)
    assert result.tolist() == [False, True, False]


@pytest.mark.parametrize("coverage,confidence", [(0.0, 0.9), (1.0, 0.1)])
def test_empty_acceptance(coverage: float, confidence: float) -> None:
    assert not select_with_majority_guard(
        np.ones(3), np.full(3, confidence), [0, 1, 2], coverage
    ).any()


@pytest.mark.parametrize("confidence", [[np.nan, 0.9], [-0.1, 0.9], [1.1, 0.9], [0.9]])
def test_rejects_invalid_confidence(confidence: list[float]) -> None:
    with pytest.raises(ValueError, match="Confidence"):
        select_with_majority_guard(np.ones(2), np.array(confidence), [0, 1], 0.5)


def test_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_with_majority_guard(np.ones(2), np.ones(2), [1, 1], 0.5)


def test_rejects_negative_cost() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        select_with_majority_guard(np.array([-1.0, 0.0]), np.ones(2), [0, 1], 0.5)


def test_matches_original_when_all_messages_are_eligible() -> None:
    costs = np.array([2.0, 1.0, 1.0, 0.0])
    ids = [3, 2, 1, 0]
    for coverage in (0.0, 0.5, 0.7, 1.0):
        np.testing.assert_array_equal(
            select_with_majority_guard(costs, np.full(4, 0.9), ids, coverage),
            select_at_coverage(costs, ids, coverage),
        )
