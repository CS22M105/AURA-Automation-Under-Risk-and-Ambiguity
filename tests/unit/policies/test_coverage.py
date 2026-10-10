import numpy as np
import pytest

from aura.policies.coverage import select_at_coverage


def test_accepts_lowest_scores_with_floor_budget() -> None:
    mask = select_at_coverage(np.array([0.8, 0.1, 0.3]), [9, 2, 7], 0.8)
    np.testing.assert_array_equal(mask, [False, True, True])


def test_ties_are_invariant_to_input_order() -> None:
    ids = np.array([4, 1, 3, 2])
    scores = np.ones(4)
    mask = select_at_coverage(scores, ids, 0.5)
    order = np.array([3, 2, 0, 1])
    shuffled = select_at_coverage(scores[order], ids[order], 0.5)
    assert set(ids[mask]) == set(ids[order][shuffled]) == {1, 2}


def test_zero_and_full_coverage() -> None:
    assert not select_at_coverage(np.ones(5), range(5), 0).any()
    assert select_at_coverage(np.ones(5), range(5), 1).all()


def test_decimal_budget_avoids_binary_float_rounding_loss() -> None:
    assert select_at_coverage(np.ones(100), range(100), 0.29).sum() == 29


@pytest.mark.parametrize("coverage", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_coverage_rejected(coverage: float) -> None:
    with pytest.raises(ValueError):
        select_at_coverage(np.ones(2), [1, 2], coverage)


def test_bad_scores_or_ids_rejected() -> None:
    for scores, ids in [
        (np.array([np.nan]), [1]),
        (np.ones(2), [1, 1]),
        (np.ones(2), [1]),
        (np.array([]), []),
    ]:
        with pytest.raises(ValueError):
            select_at_coverage(scores, ids, 0.5)
