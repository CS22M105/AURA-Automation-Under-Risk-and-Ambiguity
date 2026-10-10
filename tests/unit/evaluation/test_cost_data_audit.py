import numpy as np
import pytest

from aura.evaluation.cost_data_audit import summarize_cost_support


def test_counts_include_zero_support_cost_levels_and_intents() -> None:
    matrix = np.array([[0.0, 2.0, 5.0], [1.0, 0.0, 3.0], [5.0, 2.0, 0.0]])
    result = summarize_cost_support([0, 0, 1, 1], [2, 2, 1, 0], matrix, [0, 1, 2])
    assert result["messages"] == 4
    assert result["errors"] == 3
    assert result["maximum_cost_messages"] == 2
    assert result["true_intents_without_errors"] == 1
    assert result["distinct_error_pairs"] == 2
    assert result["singleton_error_pairs"] == 1
    assert result["scenario_total_cost"] == 11
    assert result["cost_levels"] == [
        {"cost": 0.0, "messages": 1},
        {"cost": 1.0, "messages": 1},
        {"cost": 2.0, "messages": 0},
        {"cost": 3.0, "messages": 0},
        {"cost": 5.0, "messages": 2},
    ]


def test_all_correct_is_handled_without_inventing_error_support() -> None:
    result = summarize_cost_support([0, 1], [0, 1], np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])
    assert result["distinct_error_pairs"] == 0
    assert result["maximum_cost_messages"] == 0
    assert result["true_intents_without_errors"] == 2


def test_cost_matrix_respects_class_order() -> None:
    result = summarize_cost_support([10], [20], np.array([[0.0, 5.0], [2.0, 0.0]]), [20, 10])
    assert result["scenario_total_cost"] == 2
    assert result["maximum_cost_messages"] == 0


@pytest.mark.parametrize("truth,predicted", [([], []), ([0], [2]), ([0], [0, 1])])
def test_rejects_invalid_labels(truth: list[int], predicted: list[int]) -> None:
    with pytest.raises(ValueError):
        summarize_cost_support(truth, predicted, np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])


@pytest.mark.parametrize(
    "matrix", [np.ones((2, 2)), np.zeros((2, 2)), np.array([[0.0, np.nan], [1.0, 0.0]])]
)
def test_rejects_invalid_cost_matrices(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError):
        summarize_cost_support([0], [1], matrix, [0, 1])
