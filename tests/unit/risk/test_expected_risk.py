from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aura.models.predictions import PredictionValidationError, create_prediction_batch
from aura.risk.expected_risk import ExpectedRiskError, compute_expected_routing_risk
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def test_directed_cost_uses_predicted_column_and_explains_each_contribution() -> None:
    predictions = create_prediction_batch([101, 205], [1, 22], np.array([[0.8, 0.2], [0.2, 0.8]]))
    result = compute_expected_routing_risk(predictions, np.array([[0.0, 2.0], [5.0, 0.0]]), [1, 22])

    np.testing.assert_allclose(result.expected_costs, [1.0, 0.4])
    np.testing.assert_allclose(result.contributions, [[0.0, 1.0], [0.4, 0.0]])
    np.testing.assert_array_equal(result.predicted_labels, [1, 22])
    assert result.row_indices == (101, 205)
    assert result.classes == (1, 22)


def test_reordered_probability_columns_keep_risk_and_labels_aligned() -> None:
    predictions = create_prediction_batch([7], [22, 1], np.array([[0.2, 0.8]]))
    result = compute_expected_routing_risk(predictions, np.array([[0.0, 2.0], [5.0, 0.0]]), [1, 22])
    np.testing.assert_allclose(result.expected_costs, [1.0])
    np.testing.assert_allclose(result.contributions, [[1.0, 0.0]])


def test_real_scenario_matrices_support_flat_identity_and_different_risks() -> None:
    root = Path(__file__).parents[3]
    specification = load_workflow_specification(root / "config/intent_workflows.yaml")
    scenarios = load_consequence_scenario_set(root / "config/consequence_scenarios.yaml")
    probabilities = np.zeros((2, 77))
    probabilities[0, [1, 22]] = [0.8, 0.2]
    probabilities[1, [1, 22]] = [0.2, 0.8]
    predictions = create_prediction_batch([9, 10], range(77), probabilities)

    for name, expected in (
        ("flat", [0.2, 0.2]),
        ("moderate", [1.0, 0.4]),
        ("high_protection", [2.0, 0.6]),
    ):
        matrix = build_cost_matrix(specification, scenarios, name)
        result = compute_expected_routing_risk(predictions, matrix, range(77))
        np.testing.assert_allclose(result.expected_costs, expected)
        np.testing.assert_allclose(result.contributions.sum(axis=1), result.expected_costs)
        np.testing.assert_array_equal(result.predicted_labels, predictions.predicted_labels)


@pytest.mark.parametrize(
    "costs",
    [
        np.ones((2, 3)),
        np.array([[0.0, -1.0], [1.0, 0.0]]),
        np.array([[0.0, np.nan], [1.0, 0.0]]),
        np.array([[0.0, np.inf], [1.0, 0.0]]),
        np.array([[1.0, 1.0], [1.0, 0.0]]),
    ],
)
def test_invalid_cost_matrices_are_rejected(costs: np.ndarray) -> None:
    predictions = create_prediction_batch([0], [1, 22], np.array([[0.8, 0.2]]))
    with pytest.raises(ExpectedRiskError):
        compute_expected_routing_risk(predictions, costs, [1, 22])


@pytest.mark.parametrize("labels", [[1, 1], [1, 23], [1], [True, 22]])
def test_invalid_cost_class_alignment_is_rejected(labels: list[int]) -> None:
    predictions = create_prediction_batch([0], [1, 22], np.array([[0.8, 0.2]]))
    with pytest.raises(ExpectedRiskError):
        compute_expected_routing_risk(predictions, np.array([[0.0, 2.0], [5.0, 0.0]]), labels)


def test_mutated_probabilities_and_inconsistent_prediction_are_rejected() -> None:
    predictions = create_prediction_batch([0], [1, 22], np.array([[0.8, 0.2]]))
    costs = np.array([[0.0, 2.0], [5.0, 0.0]])
    with pytest.raises(ExpectedRiskError, match="argmax"):
        compute_expected_routing_risk(
            replace(predictions, predicted_labels=np.array([22])), costs, [1, 22]
        )
    predictions.probabilities[0, 0] = np.nan
    with pytest.raises(PredictionValidationError):
        compute_expected_routing_risk(predictions, costs, [1, 22])
