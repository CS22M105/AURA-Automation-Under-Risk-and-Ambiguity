"""Probability reconstruction and class-alignment regression tests."""

import numpy as np
import pytest
from scipy.special import softmax

from aura.evaluation.routing_diagnostics import diagnose_routing_probabilities
from aura.models.predictions import create_prediction_batch


def test_reconstructs_raw_probabilities_and_aligns_permuted_classes() -> None:
    logits = np.array([[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]])
    temperature = 0.7
    classes = [2, 0, 1]
    batch = create_prediction_batch([10, 20], classes, softmax(logits / temperature, axis=1))
    costs = np.array([[0.0, 2.0, 3.0], [5.0, 0.0, 4.0], [1.0, 2.0, 0.0]])
    result = diagnose_routing_probabilities(batch, [1, 2], temperature, costs, [0, 1, 2])
    raw = softmax(logits, axis=1)
    assert result.raw_true_probability_reconstructed.tolist() == pytest.approx(
        [raw[0, 2], raw[1, 0]]
    )
    assert result.observed_cost.tolist() == [5.0, 0.0]
    assert result.true_rank.tolist() == [2, 1]
    assert result.is_error.tolist() == [True, False]
    assert result.true_intent_cost_contribution.iloc[0] == pytest.approx(
        batch.probabilities[0, 2] * 5
    )
    assert result.true_intent_cost_contribution.iloc[1] == 0
    assert result.raw_expected_cost_reconstructed.iloc[0] == pytest.approx(
        raw[0, 0] + raw[0, 2] * 5
    )


def test_unit_temperature_changes_nothing_and_rank_ties_share_rank() -> None:
    batch = create_prediction_batch([0], [0, 1], np.array([[0.5, 0.5]]))
    result = diagnose_routing_probabilities(
        batch, [1], 1.0, np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1]
    )
    assert result.true_probability_change.iloc[0] == 0
    assert result.true_rank.iloc[0] == 1
    assert result.calibrated_nll.iloc[0] == result.raw_nll_reconstructed.iloc[0]


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_temperature(temperature: float) -> None:
    batch = create_prediction_batch([0], [0, 1], np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="Temperature"):
        diagnose_routing_probabilities(
            batch, [0], temperature, np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1]
        )


@pytest.mark.parametrize("truth", [[2], [], [0, 1]])
def test_rejects_invalid_truth(truth: list[int]) -> None:
    batch = create_prediction_batch([0], [0, 1], np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError):
        diagnose_routing_probabilities(
            batch, truth, 1.0, np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1]
        )


def test_refuses_to_invent_probabilities_when_underflow_occurred() -> None:
    batch = create_prediction_batch([0], [0, 1], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="Strictly positive"):
        diagnose_routing_probabilities(batch, [1], 0.7, np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])
