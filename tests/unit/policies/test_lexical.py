"""Regression tests for prediction-aligned lexical gating and its cost extension."""

import numpy as np
import pytest

from aura.models.predictions import create_prediction_batch
from aura.policies.lexical import (
    build_lexical_witness,
    cost_aware_gate_score,
    fit_correctness_gate,
    lexical_gate_features,
)
from aura.risk.expected_risk import compute_expected_routing_risk


def test_signed_support_follows_semantic_label_with_permuted_classes() -> None:
    semantic = np.array([[4.0, 1.0, 2.0], [0.0, 5.0, 2.0]])
    lexical = np.array([[1.0, 3.0, 0.0], [2.0, 0.0, 1.0]])
    before = lexical.copy()
    result = lexical_gate_features(semantic, [10, 20, 30], lexical, [20, 10, 30])
    np.testing.assert_allclose(result, [[2.0, 2.0], [3.0, 1.0]])
    np.testing.assert_array_equal(lexical, before)


def test_disagreement_produces_negative_support_not_unsigned_margin() -> None:
    result = lexical_gate_features(np.array([[3.0, 1.0]]), [0, 1], np.array([[1.0, 4.0]]), [0, 1])
    np.testing.assert_allclose(result, [[2.0, -3.0]])


def test_tied_semantic_logits_follow_first_argmax() -> None:
    result = lexical_gate_features(np.array([[1.0, 1.0]]), [1, 0], np.array([[3.0, 1.0]]), [0, 1])
    np.testing.assert_allclose(result, [[0.0, -2.0]])


def test_margin_is_invariant_to_per_row_logit_offsets() -> None:
    logits = np.array([[3.0, 1.0], [2.0, 0.0]])
    lexical = logits.copy()
    np.testing.assert_allclose(
        lexical_gate_features(logits, [0, 1], lexical, [0, 1]),
        lexical_gate_features(logits + np.array([[100.0], [-20.0]]), [0, 1], lexical, [0, 1]),
    )


@pytest.mark.parametrize("classes", [[0, 0], [0, 2], [0]])
def test_rejects_invalid_class_alignment(classes: list[int]) -> None:
    with pytest.raises(ValueError):
        lexical_gate_features(np.ones((2, 2)), [0, 1], np.ones((2, len(classes))), classes)


def test_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        lexical_gate_features(np.array([[np.nan, 0.0]]), [0, 1], np.ones((1, 2)), [0, 1])


def test_cost_extension_recovers_original_when_gate_error_equals_alternative_mass() -> None:
    batch = create_prediction_batch([3, 4], [2, 0, 1], np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]))
    matrix = np.array([[0.0, 3.0, 2.0], [5.0, 0.0, 1.0], [2.0, 4.0, 0.0]])
    expected = compute_expected_routing_risk(batch, matrix, [0, 1, 2]).expected_costs
    actual = cost_aware_gate_score(batch, np.array([0.3, 0.2]), matrix, [0, 1, 2])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(batch.predicted_labels, [0, 2])


def test_flat_unit_cost_returns_learned_error_probability() -> None:
    batch = create_prediction_batch([0], [0, 1, 2], np.array([[0.7, 0.2, 0.1]]))
    result = cost_aware_gate_score(batch, np.array([0.4]), np.ones((3, 3)) - np.eye(3), [0, 1, 2])
    np.testing.assert_allclose(result, [0.4])


@pytest.mark.parametrize("q", [[-0.1], [1.1], [np.nan], [0.1, 0.2]])
def test_rejects_invalid_error_probability(q: list[float]) -> None:
    batch = create_prediction_batch([0], [0, 1], np.array([[0.7, 0.3]]))
    with pytest.raises(ValueError, match="Error probability"):
        cost_aware_gate_score(batch, np.array(q), np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])


def test_rejects_absent_alternative_mass() -> None:
    batch = create_prediction_batch([0], [0, 1], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="mass"):
        cost_aware_gate_score(batch, np.array([0.1]), np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])


def test_correctness_gate_standardizes_only_fitting_data() -> None:
    x = np.array([[0.0, -2.0], [0.1, -1.0], [2.0, 1.0], [3.0, 2.0]])
    gate = fit_correctness_gate(x, [0, 0, 1, 1])
    np.testing.assert_allclose(gate.named_steps["scale"].mean_, x.mean(axis=0))
    before = gate.named_steps["scale"].mean_.copy()
    p = gate.predict_proba(np.array([[20.0, 30.0]]))
    np.testing.assert_allclose(p.sum(axis=1), [1.0])
    np.testing.assert_array_equal(gate.classes_, [0, 1])
    np.testing.assert_array_equal(gate.named_steps["scale"].mean_, before)


@pytest.mark.parametrize("target", [[1, 1], [0, 0], [0], [0, 2]])
def test_gate_requires_both_correctness_classes(target: list[int]) -> None:
    with pytest.raises(ValueError):
        fit_correctness_gate(np.ones((2, 2)), target)


def test_witness_pipeline_can_fit_three_intents() -> None:
    witness = build_lexical_witness()
    witness.fit(
        [
            "card payment",
            "card refund",
            "cash withdrawal",
            "cash deposit",
            "bank transfer",
            "bank recipient",
        ],
        [0, 0, 1, 1, 2, 2],
    )
    scores = witness.decision_function(["card payment"])
    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()
