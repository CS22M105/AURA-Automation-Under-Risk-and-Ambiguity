"""Local-evidence routing identities, alignment checks and numerical stability."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from aura.policies.evidence import (
    EvidenceSelectors,
    fit_evidence_selectors,
    fit_posterior_pool,
    fixed_route_risk,
    retrieval_gaps,
)


def test_retrieval_uses_predicted_intent_and_declared_action() -> None:
    reference = csr_matrix(np.eye(3))
    query = csr_matrix([[1.0, 0.0, 0.0], [0.0, 0.8, 0.6], [0.0, 0.8, 0.6]])
    gaps = retrieval_gaps(
        query,
        reference,
        [0, 1, 2],
        [0, 0, 1],
        ["protect", "information", "information"],
        batch_size=1,
    )
    np.testing.assert_allclose(gaps, [[1.0, 1.0], [-0.8, -0.8], [0.2, 0.8]])


def test_retrieval_is_invariant_to_vector_length() -> None:
    a = retrieval_gaps(
        csr_matrix([[2.0, 0.0, 0.0]]),
        csr_matrix(np.eye(3) * 3),
        [0, 1, 2],
        [0],
        ["one", "two", "two"],
    )
    np.testing.assert_allclose(a, [[1.0, 1.0]])


def test_rejects_reference_missing_an_intent() -> None:
    with pytest.raises(ValueError, match="references"):
        retrieval_gaps(csr_matrix([[1.0, 0.0]]), csr_matrix(np.eye(2)), [0, 0], [0], ["one", "two"])


def test_flat_unit_cost_recovers_error_score() -> None:
    x = np.array([[0.0, -2.0], [0.1, -1.0], [2.0, 1.0], [3.0, 2.0]])
    errors = [1, 1, 0, 0]
    model = fit_evidence_selectors(x, errors, np.array(errors, dtype=float))
    scores = model.scores(x)
    np.testing.assert_allclose(np.exp(scores["log_cost"]), scores["error"])
    np.testing.assert_array_equal(model.error_gate.classes_, [0, 1])
    np.testing.assert_allclose(model.error_gate["scale"].mean_, model.cost_gate["scale"].mean_)


def test_log_risk_remains_finite_for_extreme_odds() -> None:
    error, cost = Mock(), Mock()
    error.decision_function.return_value = np.array([100_000.0, -100_000.0])
    cost.decision_function.return_value = np.array([100_000.0, -100_000.0])
    error.predict_proba.return_value = np.array([[0.0, 1.0], [1.0, 0.0]])
    scores = EvidenceSelectors(error, cost).scores(np.zeros((2, 2)))
    np.testing.assert_allclose(scores["log_cost"], [0.0, -100_000.0])


@pytest.mark.parametrize(
    "errors,costs",
    [([0, 0], [0.0, 0.0]), ([0, 1], [1.0, 0.0]), ([0, 1], [0.0, np.nan]), ([0, 2], [0.0, 2.0])],
)
def test_rejects_invalid_error_cost_targets(errors: list[int], costs: list[float]) -> None:
    with pytest.raises(ValueError):
        fit_evidence_selectors(np.ones((2, 2)), errors, np.array(costs))


def test_pooled_risk_does_not_replace_base_prediction() -> None:
    # Pool favors class 1, while the retained classifier route is class 0.
    probability = np.array([[0.2, 0.7, 0.1]])
    matrix = np.array([[0.0, 1.0, 2.0], [5.0, 0.0, 3.0], [2.0, 1.0, 0.0]])
    np.testing.assert_allclose(fixed_route_risk(probability, [0], matrix), [3.7])


def test_pool_is_convex_and_fitted_without_evaluation_labels() -> None:
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
    lexical = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    pool = fit_posterior_pool(logits, lexical, [0, 1, 2, 1])
    p = pool.probabilities(logits, lexical)
    assert 0 <= pool.lexical_weight <= 1
    np.testing.assert_allclose(p.sum(axis=1), 1)
    with pytest.raises(ValueError, match="weight"):
        replace(pool, lexical_weight=2).probabilities(logits, lexical)


def test_rejects_nonmatrix_pool_logits() -> None:
    with pytest.raises(ValueError, match="matrices"):
        fit_posterior_pool(np.ones(3), np.ones(3), [0])
