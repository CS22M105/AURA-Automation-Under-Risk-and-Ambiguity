"""Exploratory local-evidence and cost-tilted routing, preserving base labels."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from aura.calibration.temperature import TemperatureScaler, fit_temperature_scaler


def retrieval_gaps(
    query_features: sparse.spmatrix,
    reference_features: sparse.spmatrix,
    reference_labels: Sequence[int],
    predictions: Sequence[int],
    actions: Sequence[str],
    *,
    batch_size: int = 128,
) -> NDArray[np.float64]:
    """Compare nearest intent/action supporters with their nearest competitors."""
    reference = normalize(reference_features, norm="l2", copy=True)
    query = normalize(query_features, norm="l2", copy=True)
    labels, predicted = np.asarray(reference_labels), np.asarray(predictions)
    action_names = np.asarray(actions)
    if (
        labels.shape != (reference.shape[0],)
        or predicted.shape != (query.shape[0],)
        or labels.dtype.kind not in "iu"
        or predicted.dtype.kind not in "iu"
        or len(action_names) < 2
        or set(labels) != set(range(len(action_names)))
        or not set(predicted).issubset(labels)
        or query.shape[1] != reference.shape[1]
        or not np.isfinite(reference.data).all()
        or not np.isfinite(query.data).all()
        or len(set(actions)) < 2
        or batch_size < 1
    ):
        raise ValueError("Expected aligned finite features, all intent references and actions")
    gaps = np.empty((len(predicted), 2), dtype=np.float64)
    reference_actions = action_names[labels]
    for start in range(0, len(predicted), batch_size):
        stop = min(start + batch_size, len(predicted))
        similarity = (query[start:stop] @ reference.T).toarray()
        for offset, route in enumerate(predicted[start:stop]):
            same_intent = labels == route
            same_action = reference_actions == action_names[route]
            scores = similarity[offset]
            gaps[start + offset] = [
                scores[same_intent].max() - scores[~same_intent].max(),
                scores[same_action].max() - scores[~same_action].max(),
            ]
    return gaps


def _fit_error_gate(
    features: NDArray[np.float64],
    errors: NDArray[np.int64],
    weights: NDArray[np.float64] | None = None,
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    )
    if weights is None:
        model.fit(features, errors)
    else:
        model.fit(features, errors, classifier__sample_weight=weights / weights.mean())
    return model


@dataclass(frozen=True)
class EvidenceSelectors:
    """Matched error and cost-tilted gates; log-risk is a ranking proxy only."""

    error_gate: Pipeline
    cost_gate: Pipeline

    def scores(self, features: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Return error probability and stable log cost proxy, without query labels."""
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Expected a finite feature matrix")
        error_log_odds = np.asarray(self.error_gate.decision_function(values), dtype=np.float64)
        cost_log_odds = np.asarray(self.cost_gate.decision_function(values), dtype=np.float64)
        return {
            "error": np.asarray(self.error_gate.predict_proba(values)[:, 1], dtype=np.float64),
            "log_cost": cost_log_odds - np.logaddexp(0.0, error_log_odds),
        }


def fit_evidence_selectors(
    features: NDArray[np.float64], errors: Sequence[int], realized_cost: NDArray[np.float64]
) -> EvidenceSelectors:
    """Fit identical-capacity gates; positive weights come only from declared costs."""
    values = np.asarray(features, dtype=np.float64)
    target, costs = np.asarray(errors), np.asarray(realized_cost, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] not in (2, 4)
        or not np.isfinite(values).all()
        or target.shape != (len(values),)
        or target.dtype.kind not in "biu"
        or set(target) != {0, 1}
        or costs.shape != target.shape
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
        or not np.array_equal(costs > 0, target == 1)
    ):
        raise ValueError("Expected finite 2/4 features and aligned error/cost targets")
    return EvidenceSelectors(
        error_gate=_fit_error_gate(values, target),
        cost_gate=_fit_error_gate(values, target, np.where(target == 1, costs, 1.0)),
    )


@dataclass(frozen=True)
class PosteriorPool:
    """Calibration-likelihood-fitted convex pool, not a replacement base classifier."""

    semantic_temperature: TemperatureScaler
    lexical_temperature: TemperatureScaler
    lexical_weight: float

    def probabilities(
        self, semantic_logits: NDArray[np.float64], lexical_logits: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if not np.isfinite(self.lexical_weight) or not 0 <= self.lexical_weight <= 1:
            raise ValueError("Posterior pool weight must be between zero and one")
        semantic = self.semantic_temperature.transform(semantic_logits)
        lexical = self.lexical_temperature.transform(lexical_logits)
        if semantic.shape != lexical.shape:
            raise ValueError("Semantic and lexical probability rows must align")
        return (1 - self.lexical_weight) * semantic + self.lexical_weight * lexical


def fit_posterior_pool(
    semantic_logits: NDArray[np.float64],
    lexical_logits: NDArray[np.float64],
    labels: Sequence[int],
) -> PosteriorPool:
    """Learn temperatures and mixture weight only on supplied calibration rows."""
    semantic_logits = np.asarray(semantic_logits, dtype=np.float64)
    lexical_logits = np.asarray(lexical_logits, dtype=np.float64)
    if semantic_logits.ndim != 2 or semantic_logits.shape != lexical_logits.shape:
        raise ValueError("Semantic and lexical logits must be aligned matrices")
    classes = tuple(range(semantic_logits.shape[1]))
    semantic_temperature = fit_temperature_scaler(semantic_logits, labels, classes)
    lexical_temperature = fit_temperature_scaler(lexical_logits, labels, classes)
    semantic = semantic_temperature.transform(semantic_logits)
    lexical = lexical_temperature.transform(lexical_logits)
    if semantic.shape != lexical.shape:
        raise ValueError("Semantic and lexical rows must align")
    truth = np.asarray(labels)
    correct_semantic = semantic[np.arange(len(truth)), truth]
    correct_lexical = lexical[np.arange(len(truth)), truth]

    def objective(weight: float) -> float:
        return float(-np.log((1 - weight) * correct_semantic + weight * correct_lexical).mean())

    result = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Posterior-pool fit did not converge")
    weight = min([0.0, float(result.x), 1.0], key=objective)
    return PosteriorPool(semantic_temperature, lexical_temperature, weight)


def fixed_route_risk(
    probabilities: NDArray[np.float64], predictions: Sequence[int], matrix: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Score the original routes even when pooled probabilities favor another class."""
    values = np.asarray(probabilities, dtype=np.float64)
    routes, costs = np.asarray(predictions), np.asarray(matrix, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
        or not np.allclose(values.sum(axis=1), 1)
        or routes.shape != (len(values),)
        or routes.dtype.kind not in "iu"
        or np.any(routes < 0)
        or np.any(routes >= values.shape[1])
        or costs.shape != (values.shape[1], values.shape[1])
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
        or np.any(np.diag(costs) != 0)
    ):
        raise ValueError("Expected aligned probabilities, original routes and valid costs")
    return np.asarray((values * costs[:, routes].T).sum(axis=1), dtype=np.float64)
