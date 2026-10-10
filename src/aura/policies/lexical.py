"""Research-inspired lexical correctness gating with fixed classifier predictions."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from aura.models.predictions import PredictionBatch
from aura.risk.expected_risk import compute_expected_routing_risk


def build_lexical_witness() -> Pipeline:
    """Build a word/character SVM witness; fit only on model-training messages."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 2), min_df=2, max_features=30_000, sublinear_tf=True
                            ),
                        ),
                        (
                            "character",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=50_000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("classifier", LinearSVC(C=1.0, random_state=42, max_iter=10_000)),
        ]
    )


def lexical_gate_features(
    semantic_scores: NDArray[np.float64],
    semantic_classes: Sequence[int],
    lexical_scores: NDArray[np.float64],
    lexical_classes: Sequence[int],
) -> NDArray[np.float64]:
    """Return semantic margin and signed lexical support for the semantic prediction."""
    semantic = np.asarray(semantic_scores, dtype=np.float64)
    lexical = np.asarray(lexical_scores, dtype=np.float64)
    classes, witness_classes = tuple(semantic_classes), tuple(lexical_classes)
    for labels in (classes, witness_classes):
        if len(labels) < 2 or len(set(labels)) != len(labels):
            raise ValueError("Classes must contain at least two unique labels")
        if any(
            isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) for k in labels
        ):
            raise ValueError("Classes must be integer labels")
    if set(classes) != set(witness_classes):
        raise ValueError("Semantic and lexical classes must match")
    if (
        semantic.ndim != 2
        or semantic.shape[0] == 0
        or semantic.shape[1] != len(classes)
        or lexical.shape != (semantic.shape[0], len(witness_classes))
        or not np.isfinite(semantic).all()
        or not np.isfinite(lexical).all()
    ):
        raise ValueError("Scores must be aligned nonempty finite matrices")
    predicted_columns = semantic.argmax(axis=1)
    rows = np.arange(len(semantic))
    top_two = np.partition(semantic, -2, axis=1)[:, -2:]
    semantic_margin = top_two[:, 1] - top_two[:, 0]
    positions = {label: column for column, label in enumerate(witness_classes)}
    routed_columns = [positions[classes[int(column)]] for column in predicted_columns]
    competitors = lexical.copy()
    competitors[rows, routed_columns] = -np.inf
    support = lexical[rows, routed_columns] - competitors.max(axis=1)
    return np.column_stack((semantic_margin, support))


def fit_correctness_gate(features: NDArray[np.float64], correct: Sequence[int]) -> Pipeline:
    """Fit a small standardized gate on held-out correctness, never evaluation labels."""
    values = np.asarray(features, dtype=np.float64)
    target = np.asarray(correct)
    if values.ndim != 2 or values.shape[1] not in (1, 2) or not np.isfinite(values).all():
        raise ValueError("Gate features must have one or two finite columns")
    if target.shape != (len(values),) or target.dtype.kind not in "biu" or set(target) != {0, 1}:
        raise ValueError("Gate fitting needs aligned correct and incorrect examples")
    gate = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]
    )
    gate.fit(values, target)
    return gate


def cost_aware_gate_score(
    predictions: PredictionBatch,
    error_probability: NDArray[np.float64],
    cost_matrix: NDArray[np.float64],
    cost_classes: Sequence[int],
) -> NDArray[np.float64]:
    """Experimental score: learned error probability times conditional error severity.

    Keeps the original model's relative probabilities among alternative intents.
    It is a ranking proxy, not a validated or guaranteed expected banking loss.
    """
    risk = compute_expected_routing_risk(predictions, cost_matrix, cost_classes)
    q = np.asarray(error_probability, dtype=np.float64)
    if (
        q.shape != risk.expected_costs.shape
        or not np.isfinite(q).all()
        or np.any((q < 0) | (q > 1))
    ):
        raise ValueError("Error probability must be an aligned finite vector in [0, 1]")
    alternatives = predictions.probabilities.copy()
    alternatives[np.arange(len(q)), alternatives.argmax(axis=1)] = 0
    mass = alternatives.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("Alternative intent mass must be positive")
    return np.asarray(q * (risk.expected_costs / mass), dtype=np.float64)
