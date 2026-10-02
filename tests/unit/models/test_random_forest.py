import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from aura.models.predictions import predict_with_sklearn_pipeline
from aura.models.random_forest import (
    RandomForestConfig,
    build_random_forest_pipeline,
)


def create_training_examples() -> tuple[list[str], list[int]]:
    """Create small three-class text data for tree-pipeline tests."""
    return (
        [
            "cash withdrawal missing",
            "cash withdrawal declined",
            "cash withdrawal fee",
            "bank transfer pending",
            "bank transfer declined",
            "bank transfer recipient",
            "card delivery late",
            "card delivery tracking",
            "card delivery address",
        ],
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
    )


def test_builds_expected_random_forest_pipeline() -> None:
    pipeline = build_random_forest_pipeline()

    assert list(pipeline.named_steps) == ["tfidf", "svd", "classifier"]
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["svd"], TruncatedSVD)
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)


def test_applies_custom_random_forest_configuration() -> None:
    config = RandomForestConfig(
        ngram_range=(1, 1),
        min_document_frequency=1,
        max_tfidf_features=100,
        svd_components=4,
        svd_iterations=3,
        tree_count=12,
        maximum_depth=5,
        minimum_samples_per_leaf=1,
        maximum_features_per_split=0.5,
        random_state=7,
        parallel_jobs=1,
    )

    pipeline = build_random_forest_pipeline(config)
    vectorizer = pipeline.named_steps["tfidf"]
    svd = pipeline.named_steps["svd"]
    classifier = pipeline.named_steps["classifier"]

    assert vectorizer.ngram_range == (1, 1)
    assert vectorizer.min_df == 1
    assert vectorizer.max_features == 100
    assert svd.n_components == 4
    assert svd.n_iter == 3
    assert classifier.n_estimators == 12
    assert classifier.max_depth == 5
    assert classifier.min_samples_leaf == 1
    assert classifier.max_features == 0.5
    assert classifier.random_state == 7
    assert classifier.n_jobs == 1


def test_fits_and_uses_common_prediction_contract() -> None:
    messages, labels = create_training_examples()
    pipeline = build_random_forest_pipeline(
        RandomForestConfig(
            min_document_frequency=1,
            svd_components=4,
            tree_count=20,
            minimum_samples_per_leaf=1,
            parallel_jobs=1,
        )
    )

    pipeline.fit(messages, labels)
    predictions = predict_with_sklearn_pipeline(
        pipeline,
        messages,
        list(range(len(messages))),
    )

    assert predictions.classes == (0, 1, 2)
    assert predictions.probabilities.shape == (9, 3)
    assert predictions.decision_scores is None
    np.testing.assert_allclose(predictions.probabilities.sum(axis=1), np.ones(9))
