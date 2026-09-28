import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from aura.models.baseline import BaselineConfig, build_baseline_pipeline


def test_builds_expected_pipeline() -> None:
    pipeline = build_baseline_pipeline()

    assert list(pipeline.named_steps) == ["tfidf", "classifier"]
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_applies_custom_configuration() -> None:
    config = BaselineConfig(
        ngram_range=(1, 1),
        min_document_frequency=1,
        max_features=100,
        regularization_strength=0.5,
        maximum_iterations=250,
        random_state=7,
    )

    pipeline = build_baseline_pipeline(config)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    assert vectorizer.ngram_range == (1, 1)
    assert vectorizer.min_df == 1
    assert vectorizer.max_features == 100
    assert classifier.C == 0.5
    assert classifier.max_iter == 250
    assert classifier.random_state == 7


def test_fits_and_returns_class_probabilities() -> None:
    messages = [
        "cash withdrawal is missing",
        "cash withdrawal was declined",
        "cash withdrawal fee charged",
        "bank transfer is pending",
        "bank transfer was declined",
        "bank transfer recipient missing",
    ]
    labels = [0, 0, 0, 1, 1, 1]
    pipeline = build_baseline_pipeline()

    pipeline.fit(messages, labels)
    predictions = pipeline.predict(messages)
    probabilities = pipeline.predict_proba(messages)

    assert predictions.shape == (6,)
    assert probabilities.shape == (6, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(6))
