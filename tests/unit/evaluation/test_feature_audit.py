import pytest
from sklearn.pipeline import Pipeline

from aura.evaluation.feature_audit import FeatureAuditError, audit_baseline_features
from aura.models.baseline import BaselineConfig, build_baseline_pipeline


def create_fitted_baseline() -> tuple[list[str], list[int], Pipeline]:
    """Create a small fitted three-class baseline pipeline."""
    messages = [
        "cash withdrawal missing",
        "cash withdrawal declined",
        "bank transfer pending",
        "bank transfer declined",
        "card delivery late",
        "card delivery tracking",
    ]
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = build_baseline_pipeline(BaselineConfig(min_document_frequency=1))
    pipeline.fit(messages, labels)
    return messages, labels, pipeline


def test_audits_text_vocabulary_and_linear_features() -> None:
    messages, labels, pipeline = create_fitted_baseline()

    audit = audit_baseline_features(
        messages,
        labels,
        {0: "cash", 1: "transfer", 2: "card"},
        pipeline.named_steps["tfidf"],
        pipeline.named_steps["classifier"],
        top_features_per_intent=2,
    )

    assert audit.overall_text.sample_count == 6
    assert audit.overall_text.word_count.minimum == 3
    assert audit.class_balance.class_count == 3
    assert audit.class_balance.largest_to_smallest_ratio == pytest.approx(1.0)
    assert audit.vocabulary.feature_count > 0
    assert audit.vocabulary.unigram_count + audit.vocabulary.bigram_count == (
        audit.vocabulary.feature_count
    )
    assert len(audit.per_intent) == 3
    assert len(audit.top_features) == 6
    assert [feature.rank for feature in audit.top_features[:2]] == [1, 2]


def test_rejects_missing_intent_name() -> None:
    messages, labels, pipeline = create_fitted_baseline()

    with pytest.raises(FeatureAuditError, match="names are missing"):
        audit_baseline_features(
            messages,
            labels,
            {0: "cash", 1: "transfer"},
            pipeline.named_steps["tfidf"],
            pipeline.named_steps["classifier"],
        )
