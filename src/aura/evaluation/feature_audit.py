"""Feature and input-data diagnostics for the linear baseline."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class FeatureAuditError(ValueError):
    """Raised when a feature audit cannot be computed safely."""


@dataclass(frozen=True)
class DistributionSummary:
    """Compact descriptive statistics for a numeric variable."""

    minimum: float
    maximum: float
    mean: float
    median: float
    percentile_95: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-serializable values."""
        return asdict(self)


@dataclass(frozen=True)
class TextLengthProfile:
    """Word and character lengths for a collection of messages."""

    sample_count: int
    word_count: DistributionSummary
    character_count: DistributionSummary


@dataclass(frozen=True)
class IntentTextProfile:
    """Training frequency and message lengths for one intent."""

    label: int
    intent: str
    sample_count: int
    mean_word_count: float
    median_word_count: float
    percentile_95_word_count: float
    mean_character_count: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return CSV- and JSON-serializable values."""
        return asdict(self)


@dataclass(frozen=True)
class VocabularyStatistics:
    """Statistics for the fitted TF-IDF representation."""

    feature_count: int
    unigram_count: int
    bigram_count: int
    matrix_nonzero_count: int
    matrix_density: float
    document_frequency: DistributionSummary
    inverse_document_frequency: DistributionSummary


@dataclass(frozen=True)
class ClassBalanceStatistics:
    """Training-frequency balance across intent classes."""

    class_count: int
    smallest_class_size: int
    largest_class_size: int
    largest_to_smallest_ratio: float


@dataclass(frozen=True)
class LinearFeatureWeight:
    """One influential TF-IDF feature for an intent class."""

    label: int
    intent: str
    rank: int
    feature: str
    coefficient: float
    document_frequency: int
    inverse_document_frequency: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return CSV- and JSON-serializable values."""
        return asdict(self)


@dataclass(frozen=True)
class BaselineFeatureAudit:
    """Complete feature audit for a fitted linear text classifier."""

    overall_text: TextLengthProfile
    class_balance: ClassBalanceStatistics
    vocabulary: VocabularyStatistics
    per_intent: tuple[IntentTextProfile, ...]
    top_features: tuple[LinearFeatureWeight, ...]


def _summarize(values: NDArray[np.float64]) -> DistributionSummary:
    if values.ndim != 1 or len(values) == 0:
        raise FeatureAuditError("Distribution values must be a non-empty vector")
    return DistributionSummary(
        minimum=float(values.min()),
        maximum=float(values.max()),
        mean=float(values.mean()),
        median=float(np.median(values)),
        percentile_95=float(np.percentile(values, 95)),
    )


def audit_baseline_features(
    messages: Sequence[str],
    labels: Sequence[int],
    intent_names: Mapping[int, str],
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    top_features_per_intent: int = 15,
) -> BaselineFeatureAudit:
    """Audit baseline inputs, fitted vocabulary, and linear coefficients."""
    resolved_messages = list(messages)
    resolved_labels = np.asarray(labels, dtype=np.int64)
    classes = np.asarray(classifier.classes_, dtype=np.int64)

    if not resolved_messages or len(resolved_messages) != len(resolved_labels):
        raise FeatureAuditError("Messages and labels must be non-empty and aligned")
    if top_features_per_intent < 1:
        raise FeatureAuditError("Top features per intent must be positive")
    missing_names = set(int(label) for label in classes) - set(intent_names)
    if missing_names:
        raise FeatureAuditError(f"Intent names are missing for labels: {sorted(missing_names)[:5]}")
    if set(resolved_labels) != set(classes):
        raise FeatureAuditError("Training labels must contain every classifier class")

    word_counts = np.asarray(
        [len(message.split()) for message in resolved_messages],
        dtype=np.float64,
    )
    character_counts = np.asarray(
        [len(message) for message in resolved_messages],
        dtype=np.float64,
    )
    overall_text = TextLengthProfile(
        sample_count=len(resolved_messages),
        word_count=_summarize(word_counts),
        character_count=_summarize(character_counts),
    )

    class_sizes = np.asarray(
        [np.count_nonzero(resolved_labels == label) for label in classes],
        dtype=np.int64,
    )
    class_balance = ClassBalanceStatistics(
        class_count=len(classes),
        smallest_class_size=int(class_sizes.min()),
        largest_class_size=int(class_sizes.max()),
        largest_to_smallest_ratio=float(class_sizes.max() / class_sizes.min()),
    )

    per_intent: list[IntentTextProfile] = []
    for label in classes:
        members = resolved_labels == label
        intent_words = word_counts[members]
        intent_characters = character_counts[members]
        per_intent.append(
            IntentTextProfile(
                label=int(label),
                intent=intent_names[int(label)],
                sample_count=int(members.sum()),
                mean_word_count=float(intent_words.mean()),
                median_word_count=float(np.median(intent_words)),
                percentile_95_word_count=float(np.percentile(intent_words, 95)),
                mean_character_count=float(intent_characters.mean()),
            )
        )

    feature_names = np.asarray(vectorizer.get_feature_names_out(), dtype=str)
    matrix = vectorizer.transform(resolved_messages)
    document_frequencies = np.asarray(matrix.getnnz(axis=0), dtype=np.int64)
    inverse_document_frequencies = np.asarray(vectorizer.idf_, dtype=np.float64)
    if classifier.coef_.shape != (len(classes), len(feature_names)):
        raise FeatureAuditError("Classifier coefficients do not match classes and features")

    vocabulary = VocabularyStatistics(
        feature_count=len(feature_names),
        unigram_count=int(np.count_nonzero(np.char.find(feature_names, " ") == -1)),
        bigram_count=int(np.count_nonzero(np.char.find(feature_names, " ") >= 0)),
        matrix_nonzero_count=int(matrix.nnz),
        matrix_density=float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        document_frequency=_summarize(document_frequencies.astype(np.float64)),
        inverse_document_frequency=_summarize(inverse_document_frequencies),
    )

    feature_limit = min(top_features_per_intent, len(feature_names))
    top_features: list[LinearFeatureWeight] = []
    for class_position, label in enumerate(classes):
        coefficients = np.asarray(classifier.coef_[class_position], dtype=np.float64)
        ranked_positions = np.argsort(-coefficients, kind="stable")[:feature_limit]
        for rank, feature_position in enumerate(ranked_positions, start=1):
            top_features.append(
                LinearFeatureWeight(
                    label=int(label),
                    intent=intent_names[int(label)],
                    rank=rank,
                    feature=str(feature_names[feature_position]),
                    coefficient=float(coefficients[feature_position]),
                    document_frequency=int(document_frequencies[feature_position]),
                    inverse_document_frequency=float(
                        inverse_document_frequencies[feature_position]
                    ),
                )
            )

    return BaselineFeatureAudit(
        overall_text=overall_text,
        class_balance=class_balance,
        vocabulary=vocabulary,
        per_intent=tuple(per_intent),
        top_features=tuple(top_features),
    )
