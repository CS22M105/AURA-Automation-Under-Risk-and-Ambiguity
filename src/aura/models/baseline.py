"""TF-IDF and Logistic Regression baseline for intent classification."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from aura.data.splitting import DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the reproducible baseline classifier."""

    ngram_range: tuple[int, int] = (1, 2)
    min_document_frequency: int = 2
    max_features: int | None = 50_000
    regularization_strength: float = 1.0
    maximum_iterations: int = 1_000
    random_state: int = DEFAULT_RANDOM_STATE


def build_baseline_pipeline(
    config: BaselineConfig | None = None,
) -> Pipeline:
    """Build an unfitted TF-IDF and Logistic Regression pipeline."""
    resolved_config = config or BaselineConfig()

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=resolved_config.ngram_range,
                    min_df=resolved_config.min_document_frequency,
                    max_features=resolved_config.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=resolved_config.regularization_strength,
                    class_weight="balanced",
                    max_iter=resolved_config.maximum_iterations,
                    random_state=resolved_config.random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
