"""TF-IDF, latent semantic, and Random Forest comparison model."""

from dataclasses import dataclass

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from aura.data.splitting import DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class RandomForestConfig:
    """Configuration for the reproducible tree-based comparison model."""

    ngram_range: tuple[int, int] = (1, 2)
    min_document_frequency: int = 2
    max_tfidf_features: int | None = 50_000
    svd_components: int = 256
    svd_iterations: int = 7
    tree_count: int = 400
    maximum_depth: int | None = None
    minimum_samples_per_leaf: int = 2
    maximum_features_per_split: str | int | float | None = "sqrt"
    random_state: int = DEFAULT_RANDOM_STATE
    parallel_jobs: int = -1


def build_random_forest_pipeline(
    config: RandomForestConfig | None = None,
) -> Pipeline:
    """Build an unfitted TF-IDF, SVD, and Random Forest pipeline."""
    resolved_config = config or RandomForestConfig()

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=resolved_config.ngram_range,
                    min_df=resolved_config.min_document_frequency,
                    max_features=resolved_config.max_tfidf_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "svd",
                TruncatedSVD(
                    n_components=resolved_config.svd_components,
                    n_iter=resolved_config.svd_iterations,
                    random_state=resolved_config.random_state,
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=resolved_config.tree_count,
                    criterion="log_loss",
                    max_depth=resolved_config.maximum_depth,
                    min_samples_leaf=resolved_config.minimum_samples_per_leaf,
                    max_features=resolved_config.maximum_features_per_split,
                    class_weight="balanced_subsample",
                    random_state=resolved_config.random_state,
                    n_jobs=resolved_config.parallel_jobs,
                ),
            ),
        ]
    )
