"""Persistence helpers for trusted local model artifacts."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


class ModelArtifactError(ValueError):
    """Raised when a model artifact has an invalid structure."""


@dataclass(frozen=True)
class ModelArtifact:
    """A fitted pipeline and its reproducibility metadata."""

    pipeline: Pipeline
    metadata: dict[str, object]


def save_model_artifact(
    path: Path,
    pipeline: Pipeline,
    metadata: Mapping[str, object],
) -> None:
    """Save a fitted pipeline and metadata as a local joblib artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "metadata": dict(metadata)},
        path,
    )


def load_model_artifact(path: Path) -> ModelArtifact:
    """Load a model artifact created by AURA.

    Joblib artifacts can execute code during loading, so callers must only load
    artifacts produced by this project or another trusted source.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")

    payload: object = joblib.load(path)
    if not isinstance(payload, dict):
        raise ModelArtifactError("Model artifact must contain a dictionary")

    pipeline = payload.get("pipeline")
    metadata = payload.get("metadata")
    if not isinstance(pipeline, Pipeline):
        raise ModelArtifactError("Model artifact does not contain a scikit-learn pipeline")
    if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
        raise ModelArtifactError("Model artifact metadata must be a string-keyed dictionary")

    return ModelArtifact(pipeline=pipeline, metadata=dict(metadata))
