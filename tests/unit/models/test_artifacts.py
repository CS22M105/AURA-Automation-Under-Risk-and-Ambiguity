from pathlib import Path

import pytest

from aura.models.artifacts import load_model_artifact, save_model_artifact
from aura.models.baseline import BaselineConfig, build_baseline_pipeline


def test_round_trips_model_artifact(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    pipeline = build_baseline_pipeline(BaselineConfig(min_document_frequency=1))
    pipeline.fit(["cash missing", "cash pending", "card late", "card missing"], [0, 0, 1, 1])

    save_model_artifact(path, pipeline, {"model_type": "test_model"})
    artifact = load_model_artifact(path)

    assert artifact.metadata == {"model_type": "test_model"}
    assert artifact.pipeline.predict(["cash missing"]).shape == (1,)


def test_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_model_artifact(tmp_path / "missing.joblib")
