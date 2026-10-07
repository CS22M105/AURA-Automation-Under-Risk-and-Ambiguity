from pathlib import Path

import pytest

from aura.models.transformer_artifacts import calculate_transformer_artifact_sha256


def test_transformer_artifact_hash_is_deterministic_and_content_sensitive(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "transformer"
    artifact_directory.mkdir()
    (artifact_directory / "config.json").write_text("configuration", encoding="utf-8")
    (artifact_directory / "model.safetensors").write_bytes(b"weights")

    first_hash = calculate_transformer_artifact_sha256(artifact_directory)
    second_hash = calculate_transformer_artifact_sha256(artifact_directory)
    (artifact_directory / "model.safetensors").write_bytes(b"changed weights")

    assert first_hash == second_hash
    assert calculate_transformer_artifact_sha256(artifact_directory) != first_hash


def test_transformer_artifact_hash_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        calculate_transformer_artifact_sha256(tmp_path / "missing")
