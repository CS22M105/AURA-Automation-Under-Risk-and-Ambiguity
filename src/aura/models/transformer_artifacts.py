"""Persistence helpers for trusted local transformer artifacts."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from aura.models.distilbert import TransformerTrainingResult


@dataclass(frozen=True)
class TransformerArtifact:
    """A locally saved transformer model, tokenizer, and AURA metadata."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    metadata: dict[str, Any]


def calculate_transformer_artifact_sha256(directory: Path) -> str:
    """Fingerprint all files in a transformer artifact directory deterministically."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Transformer artifact directory does not exist: {directory}")

    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Transformer artifact directory is empty: {directory}")

    digest = sha256()
    for path in files:
        relative_path = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def save_transformer_artifact(
    directory: Path,
    result: TransformerTrainingResult,
    metadata: dict[str, object],
) -> None:
    """Save a transformer and its experiment metadata to one directory."""
    directory.mkdir(parents=True, exist_ok=True)
    result.model.save_pretrained(directory)
    result.tokenizer.save_pretrained(directory)
    (directory / "aura_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_transformer_artifact(directory: Path) -> TransformerArtifact:
    """Load a trusted transformer artifact produced by AURA."""
    metadata_path = directory / "aura_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Transformer metadata does not exist: {metadata_path}")
    metadata: object = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
        raise ValueError("Transformer metadata must be a string-keyed JSON object")

    return TransformerArtifact(
        model=AutoModelForSequenceClassification.from_pretrained(directory),
        tokenizer=AutoTokenizer.from_pretrained(directory),
        metadata=metadata,
    )
