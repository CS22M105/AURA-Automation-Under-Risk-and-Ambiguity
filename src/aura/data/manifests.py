"""Loading and integrity checks for development split manifests."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


class SplitManifestError(ValueError):
    """Raised when a development split manifest is invalid."""


@dataclass(frozen=True)
class DevelopmentSplitManifest:
    """Validated indices and provenance for development partitions."""

    source_file: str
    source_sha256: str
    random_state: int
    total_rows: int
    model_training: tuple[int, ...]
    calibration: tuple[int, ...]
    policy_validation: tuple[int, ...]


def calculate_sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of a file."""
    digest = sha256()

    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _require_value(payload: dict[str, Any], key: str, expected_type: type) -> Any:
    value = payload.get(key)
    if not isinstance(value, expected_type) or isinstance(value, bool):
        raise SplitManifestError(f"Manifest field '{key}' has an invalid type")
    return value


def _require_indices(partitions: dict[str, Any], name: str) -> tuple[int, ...]:
    raw_indices = partitions.get(name)
    if not isinstance(raw_indices, list) or not raw_indices:
        raise SplitManifestError(f"Manifest partition '{name}' must be a non-empty list")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in raw_indices):
        raise SplitManifestError(f"Manifest partition '{name}' contains a non-integer index")
    return tuple(raw_indices)


def load_development_split_manifest(path: Path) -> DevelopmentSplitManifest:
    """Load and validate a development split manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SplitManifestError(f"Split manifest is not valid JSON: {path}") from error

    if not isinstance(payload, dict):
        raise SplitManifestError("Split manifest must contain a JSON object")
    if _require_value(payload, "schema_version", int) != 1:
        raise SplitManifestError("Unsupported split manifest schema version")

    raw_partitions = _require_value(payload, "partitions", dict)
    model_training = _require_indices(raw_partitions, "model_training")
    calibration = _require_indices(raw_partitions, "calibration")
    policy_validation = _require_indices(raw_partitions, "policy_validation")
    total_rows = _require_value(payload, "total_rows", int)

    all_indices = model_training + calibration + policy_validation
    if len(all_indices) != total_rows:
        raise SplitManifestError("Partition sizes do not equal the manifest row count")
    if len(set(all_indices)) != len(all_indices):
        raise SplitManifestError("Manifest partitions contain overlapping or duplicate indices")
    if set(all_indices) != set(range(total_rows)):
        raise SplitManifestError("Manifest partitions do not cover the expected dataset indices")

    return DevelopmentSplitManifest(
        source_file=_require_value(payload, "source_file", str),
        source_sha256=_require_value(payload, "source_sha256", str),
        random_state=_require_value(payload, "random_state", int),
        total_rows=total_rows,
        model_training=model_training,
        calibration=calibration,
        policy_validation=policy_validation,
    )


def validate_manifest_source(
    manifest: DevelopmentSplitManifest,
    dataset_path: Path,
    row_count: int,
) -> None:
    """Verify that a manifest belongs to the supplied dataset."""
    if row_count != manifest.total_rows:
        raise SplitManifestError(
            f"Dataset has {row_count} rows; manifest expects {manifest.total_rows}"
        )
    if calculate_sha256(dataset_path) != manifest.source_sha256:
        raise SplitManifestError("Dataset fingerprint does not match the split manifest")
