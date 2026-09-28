import json
from pathlib import Path

import pytest

from aura.data.manifests import (
    SplitManifestError,
    load_development_split_manifest,
    validate_manifest_source,
)


def write_manifest(path: Path) -> None:
    """Write a minimal valid split manifest."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_file": "train.csv",
                "source_sha256": "incorrect-until-overridden",
                "random_state": 42,
                "total_rows": 6,
                "partitions": {
                    "model_training": [0, 1, 2],
                    "calibration": [3],
                    "policy_validation": [4, 5],
                },
            }
        ),
        encoding="utf-8",
    )


def test_loads_valid_manifest(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    write_manifest(path)

    manifest = load_development_split_manifest(path)

    assert manifest.random_state == 42
    assert manifest.model_training == (0, 1, 2)
    assert manifest.calibration == (3,)
    assert manifest.policy_validation == (4, 5)


def test_rejects_overlapping_partitions(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    write_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["partitions"]["calibration"] = [2]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SplitManifestError, match="overlapping or duplicate"):
        load_development_split_manifest(path)


def test_rejects_wrong_dataset_row_count(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    dataset_path = tmp_path / "train.csv"
    write_manifest(path)
    dataset_path.write_text("example", encoding="utf-8")
    manifest = load_development_split_manifest(path)

    with pytest.raises(SplitManifestError, match="manifest expects 6"):
        validate_manifest_source(manifest, dataset_path, row_count=5)
