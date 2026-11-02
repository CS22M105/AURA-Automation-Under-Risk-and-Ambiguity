"""A verified ZIP must contain every saved byte, without needing models or Colab."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

HELPER = Path(__file__).parents[2] / "tools/notebooks/download_saved_export.py"
SPEC = importlib.util.spec_from_file_location("saved_export_download", HELPER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def saved_export(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    folder = tmp_path / "saved-run"
    (folder / "models/ettin").mkdir(parents=True)
    (folder / "models/ettin/model.safetensors").write_bytes(b"synthetic-weights-" * 100)
    (folder / "package.json").write_text('{"run_id":"synthetic-export-test"}\n')
    manifest = {
        "schema_version": 1,
        "file_sha256": {
            p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*")
            if p.is_file()
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder, manifest


def test_zip_contains_all_files_and_original_manifest(
    saved_export: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    source, manifest = saved_export
    result = MODULE.create_verified_download(source, tmp_path / "complete.zip")
    with zipfile.ZipFile(result) as archive:
        for name, expected in manifest["file_sha256"].items():
            assert (
                hashlib.sha256(archive.read(f"saved-run-verified/{name}")).hexdigest() == expected
            )
        assert (
            archive.read("saved-run-verified/manifest.json")
            == (source / "manifest.json").read_bytes()
        )
        assert len(archive.namelist()) == len(manifest["file_sha256"]) + 1


def test_missing_file_never_creates_zip(
    saved_export: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    source, _ = saved_export
    (source / "package.json").unlink()
    with pytest.raises(FileNotFoundError, match="Drive export itself is incomplete"):
        MODULE.create_verified_download(source, tmp_path / "complete.zip")
    assert not (tmp_path / "complete.zip").exists()


def test_corrupt_file_never_publishes_partial_zip(
    saved_export: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    source, _ = saved_export
    (source / "models/ettin/model.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="differs from its manifest"):
        MODULE.create_verified_download(source, tmp_path / "complete.zip")
    assert not (tmp_path / "complete.zip").exists()
    assert not (tmp_path / "complete.zip.partial").exists()


def test_existing_download_is_preserved(
    saved_export: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    source, _ = saved_export
    destination = tmp_path / "complete.zip"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        MODULE.create_verified_download(source, destination)
    assert destination.read_bytes() == b"existing"


def test_manifest_cannot_reference_files_outside_export(
    saved_export: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    source, manifest = saved_export
    manifest["file_sha256"]["../other.txt"] = "0" * 64
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid export file path"):
        MODULE.create_verified_download(source, tmp_path / "complete.zip")
