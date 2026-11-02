"""Private-document relocation keeps original archival hashes meaningful."""

import hashlib
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_archived_protocol.py"


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Any, Path, Path]:
    root = tmp_path / "AURA"
    private = tmp_path / "Plan/docs"
    private.mkdir(parents=True)
    document = private / "FINAL_EVALUATION_PROTOCOL.md"
    document.write_text("original research protocol")
    model = root / "artifacts/models/distilbert/weights.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"original model")
    config = root / "config/final_evaluation.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"protocol_id": "frozen"}))
    files = {
        "docs/FINAL_EVALUATION_PROTOCOL.md": document,
        "artifacts/models/distilbert/weights.bin": model,
        "config/final_evaluation.json": config,
    }
    lock = root / "reports/protocols/aura-final-evaluation-v1.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "protocol_id": "frozen",
                "file_sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in files.items()
                },
                "transformer_files": ["artifacts/models/distilbert/weights.bin"],
            }
        )
    )
    return runpy.run_path(str(SCRIPT))["verify_archive"], root, private


def test_private_document_is_verified_without_copying_into_repository(archive: tuple) -> None:
    verify, root, private = archive
    assert verify(root, private) == 3
    assert not (root / "docs").exists()


def test_missing_private_document_is_not_silently_skipped(archive: tuple) -> None:
    verify, root, private = archive
    with pytest.raises(ValueError, match="missing"):
        verify(root, private / "absent")


def test_changed_private_document_still_fails_original_hash(archive: tuple) -> None:
    verify, root, private = archive
    (private / "FINAL_EVALUATION_PROTOCOL.md").write_text("altered")
    with pytest.raises(ValueError, match="changed"):
        verify(root, private)
