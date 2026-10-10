import json
import runpy
from pathlib import Path

import pytest

from aura.data.manifests import calculate_sha256

CHECK_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/check_final_protocol.py"


def prepare_lock(root: Path) -> None:
    config = root / "config/final_evaluation.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"protocol_id": "aura-final-evaluation-v1"}))
    model = root / "artifacts/models/distilbert/weights.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"frozen model")
    lock = root / "reports/protocols/aura-final-evaluation-v1.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "protocol_id": "aura-final-evaluation-v1",
                "file_sha256": {
                    p.relative_to(root).as_posix(): calculate_sha256(p) for p in (config, model)
                },
                "transformer_files": [model.relative_to(root).as_posix()],
            }
        )
    )


def test_check_succeeds_without_test_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_lock(tmp_path)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(CHECK_SCRIPT), run_name="__main__")


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_check_rejects_artifact_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    prepare_lock(tmp_path)
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "artifacts/models/distilbert/weights.bin"
    if change == "modified":
        model.write_bytes(b"new model")
    elif change == "missing":
        model.unlink()
    else:
        (model.parent / "extra.json").write_text("{}")
    with pytest.raises(ValueError):
        runpy.run_path(str(CHECK_SCRIPT), run_name="__main__")


def test_check_rejects_protocol_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepare_lock(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config/final_evaluation.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen input"):
        runpy.run_path(str(CHECK_SCRIPT), run_name="__main__")
