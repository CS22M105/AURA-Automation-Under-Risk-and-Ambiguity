"""Check frozen protocol inputs without opening the test dataset or loading models."""

import json
from pathlib import Path

from aura.data.manifests import calculate_sha256


def main() -> None:
    """Fail on missing/changed frozen files or changed transformer inventory."""
    lock = json.loads(Path("reports/protocols/aura-final-evaluation-v1.json").read_text())
    for name, expected in lock["file_sha256"].items():
        path = Path(name)
        if not path.is_file() or calculate_sha256(path) != expected:
            raise ValueError(f"Frozen input missing or changed: {name}")
    directory = Path("artifacts/models/distilbert")
    actual = sorted(path.as_posix() for path in directory.rglob("*") if path.is_file())
    if actual != lock["transformer_files"]:
        raise ValueError("Transformer artifact file inventory changed")
    config = json.loads(Path("config/final_evaluation.json").read_text())
    if config["protocol_id"] != lock["protocol_id"]:
        raise ValueError("Protocol identifier mismatch")
    print(f"Verified {len(lock['file_sha256'])} frozen files for {lock['protocol_id']}")
    print("No test data opened. Final scoring has not been run by this check.")


if __name__ == "__main__":
    main()
