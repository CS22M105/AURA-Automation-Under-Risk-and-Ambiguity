"""Verify original frozen inputs after private documentation moved out of Git.

The original lock and checker remain byte-for-byte unchanged. This relocation
check reads no test data and does not execute a new final evaluation.
"""

import hashlib
import json
from argparse import ArgumentParser
from pathlib import Path


def verify_archive(root: Path, private_docs: Path) -> int:
    """Resolve only the historical docs prefix into the supplied private folder."""
    lock = json.loads((root / "reports/protocols/aura-final-evaluation-v1.json").read_text())
    for name, expected in lock["file_sha256"].items():
        relative = Path(name)
        path = (
            private_docs / Path(*relative.parts[1:])
            if relative.parts[0] == "docs"
            else root / relative
        )
        if not path.is_file():
            raise ValueError(f"Frozen input missing: {name} (resolved to {path})")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Frozen input changed: {name}")
    actual = sorted(
        path.relative_to(root).as_posix()
        for path in (root / "artifacts/models/distilbert").rglob("*")
        if path.is_file()
    )
    if actual != lock["transformer_files"]:
        raise ValueError("Frozen transformer inventory changed")
    config = json.loads((root / "config/final_evaluation.json").read_text())
    if config["protocol_id"] != lock["protocol_id"]:
        raise ValueError("Protocol identifier mismatch")
    return len(lock["file_sha256"])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--private-docs", type=Path, default=root.parent / "Plan" / "docs")
    args = parser.parse_args()
    count = verify_archive(root, args.private_docs)
    print(f"Verified {count} original frozen inputs, including relocated private documentation.")
    print("Original locks unchanged. No test data opened or evaluation run.")


if __name__ == "__main__":
    main()
