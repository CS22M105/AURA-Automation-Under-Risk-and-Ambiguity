import hashlib
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from aura.evaluation.result_snapshot import archive_results, presentation_cost_table


def test_archive_preserves_bytes_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "result.json"
    source.write_text('{"cost": 33}\n')
    original = source.read_bytes()
    output = tmp_path / "snapshots" / "v1.zip"
    hashes = archive_results(tmp_path, [source], output)
    assert hashes == {"result.json": hashlib.sha256(original).hexdigest()}
    source.write_text('{"cost": 99}\n')
    with ZipFile(output) as archive:
        assert archive.read("result.json") == original
    with pytest.raises(FileExistsError):
        archive_results(tmp_path, [source], output)


def test_archive_rejects_outside_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("outside")
    with pytest.raises(ValueError, match="within"):
        archive_results(root, [external], root / "v1.zip")


def test_archive_rejects_empty_inventory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires"):
        archive_results(tmp_path, [], tmp_path / "v1.zip")


def comparisons() -> pd.DataFrame:
    rows = []
    for experiment, policies in (
        ("distilbert", ["confidence", "flat", "moderate", "high_protection"]),
        ("guarded_distilbert", ["guarded_moderate"]),
        (
            "lexical_distilbert",
            ["logit_margin", "semantic_gate", "signed_gate", "signed_cost_gate"],
        ),
    ):
        for policy in policies:
            for coverage in (0.5, 0.7, 0.8, 0.9, 1.0):
                rows.append(
                    {
                        "experiment": experiment,
                        "policy": policy,
                        "target_coverage": coverage,
                        "evaluation_scenario": "moderate",
                        "accepted_total_cost": 4.0,
                        "coverage_shortfall_count": int(
                            policy == "guarded_moderate" and coverage == 1.0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_presentation_never_labels_guard_shortfall_as_full_coverage() -> None:
    table = presentation_cost_table(comparisons())
    assert "| Majority safeguard | 4 | 4 | 4 | 4 | Not reached |" in table
    assert "| Original AURA (moderate) | 4 | 4 | 4 | 4 | 4 |" in table


def test_presentation_rejects_missing_or_duplicate_rows() -> None:
    frame = comparisons()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        presentation_cost_table(frame.iloc[1:])
    with pytest.raises(ValueError, match="Missing or duplicate"):
        presentation_cost_table(pd.concat([frame, frame.iloc[:1]]))
