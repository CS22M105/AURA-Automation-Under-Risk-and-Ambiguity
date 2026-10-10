"""Write-once local archives and coverage-aware presentation tables."""

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from aura.data.manifests import calculate_sha256


def archive_results(root: Path, files: list[Path], destination: Path) -> dict[str, str]:
    """Create an exclusive ZIP and verify every archived byte against its source hash."""
    root = root.resolve()
    resolved = sorted({path.resolve() for path in files})
    if not resolved:
        raise ValueError("Snapshot requires files")
    for path in resolved:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Snapshot files must be regular files within the repository")
    hashes = {path.relative_to(root).as_posix(): calculate_sha256(path) for path in resolved}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, mode="x", compression=ZIP_DEFLATED) as archive:
        for path in resolved:
            archive.write(path, path.relative_to(root).as_posix())
    with ZipFile(destination) as archive:
        if set(archive.namelist()) != set(hashes):
            raise ValueError("Archive inventory mismatch")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive verification failed: {name}")
    return hashes


def presentation_cost_table(comparisons: pd.DataFrame) -> str:
    """Format moderate costs, explicitly excluding guard shortfalls from 100% columns."""
    methods = [
        ("distilbert", "confidence", "Confidence baseline"),
        ("distilbert", "flat", "Flat-cost routing"),
        ("distilbert", "moderate", "Original AURA (moderate)"),
        ("distilbert", "high_protection", "AURA (high-protection ranking)"),
        ("guarded_distilbert", "guarded_moderate", "Majority safeguard"),
        ("lexical_distilbert", "logit_margin", "Logit margin"),
        ("lexical_distilbert", "semantic_gate", "Semantic-only gate"),
        ("lexical_distilbert", "signed_gate", "Signed lexical gate"),
        ("lexical_distilbert", "signed_cost_gate", "Cost-aware lexical gate"),
    ]
    lines = [
        "| Routing method | 50% | 70% | 80% | 90% | 100% |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment, policy, label in methods:
        values = []
        for coverage in (0.5, 0.7, 0.8, 0.9, 1.0):
            selected = comparisons[
                (comparisons.experiment == experiment)
                & (comparisons.policy == policy)
                & (comparisons.evaluation_scenario == "moderate")
                & (comparisons.target_coverage == coverage)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"Missing or duplicate presentation row: {experiment}/{policy}/{coverage}"
                )
            row = selected.iloc[0]
            values.append(
                "Not reached"
                if row.coverage_shortfall_count > 0
                else f"{row.accepted_total_cost:g}"
            )
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(lines)
