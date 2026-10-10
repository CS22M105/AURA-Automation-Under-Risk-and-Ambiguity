"""Preserve routing experiments and generate Git-trackable presentation results."""

import json
import re
import subprocess
from argparse import ArgumentParser
from datetime import UTC, datetime
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

import pandas as pd

from aura.data.manifests import calculate_sha256
from aura.evaluation.result_snapshot import archive_results, presentation_cost_table


def verify_file(path: Path, expected: str) -> None:
    """Refuse to preserve artifacts that differ from their recorded provenance."""
    if calculate_sha256(path) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path}")


def main() -> None:
    """Archive without overwriting prior versions; omit raw datasets and base weights."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.snapshot_id):
        raise ValueError("Snapshot ID must contain only letters, numbers, underscores, or hyphens")
    root = Path.cwd()
    report_dir = root / "reports" / "routing" / args.snapshot_id
    archive_path = root / "artifacts" / "snapshots" / f"{args.snapshot_id}.zip"
    if report_dir.exists() or archive_path.exists():
        raise FileExistsError("Snapshot ID already exists; choose a new version")
    source_path = root / "artifacts/risk/distilbert/summary.json"
    source = json.loads(source_path.read_text())
    if source["partition"] != "policy_validation" or source["evaluation_stage"] != "development":
        raise ValueError("Expected development policy-validation results")
    n = source["sample_count"]
    for name, relative_path in {
        "scores": "artifacts/risk/distilbert/policy_validation_scores.csv",
        "probabilities": "artifacts/risk/distilbert/policy_validation_probabilities.npz",
        "workflows": "config/intent_workflows.yaml",
        "scenarios": "config/consequence_scenarios.yaml",
        "calibration": "artifacts/calibration/distilbert_temperature.json",
        "manifest": "data/splits/development_seed_42.json",
    }.items():
        verify_file(root / relative_path, source["provenance"][f"{name}_sha256"])
    comparisons = []
    policy_dirs = []
    for experiment in ("distilbert", "guarded_distilbert", "lexical_distilbert"):
        directory = root / "artifacts/policies" / experiment
        policy_dirs.append(directory)
        metadata = json.loads((directory / "summary.json").read_text())
        verify_file(source_path, metadata["source_summary_sha256"])
        if (
            metadata["partition"] != "policy_validation"
            or metadata["evaluation_stage"] != "development"
        ):
            raise ValueError("Policy partition mismatch")
        recorded = metadata.get("output_hashes", {})
        if not recorded:
            recorded = {
                f"{name}.csv": metadata[f"{name}_sha256"] for name in ("comparison", "decisions")
            }
        for name, expected in recorded.items():
            verify_file(directory / name, expected)
        frame = pd.read_csv(directory / "comparison.csv")
        keys = ["policy", "evaluation_scenario", "target_coverage"]
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate comparison rows")
        frame["experiment"] = experiment
        frame["evaluation_stage"] = "development"
        frame["partition"] = "policy_validation"
        frame["sample_count"] = n
        frame["requested_count"] = [
            int((Decimal(str(c)) * n).to_integral_value(rounding=ROUND_FLOOR))
            for c in frame.target_coverage
        ]
        frame["coverage_shortfall_count"] = frame.requested_count - frame.accepted_count
        if (frame.coverage_shortfall_count < 0).any():
            raise ValueError("Accepted count exceeds target budget")
        frame["source_comparison"] = (directory / "comparison.csv").relative_to(root).as_posix()
        comparisons.append(frame)
    combined = pd.concat(comparisons, ignore_index=True)
    table = presentation_cost_table(combined)
    guarded_final = combined[
        (combined.experiment == "guarded_distilbert")
        & (combined.target_coverage == 1.0)
        & (combined.evaluation_scenario == "moderate")
    ].set_index("policy")
    guard = guarded_final.loc[["guarded_moderate"]].iloc[0]
    matched_aura = guarded_final.loc[["moderate_matched_budget"]].iloc[0]
    matched_confidence = guarded_final.loc[["confidence_matched_budget"]].iloc[0]
    if not (
        guard.accepted_count == matched_aura.accepted_count == matched_confidence.accepted_count
    ):
        raise ValueError("Guard controls do not have matching acceptance counts")
    interval_frames = []
    for name, path, candidate, reference in (
        (
            "original",
            root / "artifacts/audits/routing_uncertainty/intervals.csv",
            "moderate",
            "confidence",
        ),
        (
            "lexical",
            root / "artifacts/policies/lexical_distilbert/intervals.csv",
            "signed_cost_gate",
            "original_aura",
        ),
    ):
        frame = pd.read_csv(path)
        if "evaluation_scenario" not in frame:
            frame["evaluation_scenario"] = "moderate"
        frame["comparison"] = name
        frame["candidate"] = candidate
        frame["reference"] = reference
        frame["metric"] = "candidate minus reference accepted cost per input"
        interval_frames.append(frame)
    audit_dirs = [
        root / "artifacts/audits" / name
        for name in ("routing_changes", "routing_uncertainty", "routing_diagnostics")
    ]
    for directory in audit_dirs:
        metadata = json.loads((directory / "summary.json").read_text())
        for name, expected in metadata.get("output_hashes", {}).items():
            verify_file(directory / name, expected)
        for key, name in (
            ("changed_messages_sha256", "changed_messages.csv"),
            ("intervals_sha256", "intervals.csv"),
        ):
            if key in metadata:
                verify_file(directory / name, metadata[key])
        if "policy_summary_sha256" in metadata:
            verify_file(
                root / "artifacts/policies/distilbert/summary.json",
                metadata["policy_summary_sha256"],
            )
        if "source_summary_sha256" in metadata:
            verify_file(source_path, metadata["source_summary_sha256"])
    files: list[Path] = []
    for directory in [*policy_dirs, *audit_dirs, source_path.parent]:
        files.extend(path for path in directory.rglob("*") if path.is_file())
    for source_directory in ("src", "scripts", "tests", "docs", "config"):
        files.extend(
            path
            for path in (root / source_directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".toml"}
        )
    files.extend(
        root / path
        for path in (
            "pyproject.toml",
            ".gitignore",
            ".pre-commit-config.yaml",
            "README.md",
            "artifacts/calibration/distilbert_temperature.json",
            "data/splits/development_seed_42.json",
        )
    )
    # Reserve the version before writing; incomplete runs must use a new ID as well.
    report_dir.mkdir(parents=True, exist_ok=False)
    combined.to_csv(report_dir / "comparisons.csv", index=False)
    pd.concat(interval_frames, ignore_index=True).to_csv(report_dir / "intervals.csv", index=False)
    (report_dir / "README.md").write_text(
        f"# AURA Routing Results: {args.snapshot_id}\n\n"
        f"Development / policy-validation only; {n:,} messages, frozen DistilBERT.\n"
        "These are research-scenario costs, not measured banking losses or final-test results.\n\n"
        "## Moderate-Scenario Cost\n\n" + table + "\n\n"
        "Coverage headers are targets with floor-rounded budgets. Every numeric cell in a column "
        "uses the same acceptance count. At the 100% target, the guard actually accepts "
        f"{int(guard.accepted_count):,} messages ({guard.actual_coverage:.2%}), costing "
        f"{guard.accepted_total_cost:g}; at that count original AURA costs "
        f"{matched_aura.accepted_total_cost:g} and confidence costs "
        f"{matched_confidence.accepted_total_cost:g}.\n\n"
        "Compare every coverage level; do not infer uniform superiority from a selected result.\n\n"
        "## Included Evidence\n\n"
        "`comparisons.csv` retains all experiment rows, matched-budget controls, scenarios, "
        "coverage levels, counts, and error totals. Repeated baselines are intentional. "
        "`intervals.csv` names the candidate and reference for each available interval; "
        "these pointwise intervals do not cover every pair of variants and are not adjusted "
        "for repeated development experimentation.\n\n"
        "## Preservation\n\n"
        f"Local archive: `artifacts/snapshots/{args.snapshot_id}.zip`. Its checksum and "
        "per-file hashes are in `manifest.json`. The archive includes saved routing outputs, "
        "learned gates, diagnostic messages, source code, tests, configuration, and documents. "
        "It excludes the raw dataset, base DistilBERT weights, virtual environment, and classifier "
        "comparison artifacts. This is a routing-results snapshot, "
        "not a complete training backup.\n\n"
        "This directory is Git-trackable. The ZIP is Git-ignored and remains local; "
        "no remote backup or upload has been performed. The snapshot command refuses to "
        "overwrite an existing ID. Hashes detect modification but do not prevent manual editing.\n"
    )
    files.extend(report_dir / name for name in ("comparisons.csv", "intervals.csv", "README.md"))
    hashes = archive_results(root, files, archive_path)
    metadata = {
        "snapshot_id": args.snapshot_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status_at_snapshot": subprocess.check_output(["git", "status", "--short"], text=True),
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "archive": archive_path.relative_to(root).as_posix(),
        "archive_sha256": calculate_sha256(archive_path),
        "files": hashes,
    }
    (report_dir / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Archived and verified {len(hashes)} files: {archive_path.relative_to(root)}")
    print(f"Git-trackable presentation results: {report_dir.relative_to(root)}")


if __name__ == "__main__":
    main()
