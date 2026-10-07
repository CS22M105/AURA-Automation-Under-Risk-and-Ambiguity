"""Validate and summarize the BANKING77 workflow specification."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from aura.risk.workflows import load_workflow_specification


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Validate AURA's intent workflow specification.")
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("config/intent_workflows.yaml"),
    )
    return parser


def main() -> None:
    """Validate the specification and print its review state."""
    path: Path = build_parser().parse_args().specification
    specification = load_workflow_specification(path)
    family_counts = Counter(workflow.operational_family for workflow in specification.workflows)
    flag_counts = Counter(
        flag for workflow in specification.workflows for flag in workflow.consequence_flags
    )
    status_counts = Counter(workflow.review_status for workflow in specification.workflows)

    print(f"Specification: {path}")
    print(f"Specification ID: {specification.specification_id}")
    print(f"Schema version: {specification.schema_version}")
    print(f"Frozen on: {specification.governance.frozen_on}")
    print(f"Review basis: {specification.governance.review_basis}")
    print(f"External domain review: {specification.governance.external_domain_review}")
    print(f"Intended use: {specification.governance.intended_use}")
    print(f"Workflows: {len(specification.workflows)}")
    print(f"Evidence sources: {len(specification.evidence_sources)}")
    print(f"Families: {dict(sorted(family_counts.items()))}")
    print(f"Consequence flags: {dict(sorted(flag_counts.items()))}")
    print(f"Review status: {dict(sorted(status_counts.items()))}")
    print("Workflow specification validation: PASSED")


if __name__ == "__main__":
    main()
