from collections import Counter
from pathlib import Path

import pytest

from aura.data.intents import BANKING77_LABELS
from aura.risk.workflows import (
    WorkflowSpecificationError,
    load_workflow_specification,
    require_frozen_specification,
)

PROJECT_ROOT = Path(__file__).parents[3]
WORKFLOW_SPECIFICATION = PROJECT_ROOT / "config" / "intent_workflows.yaml"


def test_reviewable_specification_covers_canonical_banking77_labels() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    observed_mapping = {workflow.label: workflow.intent for workflow in specification.workflows}

    assert specification.schema_version == 1
    assert observed_mapping == BANKING77_LABELS
    assert len(specification.workflows) == 77
    assert Counter(workflow.review_status for workflow in specification.workflows) == {
        "provisional": 49,
        "reviewed": 28,
    }


def test_provisional_specification_cannot_be_used_for_routing() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    with pytest.raises(WorkflowSpecificationError, match="not frozen"):
        require_frozen_specification(specification)


def test_transaction_review_does_not_overstate_unsupported_consequences() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    workflows = {workflow.label: workflow for workflow in specification.workflows}

    for label in (6, 52):
        assert workflows[label].review_status == "reviewed"
        assert not workflows[label].consequence_flags
        assert not workflows[label].evidence_sources

    for label in (15, 17, 19, 34, 76):
        assert workflows[label].evidence_sources == ("uk_psr_charges_exchange",)


def test_consequence_flag_requires_supporting_evidence(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.yaml"
    path.write_text(
        """\
schema_version: 1
evidence_sources:
  source:
    title: Example source
    url: https://example.com
    supports: Example support
    supported_consequence_flags: [protective_action]
workflows:
  - label: 0
    intent: activate_my_card
    operational_family: card_servicing
    required_action: manage_payment_instrument
    consequence_flags: [protective_action]
    evidence_sources: []
    rationale: Example rationale
    review_status: provisional
""",
        encoding="utf-8",
    )

    with pytest.raises(WorkflowSpecificationError, match="without supporting evidence"):
        load_workflow_specification(path)


def test_unknown_consequence_flag_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unknown-flag.yaml"
    path.write_text(
        """\
schema_version: 1
evidence_sources:
  source:
    title: Example source
    url: https://example.com
    supports: Example support
    supported_consequence_flags: [protective_action]
workflows:
  - label: 0
    intent: activate_my_card
    operational_family: card_servicing
    required_action: manage_payment_instrument
    consequence_flags: [invented_consequence]
    evidence_sources: [source]
    rationale: Example rationale
    review_status: provisional
""",
        encoding="utf-8",
    )

    with pytest.raises(WorkflowSpecificationError, match="unknown consequence flags"):
        load_workflow_specification(path)


def test_evidence_must_support_the_claimed_consequence_flag(tmp_path: Path) -> None:
    path = tmp_path / "unrelated-evidence.yaml"
    path.write_text(
        """\
schema_version: 1
evidence_sources:
  source:
    title: Example source
    url: https://example.com
    supports: Example support
    supported_consequence_flags: [protective_action]
workflows:
  - label: 0
    intent: activate_my_card
    operational_family: card_servicing
    required_action: manage_payment_instrument
    consequence_flags: [compliance_sensitive]
    evidence_sources: [source]
    rationale: Example rationale
    review_status: provisional
""",
        encoding="utf-8",
    )

    with pytest.raises(WorkflowSpecificationError, match="not supported by its evidence"):
        load_workflow_specification(path)
