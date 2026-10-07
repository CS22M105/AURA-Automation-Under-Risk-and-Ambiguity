from collections import Counter
from dataclasses import replace
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


def test_frozen_specification_covers_canonical_banking77_labels() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    observed_mapping = {workflow.label: workflow.intent for workflow in specification.workflows}

    assert specification.schema_version == 1
    assert specification.specification_id == "aura-banking77-workflows-v1"
    assert specification.governance.intended_use == "research_only"
    assert not specification.governance.external_domain_review
    assert observed_mapping == BANKING77_LABELS
    assert len(specification.workflows) == 77
    assert Counter(workflow.review_status for workflow in specification.workflows) == {"frozen": 77}


def test_frozen_specification_can_be_used_for_research_routing() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    require_frozen_specification(specification)


def test_later_unfrozen_edit_blocks_routing() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    changed_workflow = replace(specification.workflows[0], review_status="reviewed")
    changed_specification = replace(
        specification,
        workflows=(changed_workflow, *specification.workflows[1:]),
    )

    with pytest.raises(WorkflowSpecificationError, match="not frozen"):
        require_frozen_specification(changed_specification)


def test_rejects_non_research_governance(tmp_path: Path) -> None:
    path = tmp_path / "production-specification.yaml"
    payload = WORKFLOW_SPECIFICATION.read_text(encoding="utf-8").replace(
        "intended_use: research_only",
        "intended_use: production",
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(WorkflowSpecificationError, match="must be 'research_only'"):
        load_workflow_specification(path)


def test_transaction_review_does_not_overstate_unsupported_consequences() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    workflows = {workflow.label: workflow for workflow in specification.workflows}

    for label in (6, 52):
        assert workflows[label].review_status == "frozen"
        assert not workflows[label].consequence_flags
        assert not workflows[label].evidence_sources

    for label in (15, 17, 19, 34, 76):
        assert workflows[label].evidence_sources == ("uk_psr_charges_exchange",)


def test_identity_review_separates_top_up_support_from_due_diligence() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    workflows = {workflow.label: workflow for workflow in specification.workflows}

    top_up = workflows[71]
    assert top_up.operational_family == "funding_top_up"
    assert top_up.required_action == "support_funding"
    assert not top_up.consequence_flags
    assert not top_up.evidence_sources

    why_verify = workflows[74]
    assert why_verify.operational_family == "identity_compliance"
    assert why_verify.required_action == "provide_information"
    assert not why_verify.consequence_flags


def test_funding_review_distinguishes_information_from_unresolved_funds() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    workflows = {workflow.label: workflow for workflow in specification.workflows}

    for label in (56, 57):
        assert workflows[label].required_action == "provide_information"
        assert not workflows[label].consequence_flags
        assert workflows[label].evidence_sources == ("uk_psr_charges_exchange",)

    failed = workflows[59]
    assert failed.required_action == "support_funding"
    assert not failed.consequence_flags
    assert not failed.evidence_sources

    for label in (47, 61):
        assert workflows[label].consequence_flags == frozenset({"transaction_correction"})


def test_card_review_uses_dataset_meaning_for_opaque_intent_names() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    workflows = {workflow.label: workflow for workflow in specification.workflows}

    digital_wallet = workflows[2]
    assert digital_wallet.operational_family == "funding_top_up"
    assert digital_wallet.required_action == "support_funding"

    physical_card = workflows[38]
    assert physical_card.operational_family == "card_servicing"
    assert physical_card.required_action == "manage_payment_instrument"
    assert "PIN" in physical_card.rationale

    frozen_labels = {0, 2, 9, 11, 13, 14, 18, 21, 23, 37, 38, 39, 40, 43, 49, 72}
    assert all(workflows[label].review_status == "frozen" for label in frozen_labels)


def test_routine_information_review_remains_non_consequential() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    routine_workflows = [
        workflow
        for workflow in specification.workflows
        if workflow.operational_family == "routine_information"
    ]

    assert len(routine_workflows) == 15
    assert all(workflow.required_action == "provide_information" for workflow in routine_workflows)
    assert all(not workflow.consequence_flags for workflow in routine_workflows)
    assert all(workflow.review_status == "frozen" for workflow in routine_workflows)


def test_account_and_transfer_review_completes_all_intents() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    workflows = {workflow.label: workflow for workflow in specification.workflows}

    assert workflows[30].required_action == "update_account"
    assert workflows[55].required_action == "terminate_account"

    transfer_fee = workflows[64]
    assert transfer_fee.operational_family == "transfer_servicing"
    assert transfer_fee.evidence_sources == ("uk_psr_charges_exchange",)

    transfer_into_account = workflows[65]
    assert transfer_into_account.operational_family == "funding_top_up"
    assert transfer_into_account.required_action == "support_funding"

    assert all(workflow.review_status == "frozen" for workflow in workflows.values())


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
