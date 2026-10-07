"""Loading and validation for evidence-backed banking workflow assignments."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from aura.data.intents import BANKING77_LABELS

OPERATIONAL_FAMILIES: frozenset[str] = frozenset(
    {
        "account_administration",
        "card_servicing",
        "funding_top_up",
        "identity_compliance",
        "routine_information",
        "security_protection",
        "transaction_correction",
        "transfer_servicing",
        "unauthorized_transaction",
    }
)
REQUIRED_ACTIONS: frozenset[str] = frozenset(
    {
        "configure_funding",
        "configure_transfer",
        "fulfil_card_request",
        "investigate_unauthorized_transaction",
        "manage_payment_instrument",
        "process_refund_or_cancellation",
        "provide_information",
        "restore_access_or_instrument",
        "secure_access_or_instrument",
        "support_funding",
        "terminate_account",
        "trace_or_correct_transaction",
        "update_account",
        "verify_identity_or_funds",
    }
)
CONSEQUENCE_FLAGS: frozenset[str] = frozenset(
    {
        "compliance_sensitive",
        "protective_action",
        "transaction_correction",
        "unauthorized_transaction",
    }
)
REVIEW_STATUSES: frozenset[str] = frozenset({"provisional", "reviewed", "frozen"})


class WorkflowSpecificationError(ValueError):
    """Raised when the workflow specification is incomplete or inconsistent."""


@dataclass(frozen=True)
class EvidenceSource:
    """A primary source supporting a consequence assignment."""

    source_id: str
    title: str
    url: str
    supports: str
    supported_consequence_flags: frozenset[str]


@dataclass(frozen=True)
class IntentWorkflow:
    """The operational handling assigned to one BANKING77 intent."""

    label: int
    intent: str
    operational_family: str
    required_action: str
    consequence_flags: frozenset[str]
    evidence_sources: tuple[str, ...]
    rationale: str
    review_status: str


@dataclass(frozen=True)
class SpecificationGovernance:
    """The review basis and limitations attached to a frozen specification."""

    frozen_on: str
    review_basis: str
    external_domain_review: bool
    intended_use: str
    limitation: str


@dataclass(frozen=True)
class WorkflowSpecification:
    """A complete, versioned set of workflow and evidence assignments."""

    schema_version: int
    specification_id: str
    governance: SpecificationGovernance
    evidence_sources: dict[str, EvidenceSource]
    workflows: tuple[IntentWorkflow, ...]


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise WorkflowSpecificationError(f"{context} must be a string-keyed mapping")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowSpecificationError(f"{context} must be a non-blank string")
    return value


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise WorkflowSpecificationError(f"{context} must be a boolean")
    return value


def _require_string_list(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise WorkflowSpecificationError(f"{context} must be a list of non-blank strings")
    if len(set(value)) != len(value):
        raise WorkflowSpecificationError(f"{context} must not contain duplicates")
    return tuple(value)


def _parse_evidence_sources(value: object) -> dict[str, EvidenceSource]:
    raw_sources = _require_mapping(value, "evidence_sources")
    if not raw_sources:
        raise WorkflowSpecificationError("evidence_sources must not be empty")

    sources: dict[str, EvidenceSource] = {}
    for source_id, raw_source in raw_sources.items():
        source = _require_mapping(raw_source, f"evidence source '{source_id}'")
        supported_flags = frozenset(
            _require_string_list(
                source.get("supported_consequence_flags"),
                f"evidence source '{source_id}' supported_consequence_flags",
            )
        )
        unknown_flags = supported_flags - CONSEQUENCE_FLAGS
        if unknown_flags:
            raise WorkflowSpecificationError(
                f"evidence source '{source_id}' supports unknown consequence flags: "
                f"{sorted(unknown_flags)}"
            )
        sources[source_id] = EvidenceSource(
            source_id=source_id,
            title=_require_string(source.get("title"), f"evidence source '{source_id}' title"),
            url=_require_string(source.get("url"), f"evidence source '{source_id}' url"),
            supports=_require_string(
                source.get("supports"),
                f"evidence source '{source_id}' supports",
            ),
            supported_consequence_flags=supported_flags,
        )
    return sources


def _parse_governance(value: object) -> SpecificationGovernance:
    raw = _require_mapping(value, "governance")
    intended_use = _require_string(raw.get("intended_use"), "governance intended_use")
    if intended_use != "research_only":
        raise WorkflowSpecificationError("governance intended_use must be 'research_only'")
    return SpecificationGovernance(
        frozen_on=_require_string(raw.get("frozen_on"), "governance frozen_on"),
        review_basis=_require_string(raw.get("review_basis"), "governance review_basis"),
        external_domain_review=_require_bool(
            raw.get("external_domain_review"),
            "governance external_domain_review",
        ),
        intended_use=intended_use,
        limitation=_require_string(raw.get("limitation"), "governance limitation"),
    )


def _parse_workflow(
    value: object,
    evidence_sources: Mapping[str, EvidenceSource],
) -> IntentWorkflow:
    raw = _require_mapping(value, "workflow entry")
    label = raw.get("label")
    if not isinstance(label, int) or isinstance(label, bool):
        raise WorkflowSpecificationError("workflow label must be an integer")

    intent = _require_string(raw.get("intent"), f"workflow {label} intent")
    family = _require_string(
        raw.get("operational_family"),
        f"workflow {label} operational_family",
    )
    action = _require_string(raw.get("required_action"), f"workflow {label} required_action")
    flags = frozenset(
        _require_string_list(raw.get("consequence_flags"), f"workflow {label} consequence_flags")
    )
    source_ids = _require_string_list(
        raw.get("evidence_sources"),
        f"workflow {label} evidence_sources",
    )
    rationale = _require_string(raw.get("rationale"), f"workflow {label} rationale")
    review_status = _require_string(
        raw.get("review_status"),
        f"workflow {label} review_status",
    )

    if family not in OPERATIONAL_FAMILIES:
        raise WorkflowSpecificationError(f"workflow {label} has unknown family '{family}'")
    if action not in REQUIRED_ACTIONS:
        raise WorkflowSpecificationError(f"workflow {label} has unknown action '{action}'")
    unknown_flags = flags - CONSEQUENCE_FLAGS
    if unknown_flags:
        raise WorkflowSpecificationError(
            f"workflow {label} has unknown consequence flags: {sorted(unknown_flags)}"
        )
    unknown_sources = set(source_ids) - set(evidence_sources)
    if unknown_sources:
        raise WorkflowSpecificationError(
            f"workflow {label} references unknown evidence: {sorted(unknown_sources)}"
        )
    if flags and not source_ids:
        raise WorkflowSpecificationError(
            f"workflow {label} has consequence flags without supporting evidence"
        )
    evidence_supported_flags = frozenset(
        flag
        for source_id in source_ids
        for flag in evidence_sources[source_id].supported_consequence_flags
    )
    unsupported_flags = flags - evidence_supported_flags
    if unsupported_flags:
        raise WorkflowSpecificationError(
            f"workflow {label} has consequence flags not supported by its evidence: "
            f"{sorted(unsupported_flags)}"
        )
    if review_status not in REVIEW_STATUSES:
        raise WorkflowSpecificationError(
            f"workflow {label} has unknown review status '{review_status}'"
        )

    return IntentWorkflow(
        label=label,
        intent=intent,
        operational_family=family,
        required_action=action,
        consequence_flags=flags,
        evidence_sources=source_ids,
        rationale=rationale,
        review_status=review_status,
    )


def load_workflow_specification(path: Path) -> WorkflowSpecification:
    """Load and strictly validate a workflow specification from YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Workflow specification does not exist: {path}")
    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise WorkflowSpecificationError(
            f"Workflow specification is invalid YAML: {path}"
        ) from error

    root = _require_mapping(payload, "workflow specification")
    if root.get("schema_version") != 1:
        raise WorkflowSpecificationError("Unsupported workflow specification schema version")
    evidence_sources = _parse_evidence_sources(root.get("evidence_sources"))
    raw_workflows = root.get("workflows")
    if not isinstance(raw_workflows, list) or not raw_workflows:
        raise WorkflowSpecificationError("workflows must be a non-empty list")
    workflows = tuple(
        _parse_workflow(raw_workflow, evidence_sources) for raw_workflow in raw_workflows
    )

    labels = [workflow.label for workflow in workflows]
    intents = [workflow.intent for workflow in workflows]
    if len(set(labels)) != len(labels):
        raise WorkflowSpecificationError("workflow labels must be unique")
    if len(set(intents)) != len(intents):
        raise WorkflowSpecificationError("workflow intent names must be unique")

    observed_mapping = {workflow.label: workflow.intent for workflow in workflows}
    if observed_mapping != BANKING77_LABELS:
        missing = sorted(set(BANKING77_LABELS) - set(observed_mapping))
        unexpected = sorted(set(observed_mapping) - set(BANKING77_LABELS))
        mismatched = sorted(
            label
            for label in set(observed_mapping).intersection(BANKING77_LABELS)
            if observed_mapping[label] != BANKING77_LABELS[label]
        )
        raise WorkflowSpecificationError(
            "workflow labels do not match canonical BANKING77 mapping; "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )

    specification_id = _require_string(root.get("specification_id"), "specification_id")
    governance = _parse_governance(root.get("governance"))
    return WorkflowSpecification(
        schema_version=1,
        specification_id=specification_id,
        governance=governance,
        evidence_sources=evidence_sources,
        workflows=workflows,
    )


def require_frozen_specification(specification: WorkflowSpecification) -> None:
    """Prevent non-frozen workflow judgments from entering routing experiments."""
    unresolved = [
        workflow.label for workflow in specification.workflows if workflow.review_status != "frozen"
    ]
    if unresolved:
        raise WorkflowSpecificationError(
            f"Workflow specification is not frozen; unresolved labels={unresolved}"
        )
