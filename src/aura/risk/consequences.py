"""Qualitative consequences of routing one BANKING77 intent as another."""

from dataclasses import dataclass

from aura.risk.workflows import (
    IntentWorkflow,
    WorkflowSpecification,
    require_frozen_specification,
)


class ConsequenceProfileError(ValueError):
    """Raised when a consequence profile cannot be constructed."""


@dataclass(frozen=True)
class ConsequenceProfile:
    """A non-numeric description of one true-intent/predicted-intent pair."""

    true_label: int
    true_intent: str
    predicted_label: int
    predicted_intent: str
    is_correct: bool
    family_mismatch: bool
    action_mismatch: bool
    omitted_consequence_flags: frozenset[str]
    additional_route_flags: frozenset[str]

    @property
    def has_qualitative_omission(self) -> bool:
        """Return whether the predicted route misses an action or consequence dimension."""
        return self.action_mismatch or bool(self.omitted_consequence_flags)


def _create_profile(
    true_workflow: IntentWorkflow,
    predicted_workflow: IntentWorkflow,
) -> ConsequenceProfile:
    return ConsequenceProfile(
        true_label=true_workflow.label,
        true_intent=true_workflow.intent,
        predicted_label=predicted_workflow.label,
        predicted_intent=predicted_workflow.intent,
        is_correct=true_workflow.label == predicted_workflow.label,
        family_mismatch=(true_workflow.operational_family != predicted_workflow.operational_family),
        action_mismatch=true_workflow.required_action != predicted_workflow.required_action,
        omitted_consequence_flags=(
            true_workflow.consequence_flags - predicted_workflow.consequence_flags
        ),
        additional_route_flags=(
            predicted_workflow.consequence_flags - true_workflow.consequence_flags
        ),
    )


def build_consequence_profile(
    specification: WorkflowSpecification,
    true_label: int,
    predicted_label: int,
) -> ConsequenceProfile:
    """Build the qualitative profile for one directed routing outcome."""
    require_frozen_specification(specification)
    workflows = {workflow.label: workflow for workflow in specification.workflows}
    unknown_labels = sorted({true_label, predicted_label} - set(workflows))
    if unknown_labels:
        raise ConsequenceProfileError(f"Unknown BANKING77 labels: {unknown_labels}")
    return _create_profile(workflows[true_label], workflows[predicted_label])


def build_pairwise_consequence_profiles(
    specification: WorkflowSpecification,
) -> dict[tuple[int, int], ConsequenceProfile]:
    """Build all directed profiles for the frozen 77-by-77 routing space."""
    require_frozen_specification(specification)
    workflows = sorted(specification.workflows, key=lambda workflow: workflow.label)
    return {
        (true_workflow.label, predicted_workflow.label): _create_profile(
            true_workflow,
            predicted_workflow,
        )
        for true_workflow in workflows
        for predicted_workflow in workflows
    }
