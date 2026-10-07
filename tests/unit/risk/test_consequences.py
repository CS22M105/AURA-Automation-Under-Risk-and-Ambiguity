from dataclasses import replace
from pathlib import Path

import pytest

from aura.risk.consequences import (
    ConsequenceProfileError,
    build_consequence_profile,
    build_pairwise_consequence_profiles,
)
from aura.risk.workflows import (
    WorkflowSpecificationError,
    load_workflow_specification,
)

PROJECT_ROOT = Path(__file__).parents[3]
WORKFLOW_SPECIFICATION = PROJECT_ROOT / "config" / "intent_workflows.yaml"


def test_correct_route_has_no_qualitative_gap() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    profile = build_consequence_profile(specification, true_label=22, predicted_label=22)

    assert profile.is_correct
    assert not profile.family_mismatch
    assert not profile.action_mismatch
    assert not profile.omitted_consequence_flags
    assert not profile.additional_route_flags
    assert not profile.has_qualitative_omission


def test_wrong_intents_can_share_the_same_operational_route() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    profile = build_consequence_profile(specification, true_label=16, predicted_label=20)

    assert not profile.is_correct
    assert not profile.family_mismatch
    assert not profile.action_mismatch
    assert not profile.omitted_consequence_flags
    assert not profile.additional_route_flags
    assert not profile.has_qualitative_omission


def test_consequence_profile_is_directional() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    missed_compromise = build_consequence_profile(
        specification,
        true_label=22,
        predicted_label=1,
    )
    unnecessary_security_route = build_consequence_profile(
        specification,
        true_label=1,
        predicted_label=22,
    )

    assert missed_compromise.omitted_consequence_flags == frozenset({"protective_action"})
    assert not missed_compromise.additional_route_flags
    assert not unnecessary_security_route.omitted_consequence_flags
    assert unnecessary_security_route.additional_route_flags == frozenset({"protective_action"})


def test_same_family_can_still_omit_the_required_action() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    profile = build_consequence_profile(specification, true_label=22, predicted_label=44)

    assert not profile.family_mismatch
    assert profile.action_mismatch
    assert profile.omitted_consequence_flags == frozenset({"protective_action"})
    assert profile.has_qualitative_omission


def test_builds_complete_directed_pairwise_space() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    profiles = build_pairwise_consequence_profiles(specification)

    assert len(profiles) == 77 * 77
    assert sum(profile.is_correct for profile in profiles.values()) == 77
    assert profiles[(22, 1)] != profiles[(1, 22)]


def test_rejects_unknown_label() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)

    with pytest.raises(ConsequenceProfileError, match="Unknown BANKING77 labels"):
        build_consequence_profile(specification, true_label=999, predicted_label=1)


def test_rejects_unfrozen_specification() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    changed_workflow = replace(specification.workflows[0], review_status="reviewed")
    changed_specification = replace(
        specification,
        workflows=(changed_workflow, *specification.workflows[1:]),
    )

    with pytest.raises(WorkflowSpecificationError, match="not frozen"):
        build_pairwise_consequence_profiles(changed_specification)
