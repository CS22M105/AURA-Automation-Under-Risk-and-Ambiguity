"""Versioned sensitivity scenarios for qualitative routing consequences."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from aura.risk.consequences import ConsequenceProfile, build_pairwise_consequence_profiles
from aura.risk.workflows import CONSEQUENCE_FLAGS, WorkflowSpecification


class ConsequenceScenarioError(ValueError):
    """Raised when a consequence scenario is invalid or incompatible."""


@dataclass(frozen=True)
class CostScenario:
    """Relative cost floors for one sensitivity scenario."""

    scenario_id: str
    description: str
    base_incorrect_cost: int
    action_mismatch_cost: int
    omitted_consequence_costs: dict[str, int]


@dataclass(frozen=True)
class ConsequenceScenarioSet:
    """A versioned collection of scenarios tied to one workflow specification."""

    schema_version: int
    scenario_set_id: str
    workflow_specification_id: str
    unit: str
    construction: str
    scenarios: dict[str, CostScenario]


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConsequenceScenarioError(f"{context} must be a string-keyed mapping")
    return value


def _require_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConsequenceScenarioError(f"{context} must be a non-blank string")
    return value


def _require_positive_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConsequenceScenarioError(f"{context} must be a positive integer")
    return value


def _parse_scenario(scenario_id: str, value: object) -> CostScenario:
    raw = _require_mapping(value, f"scenario '{scenario_id}'")
    base_cost = _require_positive_int(
        raw.get("base_incorrect_cost"),
        f"scenario '{scenario_id}' base_incorrect_cost",
    )
    action_cost = _require_positive_int(
        raw.get("action_mismatch_cost"),
        f"scenario '{scenario_id}' action_mismatch_cost",
    )
    raw_flag_costs = _require_mapping(
        raw.get("omitted_consequence_costs"),
        f"scenario '{scenario_id}' omitted_consequence_costs",
    )
    if set(raw_flag_costs) != set(CONSEQUENCE_FLAGS):
        raise ConsequenceScenarioError(
            f"scenario '{scenario_id}' must define exactly these consequence flags: "
            f"{sorted(CONSEQUENCE_FLAGS)}"
        )
    flag_costs = {
        flag: _require_positive_int(cost, f"scenario '{scenario_id}' cost for '{flag}'")
        for flag, cost in raw_flag_costs.items()
    }
    if action_cost < base_cost or any(cost < base_cost for cost in flag_costs.values()):
        raise ConsequenceScenarioError(
            f"scenario '{scenario_id}' cost floors must not be below its base cost"
        )
    return CostScenario(
        scenario_id=scenario_id,
        description=_require_string(
            raw.get("description"),
            f"scenario '{scenario_id}' description",
        ),
        base_incorrect_cost=base_cost,
        action_mismatch_cost=action_cost,
        omitted_consequence_costs=flag_costs,
    )


def load_consequence_scenario_set(path: Path) -> ConsequenceScenarioSet:
    """Load and validate a versioned consequence-scenario YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"Consequence scenario file does not exist: {path}")
    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConsequenceScenarioError(f"Consequence scenarios are invalid YAML: {path}") from error

    root = _require_mapping(payload, "consequence scenario set")
    if root.get("schema_version") != 1:
        raise ConsequenceScenarioError("Unsupported consequence scenario schema version")
    construction = _require_string(root.get("construction"), "construction")
    if construction != "maximum_floor":
        raise ConsequenceScenarioError("construction must be 'maximum_floor'")
    raw_scenarios = _require_mapping(root.get("scenarios"), "scenarios")
    if not raw_scenarios:
        raise ConsequenceScenarioError("scenarios must not be empty")
    scenarios = {
        scenario_id: _parse_scenario(scenario_id, raw_scenario)
        for scenario_id, raw_scenario in raw_scenarios.items()
    }
    flat = scenarios.get("flat")
    if flat is None:
        raise ConsequenceScenarioError("scenarios must include the 'flat' control")
    flat_costs = {
        flat.base_incorrect_cost,
        flat.action_mismatch_cost,
        *flat.omitted_consequence_costs.values(),
    }
    if len(flat_costs) != 1:
        raise ConsequenceScenarioError("the 'flat' control must give every error equal cost")

    return ConsequenceScenarioSet(
        schema_version=1,
        scenario_set_id=_require_string(root.get("scenario_set_id"), "scenario_set_id"),
        workflow_specification_id=_require_string(
            root.get("workflow_specification_id"),
            "workflow_specification_id",
        ),
        unit=_require_string(root.get("unit"), "unit"),
        construction=construction,
        scenarios=scenarios,
    )


def calculate_profile_cost(profile: ConsequenceProfile, scenario: CostScenario) -> int:
    """Convert one qualitative profile into a relative scenario cost."""
    if profile.is_correct:
        return 0
    candidate_costs = [scenario.base_incorrect_cost]
    if profile.action_mismatch:
        candidate_costs.append(scenario.action_mismatch_cost)
    candidate_costs.extend(
        scenario.omitted_consequence_costs[flag] for flag in profile.omitted_consequence_flags
    )
    return max(candidate_costs)


def build_cost_matrix(
    specification: WorkflowSpecification,
    scenario_set: ConsequenceScenarioSet,
    scenario_id: str,
) -> NDArray[np.float64]:
    """Build a directed 77-by-77 relative cost matrix for one scenario."""
    if scenario_set.workflow_specification_id != specification.specification_id:
        raise ConsequenceScenarioError(
            "Scenario set targets workflow specification "
            f"'{scenario_set.workflow_specification_id}', not '{specification.specification_id}'"
        )
    try:
        scenario = scenario_set.scenarios[scenario_id]
    except KeyError as error:
        raise ConsequenceScenarioError(f"Unknown consequence scenario: '{scenario_id}'") from error

    profiles = build_pairwise_consequence_profiles(specification)
    matrix = np.zeros(
        (len(specification.workflows), len(specification.workflows)), dtype=np.float64
    )
    for (true_label, predicted_label), profile in profiles.items():
        matrix[true_label, predicted_label] = calculate_profile_cost(profile, scenario)
    return matrix
