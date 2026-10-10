from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from aura.risk.scenarios import (
    ConsequenceScenarioError,
    build_cost_matrix,
    load_consequence_scenario_set,
)
from aura.risk.workflows import load_workflow_specification

PROJECT_ROOT = Path(__file__).parents[3]
WORKFLOW_SPECIFICATION = PROJECT_ROOT / "config" / "intent_workflows.yaml"
CONSEQUENCE_SCENARIOS = PROJECT_ROOT / "config" / "consequence_scenarios.yaml"


def test_loads_versioned_sensitivity_scenarios() -> None:
    scenario_set = load_consequence_scenario_set(CONSEQUENCE_SCENARIOS)

    assert scenario_set.scenario_set_id == "aura-consequence-scenarios-v1"
    assert scenario_set.workflow_specification_id == "aura-banking77-workflows-v1"
    assert scenario_set.unit == "relative_research_cost"
    assert set(scenario_set.scenarios) == {"flat", "moderate", "high_protection"}


def test_flat_scenario_treats_every_incorrect_intent_equally() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    scenario_set = load_consequence_scenario_set(CONSEQUENCE_SCENARIOS)

    matrix = build_cost_matrix(specification, scenario_set, "flat")

    assert matrix.shape == (77, 77)
    assert np.all(np.diag(matrix) == 0.0)
    assert np.all(matrix[~np.eye(77, dtype=bool)] == 1.0)


def test_moderate_scenario_is_directional_and_route_aware() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    scenario_set = load_consequence_scenario_set(CONSEQUENCE_SCENARIOS)

    matrix = build_cost_matrix(specification, scenario_set, "moderate")

    assert matrix[22, 1] == 5.0
    assert matrix[1, 22] == 2.0
    assert matrix[16, 20] == 1.0
    assert matrix[22, 44] == 5.0


def test_high_protection_scenario_stresses_protective_omissions() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    scenario_set = load_consequence_scenario_set(CONSEQUENCE_SCENARIOS)

    matrix = build_cost_matrix(specification, scenario_set, "high_protection")

    assert matrix[22, 1] == 10.0
    assert matrix[1, 22] == 3.0


def test_rejects_scenario_set_for_another_workflow_version() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    scenario_set = load_consequence_scenario_set(CONSEQUENCE_SCENARIOS)
    incompatible = replace(scenario_set, workflow_specification_id="another-workflow-version")

    with pytest.raises(ConsequenceScenarioError, match="targets workflow specification"):
        build_cost_matrix(specification, incompatible, "moderate")


def test_rejects_unknown_scenario() -> None:
    specification = load_workflow_specification(WORKFLOW_SPECIFICATION)
    scenario_set = load_consequence_scenario_set(CONSEQUENCE_SCENARIOS)

    with pytest.raises(ConsequenceScenarioError, match="Unknown consequence scenario"):
        build_cost_matrix(specification, scenario_set, "invented")


def test_rejects_incomplete_consequence_costs(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.yaml"
    payload = CONSEQUENCE_SCENARIOS.read_text(encoding="utf-8").replace(
        "      compliance_sensitive: 3\n",
        "",
        1,
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ConsequenceScenarioError, match="must define exactly"):
        load_consequence_scenario_set(path)
