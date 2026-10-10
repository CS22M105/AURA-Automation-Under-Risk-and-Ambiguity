"""Compare offline routing policies at matched coverage on saved development scores."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from aura.data.manifests import calculate_sha256
from aura.policies.coverage import select_at_coverage
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def main() -> None:
    """Save all policy/scenario combinations; labels are used only after selection."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/risk/distilbert"))
    parser.add_argument("--workflows", type=Path, default=Path("config/intent_workflows.yaml"))
    parser.add_argument("--scenarios", type=Path, default=Path("config/consequence_scenarios.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/policies/distilbert"))
    parser.add_argument("--coverage", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 1.0])
    args = parser.parse_args()
    score_path = args.input / "policy_validation_scores.csv"
    summary_path = args.input / "summary.json"
    source = json.loads(summary_path.read_text())
    if (
        source.get("partition") != "policy_validation"
        or source.get("evaluation_stage") != "development"
    ):
        raise ValueError("Expected development policy-validation scores")
    for name, path in {
        "scores": score_path,
        "workflows": args.workflows,
        "scenarios": args.scenarios,
    }.items():
        if source["provenance"].get(f"{name}_sha256") != calculate_sha256(path):
            raise ValueError(f"Input provenance mismatch: {name}")
    frame = pd.read_csv(score_path)
    if len(frame) != source["sample_count"]:
        raise ValueError("Score row count does not match source summary")
    specification = load_workflow_specification(args.workflows)
    scenarios = load_consequence_scenario_set(args.scenarios)
    if (
        source["workflow_specification_id"] != specification.specification_id
        or source["scenario_set_id"] != scenarios.scenario_set_id
    ):
        raise ValueError("Specification identifiers do not match scored artifacts")
    ids = frame.row_id.to_numpy()
    truth = frame.true_label.to_numpy()
    predicted = frame.predicted_label.to_numpy()
    for labels in (truth, predicted):
        if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 77)):
            raise ValueError("Invalid BANKING77 labels")
    scores = {"confidence": -frame.confidence.to_numpy()}
    scores.update({name: frame[f"{name}_risk"].to_numpy() for name in scenarios.scenarios})
    realized_costs = {
        name: build_cost_matrix(specification, scenarios, name)[truth, predicted]
        for name in scenarios.scenarios
    }
    comparisons = []
    decisions = []
    for coverage in args.coverage:
        for policy, values in scores.items():
            accepted = select_at_coverage(values, ids, coverage)
            count = int(accepted.sum())
            errors = int(np.sum((truth != predicted) & accepted))
            decisions.append(
                pd.DataFrame(
                    {
                        "row_id": ids,
                        "policy": policy,
                        "target_coverage": coverage,
                        "accepted": accepted,
                    }
                )
            )
            for evaluation_scenario, costs in realized_costs.items():
                total = float(costs[accepted].sum())
                comparisons.append(
                    {
                        "policy": policy,
                        "evaluation_scenario": evaluation_scenario,
                        "target_coverage": coverage,
                        "actual_coverage": count / len(frame),
                        "accepted_count": count,
                        "deferred_count": len(frame) - count,
                        "accepted_errors": errors,
                        "accepted_accuracy": 1 - errors / count if count else None,
                        "accepted_total_cost": total,
                        "cost_per_accepted": total / count if count else None,
                        "accepted_cost_per_input": total / len(frame),
                    }
                )
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(comparisons)
    report.to_csv(output / "comparison.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(output / "decisions.csv", index=False)
    metadata = {
        "schema_version": 1,
        "partition": "policy_validation",
        "evaluation_stage": "development",
        "sample_count": len(frame),
        "target_coverages": args.coverage,
        "selection": "floor budget; ties by ascending row_id",
        "interpretation": (
            "Observed accepted-case costs; excludes human deferral cost and outcomes."
        ),
        "source_summary_sha256": calculate_sha256(summary_path),
        "source": source,
        "comparison_sha256": calculate_sha256(output / "comparison.csv"),
        "decisions_sha256": calculate_sha256(output / "decisions.csv"),
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(report[report.evaluation_scenario == "moderate"].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
