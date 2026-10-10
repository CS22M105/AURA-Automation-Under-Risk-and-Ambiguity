"""Evaluate a fixed majority safeguard without overwriting baseline artifacts."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from aura.data.manifests import calculate_sha256
from aura.policies.coverage import select_at_coverage
from aura.policies.guarded import select_with_majority_guard
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def main() -> None:
    """Report original-budget and realized-budget controls under every fixed scenario."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/risk/distilbert"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/policies/guarded_distilbert")
    )
    args = parser.parse_args()
    summary_path = args.input / "summary.json"
    source = json.loads(summary_path.read_text())
    if source["partition"] != "policy_validation" or source["evaluation_stage"] != "development":
        raise ValueError("Expected development policy-validation scores")
    paths = {
        "scores": args.input / "policy_validation_scores.csv",
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
    }
    for name, path in paths.items():
        if source["provenance"][f"{name}_sha256"] != calculate_sha256(path):
            raise ValueError(f"Changed input: {name}")
    frame = pd.read_csv(paths["scores"])
    if len(frame) != source["sample_count"]:
        raise ValueError("Score count mismatch")
    ids = frame.row_id.to_numpy()
    risk = frame.moderate_risk.to_numpy()
    confidence = frame.confidence.to_numpy()
    specification = load_workflow_specification(paths["workflows"])
    scenarios = load_consequence_scenario_set(paths["scenarios"])
    selections = []
    reports = []
    for coverage in (0.5, 0.7, 0.8, 0.9, 1.0):
        guarded = select_with_majority_guard(risk, confidence, ids, coverage)
        count = int(guarded.sum())
        masks = {"guarded_moderate": guarded}
        for name, scores in (("confidence", -confidence), ("moderate", risk)):
            masks[f"{name}_requested_budget"] = select_at_coverage(scores, ids, coverage)
            matched = np.zeros(len(frame), dtype=np.bool_)
            matched[np.lexsort((ids, scores))[:count]] = True
            masks[f"{name}_matched_budget"] = matched
        # Labels are consulted only after the acceptance sets have been fixed.
        truth, predicted = frame.true_label.to_numpy(), frame.predicted_label.to_numpy()
        for labels in (truth, predicted):
            if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 77)):
                raise ValueError("Invalid BANKING77 labels")
        requested_count = int(masks["moderate_requested_budget"].sum())
        for policy, accepted in masks.items():
            accepted_count = int(accepted.sum())
            errors = int(np.sum(accepted & (truth != predicted)))
            selections.append(
                pd.DataFrame(
                    {
                        "row_id": ids,
                        "policy": policy,
                        "target_coverage": coverage,
                        "accepted": accepted,
                    }
                )
            )
            for scenario in scenarios.scenarios:
                costs = build_cost_matrix(specification, scenarios, scenario)[truth, predicted]
                total = float(costs[accepted].sum())
                reports.append(
                    {
                        "policy": policy,
                        "evaluation_scenario": scenario,
                        "target_coverage": coverage,
                        "requested_count": requested_count,
                        "accepted_count": accepted_count,
                        "actual_coverage": accepted_count / len(frame),
                        "coverage_shortfall_count": requested_count - accepted_count,
                        "accepted_errors": errors,
                        "accepted_total_cost": total,
                        "accepted_cost_per_input": total / len(frame),
                        "cost_per_accepted": total / accepted_count if accepted_count else None,
                        "accepted_accuracy": 1 - errors / accepted_count
                        if accepted_count
                        else None,
                        "changed_vs_original_aura": int(
                            np.sum(accepted != masks["moderate_requested_budget"])
                        ),
                    }
                )
    args.output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(reports)
    report.to_csv(args.output / "comparison.csv", index=False)
    pd.concat(selections, ignore_index=True).to_csv(args.output / "decisions.csv", index=False)
    metadata = {
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "source_summary_sha256": calculate_sha256(summary_path),
        "rule": "confidence > 0.5; ascending moderate risk; ties by original row ID",
        "rationale": "Predicted intent has more probability mass than all alternatives combined",
        "threshold_status": "Structural exploratory rule; not fitted or safety-validated",
        "eligible_count": int(np.sum(confidence > 0.5)),
        "sample_count": len(frame),
        "limitations": [
            "Post-diagnostic development experiment, not an independent confirmatory test",
            "No new uncertainty intervals; previous intervals do not apply to changed policies",
            "Conditional on research scenario costs; excludes human review outcomes and cost",
            "Offline batch ranking, not a frozen deployment threshold",
        ],
        "output_hashes": {
            name: calculate_sha256(args.output / name)
            for name in ("comparison.csv", "decisions.csv")
        },
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(report[report.evaluation_scenario == "moderate"].to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
