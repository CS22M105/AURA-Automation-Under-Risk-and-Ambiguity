"""Estimate exploratory uncertainty for moderate AURA versus confidence routing."""

import json
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from aura.data.manifests import calculate_sha256
from aura.evaluation.routing_uncertainty import bootstrap_routing_cost
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def main() -> None:
    """Verify provenance, reproduce point estimates, and save bootstrap intervals."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("artifacts/risk/distilbert"))
    parser.add_argument("--policies", type=Path, default=Path("artifacts/policies/distilbert"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audits/routing_uncertainty"))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    policy_path = args.policies / "summary.json"
    score_path = args.scores / "summary.json"
    policy = json.loads(policy_path.read_text())
    source = json.loads(score_path.read_text())
    if (
        policy["source_summary_sha256"] != calculate_sha256(score_path)
        or policy["source"] != source
    ):
        raise ValueError("Scoring provenance differs from policy comparison")
    if source["partition"] != "policy_validation" or source["evaluation_stage"] != "development":
        raise ValueError("Expected development policy-validation data")
    paths = {
        "scores": args.scores / "policy_validation_scores.csv",
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
    }
    for name, path in paths.items():
        if source["provenance"][f"{name}_sha256"] != calculate_sha256(path):
            raise ValueError(f"Changed input: {name}")
    comparison_path = args.policies / "comparison.csv"
    if policy["comparison_sha256"] != calculate_sha256(comparison_path):
        raise ValueError("Changed comparison artifact")
    frame = pd.read_csv(paths["scores"])
    comparison = pd.read_csv(comparison_path)
    if len(frame) != source["sample_count"]:
        raise ValueError("Sample count mismatch")
    truth, predicted = frame.true_label.to_numpy(), frame.predicted_label.to_numpy()
    for labels in (truth, predicted):
        if labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= 77)):
            raise ValueError("Invalid BANKING77 labels")
    matrix = build_cost_matrix(
        load_workflow_specification(paths["workflows"]),
        load_consequence_scenario_set(paths["scenarios"]),
        "moderate",
    )
    results = bootstrap_routing_cost(
        -frame.confidence.to_numpy(),
        frame.moderate_risk.to_numpy(),
        matrix[truth, predicted],
        frame.row_id.to_numpy(),
        policy["target_coverages"],
        n_resamples=args.resamples,
        seed=args.seed,
    )
    for result in results:
        totals = {}
        for name in ("confidence", "moderate"):
            rows = comparison[
                (comparison.policy == name)
                & (comparison.evaluation_scenario == "moderate")
                & (comparison.target_coverage == result.target_coverage)
            ]
            if len(rows) != 1 or rows.accepted_count.iloc[0] != result.accepted_count:
                raise ValueError("Comparison coverage mismatch")
            totals[name] = float(rows.accepted_total_cost.iloc[0])
        expected = (totals["moderate"] - totals["confidence"]) / len(frame)
        if not np.isclose(expected, result.difference, rtol=0, atol=1e-12):
            raise ValueError("Point estimate does not reproduce saved comparison")
    report = pd.DataFrame([asdict(result) for result in results])
    args.output.mkdir(parents=True, exist_ok=True)
    output_csv = args.output / "intervals.csv"
    report.to_csv(output_csv, index=False)
    metadata = {
        "partition": "policy_validation",
        "evaluation_stage": "development",
        "sample_count": len(frame),
        "policy_summary_sha256": calculate_sha256(policy_path),
        "intervals_sha256": calculate_sha256(output_csv),
        "candidate": "moderate",
        "reference": "confidence",
        "evaluation_scenario": "moderate",
        "metric": "candidate minus reference accepted cost per input",
        "method": "paired IID message bootstrap; rerank at equal coverage; percentile intervals",
        "confidence_level": 0.95,
        "n_resamples": args.resamples,
        "seed": args.seed,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "limitations": [
            "Exploratory pointwise intervals; no simultaneous or post-selection adjustment",
            "Conditional on fixed model, calibration, costs, and empirical message distribution",
            "Assumes independent messages; does not model customer or duplicate-text clusters",
            "Excludes review costs, reviewer errors, and uncertainty in consequence assumptions",
            "Offline coverage is reselected; not an evaluation of a frozen deployment threshold",
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(report.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
