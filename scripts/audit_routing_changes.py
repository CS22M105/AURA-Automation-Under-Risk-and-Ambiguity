"""Reconcile confidence/AURA acceptance swaps against the saved development run."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from aura.data.manifests import calculate_sha256
from aura.models.predictions import create_prediction_batch
from aura.policies.coverage import select_at_coverage
from aura.risk.consequences import build_pairwise_consequence_profiles
from aura.risk.expected_risk import compute_expected_routing_risk
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def require(condition: bool, message: str) -> None:
    """Fail an audit explicitly rather than relying on optional Python assertions."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Export all changed rows and an exact decomposition of observed cost changes."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("artifacts/risk/distilbert"))
    parser.add_argument("--policies", type=Path, default=Path("artifacts/policies/distilbert"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audits/routing_changes"))
    args = parser.parse_args()
    policy_summary_path = args.policies / "summary.json"
    metadata = json.loads(policy_summary_path.read_text())
    score_summary_path = args.scores / "summary.json"
    require(
        metadata["source_summary_sha256"] == calculate_sha256(score_summary_path),
        "Score summary changed",
    )
    source = json.loads(score_summary_path.read_text())
    require(source == metadata["source"], "Embedded scoring provenance differs")
    require(
        source["partition"] == "policy_validation" and source["evaluation_stage"] == "development",
        "Expected development policy-validation data",
    )
    paths = {
        "scores": args.scores / "policy_validation_scores.csv",
        "probabilities": args.scores / "policy_validation_probabilities.npz",
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
    }
    for name, path in paths.items():
        require(
            source["provenance"][f"{name}_sha256"] == calculate_sha256(path),
            f"Changed input: {name}",
        )
    for name in ("decisions", "comparison"):
        require(
            metadata[f"{name}_sha256"] == calculate_sha256(args.policies / f"{name}.csv"),
            f"Changed policy artifact: {name}",
        )

    frame = pd.read_csv(paths["scores"])
    decisions = pd.read_csv(args.policies / "decisions.csv")
    comparison = pd.read_csv(args.policies / "comparison.csv")
    require(len(frame) == source["sample_count"] and frame.row_id.is_unique, "Invalid score rows")
    require(decisions.accepted.dtype == bool, "Acceptance column must contain booleans")
    specification = load_workflow_specification(paths["workflows"])
    scenarios = load_consequence_scenario_set(paths["scenarios"])
    profiles = build_pairwise_consequence_profiles(specification)
    matrix = build_cost_matrix(specification, scenarios, "moderate")
    with np.load(paths["probabilities"], allow_pickle=False) as archive:
        require(np.array_equal(archive["row_indices"], frame.row_id), "Probability rows misaligned")
        predictions = create_prediction_batch(
            archive["row_indices"].tolist(), archive["classes"].tolist(), archive["probabilities"]
        )
    require(
        np.array_equal(predictions.predicted_labels, frame.predicted_label), "Predictions differ"
    )
    confidence = predictions.probabilities.max(axis=1)
    risk = compute_expected_routing_risk(predictions, matrix, range(77)).expected_costs
    require(
        bool(np.allclose(confidence, frame.confidence, rtol=0, atol=1e-12)), "Confidence differs"
    )
    require(bool(np.allclose(risk, frame.moderate_risk, rtol=0, atol=1e-12)), "Risk replay differs")
    truth = frame.true_label.to_numpy()
    predicted = frame.predicted_label.to_numpy()
    require(
        truth.dtype.kind in "iu" and bool(np.all((truth >= 0) & (truth < 77))),
        "Invalid truth labels",
    )
    costs = matrix[truth, predicted]
    errors = truth != predicted
    changed_frames = []
    summaries = []
    for coverage in (0.7, 0.8):
        masks = {}
        for policy in ("confidence", "moderate"):
            saved = decisions[
                (decisions.target_coverage == coverage) & (decisions.policy == policy)
            ]
            require(
                len(saved) == len(frame) and saved.row_id.is_unique, "Missing/duplicate decisions"
            )
            saved_mask = saved.set_index("row_id").loc[frame.row_id, "accepted"].to_numpy()
            # Replay with the original saved score precision to preserve tie ordering.
            original = (
                -frame.confidence.to_numpy()
                if policy == "confidence"
                else frame.moderate_risk.to_numpy()
            )
            replay = select_at_coverage(original, frame.row_id.tolist(), coverage)
            require(np.array_equal(replay, saved_mask), f"Selection replay differs: {policy}")
            masks[policy] = replay
            recorded = comparison[
                (comparison.target_coverage == coverage)
                & (comparison.policy == policy)
                & (comparison.evaluation_scenario == "moderate")
            ]
            require(len(recorded) == 1, "Missing comparison row")
            require(
                float(costs[replay].sum()) == recorded.accepted_total_cost.iloc[0]
                and int(errors[replay].sum()) == recorded.accepted_errors.iloc[0]
                and int(replay.sum()) == recorded.accepted_count.iloc[0],
                "Totals differ",
            )
        baseline, aura = masks["confidence"], masks["moderate"]
        groups = {
            "both_accept": baseline & aura,
            "confidence_only": baseline & ~aura,
            "aura_only": aura & ~baseline,
            "both_defer": ~baseline & ~aura,
        }
        require(int(baseline.sum()) == int(aura.sum()), "Unequal coverage budgets")
        totals = {
            name: {
                "messages": int(mask.sum()),
                "errors": int(errors[mask].sum()),
                "cost": float(costs[mask].sum()),
            }
            for name, mask in groups.items()
        }
        delta = float(costs[aura].sum() - costs[baseline].sum())
        require(
            delta == totals["aura_only"]["cost"] - totals["confidence_only"]["cost"],
            "Swap costs fail to reconcile",
        )
        summaries.append(
            {"target_coverage": coverage, "groups": totals, "aura_minus_confidence_cost": delta}
        )
        for group in ("confidence_only", "aura_only"):
            mask = groups[group]
            rows = frame.loc[mask].copy()
            rows["target_coverage"] = coverage
            rows["selection_group"] = group
            rows["observed_moderate_cost"] = costs[mask]
            rows["cost_change_contribution"] = costs[mask] * (1 if group == "aura_only" else -1)
            rows["omitted_flags_given_true_label"] = [
                ",".join(sorted(profiles[(int(i), int(j))].omitted_consequence_flags))
                for i, j in zip(truth[mask], predicted[mask], strict=True)
            ]
            changed_frames.append(rows)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(changed_frames, ignore_index=True).to_csv(
        output / "changed_messages.csv", index=False
    )
    report = {
        "partition": "policy_validation",
        "evaluation_scenario": "moderate",
        "policy_summary_sha256": calculate_sha256(policy_summary_path),
        "changed_messages_sha256": calculate_sha256(output / "changed_messages.csv"),
        "verified": "Input checksums, probability/risk replay, selection replay, totals and swaps",
        "limitation": "Conditional on fixed research costs; neither causal harm nor significance.",
        "comparisons": summaries,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Audit outputs: {output}")


if __name__ == "__main__":
    main()
