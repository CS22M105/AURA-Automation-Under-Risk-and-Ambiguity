"""Audit probabilities and consequence assumptions without modifying routing."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from aura.calibration.artifacts import load_calibration_artifact
from aura.data.intents import BANKING77_LABELS
from aura.data.manifests import calculate_sha256
from aura.evaluation.routing_diagnostics import diagnose_routing_probabilities
from aura.models.predictions import create_prediction_batch
from aura.policies.coverage import select_at_coverage
from aura.risk.consequences import build_pairwise_consequence_profiles
from aura.risk.expected_risk import compute_expected_routing_risk
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def main() -> None:
    """Export every message plus every alternative's contribution for error rows."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/risk/distilbert"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/audits/routing_diagnostics"))
    args = parser.parse_args()
    summary_path = args.input / "summary.json"
    source = json.loads(summary_path.read_text())
    if source["partition"] != "policy_validation" or source["evaluation_stage"] != "development":
        raise ValueError("Expected development policy-validation data")
    paths = {
        "scores": args.input / "policy_validation_scores.csv",
        "probabilities": args.input / "policy_validation_probabilities.npz",
        "calibration": Path("artifacts/calibration/distilbert_temperature.json"),
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
    }
    for name, path in paths.items():
        if source["provenance"][f"{name}_sha256"] != calculate_sha256(path):
            raise ValueError(f"Changed input: {name}")
    calibration = load_calibration_artifact(paths["calibration"])
    if calibration.scaler.temperature != source["temperature"]:
        raise ValueError("Calibration temperature mismatch")
    frame = pd.read_csv(paths["scores"])
    with np.load(paths["probabilities"], allow_pickle=False) as archive:
        batch = create_prediction_batch(
            archive["row_indices"].tolist(), archive["classes"].tolist(), archive["probabilities"]
        )
    if (
        len(frame) != source["sample_count"]
        or not np.array_equal(frame.row_id, batch.row_indices)
        or not np.array_equal(frame.predicted_label, batch.predicted_labels)
        or batch.classes != calibration.scaler.classes
        or set(batch.classes) != set(range(77))
    ):
        raise ValueError("Prediction alignment mismatch")
    specification = load_workflow_specification(paths["workflows"])
    scenarios = load_consequence_scenario_set(paths["scenarios"])
    matrix = build_cost_matrix(specification, scenarios, "moderate")
    profiles = build_pairwise_consequence_profiles(specification)
    workflows = {w.label: w for w in specification.workflows}
    diagnostics = diagnose_routing_probabilities(
        batch, frame.true_label.tolist(), calibration.scaler.temperature, matrix, range(77)
    )
    if not np.allclose(
        diagnostics.calibrated_expected_cost, frame.moderate_risk, rtol=0, atol=1e-12
    ):
        raise ValueError("Expected risk replay mismatch")
    if not np.allclose(batch.probabilities.max(axis=1), frame.confidence, rtol=0, atol=1e-12):
        raise ValueError("Confidence replay mismatch")
    rows = frame.merge(diagnostics, on="row_id", validate="one_to_one")
    pairs = [
        profiles[(int(i), int(j))]
        for i, j in zip(frame.true_label, frame.predicted_label, strict=True)
    ]
    rows["required_true_action"] = [workflows[int(i)].required_action for i in frame.true_label]
    rows["predicted_route_action"] = [
        workflows[int(i)].required_action for i in frame.predicted_label
    ]
    rows["action_mismatch"] = [p.action_mismatch for p in pairs]
    rows["omitted_flags"] = [",".join(sorted(p.omitted_consequence_flags)) for p in pairs]
    rows["shared_flags"] = [
        ",".join(
            sorted(
                workflows[p.true_label].consequence_flags
                & workflows[p.predicted_label].consequence_flags
            )
        )
        for p in pairs
    ]
    rows["cost_floor_explanation"] = [
        "correct:0"
        if p.is_correct
        else ";".join(
            [f"base:{scenarios.scenarios['moderate'].base_incorrect_cost}"]
            + (
                [f"action_mismatch:{scenarios.scenarios['moderate'].action_mismatch_cost}"]
                if p.action_mismatch
                else []
            )
            + [
                f"omit_{flag}:{scenarios.scenarios['moderate'].omitted_consequence_costs[flag]}"
                for flag in sorted(p.omitted_consequence_flags)
            ]
        )
        for p in pairs
    ]
    for coverage in (0.5, 0.7, 0.8, 0.9, 1.0):
        for policy, scores in (("confidence", -frame.confidence), ("aura", frame.moderate_risk)):
            rows[f"{policy}_accepted_{int(coverage * 100)}"] = select_at_coverage(
                scores.to_numpy(), frame.row_id.to_numpy(), coverage
            )
    risk = compute_expected_routing_risk(batch, matrix, range(77))
    alternatives = []
    for index in np.flatnonzero(rows.is_error.to_numpy()):
        for column in np.argsort(-risk.contributions[index], kind="stable"):
            label = batch.classes[int(column)]
            alternatives.append(
                {
                    "row_id": batch.row_indices[int(index)],
                    "alternative_true_intent": BANKING77_LABELS[label],
                    "probability": float(batch.probabilities[index, column]),
                    "cost_if_true": float(matrix[label, batch.predicted_labels[index]]),
                    "expected_cost_contribution": float(risk.contributions[index, column]),
                }
            )
    groups = {"all": rows, "errors": rows[rows.is_error], "correct": rows[~rows.is_error]}
    for coverage in (70, 80):
        a, b = rows[f"aura_accepted_{coverage}"], rows[f"confidence_accepted_{coverage}"]
        groups[f"aura_only_errors_{coverage}"] = rows[a & ~b & rows.is_error]
        groups[f"confidence_only_errors_{coverage}"] = rows[b & ~a & rows.is_error]
    aggregates = {}
    for name, group in groups.items():
        aggregates[name] = {
            "count": len(group),
            "raw_mean_nll_reconstructed": float(group.raw_nll_reconstructed.mean())
            if len(group)
            else None,
            "calibrated_mean_nll": float(group.calibrated_nll.mean()) if len(group) else None,
            "true_probability_decreased_count": int((group.true_probability_change < 0).sum()),
            "true_intent_outside_top_3_count": int((group.true_rank > 3).sum()),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output / "messages.csv", index=False)
    pd.DataFrame(alternatives).to_csv(args.output / "error_contributions.csv", index=False)
    report = {
        "partition": "policy_validation",
        "evaluation_stage": "development",
        "source_summary_sha256": calculate_sha256(summary_path),
        "temperature": calibration.scaler.temperature,
        "raw_probability_method": "reconstructed as softmax(T * log(calibrated_probability))",
        "cost_rule": "maximum applicable floor, not sum",
        "limitations": (
            "Descriptive only; no causal failure attribution or calibration verdict from error "
            "subsets. Shared flags do not establish operational equivalence."
        ),
        "aggregates": aggregates,
        "output_hashes": {
            name: calculate_sha256(args.output / name)
            for name in ("messages.csv", "error_contributions.csv")
        },
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(aggregates, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
