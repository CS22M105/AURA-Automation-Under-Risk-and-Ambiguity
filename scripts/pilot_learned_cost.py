"""Post-test exploratory cost-learning pilot using development artifacts only."""

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from aura.data.manifests import calculate_sha256
from aura.evaluation.routing_uncertainty import bootstrap_routing_cost
from aura.policies.coverage import select_at_coverage
from aura.policies.learned_cost import fit_matched_cost_selectors
from aura.policies.lexical import fit_correctness_gate


def main() -> None:
    """Fit only on calibration, evaluate only on previously inspected validation."""
    output = Path("artifacts/exploratory/learned_cost_v1")
    if output.exists():
        raise FileExistsError("Pilot already exists; do not overwrite an experiment")
    audit = Path("artifacts/audits/learned_cost_data_v1")
    audit_path = audit / "summary.json"
    summary = json.loads(audit_path.read_text())
    row_path = audit / "row_support.csv"
    if calculate_sha256(row_path) != summary["output_hashes"]["row_support.csv"]:
        raise ValueError("Audited rows changed")
    source_path = Path("artifacts/risk/distilbert/summary.json")
    lexical_path = Path("artifacts/policies/lexical_distilbert/summary.json")
    for path, key in (
        (source_path, "source_summary_sha256"),
        (lexical_path, "lexical_summary_sha256"),
    ):
        if calculate_sha256(path) != summary[key]:
            raise ValueError("Development provenance changed")
    source = json.loads(source_path.read_text())
    if source["evaluation_stage"] != "development" or source["partition"] != "policy_validation":
        raise ValueError("Expected development-only artifacts")
    for name, relative_path in (
        ("scenarios", "config/consequence_scenarios.yaml"),
        ("workflows", "config/intent_workflows.yaml"),
    ):
        if calculate_sha256(Path(relative_path)) != source["provenance"][f"{name}_sha256"]:
            raise ValueError("Frozen consequence assumptions changed")
    rows = pd.read_csv(row_path)
    for flag in (
        "overlaps_model_training",
        "shared_between_development_partitions",
        "duplicate_development_text_member",
        "is_error",
    ):
        if rows[flag].dtype != bool:
            raise ValueError("Audit flags must be boolean")
    if (
        rows.shared_between_development_partitions.any()
        or rows.duplicate_development_text_member.any()
    ):
        raise ValueError("A grouped data plan is needed before using duplicated development texts")
    if not rows.row_id.is_unique or set(rows.partition) != {"calibration", "policy_validation"}:
        raise ValueError("Invalid audited partition IDs")
    eligible = rows[~rows.overlaps_model_training].copy()
    fitting = eligible[eligible.partition == "calibration"].copy()
    evaluation = eligible[eligible.partition == "policy_validation"].copy()
    if len(fitting) < 2 or len(evaluation) < 2:
        raise ValueError("Both pilot partitions must be nonempty")
    features = ["semantic_margin", "signed_lexical_support"]
    x_fit = fitting[features].to_numpy(dtype=np.float64)
    x_eval = evaluation[features].to_numpy(dtype=np.float64)
    if not np.isfinite(x_eval).all():
        raise ValueError("Invalid evaluation features")
    maximum = summary["partitions"]["calibration"]["support"]["moderate"]["maximum_scenario_cost"]
    # No validation labels enter the fit calls or score calculation.
    models = fit_matched_cost_selectors(
        x_fit, fitting.is_error.to_numpy(dtype=np.int64), fitting.moderate_cost.to_numpy(), maximum
    )
    gate = fit_correctness_gate(x_fit, (~fitting.is_error).astype(int).tolist())
    scores = {
        name: np.asarray(model.predict(x_eval), dtype=np.float64) for name, model in models.items()
    }
    scores["refit_signed_gate"] = np.asarray(gate.predict_proba(x_eval)[:, 0], dtype=np.float64)
    lexical = json.loads(lexical_path.read_text())
    baseline_path = lexical_path.parent / "scores.csv"
    if calculate_sha256(baseline_path) != lexical["output_hashes"]["scores.csv"]:
        raise ValueError("Baseline scores changed")
    baseline = pd.read_csv(baseline_path).set_index("row_id").loc[evaluation.row_id]
    if not np.array_equal(baseline.predicted_label, evaluation.predicted_label):
        raise ValueError("Baseline predictions misaligned")
    scores["confidence"] = baseline.confidence.to_numpy()
    scores["original_aura"] = baseline.original_aura.to_numpy()
    coverages = [0.5, 0.7, 0.8, 0.9, 1.0]
    masks = {
        (coverage, name): select_at_coverage(score, evaluation.row_id.to_numpy(), coverage)
        for coverage in coverages
        for name, score in scores.items()
    }
    reports = []
    for (coverage, name), mask in masks.items():
        for scenario in ("flat", "moderate", "high_protection"):
            total = float(evaluation.loc[mask, f"{scenario}_cost"].sum())
            count = int(mask.sum())
            reports.append(
                {
                    "policy": name,
                    "scenario": scenario,
                    "target_coverage": coverage,
                    "accepted_count": count,
                    "actual_coverage": count / len(evaluation),
                    "accepted_errors": int(evaluation.loc[mask, "is_error"].sum()),
                    "accepted_total_cost": total,
                    "cost_per_input": total / len(evaluation),
                }
            )
    intervals = []
    # Predeclared pilot contrasts at 80%; descriptive, not fresh confirmation.
    for reference in ("error_ridge", "refit_signed_gate"):
        interval = bootstrap_routing_cost(
            scores[reference],
            scores["cost_ridge"],
            evaluation.moderate_cost.to_numpy(),
            evaluation.row_id.tolist(),
            [0.8],
            n_resamples=10000,
            seed=42,
        )[0]
        intervals.append({"candidate": "cost_ridge", "reference": reference, **asdict(interval)})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(reports).to_csv(output / "comparison.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "intervals.csv", index=False)
    saved = evaluation.drop(columns=features).copy()
    for name, score in scores.items():
        saved[f"{name}_score"] = score
    saved.to_csv(output / "scores.csv", index=False)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "row_id": evaluation.row_id.to_numpy(),
                    "target_coverage": coverage,
                    "policy": name,
                    "accepted": mask,
                }
            )
            for (coverage, name), mask in masks.items()
        ],
        ignore_index=True,
    ).to_csv(output / "decisions.csv", index=False)
    joblib.dump({**models, "refit_signed_gate": gate}, output / "selectors.joblib")
    metadata = {
        "stage": "post_test_exploratory",
        "official_test_used": False,
        "audit_summary_sha256": calculate_sha256(audit_path),
        "features": features,
        "fitting_row_ids": fitting.row_id.tolist(),
        "evaluation_row_ids": evaluation.row_id.tolist(),
        "fitting_count": len(fitting),
        "evaluation_count": len(evaluation),
        "model": "StandardScaler + Ridge(alpha=1), no tuning; unbounded, unclipped ranking scores",
        "cost_target": "moderate scenario cost divided by its frozen maximum",
        "maximum_cost": maximum,
        "coefficients": {
            name: model.named_steps["regressor"].coef_.tolist() for name, model in models.items()
        },
        "out_of_unit_range_counts": {
            name: int(np.sum((scores[name] < 0) | (scores[name] > 1))) for name in models
        },
        "sklearn_version": sklearn.__version__,
        "limitations": [
            "Development data were already inspected; no confirmatory claims",
            "Only five maximum-cost calibration errors; no new independent data",
            "Pointwise bootstrap conditional on fitted selectors; no training uncertainty",
            "Refit signed gate uses exactly the same overlap-filtered fitting rows",
        ],
        "output_hashes": {p.name: calculate_sha256(p) for p in output.iterdir() if p.is_file()},
    }
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = pd.DataFrame(reports)
    print(report[report.scenario == "moderate"].to_string(index=False))
    print(pd.DataFrame(intervals).to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
