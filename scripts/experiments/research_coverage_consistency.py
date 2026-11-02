"""Test calibration-only coverage-robust score blending; keep official test closed."""

import json
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from research_evidence_routing import comparison_rows, new_scores
from sklearn.model_selection import StratifiedKFold

from aura.data.loading import load_banking77_csv
from aura.data.manifests import calculate_sha256
from aura.evaluation.evidence_bootstrap import bootstrap_policy_contrasts
from aura.policies.coverage import select_at_coverage
from aura.policies.coverage_robust import CoverageBlend, select_coverage_blend
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification, require_frozen_specification

COMPONENTS = ["refit_signed_cost_gate", "evidence_severity_gate", "pooled_posterior_risk"]
FAMILIES = {
    "robust_evidence_blend": [0, 1],
    "robust_pool_blend": [0, 2],
    "robust_threeway_blend": [0, 1, 2],
}


def main() -> None:
    """Select weights from calibration folds, then report every validation outcome."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/exploratory/evidence_routing_v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/exploratory/coverage_consistency_v1")
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Path to the private, frozen experiment protocol outside Git.",
    )
    args = parser.parse_args()
    if not args.protocol.is_file():
        raise FileNotFoundError(f"Private experiment protocol missing: {args.protocol}")
    if args.output.exists():
        raise FileExistsError("Do not overwrite an earlier consistency experiment")
    summary_path = args.source / "summary.json"
    summary = json.loads(summary_path.read_text())
    for name in ["scores.csv", "features.npz"]:
        if calculate_sha256(args.source / name) != summary["output_hashes"][name]:
            raise ValueError(f"Prior experiment changed: {name}")
    audited = Path("artifacts/audits/learned_cost_data_v1/row_support.csv")
    if calculate_sha256(audited) != summary["input_hashes"]["audited_rows"]:
        raise ValueError("Audited development rows changed")
    rows = pd.read_csv(audited).set_index("row_id")
    with np.load(args.source / "features.npz", allow_pickle=False) as saved:
        fit_ids, eval_ids = saved["fit_row_ids"], saved["eval_row_ids"]
        fit_features = saved["fit_features"]
    fitting = rows.loc[fit_ids]
    if set(fitting.partition) != {"calibration"} or fitting.overlaps_model_training.any():
        raise ValueError("Only eligible calibration rows may choose mixture weights")
    logits_path = Path("artifacts/policies/lexical_distilbert/gate_fit.npz")
    if calculate_sha256(logits_path) != summary["input_hashes"]["gate_fit.npz"]:
        raise ValueError("Calibration logits changed")
    with np.load(logits_path, allow_pickle=False) as saved:
        positions = {int(row): i for i, row in enumerate(saved["row_indices"])}
        logits = saved["logits"][[positions[int(row)] for row in fit_ids]]
    # Lexical scores are available from the trusted training-only witness.
    gates_path = Path("artifacts/policies/lexical_distilbert/gates.joblib")
    if calculate_sha256(gates_path) != summary["input_hashes"]["gates.joblib"]:
        raise ValueError("Training-only witness changed")
    dataset = Path("datasets/train.csv")
    if calculate_sha256(dataset) != summary["input_hashes"]["dataset"]:
        raise ValueError("Training-pool content/order changed")
    pool = load_banking77_csv(dataset)
    witness = joblib.load(gates_path)["witness"]
    lexical = witness.decision_function(pool.loc[fit_ids, "text"].tolist())
    workflow_path, scenario_path = (
        Path("config/intent_workflows.yaml"),
        Path("config/consequence_scenarios.yaml"),
    )
    for key, path in [("workflows", workflow_path), ("scenarios", scenario_path)]:
        if calculate_sha256(path) != summary["input_hashes"][key]:
            raise ValueError("Declared cost assumptions changed")
    specification = load_workflow_specification(workflow_path)
    require_frozen_specification(specification)
    scenarios = load_consequence_scenario_set(scenario_path)
    matrices = {
        name: build_cost_matrix(specification, scenarios, name) for name in scenarios.scenarios
    }
    labels, error_labels = fitting.true_label.to_numpy(), fitting.is_error.to_numpy(dtype=int)
    fold_ids = np.full(len(fitting), -1, dtype=np.int64)
    oof = np.full((len(fitting), len(COMPONENTS)), np.nan)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train, held) in enumerate(splitter.split(fit_features, error_labels)):
        predictions, _ = new_scores(
            fit_features[train],
            fit_features[held],
            logits[train],
            logits[held],
            lexical[train],
            lexical[held],
            labels[train],
            matrices["moderate"],
        )
        oof[held] = np.column_stack([predictions[name] for name in COMPONENTS])
        fold_ids[held] = fold
        print(f"Calibration-only out-of-fold predictions: {fold + 1}/5", flush=True)
    fit_costs = matrices["moderate"][labels, logits.argmax(axis=1)]
    args.output.mkdir(parents=True, exist_ok=False)
    learned: dict[str, tuple[CoverageBlend, list[str]]] = {}
    for name, columns in FAMILIES.items():
        blend, grid = select_coverage_blend(
            oof[:, columns], fit_costs, fit_ids.tolist(), fold_ids.tolist()
        )
        learned[name] = (blend, [COMPONENTS[i] for i in columns])
        grid.to_csv(args.output / f"calibration_grid_{name}.csv", index=False)
    weights = {
        name: {"components": components, "weights": blend.weights}
        for name, (blend, components) in learned.items()
    }
    (args.output / "selected_weights.json").write_text(json.dumps(weights, indent=2) + "\n")
    print("Weights fixed before validation metrics:", weights, flush=True)
    previous_scores = pd.read_csv(args.source / "scores.csv").set_index("row_id").loc[eval_ids]
    scores = {name: previous_scores[name].to_numpy() for name in previous_scores}
    for name, (blend, components) in learned.items():
        scores[name] = blend.score(previous_scores[components].to_numpy())
    pd.DataFrame({"row_id": eval_ids, **scores}).to_csv(args.output / "scores.csv", index=False)
    masks = {
        (coverage, name): select_at_coverage(score, eval_ids, coverage)
        for coverage in [0.5, 0.7, 0.8, 0.9, 1.0]
        for name, score in scores.items()
    }
    pd.concat(
        [
            pd.DataFrame(
                {"row_id": eval_ids, "coverage": coverage, "policy": name, "accepted": mask}
            )
            for (coverage, name), mask in masks.items()
        ],
        ignore_index=True,
    ).to_csv(args.output / "decisions.csv", index=False)
    evaluation = rows.loc[eval_ids]
    if set(evaluation.partition) != {"policy_validation"}:
        raise ValueError("Expected development policy validation only")
    realized = {
        name: matrix[evaluation.true_label.to_numpy(), evaluation.predicted_label.to_numpy()]
        for name, matrix in matrices.items()
    }
    report = pd.DataFrame(
        comparison_rows(scores, eval_ids, evaluation.is_error.to_numpy(), realized)
    )
    report.to_csv(args.output / "comparison.csv", index=False)
    selected = report[
        (report.evaluation_scenario == "moderate")
        & report.policy.isin(
            ["confidence", "signed_cost_gate", "evidence_severity_gate", *FAMILIES]
        )
    ]
    print(
        selected.pivot(
            index="policy", columns="target_coverage", values="accepted_total_cost"
        ).to_string(),
        flush=True,
    )
    references = ["confidence", "signed_cost_gate", "evidence_severity_gate"]
    intervals = bootstrap_policy_contrasts(
        scores,
        realized["moderate"],
        eval_ids.tolist(),
        [("robust_evidence_blend", reference) for reference in references],
        coverage=0.9,
        resamples=args.resamples,
        family_size=3,
    )
    intervals.to_csv(args.output / "intervals_90.csv", index=False)
    print(intervals.to_string(index=False), flush=True)
    band_records = []
    for name in ["confidence", "signed_cost_gate", "evidence_severity_gate", *FAMILIES]:
        band = masks[(0.9, name)] & ~masks[(0.8, name)]
        band_records.append(
            {
                "policy": name,
                "additional_accepted": int(band.sum()),
                "additional_errors": int(evaluation.is_error.to_numpy()[band].sum()),
                "additional_moderate_cost": float(realized["moderate"][band].sum()),
            }
        )
    pd.DataFrame(band_records).to_csv(args.output / "acceptance_band_80_90.csv", index=False)
    protocol = args.protocol
    (args.output / "protocol.md").write_text(protocol.read_text())
    metadata = {
        "stage": "post_test_exploratory_development",
        "official_test_used": False,
        "primary_family": "robust_evidence_blend",
        "primary_coverage": 0.9,
        "fit_rows": len(fit_ids),
        "evaluation_rows": len(eval_ids),
        "selected_weights": weights,
        "source_summary_sha256": calculate_sha256(summary_path),
        "protocol_sha256": calculate_sha256(protocol),
        "implementation_sha256": calculate_sha256(Path("src/aura/policies/coverage_robust.py")),
        "runner_sha256": calculate_sha256(Path(__file__)),
        "bootstrap_resamples": args.resamples,
        "output_hashes": {
            p.name: calculate_sha256(p) for p in args.output.iterdir() if p.is_file()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
