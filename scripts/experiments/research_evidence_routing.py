"""Development-only experiment: retrieval evidence and cost-tilted routing."""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from aura.calibration.temperature import fit_temperature_scaler
from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.evaluation.evidence_bootstrap import bootstrap_policy_contrasts
from aura.policies.coverage import select_at_coverage
from aura.policies.evidence import (
    EvidenceSelectors,
    PosteriorPool,
    fit_evidence_selectors,
    fit_posterior_pool,
    fixed_route_risk,
    retrieval_gaps,
)
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification, require_frozen_specification

PRIMARY = "evidence_tilted_cost"
COVERAGES = [0.5, 0.7, 0.8, 0.9, 1.0]
REFERENCES = ["confidence", "signed_cost_gate", "signed_gate", "evidence_gate"]


def new_scores(
    fit_features: np.ndarray,
    eval_features: np.ndarray,
    fit_logits: np.ndarray,
    eval_logits: np.ndarray,
    fit_lexical: np.ndarray,
    eval_lexical: np.ndarray,
    labels: np.ndarray,
    matrix: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, EvidenceSelectors | PosteriorPool]]:
    """Fit all new candidates before receiving any evaluation labels."""
    routes = fit_logits.argmax(axis=1)
    errors = (routes != labels).astype(np.int64)
    costs = matrix[labels, routes]
    two = fit_evidence_selectors(fit_features[:, :2], errors.tolist(), costs)
    four = fit_evidence_selectors(fit_features, errors.tolist(), costs)
    two_scores, four_scores = two.scores(eval_features[:, :2]), four.scores(eval_features)
    pool = fit_posterior_pool(fit_logits, fit_lexical, labels.tolist())
    probabilities = pool.semantic_temperature.transform(eval_logits)
    predicted = eval_logits.argmax(axis=1)
    risk = fixed_route_risk(probabilities, predicted.tolist(), matrix)
    alternatives = probabilities.copy()
    alternatives[np.arange(len(predicted)), predicted] = 0
    alternative_mass = alternatives.sum(axis=1)
    if np.any(alternative_mass <= 0):
        raise ValueError("Conditional severity requires positive alternative mass")
    scores = {
        "refit_signed_gate": two_scores["error"],
        "signed_tilted_cost": two_scores["log_cost"],
        "retrieval_gap": -eval_features[:, 2],
        "evidence_gate": four_scores["error"],
        "evidence_severity_gate": four_scores["error"] * risk / alternative_mass,
        PRIMARY: four_scores["log_cost"],
        "pooled_posterior_risk": fixed_route_risk(
            pool.probabilities(eval_logits, eval_lexical), predicted.tolist(), matrix
        ),
        "refit_signed_cost_gate": two_scores["error"] * risk / alternative_mass,
    }
    return scores, {"two_feature_selectors": two, "four_feature_selectors": four, "pool": pool}


def comparison_rows(
    scores: dict[str, np.ndarray],
    ids: np.ndarray,
    errors: np.ndarray,
    realized: dict[str, np.ndarray],
    *,
    fold: int | None = None,
) -> list[dict[str, object]]:
    """Evaluate masks already determined by scores, without changing rankings."""
    records = []
    for coverage in COVERAGES:
        for name, score in scores.items():
            mask = select_at_coverage(score, ids, coverage)
            for scenario, costs in realized.items():
                count = int(mask.sum())
                total = float(costs[mask].sum())
                records.append(
                    {
                        "policy": name,
                        "evaluation_scenario": scenario,
                        "target_coverage": coverage,
                        "accepted_count": count,
                        "actual_coverage": count / len(ids),
                        "accepted_errors": int(errors[mask].sum()),
                        "accepted_total_cost": total,
                        "accepted_cost_per_input": total / len(ids),
                        "fold": fold,
                        "evaluation_count": len(ids),
                    }
                )
    return records


def main() -> None:
    """Keep official test closed and preserve all new candidates and unfavorable results."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/exploratory/evidence_routing_v1")
    )
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Path to the private, frozen experiment protocol outside Git.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new experiment directory; no overwrites")
    if args.resamples < 2:
        raise ValueError("At least two bootstrap resamples required")
    protocol_path = args.protocol
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Private experiment protocol missing: {protocol_path}")
    audit_path = Path("artifacts/audits/learned_cost_data_v1/summary.json")
    audit = json.loads(audit_path.read_text())
    row_path = audit_path.parent / "row_support.csv"
    if calculate_sha256(row_path) != audit["output_hashes"][row_path.name]:
        raise ValueError("Audited development rows changed")
    source_paths = {
        "risk_summary": Path("artifacts/risk/distilbert/summary.json"),
        "lexical_summary": Path("artifacts/policies/lexical_distilbert/summary.json"),
        "audit_summary": audit_path,
        "audited_rows": row_path,
        "dataset": Path("datasets/train.csv"),
        "manifest": Path("data/splits/development_seed_42.json"),
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
        "protocol": protocol_path,
        "runner": Path(__file__),
        "implementation": Path("src/aura/policies/evidence.py"),
    }
    source = json.loads(source_paths["risk_summary"].read_text())
    lexical = json.loads(source_paths["lexical_summary"].read_text())
    if (
        calculate_sha256(source_paths["risk_summary"]) != audit["source_summary_sha256"]
        or calculate_sha256(source_paths["lexical_summary"]) != audit["lexical_summary_sha256"]
    ):
        raise ValueError("Audited artifact provenance changed")
    if (source["evaluation_stage"], source["partition"]) != ("development", "policy_validation"):
        raise ValueError("Expected development risk artifacts")
    for key in ["dataset", "manifest", "workflows", "scenarios"]:
        if calculate_sha256(source_paths[key]) != source["provenance"][f"{key}_sha256"]:
            raise ValueError(f"Frozen input changed: {key}")
    for filename in ["gates.joblib", "gate_fit.npz", "scores.csv"]:
        path = source_paths["lexical_summary"].parent / filename
        if calculate_sha256(path) != lexical["output_hashes"][filename]:
            raise ValueError(f"Lexical input changed: {filename}")
        source_paths[filename] = path
    pool = load_banking77_csv(source_paths["dataset"])
    manifest = load_development_split_manifest(source_paths["manifest"])
    validate_manifest_source(manifest, source_paths["dataset"], len(pool))
    rows = pd.read_csv(row_path)
    if (
        rows.shared_between_development_partitions.any()
        or rows.duplicate_development_text_member.any()
    ):
        raise ValueError("Grouped development split required")
    eligible = rows[~rows.overlaps_model_training].copy()
    fitting = eligible[eligible.partition == "calibration"].copy()
    evaluation = eligible[eligible.partition == "policy_validation"].copy()
    fit_ids, eval_ids = fitting.row_id.to_numpy(), evaluation.row_id.to_numpy()
    training = pool.iloc[list(manifest.model_training)]
    if (
        set(fit_ids) & set(eval_ids)
        or set(fit_ids) & set(training.index)
        or set(eval_ids) & set(training.index)
        or not set(fit_ids).issubset(manifest.calibration)
        or not set(eval_ids).issubset(manifest.policy_validation)
    ):
        raise ValueError("Fitting/evaluation/training IDs overlap")
    for frame in [fitting, evaluation]:
        if not np.array_equal(frame.true_label, pool.loc[frame.row_id, "label"]):
            raise ValueError("Audited query labels do not match the source dataset")
    gates = joblib.load(source_paths["gates.joblib"])
    witness = gates["witness"]
    if not np.array_equal(witness.classes_, np.arange(77)):
        raise ValueError("Witness classes do not match canonical order")
    with np.load(source_paths["gate_fit.npz"], allow_pickle=False) as saved:
        positions = {int(row): i for i, row in enumerate(saved["row_indices"])}
        fit_logits = saved["logits"][[positions[int(row)] for row in fit_ids]]
    probability_path = Path("artifacts/risk/distilbert/policy_validation_probabilities.npz")
    if calculate_sha256(probability_path) != source["provenance"]["probabilities_sha256"]:
        raise ValueError("Saved validation probabilities changed")
    source_paths["probabilities"] = probability_path
    with np.load(probability_path, allow_pickle=False) as saved:
        positions = {int(row): i for i, row in enumerate(saved["row_indices"])}
        probabilities = saved["probabilities"][[positions[int(row)] for row in eval_ids]]
    temperature_path = Path("artifacts/calibration/distilbert_temperature.json")
    source_paths["temperature"] = temperature_path
    temperature = json.loads(temperature_path.read_text())["temperature"]
    if np.any(probabilities <= 0):
        raise ValueError("Cannot invert temperature scaling with zero probabilities")
    eval_logits = temperature * np.log(probabilities)
    if not np.array_equal(fit_logits.argmax(axis=1), fitting.predicted_label):
        raise ValueError("Calibration predictions do not align")
    if not np.array_equal(eval_logits.argmax(axis=1), evaluation.predicted_label):
        raise ValueError("Evaluation predictions do not align")
    spec = load_workflow_specification(source_paths["workflows"])
    require_frozen_specification(spec)
    scenarios = load_consequence_scenario_set(source_paths["scenarios"])
    matrices = {name: build_cost_matrix(spec, scenarios, name) for name in scenarios.scenarios}
    actions = [workflow.required_action for workflow in spec.workflows]
    fit_text = pool.loc[fit_ids, "text"].tolist()
    eval_text = pool.loc[eval_ids, "text"].tolist()
    transformer = witness.named_steps["features"]
    started = time.perf_counter()
    reference_features = transformer.transform(training.text.tolist())
    fit_gaps = retrieval_gaps(
        transformer.transform(fit_text),
        reference_features,
        training.label.tolist(),
        fit_logits.argmax(axis=1).tolist(),
        actions,
    )
    eval_gaps = retrieval_gaps(
        transformer.transform(eval_text),
        reference_features,
        training.label.tolist(),
        eval_logits.argmax(axis=1).tolist(),
        actions,
    )
    feature_seconds = time.perf_counter() - started
    feature_names = ["semantic_margin", "signed_lexical_support"]
    fit_x = np.column_stack([fitting[feature_names].to_numpy(), fit_gaps])
    eval_x = np.column_stack([evaluation[feature_names].to_numpy(), eval_gaps])
    fit_lexical = witness.decision_function(fit_text)
    eval_lexical = witness.decision_function(eval_text)
    scores, fitted = new_scores(
        fit_x,
        eval_x,
        fit_logits,
        eval_logits,
        fit_lexical,
        eval_lexical,
        fitting.true_label.to_numpy(),
        matrices["moderate"],
    )
    baseline = pd.read_csv(source_paths["scores.csv"]).set_index("row_id").loc[eval_ids]
    for name in ["confidence", "original_aura", "signed_gate", "signed_cost_gate"]:
        scores[name] = baseline[name].to_numpy()
    # Save label-free rankings and masks before reading evaluation labels for metrics.
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ["runner", "implementation", "protocol"]:
        (args.output / source_paths[name].name).write_bytes(source_paths[name].read_bytes())
    protocol_copy = args.output / "protocol.md"
    protocol_copy.write_text(protocol_path.read_text())
    pd.DataFrame({"row_id": eval_ids, **scores}).to_csv(args.output / "scores.csv", index=False)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "row_id": eval_ids,
                    "policy": name,
                    "target_coverage": c,
                    "accepted": select_at_coverage(values, eval_ids, c),
                }
            )
            for c in COVERAGES
            for name, values in scores.items()
        ],
        ignore_index=True,
    ).to_csv(args.output / "decisions.csv", index=False)
    realized = {
        name: matrix[evaluation.true_label.to_numpy(), evaluation.predicted_label.to_numpy()]
        for name, matrix in matrices.items()
    }
    errors = evaluation.is_error.to_numpy()
    report = pd.DataFrame(comparison_rows(scores, eval_ids, errors, realized))
    report.to_csv(args.output / "comparison.csv", index=False)
    print("VALIDATION: moderate, 80%, all candidates (not confirmation)", flush=True)
    print(
        report[(report.target_coverage == 0.8) & (report.evaluation_scenario == "moderate")]
        .drop(columns="fold")
        .to_string(index=False),
        flush=True,
    )
    intervals = bootstrap_policy_contrasts(
        scores,
        realized["moderate"],
        eval_ids.tolist(),
        [(PRIMARY, reference) for reference in REFERENCES],
        resamples=args.resamples,
        family_size=len(REFERENCES),
    )
    intervals.to_csv(args.output / "intervals.csv", index=False)
    print("PRIMARY INTERVALS", intervals.to_string(index=False), flush=True)
    new_candidates = [
        "signed_tilted_cost",
        "retrieval_gap",
        "evidence_gate",
        "evidence_severity_gate",
        PRIMARY,
        "pooled_posterior_risk",
    ]
    diagnostic_contrasts = [
        (candidate, reference)
        for candidate in new_candidates
        for reference in ["confidence", "signed_gate", "signed_cost_gate"]
    ]
    posthoc = bootstrap_policy_contrasts(
        scores,
        realized["moderate"],
        eval_ids.tolist(),
        diagnostic_contrasts,
        resamples=args.resamples,
        family_size=len(diagnostic_contrasts),
    )
    posthoc.to_csv(args.output / "posthoc_intervals.csv", index=False)
    folds = []
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fit_errors = fitting.is_error.to_numpy(dtype=np.int64)
    fit_labels = fitting.true_label.to_numpy()
    for fold, (train_pos, held_pos) in enumerate(split.split(fit_x, fit_errors), 1):
        fold_scores, _ = new_scores(
            fit_x[train_pos],
            fit_x[held_pos],
            fit_logits[train_pos],
            fit_logits[held_pos],
            fit_lexical[train_pos],
            fit_lexical[held_pos],
            fit_labels[train_pos],
            matrices["moderate"],
        )
        scale = fit_temperature_scaler(
            fit_logits[train_pos], fit_labels[train_pos].tolist(), range(77)
        )
        fold_p = scale.transform(fit_logits[held_pos])
        fold_scores["confidence"] = -fold_p.max(axis=1)
        fold_scores["original_aura"] = fixed_route_risk(
            fold_p, fit_logits[held_pos].argmax(axis=1), matrices["moderate"]
        )
        fold_realized = {
            name: matrix[fit_labels[held_pos], fit_logits[held_pos].argmax(axis=1)]
            for name, matrix in matrices.items()
        }
        folds.extend(
            comparison_rows(
                fold_scores, fit_ids[held_pos], fit_errors[held_pos], fold_realized, fold=fold
            )
        )
        print(f"Cross-fitting fold {fold}/5 complete", flush=True)
    crossfit = pd.DataFrame(folds)
    crossfit.to_csv(args.output / "crossfit.csv", index=False)
    joblib.dump(fitted, args.output / "selectors.joblib")
    np.savez_compressed(
        args.output / "features.npz",
        fit_row_ids=fit_ids,
        eval_row_ids=eval_ids,
        fit_features=fit_x,
        eval_features=eval_x,
    )
    summary = {
        "stage": "post_test_exploratory_development",
        "official_test_used": False,
        "primary_candidate": PRIMARY,
        "primary_coverage": 0.8,
        "primary_scenario": "moderate",
        "fitting_count": len(fitting),
        "evaluation_count": len(evaluation),
        "feature_names": feature_names + ["intent_retrieval_gap", "action_retrieval_gap"],
        "feature_seconds": feature_seconds,
        "bootstrap_resamples": args.resamples,
        "pool_lexical_weight": (
            fitted["pool"].lexical_weight if isinstance(fitted["pool"], PosteriorPool) else None
        ),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "input_hashes": {key: calculate_sha256(path) for key, path in source_paths.items()},
        "limitations": [
            "Previously inspected validation; not confirmatory",
            "Scenario costs lack external banking validation",
            "No fitting uncertainty in bootstrap",
            "Only five highest-cost calibration errors",
            "No actual Ettin outputs available locally",
        ],
        "output_hashes": {
            p.name: calculate_sha256(p) for p in args.output.iterdir() if p.is_file()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Saved:", args.output, flush=True)


if __name__ == "__main__":
    main()
