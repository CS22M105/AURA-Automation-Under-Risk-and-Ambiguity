"""Fit a held-out lexical gate and evaluate frozen-prediction routing variants."""

import json
import warnings
from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.evaluation.routing_uncertainty import bootstrap_routing_cost
from aura.models.distilbert import predict_with_transformer
from aura.models.predictions import create_prediction_batch
from aura.models.transformer_artifacts import (
    calculate_transformer_artifact_sha256,
    load_transformer_artifact,
)
from aura.policies.coverage import select_at_coverage
from aura.policies.lexical import (
    build_lexical_witness,
    cost_aware_gate_score,
    fit_correctness_gate,
    lexical_gate_features,
)
from aura.risk.expected_risk import compute_expected_routing_risk
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification


def main() -> None:
    """Use training for the witness, calibration for the gate, validation for comparison."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/policies/lexical_distilbert")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.resamples < 2:
        raise ValueError("At least two bootstrap resamples required")
    root = Path("artifacts/risk/distilbert")
    source_path = root / "summary.json"
    source = json.loads(source_path.read_text())
    if source["partition"] != "policy_validation" or source["evaluation_stage"] != "development":
        raise ValueError("Expected development policy-validation scores")
    paths = {
        "scores": root / "policy_validation_scores.csv",
        "probabilities": root / "policy_validation_probabilities.npz",
        "dataset": Path("datasets/train.csv"),
        "manifest": Path("data/splits/development_seed_42.json"),
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
    }
    for name, path in paths.items():
        if source["provenance"][f"{name}_sha256"] != calculate_sha256(path):
            raise ValueError(f"Changed input: {name}")
    model_path = Path("artifacts/models/distilbert")
    model_hash = calculate_transformer_artifact_sha256(model_path)
    if model_hash != source["provenance"]["model_artifact_sha256"]:
        raise ValueError("Frozen transformer artifact changed")
    dataset = load_banking77_csv(paths["dataset"])
    manifest = load_development_split_manifest(paths["manifest"])
    validate_manifest_source(manifest, paths["dataset"], len(dataset))
    frame = pd.read_csv(paths["scores"])
    with np.load(paths["probabilities"], allow_pickle=False) as archive:
        batch = create_prediction_batch(
            archive["row_indices"].tolist(), archive["classes"].tolist(), archive["probabilities"]
        )
    evaluation = dataset.loc[list(manifest.policy_validation)]
    if (
        batch.row_indices != manifest.policy_validation
        or not np.array_equal(frame.row_id, batch.row_indices)
        or not np.array_equal(frame.predicted_label, batch.predicted_labels)
        or not np.array_equal(frame.true_label, evaluation.label)
        or not np.array_equal(frame.text, evaluation.text)
        or len(frame) != source["sample_count"]
        or set(batch.classes) != set(range(77))
    ):
        raise ValueError("Evaluation alignment mismatch")
    temperature = float(source["temperature"])
    if not np.isfinite(temperature) or temperature <= 0 or np.any(batch.probabilities <= 0):
        raise ValueError("Cannot reconstruct semantic logit differences")
    semantic_scores = temperature * np.log(batch.probabilities)
    training = dataset.loc[list(manifest.model_training)]
    gate_data = dataset.loc[list(manifest.calibration)]
    print(f"Fitting lexical witness on {len(training)} training messages", flush=True)
    witness = build_lexical_witness()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        witness.fit(training.text.tolist(), training.label.tolist())
    print(f"Inferring frozen DistilBERT on {len(gate_data)} gate-fitting messages", flush=True)
    artifact = load_transformer_artifact(model_path)
    if artifact.metadata.get("dataset_sha256") != manifest.source_sha256 or artifact.metadata.get(
        "manifest_sha256"
    ) != calculate_sha256(paths["manifest"]):
        raise ValueError("Model training provenance mismatch")
    config = artifact.metadata["configuration"]
    gate_predictions = predict_with_transformer(
        artifact.model,
        artifact.tokenizer,
        gate_data.text.tolist(),
        gate_data.index.tolist(),
        maximum_length=int(config["maximum_length"]),
        batch_size=int(config["batch_size"]),
        requested_device=args.device,
    )
    if gate_predictions.decision_scores is None or gate_predictions.classes != batch.classes:
        raise ValueError("Gate-fitting logits/classes mismatch")
    fit_features = lexical_gate_features(
        gate_predictions.decision_scores,
        gate_predictions.classes,
        witness.decision_function(gate_data.text.tolist()),
        witness.classes_,
    )
    features = lexical_gate_features(
        semantic_scores,
        batch.classes,
        witness.decision_function(evaluation.text.tolist()),
        witness.classes_,
    )
    correct = (gate_predictions.predicted_labels == gate_data.label.to_numpy()).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        semantic_gate = fit_correctness_gate(fit_features[:, :1], correct)
        signed_gate = fit_correctness_gate(fit_features, correct)
    q_semantic = np.asarray(semantic_gate.predict_proba(features[:, :1])[:, 0], dtype=np.float64)
    q_signed = np.asarray(signed_gate.predict_proba(features)[:, 0], dtype=np.float64)
    specification = load_workflow_specification(paths["workflows"])
    scenarios = load_consequence_scenario_set(paths["scenarios"])
    matrices = {
        name: build_cost_matrix(specification, scenarios, name) for name in scenarios.scenarios
    }
    original = compute_expected_routing_risk(batch, matrices["moderate"], range(77)).expected_costs
    if not np.allclose(original, frame.moderate_risk, rtol=0, atol=1e-12):
        raise ValueError("Original AURA score replay mismatch")
    if not np.allclose(batch.probabilities.max(axis=1), frame.confidence, rtol=0, atol=1e-12):
        raise ValueError("Confidence replay mismatch")
    scores = {
        "confidence": -frame.confidence.to_numpy(),
        "original_aura": frame.moderate_risk.to_numpy(),
        "logit_margin": -features[:, 0],
        "semantic_gate": q_semantic,
        "signed_gate": q_signed,
        "signed_cost_gate": cost_aware_gate_score(batch, q_signed, matrices["moderate"], range(77)),
    }
    truth = frame.true_label.to_numpy()
    costs = {name: matrix[truth, batch.predicted_labels] for name, matrix in matrices.items()}
    coverages = [0.5, 0.7, 0.8, 0.9, 1.0]
    reports, decisions, intervals = [], [], []
    for coverage in coverages:
        for name, score in scores.items():
            mask = select_at_coverage(score, batch.row_indices, coverage)
            count = int(mask.sum())
            decisions.append(
                pd.DataFrame(
                    {
                        "row_id": batch.row_indices,
                        "policy": name,
                        "target_coverage": coverage,
                        "accepted": mask,
                    }
                )
            )
            for scenario, realized in costs.items():
                total = float(realized[mask].sum())
                reports.append(
                    {
                        "policy": name,
                        "evaluation_scenario": scenario,
                        "target_coverage": coverage,
                        "accepted_count": count,
                        "actual_coverage": count / len(frame),
                        "accepted_errors": int(np.sum(mask & (truth != batch.predicted_labels))),
                        "accepted_total_cost": total,
                        "accepted_cost_per_input": total / len(frame),
                        "cost_per_accepted": total / count,
                    }
                )
    print("Computing paired intervals for signed-cost gate versus original AURA", flush=True)
    for scenario, realized in costs.items():
        for result in bootstrap_routing_cost(
            scores["original_aura"],
            scores["signed_cost_gate"],
            realized,
            batch.row_indices,
            coverages,
            n_resamples=args.resamples,
        ):
            intervals.append({"evaluation_scenario": scenario, **asdict(result)})
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(reports)
    report.to_csv(output / "comparison.csv", index=False)
    pd.concat(decisions, ignore_index=True).to_csv(output / "decisions.csv", index=False)
    pd.DataFrame(intervals).to_csv(output / "intervals.csv", index=False)
    score_frame = frame[["row_id", "true_label", "predicted_label"]].copy()
    for name, score in scores.items():
        score_frame[name] = score
    score_frame["semantic_margin"] = features[:, 0]
    score_frame["signed_lexical_support"] = features[:, 1]
    score_frame.to_csv(output / "scores.csv", index=False)
    joblib.dump(
        {"witness": witness, "semantic_gate": semantic_gate, "signed_gate": signed_gate},
        output / "gates.joblib",
    )
    np.savez_compressed(
        output / "gate_fit.npz",
        row_indices=gate_predictions.row_indices,
        features=fit_features,
        correct=correct,
        logits=gate_predictions.decision_scores,
    )
    report_metadata = {
        "evaluation_stage": "development",
        "partition": "policy_validation",
        "source_summary_sha256": calculate_sha256(source_path),
        "source": source,
        "reference": "https://arxiv.org/abs/2610.00262",
        "training_rows": len(training),
        "gate_fitting_rows": len(gate_data),
        "gate_fitting_errors": int(np.sum(correct == 0)),
        "gate_fitting_partition": "calibration (also used for temperature fitting)",
        "semantic_gate_coefficients": semantic_gate.named_steps["classifier"].coef_.tolist(),
        "signed_gate_coefficients": signed_gate.named_steps["classifier"].coef_.tolist(),
        "signed_gate_intercept": signed_gate.named_steps["classifier"].intercept_.tolist(),
        "witness_parameters": (
            "word 1-2grams/30k; char_wb 3-5grams/50k; min_df=2; sublinear_tf; SVM C=1"
        ),
        "gate_parameters": "StandardScaler + logistic regression C=1; no class weighting or search",
        "seed": 42,
        "bootstrap_resamples": args.resamples,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cost_extension": (
            "signed gate P(error) * original expected cost / alternative probability mass"
        ),
        "limitations": [
            "Adaptation to DistilBERT, not an exact reproduction; source is a 2026 preprint",
            "No independent risk-calibration stage or finite-sample safety guarantee",
            "Learned correctness probability not independently probability-calibrated",
            "Cost extension assumes alternative-intent relative probabilities remain useful",
            "Exploratory after repeated development-data inspection; official test untouched",
            "Pointwise bootstrap conditional on fitted models/gates; no multiplicity adjustment",
        ],
        "output_hashes": {
            name: calculate_sha256(output / name)
            for name in (
                "comparison.csv",
                "decisions.csv",
                "intervals.csv",
                "scores.csv",
                "gates.joblib",
                "gate_fit.npz",
            )
        },
    }
    if calculate_transformer_artifact_sha256(model_path) != model_hash:
        raise ValueError("Transformer files unexpectedly changed")
    (output / "summary.json").write_text(json.dumps(report_metadata, indent=2) + "\n")
    print(report[report.evaluation_scenario == "moderate"].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
