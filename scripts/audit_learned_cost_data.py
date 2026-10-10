"""Audit saved development predictions only; never read the official test partition."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.evaluation.cost_data_audit import summarize_cost_support
from aura.risk.consequences import build_pairwise_consequence_profiles
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import CONSEQUENCE_FLAGS, load_workflow_specification


def main() -> None:
    """Preserve classifier-held-out versus selector-held-out distinctions in all outputs."""
    source_path = Path("artifacts/risk/distilbert/summary.json")
    source = json.loads(source_path.read_text())
    lexical_path = Path("artifacts/policies/lexical_distilbert/summary.json")
    lexical = json.loads(lexical_path.read_text())
    if source["partition"] != "policy_validation" or source["evaluation_stage"] != "development":
        raise ValueError("Expected development artifacts")
    if lexical["source_summary_sha256"] != calculate_sha256(source_path):
        raise ValueError("Lexical/scoring provenance mismatch")
    paths = {
        "dataset": Path("datasets/train.csv"),
        "manifest": Path("data/splits/development_seed_42.json"),
        "workflows": Path("config/intent_workflows.yaml"),
        "scenarios": Path("config/consequence_scenarios.yaml"),
        "probabilities": Path("artifacts/risk/distilbert/policy_validation_probabilities.npz"),
    }
    for name, path in paths.items():
        if calculate_sha256(path) != source["provenance"][f"{name}_sha256"]:
            raise ValueError(f"Changed input: {name}")
    for name in ("gate_fit.npz", "scores.csv"):
        path = lexical_path.parent / name
        if calculate_sha256(path) != lexical["output_hashes"][name]:
            raise ValueError(f"Changed lexical input: {name}")
    data = load_banking77_csv(paths["dataset"])
    manifest = load_development_split_manifest(paths["manifest"])
    validate_manifest_source(manifest, paths["dataset"], len(data))
    with np.load(lexical_path.parent / "gate_fit.npz", allow_pickle=False) as archive:
        if not np.array_equal(archive["row_indices"], manifest.calibration):
            raise ValueError("Calibration row alignment mismatch")
        logits = archive["logits"]
        if logits.shape != (len(manifest.calibration), 77) or not np.isfinite(logits).all():
            raise ValueError("Invalid saved calibration logits")
        calibration_predictions = logits.argmax(axis=1)
        correct = (
            calibration_predictions == data.loc[list(manifest.calibration), "label"].to_numpy()
        )
        if not np.array_equal(correct, archive["correct"]):
            raise ValueError("Saved gate correctness does not match predictions")
        fit_features = archive["features"].copy()
        if (
            fit_features.shape != (len(manifest.calibration), 2)
            or not np.isfinite(fit_features).all()
        ):
            raise ValueError("Invalid saved gate features")
    validation_scores = pd.read_csv(lexical_path.parent / "scores.csv")
    if not np.array_equal(validation_scores.row_id, manifest.policy_validation):
        raise ValueError("Validation row alignment mismatch")
    with np.load(paths["probabilities"], allow_pickle=False) as archive:
        if not np.array_equal(
            archive["row_indices"], manifest.policy_validation
        ) or not np.array_equal(archive["classes"], np.arange(77)):
            raise ValueError("Validation probability alignment mismatch")
        validation_predictions = archive["probabilities"].argmax(axis=1)
    if not np.array_equal(validation_scores.predicted_label, validation_predictions):
        raise ValueError("Saved validation predictions disagree")
    normalized = data.text.map(lambda text: " ".join(text.lower().split()))
    train_texts = set(normalized.loc[list(manifest.model_training)])
    spec = load_workflow_specification(paths["workflows"])
    scenarios = load_consequence_scenario_set(paths["scenarios"])
    profiles = build_pairwise_consequence_profiles(spec)
    matrices = {name: build_cost_matrix(spec, scenarios, name) for name in scenarios.scenarios}
    partitions = {
        "calibration": (manifest.calibration, calibration_predictions, fit_features),
        "policy_validation": (
            manifest.policy_validation,
            validation_predictions,
            validation_scores[["semantic_margin", "signed_lexical_support"]].to_numpy(),
        ),
    }
    row_frames = []
    for partition, (ids, predicted, features) in partitions.items():
        truth = data.loc[list(ids), "label"].to_numpy()
        rows = pd.DataFrame(
            {
                "partition": partition,
                "row_id": ids,
                "true_label": truth,
                "predicted_label": predicted,
                "is_error": truth != predicted,
                "semantic_margin": features[:, 0],
                "signed_lexical_support": features[:, 1],
                "normalized_text": normalized.loc[list(ids)].to_numpy(),
            }
        )
        rows["overlaps_model_training"] = rows.normalized_text.isin(train_texts)
        for name, matrix in matrices.items():
            rows[f"{name}_cost"] = matrix[truth, predicted]
        for flag in sorted(CONSEQUENCE_FLAGS):
            rows[f"omits_{flag}"] = [
                flag in profiles[(int(i), int(j))].omitted_consequence_flags
                for i, j in zip(truth, predicted, strict=True)
            ]
        row_frames.append(rows)
    rows = pd.concat(row_frames, ignore_index=True)
    cross = rows.groupby("normalized_text").partition.nunique()
    rows["shared_between_development_partitions"] = rows.normalized_text.map(cross) > 1
    rows["duplicate_development_text_member"] = rows.normalized_text.duplicated(keep=False)
    summary: dict[str, Any] = {}
    for partition in partitions:
        selected = rows[rows.partition == partition]
        support = {
            name: summarize_cost_support(
                selected.true_label.tolist(), selected.predicted_label.tolist(), matrix, range(77)
            )
            for name, matrix in matrices.items()
        }
        summary[partition] = {
            "support": support,
            "overlaps_model_training": int(selected.overlaps_model_training.sum()),
            "overlapping_training_errors": int(
                (selected.overlaps_model_training & selected.is_error).sum()
            ),
            "shared_between_development_partitions": int(
                selected.shared_between_development_partitions.sum()
            ),
            "unique_normalized_texts": int(selected.normalized_text.nunique()),
            "omission_counts": {
                flag: int(selected[f"omits_{flag}"].sum()) for flag in sorted(CONSEQUENCE_FLAGS)
            },
        }
    output = Path("artifacts/audits/learned_cost_data_v1")
    output.mkdir(parents=True, exist_ok=False)
    rows.drop(columns="normalized_text").to_csv(output / "row_support.csv", index=False)
    records = []
    for partition, values in summary.items():
        for scenario, support in values["support"].items():
            for level in support["cost_levels"]:
                records.append({"partition": partition, "scenario": scenario, **level})
    pd.DataFrame(records).to_csv(output / "cost_counts.csv", index=False)
    pairs = (
        rows[rows.is_error]
        .groupby(["partition", "true_label", "predicted_label"])
        .size()
        .rename("messages")
        .reset_index()
    )
    pairs.to_csv(output / "error_pairs.csv", index=False)
    result = {
        "stage": "post_test_exploratory_feasibility_audit",
        "official_test_read": False,
        "source_summary_sha256": calculate_sha256(source_path),
        "lexical_summary_sha256": calculate_sha256(lexical_path),
        "partitions": summary,
        "limitations": [
            "Costs are frozen research assumptions, not measured harm",
            "Calibration is classifier-held-out, but already used for gate and temperature fitting",
            "Policy validation has been repeatedly inspected; it is not fresh confirmation",
            "Counts alone do not establish statistical power or model feasibility",
        ],
        "output_hashes": {
            name: calculate_sha256(output / name)
            for name in ("row_support.csv", "cost_counts.csv", "error_pairs.csv")
        },
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for partition, values in summary.items():
        moderate = values["support"]["moderate"]
        print(partition, json.dumps({k: v for k, v in moderate.items() if k != "per_intent"}))
        print("Omitted flags:", values["omission_counts"])
        print("Training overlaps:", values["overlaps_model_training"])
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
