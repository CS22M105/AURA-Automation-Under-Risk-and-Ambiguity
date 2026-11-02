"""Fit and compare AURA routing for one exported DistilBERT or Ettin checkpoint."""

import json
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.evaluation.encoder_routing import frame_content_sha256, load_encoder_logits
from aura.evaluation.final_routing import evaluate_final_decisions, select_final_decisions
from aura.policies.encoder_routing import fit_encoder_routing_layer
from aura.policies.lexical import build_lexical_witness
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification, require_frozen_specification


def main() -> None:
    """Fit on development only and save new, encoder-labelled exploratory results."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=["DistilBERT", "Ettin-150M"], required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; previous results are not overwritten")
    pool = load_banking77_csv(Path("datasets/train.csv"))
    manifest = load_development_split_manifest(Path("data/splits/development_seed_42.json"))
    validate_manifest_source(manifest, Path("datasets/train.csv"), len(pool))
    frames = {
        "calibration": pool.iloc[list(manifest.calibration)],
        "policy_validation": pool.iloc[list(manifest.policy_validation)],
    }
    fingerprint = frame_content_sha256(pool)
    exports = {
        name: load_encoder_logits(
            args.predictions / f"{name}.npz",
            frame,
            encoder=args.encoder,
            partition=name,
            training_fingerprint=fingerprint,
        )
        for name, frame in frames.items()
    }
    if args.include_test:
        frames["test"] = load_banking77_csv(Path("datasets/test.csv"))
        exports["test"] = load_encoder_logits(
            args.predictions / "test.npz",
            frames["test"],
            encoder=args.encoder,
            partition="test",
            training_fingerprint=fingerprint,
        )
    identities = {export.checkpoint_sha256 for export in exports.values()}
    if len(identities) != 1:
        raise ValueError("All partition logits must come from the same checkpoint")
    checkpoint = identities.pop()
    spec = load_workflow_specification(Path("config/intent_workflows.yaml"))
    require_frozen_specification(spec)
    scenarios = load_consequence_scenario_set(Path("config/consequence_scenarios.yaml"))
    matrices = {name: build_cost_matrix(spec, scenarios, name) for name in scenarios.scenarios}
    training = pool.iloc[list(manifest.model_training)]
    witness = build_lexical_witness().fit(training.text.tolist(), training.label.to_numpy())
    layer = fit_encoder_routing_layer(
        args.encoder,
        checkpoint,
        exports["calibration"].predictions,
        frames["calibration"].label.tolist(),
        frames["calibration"].text.tolist(),
        witness,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    joblib.dump(layer, args.output / "routing_layer.joblib")
    reports = []
    for partition in frames.keys() - {"calibration"}:
        frame, export = frames[partition], exports[partition]
        calibrated, scores, _ = layer.score(
            export.predictions,
            frame.text.tolist(),
            matrices,
            tuple(range(77)),
            encoder=args.encoder,
            checkpoint_sha256=checkpoint,
        )
        masks = select_final_decisions(scores, calibrated.row_indices, [0.5, 0.7, 0.8, 0.9, 1.0])
        report = evaluate_final_decisions(
            calibrated, frame.label.tolist(), masks, matrices, tuple(range(77))
        )
        report.insert(0, "encoder", args.encoder)
        report.insert(1, "partition", partition)
        report.to_csv(args.output / f"routing_{partition}.csv", index=False)
        pd.DataFrame({"row_id": calibrated.row_indices, **scores}).to_csv(
            args.output / f"scores_{partition}.csv", index=False
        )
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "row_id": calibrated.row_indices,
                        "target_coverage": coverage,
                        "policy": policy,
                        "accepted": accepted,
                    }
                )
                for (coverage, policy), accepted in masks.items()
            ],
            ignore_index=True,
        ).to_csv(args.output / f"decisions_{partition}.csv", index=False)
        np.savez_compressed(
            args.output / f"probabilities_{partition}.npz",
            row_indices=calibrated.row_indices,
            classes=calibrated.classes,
            probabilities=calibrated.probabilities,
        )
        reports.append(report)
    summary = {
        "encoder": args.encoder,
        "checkpoint_sha256": checkpoint,
        "temperature": layer.temperature.temperature,
        "fit_partition": "calibration",
        "fit_rows": len(frames["calibration"]),
        "evaluation_stage": "exploratory_encoder_comparison_not_frozen_final_protocol",
        "test_previously_exposed": True,
        "cost_unit": scenarios.unit,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "training_fingerprint": fingerprint,
        "configuration_sha256": {
            name: calculate_sha256(path)
            for name, path in {
                "manifest": Path("data/splits/development_seed_42.json"),
                "workflows": Path("config/intent_workflows.yaml"),
                "scenarios": Path("config/consequence_scenarios.yaml"),
                "runner": Path(__file__),
            }.items()
        },
        "prediction_exports_sha256": {
            name: calculate_sha256(args.predictions / f"{name}.npz") for name in exports
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    combined = pd.concat(reports, ignore_index=True)
    print(
        combined[
            (combined.evaluation_scenario == "moderate") & (combined.target_coverage == 0.8)
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
