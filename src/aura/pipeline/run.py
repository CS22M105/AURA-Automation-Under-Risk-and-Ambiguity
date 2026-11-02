"""Complete training and evaluation orchestration; no notebook or Drive dependency."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from aura.calibration.temperature import fit_temperature_scaler
from aura.data.intents import BANKING77_LABELS
from aura.data.manifests import calculate_sha256
from aura.evaluation.calibration import compute_calibration_metrics
from aura.evaluation.classification import compute_classification_metrics
from aura.evaluation.final_routing import evaluate_final_decisions, select_final_decisions
from aura.models.baseline import BaselineConfig, build_baseline_pipeline
from aura.models.predictions import PredictionBatch, predict_with_sklearn_pipeline
from aura.models.random_forest import RandomForestConfig, build_random_forest_pipeline
from aura.pipeline.config import TrainingConfig
from aura.pipeline.data import PreparedData
from aura.pipeline.encoders import load_encoder, predict_batch, predict_logits, train_encoder
from aura.policies.encoder_routing import fit_encoder_routing_layer
from aura.policies.lexical import build_lexical_witness
from aura.risk.scenarios import build_cost_matrix, load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification

COVERAGES = [0.5, 0.7, 0.8, 0.9, 1.0]


def checkpoint_digest(directory: Path) -> str:
    """Match the portable checkpoint identity, without importing the demo UI."""
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json", ".txt"}:
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def stage_model(
    model: Pipeline,
    texts: list[str],
    labels: list[int],
    path: Path,
    signature: str,
) -> tuple[Pipeline, dict[str, Any]]:
    metadata_path = path.with_suffix(".json")
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["signature"] != signature or metadata["sha256"] != calculate_sha256(path):
            raise ValueError(f"Saved training stage changed: {path}")
        return joblib.load(path), {**metadata, "reused": True}
    started = time.perf_counter()
    model.fit(texts, labels)
    joblib.dump(model, path)
    metadata = {
        "signature": signature,
        "sha256": calculate_sha256(path),
        "training_seconds": time.perf_counter() - started,
    }
    write_json(metadata_path, metadata)
    return model, {**metadata, "reused": False}


def metrics_row(name: str, frame: pd.DataFrame, batch: PredictionBatch) -> dict[str, Any]:
    classification = compute_classification_metrics(
        frame.label.tolist(), batch.probabilities, batch.classes
    ).to_dict()
    calibration = compute_calibration_metrics(
        frame.label.tolist(), batch.probabilities, batch.classes
    ).to_dict()
    return {
        "model": name,
        **classification,
        "macro_f1": classification["macro_f1"],
        "ece": calibration["expected_calibration_error"],
        "brier": calibration["multiclass_brier_score"],
    }


@dataclass
class TrainingRun:
    run_id: str
    directory: Path
    result_dir: Path
    specification: dict[str, Any]
    encoders: dict[str, Any]
    checkpoints: dict[str, Path]
    classical_models: dict[str, Any]
    witness: Pipeline
    matrices: dict[str, np.ndarray]
    routing_reports: dict[str, pd.DataFrame]
    classifier_reports: dict[str, pd.DataFrame]
    device: str

    @property
    def intents(self) -> list[str]:
        return [BANKING77_LABELS[i] for i in range(77)]

    @property
    def workflows(self) -> list[dict[str, Any]]:
        return sorted(
            self.specification["workflow_specification"]["workflows"],
            key=lambda workflow: workflow["label"],
        )

    def predict(self, messages: list[str], **kwargs: Any) -> np.ndarray:
        return predict_logits(messages, device=self.device, **kwargs)


def run_training(
    root: Path,
    config: TrainingConfig,
    data: PreparedData,
    directory: Path,
    *,
    device: str = "auto",
    resume: bool = False,
    local_files_only: bool = False,
) -> TrainingRun:
    """Train four models, fit AURA on calibration only, and evaluate fixed policies."""
    signature_payload = {
        "configuration": config.to_dict(),
        "input_hashes": {
            "training_content": config.expected_training_fingerprint,
            "split_manifest": calculate_sha256(root / config.split_manifest),
            "workflows": calculate_sha256(root / config.workflows),
            "scenarios": calculate_sha256(root / config.scenarios),
        },
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    run_meta = directory / "run.json"
    if directory.exists():
        if not resume or not run_meta.is_file():
            raise FileExistsError("Choose a new run directory or use --resume for this run.")
        if json.loads(run_meta.read_text())["signature"] != signature:
            raise ValueError("Resume refused: training configuration or input data changed.")
    else:
        directory.mkdir(parents=True)
        write_json(run_meta, {"signature": signature, **signature_payload})
    checkpoints = directory / "checkpoints"
    results = directory / "results"
    checkpoints.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    frames = data.partitions()
    training, calibration = frames["model_training"], frames["calibration"]
    evaluation = {
        name: frame for name, frame in frames.items() if name in {"policy_validation", "test"}
    }
    labels = {
        "policy_validation": "Development validation",
        "test": "Official test (already exposed)",
    }
    classifiers: dict[str, list[dict[str, Any]]] = {part: [] for part in evaluation}
    routing: dict[str, list[pd.DataFrame]] = {part: [] for part in evaluation}
    calibration_rows = []
    records = []
    fitted_classical = {}
    for name, model in {
        "Logistic Regression": build_baseline_pipeline(BaselineConfig(**config.baseline)),
        "Random Forest": build_random_forest_pipeline(RandomForestConfig(**config.random_forest)),
    }.items():
        print(f"Training/loading {name} on {len(training)} development-training rows…", flush=True)
        model, record = stage_model(
            model,
            training.text.tolist(),
            training.label.tolist(),
            checkpoints / (name.lower().replace(" ", "_") + ".joblib"),
            signature,
        )
        fitted_classical[name] = model
        records.append({"model": name, **record})
        scaler = None
        if name == "Logistic Regression":
            batch = predict_with_sklearn_pipeline(
                model, calibration.text.tolist(), calibration.index.tolist()
            )
            assert batch.decision_scores is not None
            scaler = fit_temperature_scaler(
                batch.decision_scores, calibration.label.tolist(), batch.classes
            )
            write_json(results / "logistic_regression_temperature.json", asdict(scaler))
        for partition, frame in evaluation.items():
            started = time.perf_counter()
            batch = predict_with_sklearn_pipeline(model, frame.text.tolist(), frame.index.tolist())
            classifiers[partition].append(
                {
                    **metrics_row(name, frame, batch),
                    "batch_inference_seconds": time.perf_counter() - started,
                }
            )
            if scaler is not None:
                assert batch.decision_scores is not None
                metrics = compute_calibration_metrics(
                    frame.label.tolist(), scaler.transform(batch.decision_scores), batch.classes
                ).to_dict()
                calibration_rows.append(
                    {
                        "encoder": name,
                        "partition": partition,
                        "temperature": scaler.temperature,
                        **metrics,
                    }
                )
    witness, witness_record = stage_model(
        build_lexical_witness(),
        training.text.tolist(),
        training.label.tolist(),
        checkpoints / "lexical_witness.joblib",
        signature,
    )
    records.append({"model": "Lexical witness", **witness_record})
    workflows = load_workflow_specification(root / config.workflows)
    scenarios = load_consequence_scenario_set(root / config.scenarios)
    matrices = {name: build_cost_matrix(workflows, scenarios, name) for name in scenarios.scenarios}
    states, encoder_paths = {}, {}
    for name, encoder_config in config.encoders.items():
        slug = "distilbert" if name == "DistilBERT" else "ettin"
        checkpoint = checkpoints / slug
        stage_meta = checkpoints / f"{slug}_stage.json"
        started = time.perf_counter()
        if checkpoint.exists():
            saved = json.loads(stage_meta.read_text())
            if saved["signature"] != signature or saved["sha256"] != checkpoint_digest(checkpoint):
                raise ValueError(f"Saved {name} checkpoint/configuration changed.")
            model, tokenizer = load_encoder(checkpoint, encoder_config)
            record = {**saved, "reused": True}
        else:
            print(f"Fine-tuning {name}…", flush=True)
            model, tokenizer, history = train_encoder(
                training.text.tolist(),
                training.label.tolist(),
                [BANKING77_LABELS[i] for i in range(77)],
                encoder_config,
                checkpoint,
                device=device,
                local_files_only=local_files_only,
            )
            record = {
                "signature": signature,
                "sha256": checkpoint_digest(checkpoint),
                "training_seconds": time.perf_counter() - started,
                "epoch_losses": history,
            }
            write_json(stage_meta, record)
            record = {**record, "reused": False}
        records.append({"model": name, **record})
        encoder_paths[name] = checkpoint
        options = {
            "model": model,
            "tokenizer": tokenizer,
            "config": encoder_config,
            "device": device,
        }
        cal_batch = predict_batch(calibration.text.tolist(), calibration.index.tolist(), **options)
        layer = fit_encoder_routing_layer(
            name,
            record["sha256"],
            cal_batch,
            calibration.label.tolist(),
            calibration.text.tolist(),
            witness,
        )
        route_dir = results / "routing" / slug
        route_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(layer, route_dir / "fitted_layer.joblib")
        write_json(route_dir / "temperature.json", asdict(layer.temperature))
        states[name] = {
            "slug": slug,
            "model": model,
            "tokenizer": tokenizer,
            "config": asdict(encoder_config),
            "temperature": layer.temperature.temperature,
            "checkpoint_sha256": record["sha256"],
            "gates": {key: layer.gates[key] for key in ["semantic_gate", "signed_gate"]},
        }
        for partition, frame in evaluation.items():
            started = time.perf_counter()
            raw = predict_batch(frame.text.tolist(), frame.index.tolist(), **options)
            assert raw.decision_scores is not None
            elapsed = time.perf_counter() - started
            classifiers[partition].append(
                {**metrics_row(name, frame, raw), "batch_inference_seconds": elapsed}
            )
            calibrated, scores, features = layer.score(
                raw,
                frame.text.tolist(),
                matrices,
                range(77),
                encoder=name,
                checkpoint_sha256=record["sha256"],
            )
            masks = select_final_decisions(scores, frame.index.tolist(), COVERAGES)
            report = evaluate_final_decisions(
                calibrated, frame.label.tolist(), masks, matrices, range(77)
            )
            report.insert(0, "encoder", name)
            report.insert(1, "partition", partition)
            routing[partition].append(report)
            report.to_csv(route_dir / f"comparison_{partition}.csv", index=False)
            pd.DataFrame({"row_id": frame.index, **scores}).to_csv(
                route_dir / f"scores_{partition}.csv", index=False
            )
            np.savez_compressed(
                route_dir / f"predictions_{partition}.npz",
                row_indices=frame.index,
                classes=np.arange(77),
                logits=raw.decision_scores,
                probabilities=calibrated.probabilities,
            )
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "row_id": frame.index,
                            "target_coverage": c,
                            "policy": p,
                            "accepted": accepted,
                        }
                    )
                    for (c, p), accepted in masks.items()
                ]
            ).to_csv(route_dir / f"decisions_{partition}.csv", index=False)
            calibration_rows.append(
                {
                    "encoder": name,
                    "partition": partition,
                    "temperature": layer.temperature.temperature,
                    **compute_calibration_metrics(
                        frame.label.tolist(), calibrated.probabilities, calibrated.classes
                    ).to_dict(),
                }
            )
            if partition == "policy_validation":
                states[name].update(
                    val_probabilities=calibrated.probabilities,
                    val_features=features,
                    val_scores=scores,
                )
    classifier_reports = {labels[p]: pd.DataFrame(rows) for p, rows in classifiers.items()}
    routing_reports = {labels[p]: pd.concat(rows, ignore_index=True) for p, rows in routing.items()}
    for partition in evaluation:
        classifier_reports[labels[partition]].to_csv(
            results / f"classification_{partition}.csv", index=False
        )
        routing_reports[labels[partition]].to_csv(
            results / f"encoder_routing_{partition}.csv", index=False
        )
    pd.DataFrame(calibration_rows).to_csv(results / "calibration.csv", index=False)
    write_json(
        results / "summary.json",
        {
            "run_id": directory.name,
            "origin": "native_python",
            "configuration": config.to_dict(),
            "signature": signature,
            "training_records": records,
            "test_evaluated": data.test is not None,
            "evaluation_stage": "new_exploratory_run",
            "cost_unit": "relative_research_cost",
        },
    )
    return TrainingRun(
        directory.name,
        directory,
        results,
        data.specification,
        states,
        encoder_paths,
        fitted_classical,
        witness,
        matrices,
        routing_reports,
        classifier_reports,
        device,
    )
