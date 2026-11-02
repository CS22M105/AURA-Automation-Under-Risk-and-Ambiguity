"""Export one Colab run and load its matching models/results without Colab.

This file and aura_demo_ui.py are copied into every exported package. Nothing
is trained here. Joblib files must come from your own trusted export.
"""

import hashlib
import json
import platform
import shutil
import tempfile
from argparse import ArgumentParser
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aura_demo_ui import (
    CSS,
    MAIN_POLICY_ID,
    MAIN_POLICY_LABEL,
    NotebookDemo,
    build_app,
    light_theme,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_sha256(directory: Path) -> str:
    """Use the same flat checkpoint identity as the Colab notebook."""
    digest = hashlib.sha256()
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json", ".txt"}:
            digest.update(path.name.encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def cost_matrices(specification: dict[str, Any]) -> dict[str, np.ndarray]:
    workflows = sorted(
        specification["workflow_specification"]["workflows"], key=lambda item: item["label"]
    )
    matrices = {}
    for name, scenario in specification["scenario_specification"]["scenarios"].items():
        matrix = np.zeros((len(workflows), len(workflows)), dtype=np.float64)
        for true in workflows:
            for route in workflows:
                if true["label"] == route["label"]:
                    continue
                floors = [scenario["base_incorrect_cost"]]
                if true["required_action"] != route["required_action"]:
                    floors.append(scenario["action_mismatch_cost"])
                omitted = set(true["consequence_flags"]) - set(route["consequence_flags"])
                floors.extend(scenario["omitted_consequence_costs"][flag] for flag in omitted)
                matrix[true["label"], route["label"]] = max(floors)
        matrices[name] = matrix
    return matrices


def lexical_features(logits: Any, witness_scores: Any, classes: Any) -> np.ndarray:
    semantic = np.asarray(logits, dtype=np.float64)
    positions = {int(label): column for column, label in enumerate(classes)}
    if set(positions) != set(range(semantic.shape[1])):
        raise ValueError("Lexical witness classes do not match the encoder.")
    lexical = np.asarray(witness_scores, dtype=np.float64)[
        :, [positions[i] for i in range(semantic.shape[1])]
    ]
    predicted = semantic.argmax(axis=1)
    top = np.partition(semantic, -2, axis=1)[:, -2:]
    routed = lexical[np.arange(len(lexical)), predicted]
    competitors = lexical.copy()
    competitors[np.arange(len(lexical)), predicted] = -np.inf
    return np.column_stack([top[:, 1] - top[:, 0], routed - competitors.max(axis=1)])


def routing_scores(
    probabilities: Any,
    features: Any,
    *,
    gates: Any,
    matrices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Replay all nine existing notebook scores, with no fitting or new policy."""
    p = np.asarray(probabilities, dtype=np.float64)
    predicted = p.argmax(axis=1)
    risks = {name: (p * matrix[:, predicted].T).sum(axis=1) for name, matrix in matrices.items()}
    for gate in gates.values():
        if not np.array_equal(gate.classes_, [0, 1]):
            raise ValueError("Correctness gates must encode 0=error and 1=correct.")
    q_semantic = gates["semantic_gate"].predict_proba(features[:, :1])[:, 0]
    q_signed = gates["signed_gate"].predict_proba(features)[:, 0]
    alternatives = p.copy()
    alternatives[np.arange(len(p)), predicted] = 0
    mass = alternatives.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("Alternative-intent probability mass must be positive.")
    return {
        "confidence": -p.max(axis=1),
        "flat": risks["flat"],
        "original_aura": risks["moderate"],
        "high_protection": risks["high_protection"],
        "guarded_moderate": risks["moderate"],
        "logit_margin": -features[:, 0],
        "semantic_gate": q_semantic,
        "signed_gate": q_signed,
        "signed_cost_gate": q_signed * risks["moderate"] / mass,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_notebook_package(
    namespace: dict[str, Any],
    destination: str | Path,
    *,
    create_zip: bool = True,
) -> tuple[Path, Path | None]:
    """Compatibility adapter for the optional notebook; exporting itself is generic."""
    backend = NotebookDemo.from_namespace(namespace)
    required = ["PROJECT_SPEC", "RESULT_DIR", "classical_models", "COST_MATRICES"]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError("Complete notebook preparation first. Missing: " + ", ".join(missing))
    names = {"DistilBERT": "MODEL_DIR", "Ettin-150M": "ETTIN_MODEL_DIR"}
    checkpoints = {}
    for name in backend.encoders:
        if name not in names or names[name] not in namespace:
            raise ValueError(f"Missing checkpoint directory for {name}.")
        checkpoints[name] = Path(namespace[names[name]])
    return export_demo_package(
        backend,
        destination,
        specification=namespace["PROJECT_SPEC"],
        checkpoints=checkpoints,
        classical_models=namespace["classical_models"],
        result_dir=Path(namespace["RESULT_DIR"]),
        create_zip=create_zip,
        origin="notebook",
    )


def export_demo_package(
    backend: NotebookDemo,
    destination: str | Path,
    *,
    specification: dict[str, Any],
    checkpoints: dict[str, Path],
    classical_models: dict[str, Any],
    result_dir: Path,
    create_zip: bool = True,
    origin: str = "native_python",
) -> tuple[Path, Path | None]:
    """Save a completed training run independently of notebooks or training hardware."""
    destination = Path(destination).expanduser().resolve()
    archive = destination.with_name(destination.name + ".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Choose a new export folder; earlier packages are not overwritten.")
    result_dir = result_dir.resolve()
    if not result_dir.is_dir():
        raise ValueError("This run's result directory is missing.")
    if destination == result_dir or result_dir in destination.parents:
        raise ValueError("Export outside RESULT_DIR to avoid copying an export into itself.")
    matrices = cost_matrices(specification)
    if not np.array_equal(matrices["moderate"], backend.costs):
        raise ValueError("The specification does not match the active cost matrix.")
    encoders = {}
    for name, state in backend.encoders.items():
        if name not in checkpoints:
            raise ValueError(f"Missing checkpoint directory for {name}.")
        source = checkpoints[name]
        if (
            not (source / "config.json").is_file()
            or (not any(source.glob("*.safetensors")) and not any(source.glob("*.bin")))
            or not (source / "tokenizer_config.json").is_file()
            or not any(
                (source / name).is_file()
                for name in ["tokenizer.json", "vocab.txt", "tokenizer.model"]
            )
        ):
            raise ValueError(f"Incomplete fine-tuned checkpoint for {name}.")
        if checkpoint_sha256(source) != state["checkpoint_sha256"]:
            raise ValueError(f"{name} checkpoint differs from its fitted routing layer.")
        replay = routing_scores(
            state["val_probabilities"],
            state["val_features"],
            gates=state["gates"],
            matrices=matrices,
        )
        for policy, saved in state["val_scores"].items():
            if policy not in replay or not np.allclose(
                replay[policy], saved, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"Portable {name} score replay differs for {policy}.")
        encoders[name] = {
            "slug": state["slug"],
            "checkpoint_sha256": state["checkpoint_sha256"],
            "temperature": float(state["temperature"]),
            "config": state["config"],
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".aura-export-", dir=destination.parent))
    try:
        print("Copying this run's complete result directory…", flush=True)
        shutil.copytree(result_dir, staging / "results")
        _write_json(staging / "aura_project_spec.json", specification)
        for source_name in ["aura_demo_ui.py", "aura_demo_package.py"]:
            source = Path(__file__).with_name(source_name)
            shutil.copy2(source, staging / source_name)
        (staging / "routing").mkdir()
        joblib.dump(backend.witness, staging / "routing" / "witness.joblib")
        for name, state in backend.encoders.items():
            slug = state["slug"]
            print(f"Copying fine-tuned {name}, tokenizer and fitted AURA layer…", flush=True)
            shutil.copytree(checkpoints[name], staging / "models" / slug)
            if checkpoint_sha256(staging / "models" / slug) != state["checkpoint_sha256"]:
                raise ValueError(f"{name} checkpoint changed during copying; retry the export.")
            route_dir = staging / "routing" / slug
            route_dir.mkdir()
            joblib.dump(state["gates"], route_dir / "gates.joblib")
            pd.DataFrame(state["val_scores"]).to_csv(
                route_dir / "validation_scores.csv", index=False
            )
            _write_json(route_dir / "calibration.json", encoders[name])
        classical = staging / "models" / "classical"
        classical.mkdir(parents=True)
        for name, model in classical_models.items():
            joblib.dump(model, classical / (name.lower().replace(" ", "_") + ".joblib"))
        report_paths: dict[str, dict[str, str]] = {"routing": {}, "classification": {}}
        for category, reports in [
            ("routing", backend.routing_reports),
            ("classification", backend.classifier_reports),
        ]:
            for index, (label, report) in enumerate(reports.items()):
                path = f"results/portable_{category}_{index}.csv"
                report.to_csv(staging / path, index=False)
                report_paths[category][label] = path
        packages = [
            "torch",
            "transformers",
            "scikit-learn",
            "numpy",
            "scipy",
            "pandas",
            "joblib",
            "gradio",
            "matplotlib",
        ]
        versions = {}
        for name in packages:
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                if name != "gradio":
                    raise
                # Exporting must also work when the optional live-UI install
                # cell was skipped. The local requirements still install it.
                versions[name] = "not_installed_at_export"
        _write_json(
            staging / "environment.json",
            {"python": platform.python_version(), "packages": versions},
        )
        # Preserve sklearn's serialization version, but do not copy a Colab CUDA
        # wheel requirement to macOS or force Colab's Python/pandas wheel versions.
        requirements = [
            "torch>=2.7",
            f"transformers=={versions['transformers']}",
            f"scikit-learn=={versions['scikit-learn']}",
            "numpy>=1.26,<3",
            "scipy>=1.14,<2",
            "pandas>=2.2,<4",
            "joblib>=1.4,<2",
            "matplotlib>=3.8,<4",
        ]
        if versions["transformers"].split(".")[0] == "4":
            requirements.extend(
                ["gradio==6.12.0", "gradio-client==2.4.1", "huggingface-hub==0.36.0"]
            )
        else:
            requirements.extend(["gradio==6.31.0", "gradio-client==2.7.2", "huggingface-hub>=1,<2"])
        (staging / "requirements.txt").write_text("\n".join(requirements) + "\n")
        _write_json(
            staging / "package.json",
            {
                "schema_version": 1,
                "training_origin": origin,
                "run_id": backend.run_id,
                "encoders": encoders,
                "main_policy": MAIN_POLICY_ID,
                "main_policy_name": MAIN_POLICY_LABEL,
                "scoring_scenario": "moderate",
                "reports": report_paths,
                "gate_fit_partition": "calibration",
                "cutoff_partition": "policy_validation",
                "versions": versions,
                "cost_unit": "relative_research_cost",
                "test_previously_exposed": True,
            },
        )
        (staging / "START_HERE.md").write_text(
            "# AURA portable demo\n\nThis package contains one run's fine-tuned models, "
            "tokenizers, fitted calibration/gates, classical models and saved results.\n\n"
            "Use Python 3.12 or 3.13 in a dedicated virtual environment.\n\n"
            "```bash\npython -m pip install -r requirements.txt\n"
            "python aura_demo_package.py --package .\n```\n\n"
            "Open http://127.0.0.1:7860. No retraining, Colab connection or model downloads "
            "are required after dependencies are installed. CPU inference is the default.\n\n"
            "Keep the folder together: gates, temperatures and cutoffs belong to the "
            "included checkpoints. Hashes are checked before loading trusted joblib files. "
            "The original run's environment is recorded in environment.json.\n"
        )
        _write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "file_sha256": {
                    path.relative_to(staging).as_posix(): file_sha256(path)
                    for path in sorted(staging.rglob("*"))
                    if path.is_file()
                },
            },
        )
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if create_zip:
        print("Creating the downloadable ZIP (includes model weights)…", flush=True)
        shutil.make_archive(
            str(destination), "zip", root_dir=destination.parent, base_dir=destination.name
        )
    return destination, archive if create_zip else None


def verify_package(directory: str | Path) -> dict[str, Any]:
    """Check every declared file before any joblib deserialization/model loading."""
    root = Path(directory).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported package manifest version.")
    for relative, expected in manifest["file_sha256"].items():
        path = root / relative
        if (
            root not in path.resolve().parents
            or not path.is_file()
            or file_sha256(path) != expected
        ):
            raise ValueError(f"Package file missing or changed: {relative}")
    metadata: dict[str, Any] = json.loads((root / "package.json").read_text())
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported demo package version.")
    return metadata


def configured_package_directory(config_path: Path | None = None) -> Path | None:
    """Resolve the local folder independently of the terminal working directory."""
    home = Path(__file__).resolve().parent
    if config_path is None:
        config_path = home / "config" / "local_demo.json"
    if config_path.is_file():
        value = json.loads(config_path.read_text()).get("package_directory")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("local_demo.json needs a non-empty package_directory.")
        path = Path(value).expanduser()
        return (path if path.is_absolute() else home / path).resolve()
    # The same launcher is copied inside a self-contained portable package.
    return home if (home / "manifest.json").is_file() else None


def inspect_demo_directory(directory: str | Path) -> dict[str, Any]:
    """Report a portable export or incomplete Drive backup without loading pickle files."""
    root = Path(directory).expanduser().resolve()
    report: dict[str, Any] = {"directory": str(root), "ready": False}
    if not root.is_dir():
        return {**report, "layout": "missing_directory", "missing": [str(root)]}
    if (root / "manifest.json").is_file() and (root / "package.json").is_file():
        try:
            metadata = verify_package(root)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
            return {**report, "layout": "portable_package", "error": str(error)}
        return {
            **report,
            "ready": True,
            "layout": "portable_package",
            "run_id": metadata["run_id"],
            "encoders": list(metadata["encoders"]),
            "integrity_verified": True,
        }
    if not (root / "checkpoints").is_dir():
        return {
            **report,
            "layout": "unrecognized",
            "missing": ["manifest.json", "package.json"],
            "next_step": "Extract the complete portable ZIP created by notebook section 16B.",
        }

    families = []
    for family in sorted((root / "checkpoints").iterdir()):
        if not family.is_dir():
            continue
        record: dict[str, Any] = {"checkpoint_family": family.name, "models": []}
        candidates = [("DistilBERT", family / "distilbert")]
        ettin = family / "ettin"
        if ettin.is_dir():
            candidates.extend(
                ("Ettin-150M", child) for child in sorted(ettin.iterdir()) if child.is_dir()
            )
        for name, model_dir in candidates:
            missing = [
                str((model_dir / filename).relative_to(root))
                for filename in ["config.json", "tokenizer_config.json", "completed.json"]
                if not (model_dir / filename).is_file()
            ]
            if not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("*.bin")):
                missing.append(str(model_dir.relative_to(root) / "model.safetensors"))
            if not any(
                (model_dir / filename).is_file()
                for filename in ["tokenizer.json", "vocab.txt", "tokenizer.model"]
            ):
                missing.append(str(model_dir.relative_to(root) / "tokenizer.json"))
            record["models"].append(
                {
                    "encoder": name,
                    "directory": str(model_dir.relative_to(root)),
                    "checkpoint_files_complete": not missing,
                    "missing": missing,
                }
            )
        record["lexical_bundle_present"] = any(family.glob("lexical_*.joblib"))
        record["ettin_gate_bundle_present"] = any(family.glob("ettin_routing_*.joblib"))
        record["classical_models_present"] = {
            name: (family / filename).is_file()
            for name, filename in [
                ("Logistic Regression", "logistic_regression.joblib"),
                ("Random Forest", "random_forest.joblib"),
            ]
        }
        families.append(record)
    runs = root / "runs"
    run_names = (
        sorted(path.name for path in runs.iterdir() if path.is_dir()) if runs.is_dir() else []
    )
    return {
        **report,
        "layout": "colab_drive_backup",
        "checkpoint_families": families,
        "run_directories": run_names,
        "portable_package_files_missing": [
            name
            for name in [
                "manifest.json",
                "package.json",
                "aura_project_spec.json",
                "requirements.txt",
            ]
            if not (root / name).is_file()
        ],
        "next_step": (
            "Download the complete ZIP from AURA_Colab_Demo/exports/ created by section 16B. "
            "Extract beside this backup, then set config/local_demo.json package_directory "
            "to that exported folder. Do not combine checkpoint families or substitute gates."
        ),
    }


def load_demo_package(directory: str | Path, *, device: str = "cpu") -> NotebookDemo:
    """Load saved encoder weights and gates locally, without fitting or network calls."""
    root = Path(directory).expanduser().resolve()
    if not (root / "manifest.json").is_file():
        raise ValueError(
            f"No portable manifest.json in {root}. This download is not ready for the demo. "
            "Run python aura_demo_package.py --inspect for the inventory. Download the "
            "complete section 16B export ZIP, keeping its matching models and AURA files together."
        )
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    metadata = verify_package(root)
    if version("scikit-learn") != metadata["versions"]["scikit-learn"]:
        raise ValueError(
            f"Saved gates require scikit-learn {metadata['versions']['scikit-learn']}; "
            f"this environment has {version('scikit-learn')}. Install the repository's "
            "requirements.txt for the supported demo. For a different export, use its "
            "requirements.txt in a separate environment."
        )
    specification = json.loads((root / "aura_project_spec.json").read_text())
    matrices = cost_matrices(specification)
    workflows = sorted(
        specification["workflow_specification"]["workflows"], key=lambda item: item["label"]
    )
    intents = [item["intent"] for item in workflows]
    encoders = {}
    for name, config in metadata["encoders"].items():
        checkpoint = root / "models" / config["slug"]
        if checkpoint_sha256(checkpoint) != config["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint identity mismatch: {name}")
        architecture = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        if "reference_compile" in config["config"]:
            architecture.reference_compile = config["config"]["reference_compile"]
        options = {
            key: config["config"][key] for key in ["attn_implementation"] if key in config["config"]
        }
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=architecture,
            local_files_only=True,
            **options,
        ).eval()
        if [model.config.id2label[i] for i in range(len(intents))] != intents:
            raise ValueError(f"Canonical intent mapping mismatch for {name}.")
        route_dir = root / "routing" / config["slug"]
        encoders[name] = {
            **config,
            "model": model,
            "tokenizer": AutoTokenizer.from_pretrained(checkpoint, local_files_only=True),
            "gates": joblib.load(route_dir / "gates.joblib"),
            "val_scores": pd.read_csv(route_dir / "validation_scores.csv").to_dict("list"),
        }

    @torch.inference_mode()
    def predict(
        messages: Any, *, trained_model: Any, trained_tokenizer: Any, config: Any, **kwargs: Any
    ) -> np.ndarray:
        trained_model.eval().to(device)
        try:
            encoded = trained_tokenizer(
                list(messages),
                padding="max_length",
                truncation=True,
                max_length=config["maximum_length"],
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            return np.asarray(
                trained_model(**encoded).logits.detach().cpu().numpy(), dtype=np.float64
            )
        finally:
            trained_model.to("cpu")

    return NotebookDemo(
        encoders,
        intents,
        workflows,
        matrices["moderate"],
        joblib.load(root / "routing" / "witness.joblib"),
        predict,
        lexical_features,
        partial(routing_scores, matrices=matrices),
        {label: pd.read_csv(root / path) for label, path in metadata["reports"]["routing"].items()},
        {
            label: pd.read_csv(root / path)
            for label, path in metadata["reports"]["classification"].items()
        },
        metadata["run_id"],
    )


def main() -> None:
    parser = ArgumentParser(description="Load an exported AURA package and open the local demo.")
    parser.add_argument("--package", type=Path, help="Override config/local_demo.json.")
    parser.add_argument(
        "--inspect", action="store_true", help="Audit files without loading models."
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    package_dir = args.package or configured_package_directory()
    if package_dir is None:
        parser.error("Pass --package or set package_directory in config/local_demo.json.")
    if args.inspect:
        print(json.dumps(inspect_demo_directory(package_dir), indent=2))
        return
    try:
        backend = load_demo_package(package_dir, device=args.device)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    app = build_app(backend)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        css=CSS,
        theme=light_theme(),
        show_error=False,
    )


if __name__ == "__main__":
    main()
