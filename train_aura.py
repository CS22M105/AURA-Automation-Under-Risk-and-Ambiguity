"""Train all four classifiers, fit AURA and create a demo package on this machine."""

import json
from argparse import ArgumentParser
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from aura.pipeline.config import load_training_config
from aura.pipeline.data import prepare_data
from aura.pipeline.run import run_training
from aura_demo_package import export_demo_package, lexical_features, routing_scores
from aura_demo_ui import NotebookDemo


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/training.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--package", type=Path, help="New destination; existing exports are preserved."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--epochs", type=int, help="Override both encoder budgets for a new run.")
    parser.add_argument(
        "--include-test", action="store_true", help="Also evaluate the exposed official test set."
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Fetch missing BANKING77 CSVs from the official repository.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load base models only from local paths/cache.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed model stages with identical inputs/config.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate data/config without model loading or training.",
    )
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP of the new package.")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Select the new package only after successful export.",
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    if args.epochs is not None:
        config = replace(
            config,
            encoders={
                name: replace(settings, epochs=args.epochs)
                for name, settings in config.encoders.items()
            },
        )
    data = prepare_data(root, config, include_test=args.include_test, download=args.download_data)
    if args.check_only:
        print(
            json.dumps(
                {
                    "ready": True,
                    "training_origin": "native_python",
                    "partitions": {name: len(frame) for name, frame in data.partitions().items()},
                    "encoders": list(config.encoders),
                    "model_loading_or_training": False,
                },
                indent=2,
            )
        )
        return
    timestamp = datetime.now(UTC).strftime("native-%Y%m%dT%H%M%S%fZ")
    directory = (args.run_dir or root / "artifacts/training" / timestamp).resolve()
    destination = (args.package or root / "artifacts/demo_packages" / directory.name).resolve()
    if destination.exists() or destination.with_name(destination.name + ".zip").exists():
        parser.error("Choose a new --package destination; existing packages are not overwritten.")
    run = run_training(
        root,
        config,
        data,
        directory,
        device=args.device,
        resume=args.resume,
        local_files_only=args.local_files_only,
    )
    backend = NotebookDemo(
        run.encoders,
        run.intents,
        run.workflows,
        run.matrices["moderate"],
        run.witness,
        run.predict,
        lexical_features,
        partial(routing_scores, matrices=run.matrices),
        run.routing_reports,
        run.classifier_reports,
        run.run_id,
    )
    folder, archive = export_demo_package(
        backend,
        destination,
        specification=run.specification,
        checkpoints=run.checkpoints,
        classical_models=run.classical_models,
        result_dir=run.result_dir,
        create_zip=args.zip,
    )
    if args.activate:
        selected = root / "config/local_demo.json"
        selected.write_text(json.dumps({"package_directory": str(folder)}, indent=2) + "\n")
    print("Generated demo package:", folder)
    if archive:
        print("ZIP:", archive)
    print("Run: python aura_demo_package.py --package", folder)


if __name__ == "__main__":
    main()
