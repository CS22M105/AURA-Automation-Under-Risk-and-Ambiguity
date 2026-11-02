"""Train tiny random encoders and export/reload without notebooks or network.

Synthetic data and tiny random DistilBERT/ModernBERT checkpoints exercise the real
optimizer, all four model stages and packaging; these are not research results.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from aura.data.intents import BANKING77_LABELS
from aura.data.manifests import calculate_sha256
from aura.evaluation.encoder_routing import frame_content_sha256
from aura.pipeline.config import EncoderConfig, TrainingConfig
from aura.pipeline.data import prepare_data
from aura.pipeline.run import run_training

ROOT = Path(__file__).parents[2]


def load_demo_modules() -> tuple[Any, Any]:
    for name in ["aura_demo_ui", "aura_demo_package"]:
        spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules["aura_demo_ui"], sys.modules["aura_demo_package"]


@pytest.fixture
def native_inputs(tmp_path: Path) -> tuple[Path, TrainingConfig]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("gradio")
    torch.set_num_threads(2)
    rows: list[dict[str, Any]] = []
    partitions: dict[str, list[int]] = {}
    for name, repeats in [("model_training", 4), ("calibration", 1), ("policy_validation", 1)]:
        start = len(rows)
        rows.extend(
            {
                "text": f"banking intentcode{label} {name} sample{n}",
                "label": label,
                "label_text": intent,
            }
            for label, intent in BANKING77_LABELS.items()
            for n in range(repeats)
        )
        partitions[name] = list(range(start, len(rows)))
    pool = pd.DataFrame(rows)
    pool.to_csv(tmp_path / "train.csv", index=False)
    manifest = {
        "schema_version": 1,
        "source_file": "train.csv",
        "source_sha256": calculate_sha256(tmp_path / "train.csv"),
        "random_state": 42,
        "total_rows": len(pool),
        "partitions": partitions,
    }
    (tmp_path / "splits.json").write_text(json.dumps(manifest))
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nbanking\ncard\nintent\nquery\n")
    tokenizer = transformers.DistilBertTokenizerFast(vocab_file=str(vocab))
    mapping = {
        "num_labels": 77,
        "id2label": BANKING77_LABELS,
        "label2id": {name: label for label, name in BANKING77_LABELS.items()},
    }
    bases = {}
    for name in ["distilbert", "modernbert"]:
        directory = tmp_path / "base" / name
        directory.mkdir(parents=True)
        if name == "distilbert":
            architecture = transformers.DistilBertConfig(
                vocab_size=16,
                dim=16,
                hidden_dim=32,
                n_layers=1,
                n_heads=2,
                max_position_embeddings=32,
                **mapping,
            )
        else:
            architecture = transformers.ModernBertConfig(
                vocab_size=16,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=32,
                pad_token_id=0,
                cls_token_id=2,
                sep_token_id=3,
                reference_compile=False,
                **mapping,
            )
        model = transformers.AutoModelForSequenceClassification.from_config(
            architecture, attn_implementation="eager"
        )
        model.save_pretrained(directory)
        tokenizer.save_pretrained(directory)
        bases[name] = directory
    return tmp_path, TrainingConfig(
        dataset="train.csv",
        test_dataset="test.csv",
        split_manifest="splits.json",
        workflows=str(ROOT / "config/intent_workflows.yaml"),
        scenarios=str(ROOT / "config/consequence_scenarios.yaml"),
        expected_training_fingerprint=frame_content_sha256(pool),
        encoders={
            "DistilBERT": EncoderConfig(
                str(bases["distilbert"]),
                epochs=1,
                batch_size=64,
                gradient_accumulation_steps=1,
                maximum_length=16,
                attn_implementation="eager",
            ),
            "Ettin-150M": EncoderConfig(
                str(bases["modernbert"]),
                epochs=1,
                batch_size=64,
                gradient_accumulation_steps=1,
                maximum_length=16,
                attn_implementation="eager",
            ),
        },
        random_forest={"svd_components": 4, "tree_count": 2, "parallel_jobs": 1},
    )


def test_native_training_export_prediction_and_resume(
    native_inputs: tuple[Path, TrainingConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = native_inputs
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    read_text = Path.read_text

    def no_notebook_reads(path: Path, *args: Any, **kwargs: Any) -> str:
        assert path.suffix != ".ipynb" and "notebooks" not in path.parts
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_notebook_reads)
    data = prepare_data(root, config)
    assert data.test is None and not (root / "test.csv").exists()
    run = run_training(root, config, data, root / "native-run", device="cpu", local_files_only=True)
    assert set(run.encoders) == {"DistilBERT", "Ettin-150M"}
    assert len(run.classifier_reports["Development validation"]) == 4
    report = run.routing_reports["Development validation"]
    assert len(report) == 330  # 2 encoders * 11 policies/controls * 5 budgets * 3 scenarios.
    assert {"confidence", "signed_gate", "signed_cost_gate"}.issubset(set(report.policy))
    ui, package = load_demo_modules()
    backend = ui.NotebookDemo(
        run.encoders,
        run.intents,
        run.workflows,
        run.matrices["moderate"],
        run.witness,
        run.predict,
        package.lexical_features,
        partial(package.routing_scores, matrices=run.matrices),
        run.routing_reports,
        run.classifier_reports,
        run.run_id,
    )
    folder, _ = package.export_demo_package(
        backend,
        root / "generated-demo",
        specification=run.specification,
        checkpoints=run.checkpoints,
        classical_models=run.classical_models,
        result_dir=run.result_dir,
        create_zip=False,
    )
    metadata = package.verify_package(folder)
    assert metadata["training_origin"] == "native_python"
    reloaded = package.load_demo_package(folder)
    for encoder in run.encoders:
        before = backend.analyze("banking card query", encoder, 80)
        after = reloaded.analyze("banking card query", encoder, 80)
        np.testing.assert_allclose(before["probabilities"], after["probabilities"], atol=1e-7)
        assert before["decisions"][-1]["decision"] == after["decisions"][-1]["decision"]
    app = ui.build_app(reloaded)
    app.close()

    def forbid_retraining(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Completed encoders must be loaded on resume")

    monkeypatch.setattr("aura.pipeline.run.train_encoder", forbid_retraining)
    resumed = run_training(
        root, config, data, run.directory, device="cpu", resume=True, local_files_only=True
    )
    assert set(resumed.encoders) == set(run.encoders)
    saved = json.loads((run.result_dir / "summary.json").read_text())
    assert all(item["reused"] for item in saved["training_records"])
    # Only an explicit request may open/evaluate the official-shape test input.
    test_frame = pd.DataFrame(
        [
            {
                "text": f"synthetic banking test intentcode{label} sample{index}",
                "label": label,
                "label_text": intent,
            }
            for label, intent in BANKING77_LABELS.items()
            for index in range(40)
        ]
    )
    test_frame.to_csv(root / "test.csv", index=False)
    with_test = prepare_data(root, config, include_test=True)
    evaluated = run_training(
        root, config, with_test, run.directory, device="cpu", resume=True, local_files_only=True
    )
    assert len(evaluated.classifier_reports["Official test (already exposed)"]) == 4
    assert len(evaluated.routing_reports["Official test (already exposed)"]) == 330
    for encoder in run.encoders:
        for policy, scores in run.encoders[encoder]["val_scores"].items():
            np.testing.assert_allclose(evaluated.encoders[encoder]["val_scores"][policy], scores)
    changed = replace(config, baseline={"regularization_strength": 2})
    with pytest.raises(ValueError, match="Resume refused"):
        run_training(root, changed, data, run.directory, resume=True, device="cpu")


def test_native_data_rejects_reordered_inputs_and_invalid_config(
    native_inputs: tuple[Path, TrainingConfig],
) -> None:
    root, config = native_inputs
    with pytest.raises(ValueError, match="complete pipeline"):
        replace(config, encoders={}).validate()
    pool = pd.read_csv(root / "train.csv")
    pool.drop(columns="label").rename(columns={"label_text": "category"}).to_csv(
        root / "train.csv", index=False
    )
    assert len(prepare_data(root, config).pool) == len(pool)
    pool.iloc[::-1].to_csv(root / "train.csv", index=False)
    with pytest.raises(ValueError, match="content/order"):
        prepare_data(root, config)
