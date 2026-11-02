"""Portable export integrity and offline reload, using tiny synthetic models."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parents[2]
for name in ["aura_demo_ui", "aura_demo_package"]:
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
PACKAGE = sys.modules["aura_demo_package"]


@pytest.fixture
def prepared_run(tmp_path: Path) -> dict[str, Any]:
    pytest.importorskip("gradio")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    intents = ["first", "second", "third"]
    config = transformers.DistilBertConfig(
        vocab_size=10,
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        num_labels=3,
        id2label=dict(enumerate(intents)),
        label2id={name: i for i, name in enumerate(intents)},
    )
    transformers.DistilBertForSequenceClassification(config).save_pretrained(checkpoint)
    vocab = checkpoint / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nfirst\nsecond\nthird\ncard\nquery\n")
    tokenizer = transformers.DistilBertTokenizerFast(vocab_file=str(vocab))
    tokenizer.save_pretrained(checkpoint)
    workflows = [
        {"label": i, "intent": name, "required_action": name, "consequence_flags": []}
        for i, name in enumerate(intents)
    ]
    scenarios = {
        name: {
            "base_incorrect_cost": 1,
            "action_mismatch_cost": mismatch,
            "omitted_consequence_costs": {},
        }
        for name, mismatch in [("flat", 1), ("moderate", 2), ("high_protection", 3)]
    }
    project = {
        "workflow_specification": {"workflows": workflows},
        "scenario_specification": {"scenarios": scenarios},
    }
    matrices = PACKAGE.cost_matrices(project)
    random = np.random.default_rng(42)
    logits = random.normal(size=(30, 3))
    features = random.normal(size=(30, 2))
    labels = np.arange(30) % 2
    gates = {
        "semantic_gate": LogisticRegression().fit(features[:, :1], labels),
        "signed_gate": LogisticRegression().fit(features, labels),
    }
    texts = [f"{name} card query" for name in intents] * 2
    witness = Pipeline([("text", TfidfVectorizer()), ("classifier", LinearSVC())]).fit(
        texts, [0, 1, 2] * 2
    )
    probabilities = softmax(logits / 0.9, axis=1)
    scores = PACKAGE.routing_scores(probabilities, features, gates=gates, matrices=matrices)
    state = {
        "slug": "distilbert",
        "model": object(),
        "tokenizer": tokenizer,
        "config": {"maximum_length": 32},
        "temperature": 0.9,
        "gates": gates,
        "checkpoint_sha256": PACKAGE.checkpoint_sha256(checkpoint),
        "val_probabilities": probabilities,
        "val_features": features,
        "val_scores": scores,
    }
    results = tmp_path / "run"
    results.mkdir()
    (results / "retained_report.md").write_text("Synthetic test result, not project performance.")
    report = pd.DataFrame(
        [
            {"encoder": "DistilBERT", "policy": policy}
            for policy in ["confidence", "signed_gate", "signed_cost_gate"]
        ]
    )
    return {
        "RUN_ID": "SYNTHETIC-PACKAGE-TEST",
        "ENCODER_ROUTING": {"DistilBERT": state},
        "PROJECT_SPEC": project,
        "MODEL_DIR": checkpoint,
        "RESULT_DIR": results,
        "INTENTS": intents,
        "WORKFLOWS": workflows,
        "C_MODERATE": matrices["moderate"],
        "COST_MATRICES": matrices,
        "witness": witness,
        "classical_models": {"Toy": witness},
        "predict_logits": lambda *args, **kwargs: None,
        "lexical_features": PACKAGE.lexical_features,
        "routing_variant_scores": PACKAGE.routing_scores,
        "encoder_validation_report": report,
        "validation_metrics": pd.DataFrame({"model": ["Toy"], "accuracy": [0.5]}),
    }


def test_package_round_trip_is_local_and_preserves_scores(
    prepared_run: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    folder, archive = PACKAGE.export_notebook_package(prepared_run, tmp_path / "export")
    assert archive.is_file()
    assert "Synthetic test result" in (folder / "results/retained_report.md").read_text()
    metadata = PACKAGE.verify_package(folder)
    assert metadata["run_id"] == "SYNTHETIC-PACKAGE-TEST"
    loaded = PACKAGE.load_demo_package(folder)
    state = loaded.encoders["DistilBERT"]
    assert state["temperature"] == 0.9
    for policy, saved in prepared_run["ENCODER_ROUTING"]["DistilBERT"]["val_scores"].items():
        np.testing.assert_allclose(state["val_scores"][policy], saved, rtol=1e-12)
    result = loaded.analyze("first card query", "DistilBERT", 80)
    assert len(result["decisions"]) == 3
    assert result["encoder"] == "DistilBERT"
    encoded = state["tokenizer"](
        ["first card query"],
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    with torch.inference_mode():
        reference = state["model"](**encoded).logits.numpy()
    predicted = loaded.predict_logits(
        ["first card query"],
        trained_model=state["model"],
        trained_tokenizer=state["tokenizer"],
        config=state["config"],
    )
    np.testing.assert_allclose(predicted, reference, rtol=1e-12)


def test_export_rejects_wrong_checkpoint_and_overwriting(
    prepared_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    folder, _ = PACKAGE.export_notebook_package(prepared_run, tmp_path / "export", create_zip=False)
    with pytest.raises(FileExistsError):
        PACKAGE.export_notebook_package(prepared_run, folder)
    prepared_run["ENCODER_ROUTING"]["DistilBERT"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from its fitted routing"):
        PACKAGE.export_notebook_package(prepared_run, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_export_rejects_changed_policy_scores(prepared_run: dict[str, Any], tmp_path: Path) -> None:
    prepared_run["ENCODER_ROUTING"]["DistilBERT"]["val_scores"]["signed_cost_gate"] += 1
    with pytest.raises(ValueError, match="score replay differs"):
        PACKAGE.export_notebook_package(prepared_run, tmp_path / "wrong")


def test_missing_file_detected_before_deserialization(
    prepared_run: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folder, _ = PACKAGE.export_notebook_package(prepared_run, tmp_path / "export", create_zip=False)
    (folder / "routing/witness.joblib").write_bytes(b"damaged")
    monkeypatch.setattr(PACKAGE.joblib, "load", lambda path: pytest.fail("Must verify first"))
    with pytest.raises(ValueError, match="missing or changed"):
        PACKAGE.load_demo_package(folder)


def test_bundle_does_not_depend_on_original_result_directory(
    prepared_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    folder, _ = PACKAGE.export_notebook_package(prepared_run, tmp_path / "export", create_zip=False)
    metadata = json.loads((folder / "package.json").read_text())
    for category in metadata["reports"].values():
        for path in category.values():
            assert not Path(path).is_absolute()
            assert (folder / path).is_file()


def test_inventory_keeps_partial_checkpoint_families_separate(tmp_path: Path) -> None:
    root = tmp_path / "download"
    first = root / "checkpoints/first/distilbert"
    second = root / "checkpoints/second/distilbert"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "model.safetensors").write_bytes(b"synthetic")
    (second / "tokenizer.json").write_text("{}")
    report = PACKAGE.inspect_demo_directory(root)
    assert report["layout"] == "colab_drive_backup"
    assert report["ready"] is False
    assert len(report["checkpoint_families"]) == 2
    one, two = [family["models"][0] for family in report["checkpoint_families"]]
    assert any(path.endswith("tokenizer.json") for path in one["missing"])
    assert any(path.endswith("model.safetensors") for path in two["missing"])
    assert not any(path.endswith("model.safetensors") for path in one["missing"])
    assert "section 16B" in report["next_step"]


def test_local_config_paths_are_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    config = project / "config/local_demo.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"package_directory": "artifacts/demo_packages/import"}))
    monkeypatch.setattr(PACKAGE, "__file__", str(project / "aura_demo_package.py"))
    monkeypatch.chdir(tmp_path)
    assert PACKAGE.configured_package_directory() == project / "artifacts/demo_packages/import"


def test_raw_download_fails_before_loading_models(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    with pytest.raises(ValueError, match="not ready for the demo"):
        PACKAGE.load_demo_package(tmp_path)


def test_portable_inventory_verifies_export_integrity(
    prepared_run: dict[str, Any],
    tmp_path: Path,
) -> None:
    folder, _ = PACKAGE.export_notebook_package(prepared_run, tmp_path / "export", create_zip=False)
    assert PACKAGE.inspect_demo_directory(folder)["ready"] is True
    (folder / "package.json").write_text("{}")
    report = PACKAGE.inspect_demo_directory(folder)
    assert report["ready"] is False
    assert "missing or changed" in report["error"]
