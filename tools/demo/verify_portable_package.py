"""Round-trip the real archived DistilBERT package; never train or access Colab."""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for name, path in [
    ("local_harness", HERE / "verify_saved_distilbert.py"),
    ("aura_demo_package", ROOT / "aura_demo_package.py"),
]:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

backend = sys.modules["local_harness"].prepared_backend()
package = sys.modules["aura_demo_package"]
state = backend.encoders["DistilBERT"]
gates = {name: state["gates"][name] for name in ["semantic_gate", "signed_gate"]}
scores = pd.read_csv(ROOT / "artifacts/policies/lexical_distilbert/scores.csv")
with np.load(
    ROOT / "artifacts/risk/distilbert/policy_validation_probabilities.npz", allow_pickle=False
) as arrays:
    probabilities = arrays["probabilities"]
features = scores[["semantic_margin", "signed_lexical_support"]].to_numpy()
project = json.loads((ROOT / "notebooks/aura_project_spec.json").read_text())
matrices = package.cost_matrices(project)
replay = package.routing_scores(probabilities, features, gates=gates, matrices=matrices)
for key in replay.keys() & set(scores.columns):
    np.testing.assert_allclose(replay[key], scores[key], rtol=1e-10, atol=1e-12)
state.update(
    slug="distilbert",
    checkpoint_sha256=package.checkpoint_sha256(ROOT / "artifacts/models/distilbert"),
    gates=gates,
    val_features=features,
    val_probabilities=probabilities,
    val_scores=replay,
)
namespace = {
    "RUN_ID": backend.run_id,
    "ENCODER_ROUTING": backend.encoders,
    "INTENTS": backend.intents,
    "WORKFLOWS": backend.workflows,
    "C_MODERATE": backend.costs,
    "COST_MATRICES": matrices,
    "PROJECT_SPEC": project,
    "witness": backend.witness,
    "predict_logits": backend.predict_logits,
    "lexical_features": backend.lexical_features,
    "routing_variant_scores": backend.routing_scores,
    "MODEL_DIR": ROOT / "artifacts/models/distilbert",
    "RESULT_DIR": ROOT / "artifacts/policies/lexical_distilbert",
    "classical_models": {
        "Logistic Regression": joblib.load(
            ROOT / "artifacts/models/tfidf_logistic_regression.joblib"
        ),
        "Random Forest": joblib.load(ROOT / "artifacts/models/tfidf_svd_random_forest.joblib"),
    },
    "encoder_validation_report": next(iter(backend.routing_reports.values())),
    "validation_metrics": next(iter(backend.classifier_reports.values())),
}
message = "I topped up but the app declined it"
before = backend.analyze(message, "DistilBERT", 80)
with tempfile.TemporaryDirectory(prefix="aura-real-package-") as directory:
    folder, _ = package.export_notebook_package(
        namespace, Path(directory) / "export", create_zip=False
    )
    restored = package.load_demo_package(folder)
    after = restored.analyze(message, "DistilBERT", 80)
    assert before["intent"] == after["intent"]
    assert [row["decision"] for row in before["decisions"]] == [
        row["decision"] for row in after["decisions"]
    ]
    np.testing.assert_allclose(
        before["probabilities"], after["probabilities"], rtol=1e-7, atol=1e-10
    )
    for old, new in zip(before["decisions"], after["decisions"], strict=True):
        assert abs(old["score"] - new["score"]) < 1e-8
        assert abs(old["cutoff"] - new["cutoff"]) < 1e-12
    app = package.build_app(restored)
    assert app.config["title"] == "AURA | Banking Intent Routing"
    app.close()
print(
    "Real saved DistilBERT, witness, gates, temperatures, cutoffs, predictions "
    "and UI round-trip passed."
)
print("No Ettin weights were available locally; this is not an actual Ettin export claim.")
