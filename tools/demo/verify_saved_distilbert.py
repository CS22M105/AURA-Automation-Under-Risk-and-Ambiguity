"""Verify the UI against existing local DistilBERT artifacts, without training.

This is a local verification harness, not the Colab presentation entry point.
It deliberately labels its older checkpoint/results separately from the Colab run.
"""

import ast
import importlib.util
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
module_spec = importlib.util.spec_from_file_location("aura_demo_ui", ROOT / "aura_demo_ui.py")
ui = importlib.util.module_from_spec(module_spec)
sys.modules["aura_demo_ui"] = ui
module_spec.loader.exec_module(ui)


def prepared_backend():
    spec = json.loads((ROOT / "notebooks/aura_project_spec.json").read_text())
    workflows = sorted(spec["workflow_specification"]["workflows"], key=lambda row: row["label"])
    namespace = {
        "np": np,
        "NUM_CLASSES": 77,
        "WORKFLOWS": workflows,
        "SCENARIOS": spec["scenario_specification"]["scenarios"],
    }
    notebook = json.loads((ROOT / "notebooks/AURA_Colab_Demo.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith(("SCENARIOS =", "from sklearn.pipeline", "POLICY_LABELS =")):
            for node in ast.parse(source).body:
                if isinstance(node, ast.FunctionDef) and node.name in {
                    "build_cost_matrix",
                    "expected_risk",
                    "lexical_features",
                    "routing_variant_scores",
                }:
                    exec(
                        compile(ast.Module(body=[node], type_ignores=[]), "notebook", "exec"),
                        namespace,
                    )
    namespace["COST_MATRICES"] = {
        name: namespace["build_cost_matrix"](name) for name in namespace["SCENARIOS"]
    }
    namespace["C_MODERATE"] = namespace["COST_MATRICES"]["moderate"]
    directory = ROOT / "artifacts/models/distilbert"
    model = AutoModelForSequenceClassification.from_pretrained(directory, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    model.eval()
    torch.set_num_threads(2)

    @torch.inference_mode()
    def predict(messages, *, trained_model, trained_tokenizer, config, **kwargs):
        encoded = trained_tokenizer(
            messages,
            padding="max_length",
            truncation=True,
            max_length=config["maximum_length"],
            return_tensors="pt",
        )
        return trained_model(**encoded).logits.detach().cpu().numpy().astype(np.float64)

    gate_dir = ROOT / "artifacts/policies/lexical_distilbert"
    gates = joblib.load(gate_dir / "gates.joblib")
    scores = pd.read_csv(gate_dir / "scores.csv")
    temperature = json.loads(
        (ROOT / "artifacts/calibration/distilbert_temperature.json").read_text()
    )
    state = {
        "model": model,
        "tokenizer": tokenizer,
        "config": {"maximum_length": 128},
        "temperature": temperature["temperature"],
        "gates": gates,
        "val_scores": {key: scores[key].to_numpy() for key in ui.POLICIES},
    }
    report = pd.read_csv(gate_dir / "comparison.csv").assign(encoder="DistilBERT")
    report["deferred_count"] = len(scores) - report.accepted_count
    report["accepted_accuracy"] = 1 - report.accepted_errors / report.accepted_count
    metrics = []
    for name, filename in [
        ("Logistic Regression", "baseline_policy_validation.json"),
        ("Random Forest", "random_forest_policy_validation.json"),
        ("DistilBERT", "distilbert_policy_validation.json"),
    ]:
        data = json.loads((ROOT / "artifacts/metrics" / filename).read_text())
        classification = data.get("classification", data.get("metrics"))
        metrics.append({"model": name, "accuracy": classification["accuracy"]})
    return ui.NotebookDemo(
        {"DistilBERT": state},
        [row["intent"] for row in workflows],
        workflows,
        namespace["C_MODERATE"],
        gates["witness"],
        predict,
        namespace["lexical_features"],
        namespace["routing_variant_scores"],
        {"Archived development validation": report},
        {"Archived development validation": pd.DataFrame(metrics)},
        "LOCAL-VERIFICATION-ARCHIVED-DISTILBERT-NOT-OCT26-COLAB",
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    backend = prepared_backend()
    for example in ui.EXAMPLES:
        result = backend.analyze(example[0], "DistilBERT", 80)
        print(
            json.dumps(
                {
                    "message": example[0],
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "decisions": result["decisions"],
                }
            ),
            flush=True,
        )
    for coverage in ui.COVERAGES:
        for scenario in ["flat", "moderate", "high_protection"]:
            backend.selected_results(
                "Archived development validation", "DistilBERT", scenario, coverage
            )
    print("Real saved checkpoint and all 15 result selections verified.", flush=True)
    if args.serve:
        app = ui.build_app(backend)
        app.launch(
            server_name="127.0.0.1",
            server_port=7864,
            share=False,
            inbrowser=False,
            css=ui.CSS,
            theme=ui.light_theme(),
            show_error=True,
        )
