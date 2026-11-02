"""BANKING77 preparation with canonical labels and fixed, disjoint partitions."""

import io
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import yaml

from aura.data.intents import BANKING77_LABELS
from aura.data.manifests import DevelopmentSplitManifest, load_development_split_manifest
from aura.data.validation import validate_banking77_frame
from aura.evaluation.encoder_routing import frame_content_sha256
from aura.pipeline.config import TrainingConfig
from aura.risk.scenarios import load_consequence_scenario_set
from aura.risk.workflows import load_workflow_specification, require_frozen_specification

DATA_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "category" in frame and "label_text" not in frame:
        frame = frame.rename(columns={"category": "label_text"})
        lookup = {name: label for label, name in BANKING77_LABELS.items()}
        if not frame.label_text.isin(lookup).all():
            raise ValueError("Dataset contains unknown intent names.")
        frame["label"] = frame.label_text.map(lookup).astype(int)
    validate_banking77_frame(frame)
    if not frame.label_text.eq(frame.label.map(BANKING77_LABELS)).all():
        raise ValueError("Dataset labels do not match canonical BANKING77 names.")
    return frame[["text", "label", "label_text"]]


def download_missing_dataset(path: Path, partition: str) -> None:
    """Download only explicitly requested missing input data; never require Colab."""
    if path.exists():
        return
    with urlopen(f"{DATA_URL}/{partition}.csv", timeout=60) as response:
        raw = pd.read_csv(io.BytesIO(response.read()))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the upstream representation; canonical conversion happens in load_frame.
    with path.open("x") as stream:
        raw.to_csv(stream, index=False)


@dataclass(frozen=True)
class PreparedData:
    pool: pd.DataFrame
    manifest: DevelopmentSplitManifest
    specification: dict
    test: pd.DataFrame | None

    def partitions(self) -> dict[str, pd.DataFrame]:
        result = {
            name: self.pool.iloc[list(getattr(self.manifest, name))]
            for name in ["model_training", "calibration", "policy_validation"]
        }
        if self.test is not None:
            result["test"] = self.test
        return result


def prepare_data(
    root: Path,
    config: TrainingConfig,
    *,
    include_test: bool = False,
    download: bool = False,
) -> PreparedData:
    config.validate()
    if download:
        download_missing_dataset(root / config.dataset, "train")
        if include_test:
            download_missing_dataset(root / config.test_dataset, "test")
    pool = load_frame(root / config.dataset)
    manifest = load_development_split_manifest(root / config.split_manifest)
    # Content identity is invariant to CSV writer/column-order differences.
    if (
        len(pool) != manifest.total_rows
        or frame_content_sha256(pool) != config.expected_training_fingerprint
    ):
        raise ValueError("Training-pool content/order differs from the configured fingerprint.")
    for name in ["model_training", "calibration", "policy_validation"]:
        if set(pool.iloc[list(getattr(manifest, name))].label) != set(BANKING77_LABELS):
            raise ValueError(f"Partition {name} must contain all 77 canonical classes.")
    require_frozen_specification(load_workflow_specification(root / config.workflows))
    load_consequence_scenario_set(root / config.scenarios)
    specification = {
        "workflow_specification": yaml.safe_load((root / config.workflows).read_text()),
        "scenario_specification": yaml.safe_load((root / config.scenarios).read_text()),
        "split_ids": {
            name: list(getattr(manifest, name))
            for name in ["model_training", "calibration", "policy_validation"]
        },
        "expected_training_fingerprint": config.expected_training_fingerprint,
    }
    test = load_frame(root / config.test_dataset) if include_test else None
    if test is not None:
        from aura.evaluation.final_routing import validate_final_frame

        validate_final_frame(test)
    # JSON round-trip checks the exported specification has no Python-only YAML types.
    json.dumps(specification)
    return PreparedData(pool, manifest, specification, test)
