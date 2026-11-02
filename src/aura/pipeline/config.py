"""Explicit training settings shared by command-line runs on any machine."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str
    epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    maximum_length: int = 128
    learning_rate: float = 4e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    seed: int = 42
    attn_implementation: str | None = None
    reference_compile: bool = False

    def validate(self) -> None:
        if not self.model_name.strip() or any(
            value < 1
            for value in [
                self.epochs,
                self.batch_size,
                self.gradient_accumulation_steps,
                self.maximum_length,
            ]
        ):
            raise ValueError("Encoder model, epochs, batch sizes and maximum length must be valid.")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid encoder optimizer settings.")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("Warmup ratio must be between zero and one.")


def default_encoders() -> dict[str, EncoderConfig]:
    return {
        "DistilBERT": EncoderConfig("distilbert/distilbert-base-uncased"),
        "Ettin-150M": EncoderConfig(
            "jhu-clsp/ettin-encoder-150m",
            batch_size=8,
            gradient_accumulation_steps=4,
            attn_implementation="eager",
        ),
    }


@dataclass(frozen=True)
class TrainingConfig:
    dataset: str = "datasets/train.csv"
    test_dataset: str = "datasets/test.csv"
    split_manifest: str = "data/splits/development_seed_42.json"
    workflows: str = "config/intent_workflows.yaml"
    scenarios: str = "config/consequence_scenarios.yaml"
    expected_training_fingerprint: str = (
        "01612df1d62829c99b8d4ae2df715cc2b2fb39b7716bb635aba94239a3bcb9d0"
    )
    encoders: dict[str, EncoderConfig] = field(default_factory=default_encoders)
    baseline: dict[str, Any] = field(default_factory=dict)
    random_forest: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if set(self.encoders) != {"DistilBERT", "Ettin-150M"}:
            raise ValueError("The complete pipeline requires DistilBERT and Ettin-150M.")
        for config in self.encoders.values():
            config.validate()
        if len(self.expected_training_fingerprint) != 64:
            raise ValueError("A SHA-256 training-content fingerprint is required.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_training_config(path: Path) -> TrainingConfig:
    payload = json.loads(path.read_text())
    if "encoders" in payload:
        payload["encoders"] = {
            name: EncoderConfig(**settings) for name, settings in payload["encoders"].items()
        }
    config = TrainingConfig(**payload)
    config.validate()
    return config
