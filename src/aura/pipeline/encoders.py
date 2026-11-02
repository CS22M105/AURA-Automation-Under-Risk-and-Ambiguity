"""Train either encoder on ordinary CPU/CUDA/MPS machines, without notebook state."""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import softmax
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from aura.models.distilbert import select_device, set_reproducible_seed
from aura.models.predictions import PredictionBatch, create_prediction_batch
from aura.pipeline.config import EncoderConfig


def train_encoder(
    texts: list[str],
    labels: list[int],
    intents: list[str],
    config: EncoderConfig,
    destination: Path,
    *,
    device: str,
    local_files_only: bool = False,
) -> tuple[Any, Any, list[float]]:
    """Fine-tune all encoder parameters using only the supplied training partition."""
    config.validate()
    set_reproducible_seed(config.seed)
    architecture = AutoConfig.from_pretrained(config.model_name, local_files_only=local_files_only)
    architecture.num_labels = len(intents)
    architecture.id2label = dict(enumerate(intents))
    architecture.label2id = {name: i for i, name in enumerate(intents)}
    if architecture.model_type == "modernbert":
        architecture.reference_compile = config.reference_compile
    options = (
        {"attn_implementation": config.attn_implementation} if config.attn_implementation else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        config=architecture,
        local_files_only=local_files_only,
        **options,
    )
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=config.maximum_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"], torch.tensor(labels))
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    resolved_device = select_device(device)
    model.to(resolved_device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    steps = steps_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * config.warmup_ratio), steps)
    history = []
    try:
        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for index, (ids, mask, targets) in enumerate(loader):
                output = model(
                    input_ids=ids.to(resolved_device),
                    attention_mask=mask.to(resolved_device),
                    labels=targets.to(resolved_device),
                )
                losses.append(float(output.loss.detach().cpu()))
                (output.loss / config.gradient_accumulation_steps).backward()
                if (index + 1) % config.gradient_accumulation_steps == 0 or index + 1 == len(
                    loader
                ):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            history.append(float(np.mean(losses)))
            print(
                f"{config.model_name}: epoch {epoch + 1}/{config.epochs}; loss={history[-1]:.5f}",
                flush=True,
            )
    finally:
        model.to("cpu")
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    (destination / "training.json").write_text(
        json.dumps({"configuration": asdict(config), "epoch_losses": history}, indent=2)
    )
    return model, tokenizer, history


def load_encoder(directory: Path, config: EncoderConfig) -> tuple[Any, Any]:
    architecture = AutoConfig.from_pretrained(directory, local_files_only=True)
    if architecture.model_type == "modernbert":
        architecture.reference_compile = config.reference_compile
    options = (
        {"attn_implementation": config.attn_implementation} if config.attn_implementation else {}
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        directory,
        config=architecture,
        local_files_only=True,
        **options,
    ).eval()
    return model, AutoTokenizer.from_pretrained(directory, local_files_only=True)


def predict_logits(
    texts: list[str],
    *,
    trained_model: Any,
    trained_tokenizer: Any,
    config: dict[str, Any],
    device: str = "cpu",
    **kwargs: Any,
) -> np.ndarray:
    resolved_device = select_device(device)
    trained_model.eval().to(resolved_device)
    chunks = []
    try:
        with torch.inference_mode():
            for start in range(0, len(texts), config["batch_size"]):
                encoded = trained_tokenizer(
                    texts[start : start + config["batch_size"]],
                    padding="max_length",
                    truncation=True,
                    max_length=config["maximum_length"],
                    return_tensors="pt",
                )
                output = trained_model(
                    input_ids=encoded["input_ids"].to(resolved_device),
                    attention_mask=encoded["attention_mask"].to(resolved_device),
                )
                chunks.append(output.logits.detach().cpu().numpy())
    finally:
        trained_model.to("cpu")
    return np.asarray(np.concatenate(chunks), dtype=np.float64)


def predict_batch(
    texts: list[str],
    row_ids: list[int],
    *,
    model: Any,
    tokenizer: Any,
    config: EncoderConfig,
    device: str,
) -> PredictionBatch:
    logits = predict_logits(
        texts,
        trained_model=model,
        trained_tokenizer=tokenizer,
        config=asdict(config),
        device=device,
    )
    return create_prediction_batch(row_ids, range(logits.shape[1]), softmax(logits, axis=1), logits)
