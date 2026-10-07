"""DistilBERT training and prediction operations for intent classification."""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import cast

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from aura.data.splitting import DEFAULT_RANDOM_STATE
from aura.models.predictions import PredictionBatch, create_prediction_batch


@dataclass(frozen=True)
class DistilBertConfig:
    """Fixed training budget for the compact transformer comparison."""

    model_name: str = "distilbert/distilbert-base-uncased"
    maximum_length: int = 128
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    epochs: int = 5
    learning_rate: float = 4e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    random_state: int = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class TransformerTrainingResult:
    """A trained transformer and its observed optimization history."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    epoch_losses: tuple[float, ...]
    device: str


def select_device(requested_device: str = "auto") -> torch.device:
    """Resolve a requested PyTorch device, preferring available accelerators."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducible_seed(random_state: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable model training."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)


def build_intent_mappings(
    labels: Sequence[int],
    intent_names: Mapping[int, str],
) -> tuple[dict[int, str], dict[str, int]]:
    """Validate labels and build Hugging Face label mappings."""
    classes = sorted(set(int(label) for label in labels))
    if not classes:
        raise ValueError("Transformer training labels must not be empty")
    if classes != list(range(len(classes))):
        raise ValueError("Transformer labels must be contiguous and start at zero")
    if set(classes) != set(intent_names):
        raise ValueError("Intent names must cover every transformer class exactly")

    id_to_label = {label: str(intent_names[label]) for label in classes}
    if any(not name.strip() for name in id_to_label.values()):
        raise ValueError("Transformer intent names must not be blank")
    if len(set(id_to_label.values())) != len(id_to_label):
        raise ValueError("Transformer intent names must be unique")
    return id_to_label, {name: label for label, name in id_to_label.items()}


def _encode_dataset(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[str],
    labels: Sequence[int],
    maximum_length: int,
) -> TensorDataset:
    encoded = tokenizer(
        list(messages),
        padding="max_length",
        truncation=True,
        max_length=maximum_length,
        return_tensors="pt",
    )
    input_ids = cast(torch.Tensor, encoded["input_ids"])
    attention_mask = cast(torch.Tensor, encoded["attention_mask"])
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, label_tensor)


def train_distilbert(
    messages: Sequence[str],
    labels: Sequence[int],
    intent_names: Mapping[int, str],
    config: DistilBertConfig | None = None,
    requested_device: str = "auto",
) -> TransformerTrainingResult:
    """Fine-tune DistilBERT under a fixed final-checkpoint training budget."""
    resolved_config = config or DistilBertConfig()
    if len(messages) != len(labels) or not messages:
        raise ValueError("Transformer messages and labels must be non-empty and aligned")
    if (
        resolved_config.epochs < 1
        or resolved_config.batch_size < 1
        or resolved_config.gradient_accumulation_steps < 1
    ):
        raise ValueError("Transformer epochs, batch size, and accumulation steps must be positive")

    id_to_label, label_to_id = build_intent_mappings(labels, intent_names)
    set_reproducible_seed(resolved_config.random_state)
    device = select_device(requested_device)
    tokenizer = AutoTokenizer.from_pretrained(resolved_config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        resolved_config.model_name,
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    dataset = _encode_dataset(
        tokenizer,
        messages,
        labels,
        resolved_config.maximum_length,
    )
    generator = torch.Generator().manual_seed(resolved_config.random_state)
    loader = DataLoader(
        dataset,
        batch_size=resolved_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=resolved_config.learning_rate,
        weight_decay=resolved_config.weight_decay,
    )
    optimizer_steps_per_epoch = ceil(len(loader) / resolved_config.gradient_accumulation_steps)
    total_steps = optimizer_steps_per_epoch * resolved_config.epochs
    warmup_steps = int(total_steps * resolved_config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_losses: list[float] = []
    for _ in range(resolved_config.epochs):
        model.train()
        cumulative_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (input_ids, attention_mask, batch_labels) in enumerate(loader):
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                labels=batch_labels.to(device),
            )
            loss = outputs.loss
            cumulative_loss += float(loss.detach().cpu())
            (loss / resolved_config.gradient_accumulation_steps).backward()

            accumulation_complete = (
                batch_index + 1
            ) % resolved_config.gradient_accumulation_steps == 0 or batch_index + 1 == len(loader)
            if accumulation_complete:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    resolved_config.gradient_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_losses.append(cumulative_loss / len(loader))

    model.to("cpu")
    return TransformerTrainingResult(
        model=model,
        tokenizer=tokenizer,
        epoch_losses=tuple(epoch_losses),
        device=str(device),
    )


def predict_with_transformer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[str],
    row_indices: Sequence[int],
    maximum_length: int,
    batch_size: int,
    requested_device: str = "auto",
) -> PredictionBatch:
    """Generate the common prediction contract from a transformer classifier."""
    if len(messages) != len(row_indices) or not messages:
        raise ValueError("Transformer messages and row indices must be non-empty and aligned")
    device = select_device(requested_device)
    encoded = tokenizer(
        list(messages),
        padding="max_length",
        truncation=True,
        max_length=maximum_length,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        cast(torch.Tensor, encoded["input_ids"]),
        cast(torch.Tensor, encoded["attention_mask"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    torch_module = cast(torch.nn.Module, model)
    torch_module.to(device)
    model.eval()

    logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for input_ids, attention_mask in loader:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            logits.append(outputs.logits.detach().cpu())

    torch_module.to("cpu")
    combined_logits = torch.cat(logits).numpy().astype(np.float64)
    probabilities = torch.softmax(torch.from_numpy(combined_logits), dim=1).numpy()
    classes = tuple(range(combined_logits.shape[1]))
    return create_prediction_batch(
        row_indices=row_indices,
        classes=classes,
        probabilities=probabilities,
        decision_scores=combined_logits,
    )
