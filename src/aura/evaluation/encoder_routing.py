"""Safe exchange of Colab encoder logits with the local AURA routing pipeline."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from aura.data.intents import BANKING77_LABELS
from aura.models.predictions import PredictionBatch, create_prediction_batch


def frame_content_sha256(frame: pd.DataFrame) -> str:
    """Hash ordered text/intent pairs, matching the standalone notebook contract."""
    records = frame[["text", "label_text"]].astype(str).values.tolist()
    encoded = json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EncoderLogits:
    """Validated logits with explicit partition and model identity."""

    encoder: str
    checkpoint_sha256: str
    predictions: PredictionBatch


def load_encoder_logits(
    path: Path,
    frame: pd.DataFrame,
    *,
    encoder: str,
    partition: str,
    training_fingerprint: str,
) -> EncoderLogits:
    """Reject wrong row order, labels, data, partitions, or encoder exports."""
    required = {
        "schema_version",
        "encoder",
        "checkpoint_sha256",
        "partition",
        "classes",
        "intent_names",
        "row_indices",
        "logits",
        "training_fingerprint",
        "partition_fingerprint",
    }
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != required:
            raise ValueError("Unexpected encoder-logit export fields")
        if saved["schema_version"].item() != 1:
            raise ValueError("Unsupported encoder-logit schema")
        if saved["encoder"].item() != encoder or saved["partition"].item() != partition:
            raise ValueError("Encoder or partition does not match the requested experiment")
        if saved["training_fingerprint"].item() != training_fingerprint:
            raise ValueError("Training-pool fingerprint mismatch")
        if saved["partition_fingerprint"].item() != frame_content_sha256(frame):
            raise ValueError("Partition content/order fingerprint mismatch")
        classes = saved["classes"]
        ids = saved["row_indices"]
        if classes.dtype.kind not in "iu" or not np.array_equal(classes, np.arange(77)):
            raise ValueError("Expected canonical BANKING77 probability-column order")
        expected_names = [BANKING77_LABELS[i] for i in range(77)]
        if saved["intent_names"].tolist() != expected_names:
            raise ValueError("Intent names do not match the canonical label mapping")
        if ids.dtype.kind not in "iu" or not np.array_equal(ids, frame.index.to_numpy()):
            raise ValueError("Prediction row IDs must exactly match the partition order")
        logits = np.asarray(saved["logits"], dtype=np.float64)
        if logits.shape != (len(frame), 77) or not np.isfinite(logits).all():
            raise ValueError("Expected finite aligned 77-class logits")
        digest = str(saved["checkpoint_sha256"].item())
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Invalid checkpoint SHA-256 identity")
    return EncoderLogits(
        encoder=encoder,
        checkpoint_sha256=digest,
        predictions=create_prediction_batch(ids, classes, softmax(logits, axis=1), logits),
    )
