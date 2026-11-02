"""Alignment and provenance validation for encoder logits exported from Colab."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aura.data.intents import BANKING77_LABELS
from aura.evaluation.encoder_routing import frame_content_sha256, load_encoder_logits


@pytest.fixture
def export_inputs() -> tuple:
    frame = pd.DataFrame(
        {
            "text": ["message one", "message two"],
            "label_text": [BANKING77_LABELS[0], BANKING77_LABELS[1]],
        },
        index=[7, 9],
    )
    payload = {
        "schema_version": 1,
        "encoder": "Ettin-150M",
        "checkpoint_sha256": "a" * 64,
        "partition": "calibration",
        "classes": np.arange(77),
        "intent_names": np.array([BANKING77_LABELS[i] for i in range(77)]),
        "row_indices": np.array([7, 9]),
        "logits": np.zeros((2, 77)),
        "training_fingerprint": "training",
        "partition_fingerprint": frame_content_sha256(frame),
    }
    return frame, payload


def test_load_export(tmp_path: Path, export_inputs: tuple) -> None:
    frame, payload = export_inputs
    path = tmp_path / "calibration.npz"
    np.savez_compressed(path, **payload)
    loaded = load_encoder_logits(
        path, frame, encoder="Ettin-150M", partition="calibration", training_fingerprint="training"
    )
    assert loaded.predictions.row_indices == (7, 9)
    np.testing.assert_allclose(loaded.predictions.probabilities.sum(axis=1), 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder", "DistilBERT"),
        ("partition", "test"),
        ("checkpoint_sha256", "wrong"),
        ("training_fingerprint", "wrong"),
        ("partition_fingerprint", "wrong"),
        ("classes", np.arange(77)[::-1]),
        ("row_indices", np.array([9, 7])),
        ("logits", np.full((2, 77), np.nan)),
        ("intent_names", np.array(["wrong"] * 77)),
    ],
)
def test_rejects_mismatched_export(
    tmp_path: Path, export_inputs: tuple, field: str, value: object
) -> None:
    frame, payload = export_inputs
    payload[field] = value
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError):
        load_encoder_logits(
            path,
            frame,
            encoder="Ettin-150M",
            partition="calibration",
            training_fingerprint="training",
        )
