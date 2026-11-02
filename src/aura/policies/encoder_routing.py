"""Encoder-specific calibration and gates for comparative AURA experiments."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from aura.calibration.temperature import TemperatureScaler, fit_temperature_scaler
from aura.evaluation.final_routing import build_final_scores
from aura.models.predictions import PredictionBatch, create_prediction_batch
from aura.policies.lexical import fit_correctness_gate, lexical_gate_features


@dataclass(frozen=True)
class EncoderRoutingLayer:
    """One encoder's temperature and gates; never share these across encoders."""

    encoder: str
    checkpoint_sha256: str
    temperature: TemperatureScaler
    gates: dict[str, Any]

    def score(
        self,
        raw: PredictionBatch,
        messages: Sequence[str],
        matrices: Mapping[str, NDArray[np.float64]],
        cost_classes: Sequence[int],
        *,
        encoder: str,
        checkpoint_sha256: str,
    ) -> tuple[PredictionBatch, dict[str, NDArray[np.float64]], NDArray[np.float64]]:
        """Score without truth labels, rejecting a different encoder/checkpoint."""
        if encoder != self.encoder or checkpoint_sha256 != self.checkpoint_sha256:
            raise ValueError("Routing layer belongs to a different encoder or checkpoint")
        if raw.classes != self.temperature.classes or raw.decision_scores is None:
            raise ValueError("Logits and classes must match the fitted temperature")
        calibrated = create_prediction_batch(
            raw.row_indices,
            raw.classes,
            self.temperature.transform(raw.decision_scores),
            raw.decision_scores,
        )
        scores, features = build_final_scores(
            raw, calibrated, messages, self.gates, matrices, cost_classes
        )
        return calibrated, scores, features


def fit_encoder_routing_layer(
    encoder: str,
    checkpoint_sha256: str,
    calibration: PredictionBatch,
    labels: Sequence[int],
    messages: Sequence[str],
    witness: Pipeline,
) -> EncoderRoutingLayer:
    """Fit only on calibration rows; witness must already be fitted on training rows."""
    if not encoder.strip() or len(checkpoint_sha256) != 64:
        raise ValueError("An encoder name and SHA-256 checkpoint identity are required")
    try:
        int(checkpoint_sha256, 16)
    except ValueError as error:
        raise ValueError("Checkpoint identity must be a SHA-256 hexadecimal digest") from error
    if calibration.decision_scores is None or len(messages) != calibration.sample_count:
        raise ValueError("Calibration logits and aligned messages are required")
    temperature = fit_temperature_scaler(calibration.decision_scores, labels, calibration.classes)
    truth = np.asarray(labels, dtype=np.int64)
    features = lexical_gate_features(
        calibration.decision_scores,
        calibration.classes,
        np.asarray(witness.decision_function(list(messages)), dtype=np.float64),
        witness.classes_,
    )
    correct = (calibration.predicted_labels == truth).astype(np.int64)
    return EncoderRoutingLayer(
        encoder=encoder,
        checkpoint_sha256=checkpoint_sha256,
        temperature=temperature,
        gates={
            "witness": witness,
            "semantic_gate": fit_correctness_gate(features[:, :1], correct),
            "signed_gate": fit_correctness_gate(features, correct),
        },
    )
