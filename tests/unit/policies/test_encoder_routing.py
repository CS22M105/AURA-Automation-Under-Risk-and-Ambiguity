"""Encoder isolation and label-free routing for the second AURA base model."""

import numpy as np
import pytest
from scipy.special import softmax

from aura.evaluation.final_routing import FINAL_POLICIES
from aura.models.predictions import create_prediction_batch
from aura.policies.encoder_routing import fit_encoder_routing_layer
from aura.policies.lexical import build_lexical_witness


@pytest.fixture
def routing_inputs() -> tuple:
    texts = [
        "card payment",
        "card refund",
        "cash deposit",
        "cash withdrawal",
        "bank transfer",
        "bank recipient",
    ]
    truth = np.array([0, 0, 1, 1, 2, 2])
    witness = build_lexical_witness().fit(texts, truth)
    logits = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 4.0, 1.0],
            [3.0, 1.0, 0.0],
            [0.0, 1.0, 4.0],
            [1.0, 3.0, 2.0],
        ]
    )
    batch = create_prediction_batch(range(6), [0, 1, 2], softmax(logits, axis=1), logits)
    flat = np.ones((3, 3)) - np.eye(3)
    matrices = {"flat": flat, "moderate": flat * 3, "high_protection": flat * 5}
    return texts, truth, witness, batch, matrices


def test_separate_layers_fit_own_temperature_and_gates(routing_inputs: tuple) -> None:
    texts, truth, witness, batch, matrices = routing_inputs
    first = fit_encoder_routing_layer("DistilBERT", "a" * 64, batch, truth, texts, witness)
    second_logits = batch.decision_scores * 2
    second_batch = create_prediction_batch(
        range(6), [0, 1, 2], softmax(second_logits, axis=1), second_logits
    )
    second = fit_encoder_routing_layer("Ettin-150M", "b" * 64, second_batch, truth, texts, witness)
    assert second.gates["signed_gate"] is not first.gates["signed_gate"]
    assert second.gates["witness"] is first.gates["witness"]
    assert second.temperature.temperature == pytest.approx(first.temperature.temperature * 2)
    calibrated, scores, _ = second.score(
        second_batch, texts, matrices, [0, 1, 2], encoder="Ettin-150M", checkpoint_sha256="b" * 64
    )
    assert set(scores) == set(FINAL_POLICIES)
    np.testing.assert_array_equal(calibrated.predicted_labels, second_batch.predicted_labels)
    np.testing.assert_allclose(scores["flat"], 1 - calibrated.probabilities.max(axis=1))


@pytest.mark.parametrize("encoder,digest", [("Ettin-150M", "a" * 64), ("DistilBERT", "b" * 64)])
def test_cannot_apply_layer_to_other_encoder_or_weights(
    routing_inputs: tuple, encoder: str, digest: str
) -> None:
    texts, truth, witness, batch, matrices = routing_inputs
    layer = fit_encoder_routing_layer("DistilBERT", "a" * 64, batch, truth, texts, witness)
    with pytest.raises(ValueError, match="different encoder or checkpoint"):
        layer.score(batch, texts, matrices, [0, 1, 2], encoder=encoder, checkpoint_sha256=digest)


def test_scoring_does_not_refit_gates(routing_inputs: tuple) -> None:
    texts, truth, witness, batch, matrices = routing_inputs
    layer = fit_encoder_routing_layer("Ettin-150M", "a" * 64, batch, truth, texts, witness)
    before = layer.gates["signed_gate"].named_steps["classifier"].coef_.copy()
    layer.score(batch, texts, matrices, [0, 1, 2], encoder="Ettin-150M", checkpoint_sha256="a" * 64)
    np.testing.assert_array_equal(
        before, layer.gates["signed_gate"].named_steps["classifier"].coef_
    )


@pytest.mark.parametrize("digest", ["short", "z" * 64])
def test_rejects_bad_checkpoint_identity(routing_inputs: tuple, digest: str) -> None:
    texts, truth, witness, batch, _ = routing_inputs
    with pytest.raises(ValueError, match="SHA-256"):
        fit_encoder_routing_layer("Ettin-150M", digest, batch, truth, texts, witness)
