"""Descriptive probability diagnostics, not automatic root-cause assignments."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import softmax

from aura.models.predictions import PredictionBatch, create_prediction_batch
from aura.risk.expected_risk import compute_expected_routing_risk


def diagnose_routing_probabilities(
    predictions: PredictionBatch,
    truth: Sequence[int],
    temperature: float,
    costs: NDArray[np.float64],
    cost_classes: Sequence[int],
) -> pd.DataFrame:
    """Describe true-label probability, rank, NLL, and its expected-cost contribution.

    Invert scalar temperature scaling: raw = softmax(T * log(calibrated)).
    Zeros are rejected because underflow would prevent reliable reconstruction.
    Labels are used for retrospective evaluation only.
    """
    risk = compute_expected_routing_risk(predictions, costs, cost_classes)
    p = predictions.probabilities
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    if np.any(p <= 0):
        raise ValueError("Strictly positive probabilities required for reconstruction")
    labels = np.asarray(truth)
    if labels.shape != (predictions.sample_count,) or labels.dtype.kind not in "iu":
        raise ValueError("Truth must be an aligned integer vector")
    positions = {label: i for i, label in enumerate(predictions.classes)}
    if not set(labels).issubset(positions):
        raise ValueError("Unknown true label")
    columns = np.array([positions[int(label)] for label in labels])
    rows = np.arange(len(labels))
    raw = np.asarray(softmax(temperature * np.log(p), axis=1), dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("Reconstructed probabilities underflowed or are non-finite")
    raw_batch = create_prediction_batch(predictions.row_indices, predictions.classes, raw)
    raw_risk = compute_expected_routing_risk(raw_batch, costs, cost_classes)
    calibrated_true, raw_true = p[rows, columns], raw[rows, columns]
    matrix_positions = {label: i for i, label in enumerate(cost_classes)}
    realized = costs[
        [matrix_positions[int(label)] for label in labels],
        [matrix_positions[int(label)] for label in predictions.predicted_labels],
    ]
    return pd.DataFrame(
        {
            "row_id": predictions.row_indices,
            "is_error": labels != predictions.predicted_labels,
            "raw_true_probability_reconstructed": raw_true,
            "calibrated_true_probability": calibrated_true,
            "true_probability_change": calibrated_true - raw_true,
            "true_rank": 1 + np.sum(p > calibrated_true[:, None], axis=1),
            "raw_nll_reconstructed": -np.log(raw_true),
            "calibrated_nll": -np.log(calibrated_true),
            "raw_expected_cost_reconstructed": raw_risk.expected_costs,
            "calibrated_expected_cost": risk.expected_costs,
            "observed_cost": realized,
            "true_intent_cost_contribution": risk.contributions[rows, columns],
        }
    )
