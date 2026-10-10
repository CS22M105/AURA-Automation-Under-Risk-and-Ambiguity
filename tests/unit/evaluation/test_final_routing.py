"""Synthetic tests only: the official test data must remain unopened here."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from aura.data.intents import BANKING77_LABELS
from aura.evaluation.final_routing import (
    FINAL_POLICIES,
    build_final_scores,
    evaluate_final_decisions,
    normalized_overlap_audit,
    primary_interpretation,
    select_final_decisions,
    validate_final_frame,
)
from aura.models.predictions import create_prediction_batch


class FrozenWitness:
    classes_ = np.array([0, 1, 2])

    def decision_function(self, messages: list[str]) -> np.ndarray:
        return np.tile([0.0, 1.0, 2.0], (len(messages), 1))


class FrozenGate:
    classes_ = np.array([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        error = np.linspace(0.1, 0.8, len(features))
        return np.column_stack([error, 1 - error])


def inputs() -> tuple[Any, dict[str, Any], dict[str, np.ndarray]]:
    p = np.array([[0.8, 0.1, 0.1], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])
    batch = create_prediction_batch([2, 0, 1], [0, 1, 2], p, np.log(p))
    gates = {"witness": FrozenWitness(), "semantic_gate": FrozenGate(), "signed_gate": FrozenGate()}
    flat = np.ones((3, 3)) - np.eye(3)
    return batch, gates, {"flat": flat, "moderate": flat * 2, "high_protection": flat * 3}


def test_scores_preserve_predictions_and_do_not_fit_models() -> None:
    batch, gates, matrices = inputs()
    before = batch.predicted_labels.copy()
    scores, features = build_final_scores(batch, batch, ["a", "b", "c"], gates, matrices, [0, 1, 2])
    assert set(scores) == set(FINAL_POLICIES)
    assert features.shape == (3, 2)
    np.testing.assert_array_equal(batch.predicted_labels, before)
    np.testing.assert_allclose(scores["flat"], 1 + scores["confidence"])


def test_guard_controls_have_exact_actual_counts_and_empty_metrics_are_null() -> None:
    batch, gates, matrices = inputs()
    scores, _ = build_final_scores(batch, batch, ["a", "b", "c"], gates, matrices, [0, 1, 2])
    masks = select_final_decisions(scores, batch.row_indices, [0.0, 0.8, 1.0])
    assert int(masks[(1.0, "guarded_moderate")].sum()) == 2
    assert int(masks[(1.0, "original_aura_guard_matched")].sum()) == 2
    assert int(masks[(1.0, "confidence_guard_matched")].sum()) == 2
    report = evaluate_final_decisions(batch, [0, 2, 1], masks, matrices, [0, 1, 2])
    empty = report[report.target_coverage == 0]
    assert empty.accepted_accuracy.isna().all()
    assert empty.cost_per_accepted.isna().all()
    assert empty.accepted_total_cost.eq(0).all()
    full = report[(report.target_coverage == 1) & (report.evaluation_scenario == "moderate")]
    unguarded = full[~full.policy.str.contains("guard")]
    assert unguarded.accepted_total_cost.eq(4).all()


def test_evaluation_never_changes_masks_when_truth_changes() -> None:
    batch, gates, matrices = inputs()
    scores, _ = build_final_scores(batch, batch, ["a", "b", "c"], gates, matrices, [0, 1, 2])
    masks = select_final_decisions(scores, batch.row_indices, [0.8])
    copies = {key: value.copy() for key, value in masks.items()}
    evaluate_final_decisions(batch, [0, 1, 2], masks, matrices, [0, 1, 2])
    evaluate_final_decisions(batch, [2, 0, 1], masks, matrices, [0, 1, 2])
    for key, value in masks.items():
        np.testing.assert_array_equal(value, copies[key])


def test_gate_class_encoding_is_checked() -> None:
    batch, gates, matrices = inputs()
    gates["signed_gate"].classes_ = np.array([1, 0])
    with pytest.raises(ValueError, match="encode"):
        build_final_scores(batch, batch, ["a", "b", "c"], gates, matrices, [0, 1, 2])


def test_overlap_flags_keep_all_rows_and_normalize_only_case_whitespace() -> None:
    result = normalized_overlap_audit(
        ["  HELLO world ", "hello   world", "hello world!"], ["hello world"]
    )
    assert result.overlaps_training_pool.tolist() == [True, True, False]
    assert result.duplicate_test_text_member.tolist() == [True, True, False]
    assert result.duplicate_test_text_excess.tolist() == [False, True, False]
    assert result.row_id.tolist() == [0, 1, 2]


@pytest.mark.parametrize(
    "lower,upper,expected",
    [
        (-2.0, -1.0, "supports_candidate_under_prespecified_procedure"),
        (1.0, 2.0, "favors_reference_under_prespecified_procedure"),
        (-1.0, 0.0, "inconclusive_not_evidence_of_equivalence"),
        (0.0, 1.0, "inconclusive_not_evidence_of_equivalence"),
    ],
)
def test_primary_rule(lower: float, upper: float, expected: str) -> None:
    assert primary_interpretation(lower, upper) == expected


def synthetic_official_shape() -> pd.DataFrame:
    labels = np.repeat(np.arange(77), 40)
    return pd.DataFrame(
        {
            "text": ["synthetic repeated text"] * 3080,
            "label": labels,
            "label_text": [BANKING77_LABELS[int(i)] for i in labels],
        }
    )


def test_official_shape_validation_allows_duplicate_texts() -> None:
    validate_final_frame(synthetic_official_shape())


def test_official_shape_validation_rejects_wrong_mapping_and_row_count() -> None:
    frame = synthetic_official_shape()
    frame.loc[frame.label == 0, "label_text"] = "wrong_intent"
    with pytest.raises(ValueError, match="canonical"):
        validate_final_frame(frame)
    with pytest.raises(ValueError, match="3080"):
        validate_final_frame(synthetic_official_shape().iloc[:-1])
