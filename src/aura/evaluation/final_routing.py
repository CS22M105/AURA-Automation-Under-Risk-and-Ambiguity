"""Label-free final routing selection and separate labeled evaluation helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aura.data.intents import BANKING77_LABELS
from aura.data.validation import validate_banking77_frame
from aura.models.predictions import PredictionBatch
from aura.policies.coverage import select_at_coverage
from aura.policies.guarded import select_with_majority_guard
from aura.policies.lexical import cost_aware_gate_score, lexical_gate_features
from aura.risk.expected_risk import compute_expected_routing_risk

FINAL_POLICIES = (
    "confidence",
    "flat",
    "original_aura",
    "high_protection",
    "guarded_moderate",
    "logit_margin",
    "semantic_gate",
    "signed_gate",
    "signed_cost_gate",
)


def validate_final_frame(frame: pd.DataFrame) -> None:
    """Validate official BANKING77 test dimensions/mapping without dropping duplicates."""
    validate_banking77_frame(frame)
    if len(frame) != 3080 or set(frame.label) != set(range(77)):
        raise ValueError("Expected official BANKING77 test split: 3080 rows and 77 intents")
    if not frame.groupby("label").size().eq(40).all():
        raise ValueError("Expected 40 official test messages per intent")
    if not frame.label_text.eq(frame.label.map(BANKING77_LABELS)).all():
        raise ValueError("Test labels do not match the canonical BANKING77 mapping")
    if frame.index.tolist() != list(range(len(frame))):
        raise ValueError("Preserve original zero-based test row IDs")


def build_final_scores(
    raw: PredictionBatch,
    calibrated: PredictionBatch,
    messages: Sequence[str],
    gates: Mapping[str, Any],
    matrices: Mapping[str, NDArray[np.float64]],
    cost_classes: Sequence[int],
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64]]:
    """Score all frozen variants without receiving or consulting ground-truth labels."""
    if (
        raw.row_indices != calibrated.row_indices
        or raw.classes != calibrated.classes
        or len(messages) != raw.sample_count
        or raw.decision_scores is None
        or not np.array_equal(raw.predicted_labels, calibrated.predicted_labels)
    ):
        raise ValueError("Raw and calibrated predictions must align and preserve labels")
    if set(gates) != {"witness", "semantic_gate", "signed_gate"}:
        raise ValueError("Unexpected fitted gate bundle")
    witness = gates["witness"]
    features = lexical_gate_features(
        raw.decision_scores,
        raw.classes,
        np.asarray(witness.decision_function(list(messages)), dtype=np.float64),
        witness.classes_,
    )
    errors = {}
    for name, values in (("semantic_gate", features[:, :1]), ("signed_gate", features)):
        gate = gates[name]
        if not np.array_equal(gate.classes_, [0, 1]):
            raise ValueError("Correctness gate must encode 0=error, 1=correct")
        probabilities = np.asarray(gate.predict_proba(values), dtype=np.float64)
        if (
            probabilities.shape != (raw.sample_count, 2)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.allclose(probabilities.sum(axis=1), 1)
        ):
            raise ValueError("Invalid correctness gate probabilities")
        errors[name] = probabilities[:, 0]
    risks = {
        name: compute_expected_routing_risk(calibrated, matrices[name], cost_classes).expected_costs
        for name in ("flat", "moderate", "high_protection")
    }
    scores = {
        "confidence": -calibrated.probabilities.max(axis=1),
        "flat": risks["flat"],
        "original_aura": risks["moderate"],
        "high_protection": risks["high_protection"],
        "guarded_moderate": risks["moderate"],
        "logit_margin": -features[:, 0],
        **errors,
        "signed_cost_gate": cost_aware_gate_score(
            calibrated, errors["signed_gate"], matrices["moderate"], cost_classes
        ),
    }
    return scores, features


def select_final_decisions(
    scores: Mapping[str, NDArray[np.float64]],
    row_ids: Sequence[int],
    coverages: Sequence[float],
) -> dict[tuple[float, str], NDArray[np.bool_]]:
    """Fix masks for all variants and guard controls using scores only."""
    if set(scores) != set(FINAL_POLICIES):
        raise ValueError("Final score policies differ from frozen protocol")
    if not coverages or len(set(coverages)) != len(coverages):
        raise ValueError("Coverage targets must be nonempty and unique")
    masks = {}
    for coverage in coverages:
        for policy in FINAL_POLICIES:
            if policy == "guarded_moderate":
                accepted = select_with_majority_guard(
                    scores[policy], -scores["confidence"], row_ids, coverage
                )
            else:
                accepted = select_at_coverage(scores[policy], row_ids, coverage)
            masks[(coverage, policy)] = accepted
        guard_count = int(masks[(coverage, "guarded_moderate")].sum())
        for policy in ("confidence", "original_aura"):
            accepted = np.zeros(len(row_ids), dtype=np.bool_)
            accepted[np.lexsort((np.asarray(row_ids), scores[policy]))[:guard_count]] = True
            masks[(coverage, f"{policy}_guard_matched")] = accepted
    return masks


def evaluate_final_decisions(
    predictions: PredictionBatch,
    truth: Sequence[int],
    masks: Mapping[tuple[float, str], NDArray[np.bool_]],
    matrices: Mapping[str, NDArray[np.float64]],
    cost_classes: Sequence[int],
) -> pd.DataFrame:
    """Evaluate preselected masks; never select or modify a mask using labels."""
    labels = np.asarray(truth)
    if (
        labels.shape != (predictions.sample_count,)
        or labels.dtype.kind not in "iu"
        or not set(labels).issubset(predictions.classes)
    ):
        raise ValueError("Invalid evaluation labels")
    positions = {label: i for i, label in enumerate(cost_classes)}
    true_columns = [positions[int(label)] for label in labels]
    predicted_columns = [positions[int(label)] for label in predictions.predicted_labels]
    records = []
    for scenario, matrix in matrices.items():
        compute_expected_routing_risk(predictions, matrix, cost_classes)
        realized = matrix[true_columns, predicted_columns]
        for (coverage, policy), accepted in masks.items():
            if accepted.shape != labels.shape or accepted.dtype != np.bool_:
                raise ValueError("Decision masks must be aligned boolean vectors")
            count = int(accepted.sum())
            errors = int(np.sum(accepted & (labels != predictions.predicted_labels)))
            total = float(realized[accepted].sum())
            requested = int(masks[(coverage, "original_aura")].sum())
            records.append(
                {
                    "policy": policy,
                    "evaluation_scenario": scenario,
                    "target_coverage": coverage,
                    "requested_count": requested,
                    "accepted_count": count,
                    "actual_coverage": count / len(labels),
                    "deferred_count": len(labels) - count,
                    "coverage_shortfall_count": requested - count,
                    "accepted_errors": errors,
                    "accepted_accuracy": 1 - errors / count if count else None,
                    "accepted_total_cost": total,
                    "accepted_cost_per_input": total / len(labels),
                    "cost_per_accepted": total / count if count else None,
                }
            )
    return pd.DataFrame(records)


def normalized_overlap_audit(
    test_texts: Sequence[str], training_texts: Sequence[str]
) -> pd.DataFrame:
    """Flag lowercase/whitespace-normalized overlaps without excluding primary rows."""

    def normalize(text: str) -> str:
        return " ".join(text.lower().split())

    training = {normalize(text) for text in training_texts}
    normalized = pd.Series([normalize(text) for text in test_texts])
    return pd.DataFrame(
        {
            "row_id": np.arange(len(normalized)),
            "overlaps_training_pool": normalized.isin(training),
            "duplicate_test_text_member": normalized.duplicated(keep=False),
            "duplicate_test_text_excess": normalized.duplicated(keep="first"),
        }
    )


def primary_interpretation(lower: float, upper: float) -> str:
    """Apply the frozen direction rule; zero endpoints remain inconclusive."""
    if not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Invalid primary interval")
    if upper < 0:
        return "supports_candidate_under_prespecified_procedure"
    if lower > 0:
        return "favors_reference_under_prespecified_procedure"
    return "inconclusive_not_evidence_of_equivalence"
