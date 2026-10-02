import pytest

from aura.evaluation.classification import ClassificationEvaluationError
from aura.evaluation.diagnostics import compute_classification_diagnostics


def test_computes_per_intent_metrics_and_directed_confusions() -> None:
    diagnostics = compute_classification_diagnostics(
        y_true=[0, 0, 0, 1, 1, 2],
        y_predicted=[0, 1, 1, 1, 2, 2],
        classes=[0, 1, 2],
        intent_names={0: "cash", 1: "transfer", 2: "card"},
    )

    assert diagnostics.confusion_matrix.tolist() == [
        [1, 2, 0],
        [0, 1, 1],
        [0, 0, 1],
    ]
    assert diagnostics.per_intent[0].recall == pytest.approx(1 / 3)
    assert diagnostics.per_intent[2].precision == pytest.approx(0.5)
    assert diagnostics.top_confusions[0].true_intent == "cash"
    assert diagnostics.top_confusions[0].predicted_intent == "transfer"
    assert diagnostics.top_confusions[0].count == 2


def test_rejects_missing_intent_name() -> None:
    with pytest.raises(ClassificationEvaluationError, match="names are missing"):
        compute_classification_diagnostics(
            y_true=[0, 1],
            y_predicted=[0, 1],
            classes=[0, 1],
            intent_names={0: "cash"},
        )
