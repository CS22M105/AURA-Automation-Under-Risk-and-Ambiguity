import pytest

from aura.models.distilbert import DistilBertConfig, build_intent_mappings, select_device


def test_uses_fixed_compact_transformer_budget() -> None:
    config = DistilBertConfig()

    assert config.model_name == "distilbert/distilbert-base-uncased"
    assert config.maximum_length == 128
    assert config.batch_size == 16
    assert config.gradient_accumulation_steps == 2
    assert config.epochs == 5
    assert config.learning_rate == 4e-5
    assert config.random_state == 42


def test_builds_complete_intent_mappings() -> None:
    id_to_label, label_to_id = build_intent_mappings(
        [0, 1, 0, 1],
        {0: "cash_withdrawal", 1: "card_delivery"},
    )

    assert id_to_label == {0: "cash_withdrawal", 1: "card_delivery"}
    assert label_to_id == {"cash_withdrawal": 0, "card_delivery": 1}


def test_rejects_noncontiguous_transformer_labels() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        build_intent_mappings([0, 2], {0: "cash_withdrawal", 2: "card_delivery"})


def test_accepts_explicit_cpu_device() -> None:
    assert select_device("cpu").type == "cpu"
