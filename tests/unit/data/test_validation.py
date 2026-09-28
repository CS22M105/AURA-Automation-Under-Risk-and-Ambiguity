import pandas as pd
import pytest

from aura.data.validation import (
    DatasetValidationError,
    validate_banking77_frame,
    validate_required_columns,
)


def test_accepts_required_columns() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Where is my card?"],
            "label": [11],
            "label_text": ["card_arrival"],
        }
    )

    validate_required_columns(frame)


def test_rejects_missing_columns() -> None:
    frame = pd.DataFrame({"text": ["Where is my card?"]})

    with pytest.raises(
        DatasetValidationError,
        match="label, label_text",
    ):
        validate_required_columns(frame)


@pytest.mark.parametrize("invalid_text", [None, "", "   "])
def test_rejects_missing_or_blank_text(invalid_text: object) -> None:
    frame = pd.DataFrame(
        {
            "text": [invalid_text],
            "label": [11],
            "label_text": ["card_arrival"],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="1 missing or blank text",
    ):
        validate_banking77_frame(frame)


def test_rejects_non_integer_label_ids() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Where is my card?"],
            "label": ["eleven"],
            "label_text": ["card_arrival"],
        }
    )

    with pytest.raises(DatasetValidationError, match="labels must use integer IDs"):
        validate_banking77_frame(frame)


@pytest.mark.parametrize("invalid_name", [None, "", "   "])
def test_rejects_missing_or_blank_label_names(invalid_name: object) -> None:
    frame = pd.DataFrame(
        {
            "text": ["Where is my card?"],
            "label": [11],
            "label_text": [invalid_name],
        }
    )

    with pytest.raises(DatasetValidationError, match="1 missing or blank label name"):
        validate_banking77_frame(frame)


def test_rejects_numeric_label_with_multiple_intent_names() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Where is my card?", "I do not recognize this withdrawal"],
            "label": [11, 11],
            "label_text": ["card_arrival", "cash_withdrawal_not_recognised"],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Numeric labels map to multiple intent names: \\[11\\]",
    ):
        validate_banking77_frame(frame)


def test_rejects_intent_name_with_multiple_numeric_labels() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Where is my card?", "When will my card arrive?"],
            "label": [11, 12],
            "label_text": ["card_arrival", "card_arrival"],
        }
    )

    with pytest.raises(
        DatasetValidationError,
        match="Intent names map to multiple numeric labels: \\['card_arrival'\\]",
    ):
        validate_banking77_frame(frame)
