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
