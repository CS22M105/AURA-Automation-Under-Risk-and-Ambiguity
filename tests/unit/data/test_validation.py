import pandas as pd
import pytest

from aura.data.validation import (
    DatasetProfile,
    DatasetValidationError,
    validate_banking77_frame,
    validate_dataset_profile,
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


def create_profile_frame() -> pd.DataFrame:
    """Create a small valid dataset for profile tests."""
    return pd.DataFrame(
        {
            "text": ["Where is my card?", "I need to exchange cash"],
            "label": [0, 1],
            "label_text": ["card_arrival", "cash_exchange"],
        }
    )


def test_accepts_matching_dataset_profile() -> None:
    profile = DatasetProfile(
        name="test split",
        expected_rows=2,
        expected_label_ids=frozenset({0, 1}),
    )

    validate_dataset_profile(create_profile_frame(), profile)


def test_rejects_unexpected_row_count() -> None:
    profile = DatasetProfile(
        name="test split",
        expected_rows=3,
        expected_label_ids=frozenset({0, 1}),
    )

    with pytest.raises(
        DatasetValidationError,
        match="must contain 3 rows; found 2",
    ):
        validate_dataset_profile(create_profile_frame(), profile)


def test_rejects_mismatched_label_ids() -> None:
    profile = DatasetProfile(
        name="test split",
        expected_rows=2,
        expected_label_ids=frozenset({0, 2}),
    )

    with pytest.raises(
        DatasetValidationError,
        match=r"missing=\[2\], unexpected=\[1\]",
    ):
        validate_dataset_profile(create_profile_frame(), profile)


def test_rejects_duplicate_messages() -> None:
    frame = create_profile_frame()
    frame.loc[1, "text"] = frame.loc[0, "text"]

    profile = DatasetProfile(
        name="test split",
        expected_rows=2,
        expected_label_ids=frozenset({0, 1}),
    )

    with pytest.raises(
        DatasetValidationError,
        match="contains 1 duplicate message",
    ):
        validate_dataset_profile(frame, profile)
