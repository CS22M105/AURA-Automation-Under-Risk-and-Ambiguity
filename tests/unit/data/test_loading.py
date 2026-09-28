from pathlib import Path

import pandas as pd
import pytest

from aura.data.loading import load_banking77_csv
from aura.data.validation import DatasetValidationError


def test_loads_valid_banking77_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "banking77.csv"
    expected = pd.DataFrame(
        {
            "text": ["Where is my card?"],
            "label": [11],
            "label_text": ["card_arrival"],
        }
    )
    expected.to_csv(dataset_path, index=False)

    actual = load_banking77_csv(dataset_path)

    pd.testing.assert_frame_equal(actual, expected)


def test_rejects_missing_dataset_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset file does not exist"):
        load_banking77_csv(missing_path)


def test_rejects_csv_with_invalid_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.csv"
    pd.DataFrame({"text": ["Where is my card?"]}).to_csv(
        dataset_path,
        index=False,
    )

    with pytest.raises(
        DatasetValidationError,
        match="label, label_text",
    ):
        load_banking77_csv(dataset_path)
