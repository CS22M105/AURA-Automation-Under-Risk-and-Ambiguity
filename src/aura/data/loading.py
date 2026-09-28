"""Loading utilities for BANKING77 datasets."""

from pathlib import Path

import pandas as pd

from aura.data.validation import validate_banking77_frame


def load_banking77_csv(path: Path) -> pd.DataFrame:
    """Load a BANKING77 CSV file and validate its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    frame = pd.read_csv(path)
    validate_banking77_frame(frame)

    return frame
