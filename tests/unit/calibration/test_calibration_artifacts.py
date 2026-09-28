from pathlib import Path

import pytest

from aura.calibration.artifacts import (
    load_calibration_artifact,
    save_calibration_artifact,
)
from aura.calibration.temperature import TemperatureScaler


def test_round_trips_calibration_artifact(tmp_path: Path) -> None:
    path = tmp_path / "temperature.json"
    scaler = TemperatureScaler(temperature=0.75, classes=(0, 1, 2))

    save_calibration_artifact(path, scaler, {"fitted_partition": "calibration"})
    artifact = load_calibration_artifact(path)

    assert artifact.scaler == scaler
    assert artifact.metadata == {"fitted_partition": "calibration"}


def test_rejects_missing_calibration_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_calibration_artifact(tmp_path / "missing.json")
