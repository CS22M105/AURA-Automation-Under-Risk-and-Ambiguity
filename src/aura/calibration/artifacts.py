"""JSON persistence for fitted calibration parameters."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from aura.calibration.temperature import TemperatureScaler


class CalibrationArtifactError(ValueError):
    """Raised when a calibration artifact is invalid."""


@dataclass(frozen=True)
class CalibrationArtifact:
    """A fitted temperature scaler and its provenance metadata."""

    scaler: TemperatureScaler
    metadata: dict[str, object]


def save_calibration_artifact(
    path: Path,
    scaler: TemperatureScaler,
    metadata: Mapping[str, object],
) -> None:
    """Save a temperature scaler and metadata as JSON."""
    payload = {
        "schema_version": 1,
        "method": "temperature_scaling",
        "temperature": scaler.temperature,
        "classes": list(scaler.classes),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_calibration_artifact(path: Path) -> CalibrationArtifact:
    """Load a validated temperature-scaling artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Calibration artifact does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CalibrationArtifactError(f"Calibration artifact is not valid JSON: {path}") from error

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CalibrationArtifactError("Unsupported calibration artifact schema")
    if payload.get("method") != "temperature_scaling":
        raise CalibrationArtifactError("Unsupported calibration method")

    temperature = payload.get("temperature")
    classes = payload.get("classes")
    metadata = payload.get("metadata")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise CalibrationArtifactError("Calibration temperature must be numeric")
    if not isinstance(classes, list) or any(
        not isinstance(label, int) or isinstance(label, bool) for label in classes
    ):
        raise CalibrationArtifactError("Calibration classes must be integer labels")
    if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
        raise CalibrationArtifactError("Calibration metadata must be a string-keyed dictionary")

    scaler = TemperatureScaler(
        temperature=float(temperature),
        classes=tuple(classes),
    )
    return CalibrationArtifact(scaler=scaler, metadata=dict(metadata))
