"""Repeatable runtime measurements for model comparison."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class LatencyMetrics:
    """Repeated batch-inference latency measurements."""

    batch_size: int
    warmup_runs: int
    measured_runs: int
    median_batch_milliseconds: float
    p95_batch_milliseconds: float
    median_per_message_milliseconds: float

    def to_dict(self) -> dict[str, int | float]:
        """Return JSON-serializable timing values."""
        return asdict(self)


def measure_latency(
    operation: Callable[[], object],
    batch_size: int,
    measured_runs: int = 20,
    warmup_runs: int = 3,
) -> LatencyMetrics:
    """Measure median and 95th-percentile latency after warm-up calls."""
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if measured_runs < 1:
        raise ValueError("Measured runs must be positive")
    if warmup_runs < 0:
        raise ValueError("Warm-up runs must not be negative")

    for _ in range(warmup_runs):
        operation()

    durations: list[float] = []
    for _ in range(measured_runs):
        started_at = perf_counter()
        operation()
        durations.append((perf_counter() - started_at) * 1_000.0)

    median_batch = median(durations)
    return LatencyMetrics(
        batch_size=batch_size,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        median_batch_milliseconds=median_batch,
        p95_batch_milliseconds=float(np.percentile(durations, 95)),
        median_per_message_milliseconds=median_batch / batch_size,
    )
