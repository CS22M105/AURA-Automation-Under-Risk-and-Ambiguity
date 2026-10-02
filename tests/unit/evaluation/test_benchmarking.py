import pytest

from aura.evaluation.benchmarking import measure_latency


def test_measures_requested_operation_runs() -> None:
    call_count = 0

    def operation() -> None:
        nonlocal call_count
        call_count += 1

    metrics = measure_latency(operation, batch_size=5, measured_runs=4, warmup_runs=2)

    assert call_count == 6
    assert metrics.batch_size == 5
    assert metrics.measured_runs == 4
    assert metrics.warmup_runs == 2
    assert metrics.median_batch_milliseconds >= 0
    assert metrics.p95_batch_milliseconds >= 0


def test_rejects_invalid_measured_run_count() -> None:
    with pytest.raises(ValueError, match="Measured runs must be positive"):
        measure_latency(lambda: None, batch_size=1, measured_runs=0)
