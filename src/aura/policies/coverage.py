"""Offline selective routing at an exact, bounded automation budget."""

from collections.abc import Sequence
from decimal import ROUND_FLOOR, Decimal

import numpy as np
from numpy.typing import NDArray


def select_at_coverage(
    scores: NDArray[np.float64], row_ids: Sequence[int] | NDArray[np.int64], coverage: float
) -> NDArray[np.bool_]:
    """Accept the lowest scores, breaking ties by ascending unique row ID.

    Scores must be oriented so lower is better (use negative confidence for
    confidence selection). This is an offline ranking, not a deployment threshold.
    """
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(row_ids)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a non-empty finite vector")
    if ids.shape != values.shape or ids.dtype.kind not in "iu":
        raise ValueError("Row IDs must be an aligned integer vector")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Row IDs must be unique")
    if not np.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError("Coverage must be between zero and one")
    count = int((Decimal(str(coverage)) * len(values)).to_integral_value(rounding=ROUND_FLOOR))
    accepted = np.zeros(len(values), dtype=np.bool_)
    accepted[np.lexsort((ids, values))[:count]] = True
    return accepted
