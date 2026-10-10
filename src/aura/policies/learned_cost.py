"""Matched low-capacity exploratory selectors; outputs are ranking scores only."""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_matched_cost_selectors(
    features: NDArray[np.float64],
    errors: NDArray[np.int64],
    realized_cost: NDArray[np.float64],
    maximum_cost: float,
) -> dict[str, Pipeline]:
    """Fit identical ridge pipelines to error and normalized-cost targets.

    Alpha=1 is a fixed default modeling choice, not a consequence weight or tuned
    optimum. Do not clip predictions: they are ordering scores, not probabilities.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(errors)
    costs = np.asarray(realized_cost, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected two finite features and at least two fitting rows")
    if y.shape != (len(x),) or y.dtype.kind not in "biu" or set(y) != {0, 1}:
        raise ValueError("Aligned targets must contain both correct and incorrect cases")
    if not np.isfinite(maximum_cost) or maximum_cost <= 0:
        raise ValueError("Maximum scenario cost must be finite and positive")
    if (
        costs.shape != y.shape
        or not np.isfinite(costs).all()
        or np.any((costs < 0) | (costs > maximum_cost))
        or not np.array_equal(costs > 0, y == 1)
    ):
        raise ValueError("Costs must be bounded and positive exactly on errors")
    models = {}
    for name, target in (("error_ridge", y), ("cost_ridge", costs / maximum_cost)):
        model = Pipeline([("scale", StandardScaler()), ("regressor", Ridge(alpha=1.0))])
        model.fit(x, target)
        models[name] = model
    return models
