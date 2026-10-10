import numpy as np
import pytest

from aura.policies.learned_cost import fit_matched_cost_selectors


def test_flat_cost_targets_produce_identical_matched_models() -> None:
    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]])
    y = np.array([1, 0, 1, 0])
    models = fit_matched_cost_selectors(x, y, y.astype(float), 1.0)
    np.testing.assert_allclose(models["error_ridge"].predict(x), models["cost_ridge"].predict(x))
    for model in models.values():
        np.testing.assert_allclose(model.named_steps["scale"].mean_, x.mean(axis=0))
        assert model.named_steps["regressor"].alpha == 1.0


def test_cost_model_changes_with_severity_but_error_model_does_not() -> None:
    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0], [3.0, 2.0]])
    y = np.array([1, 0, 1, 0])
    a = fit_matched_cost_selectors(x, y, np.array([1.0, 0.0, 5.0, 0.0]), 5.0)
    b = fit_matched_cost_selectors(x, y, np.array([5.0, 0.0, 1.0, 0.0]), 5.0)
    np.testing.assert_allclose(a["error_ridge"].predict(x), b["error_ridge"].predict(x))
    assert not np.allclose(a["cost_ridge"].predict(x), b["cost_ridge"].predict(x))


@pytest.mark.parametrize("costs", [[0.0, 0.0], [1.0, 1.0], [6.0, 0.0], [np.nan, 0.0], [1.0]])
def test_rejects_invalid_cost_targets(costs: list[float]) -> None:
    with pytest.raises(ValueError):
        fit_matched_cost_selectors(np.ones((2, 2)), np.array([1, 0]), np.array(costs), 5.0)


def test_rejects_nonfinite_features() -> None:
    with pytest.raises(ValueError):
        fit_matched_cost_selectors(
            np.array([[np.nan, 1.0], [0.0, 1.0]]), np.array([1, 0]), np.array([5.0, 0.0]), 5.0
        )
