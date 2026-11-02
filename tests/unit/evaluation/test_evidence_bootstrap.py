"""Joint contrast draws must agree with the existing single-contrast bootstrap."""

import numpy as np
import pytest

from aura.evaluation.evidence_bootstrap import bootstrap_policy_contrasts
from aura.evaluation.routing_uncertainty import bootstrap_routing_cost


def test_joint_draws_reproduce_single_contrast() -> None:
    rng = np.random.default_rng(7)
    scores = {name: rng.random(30) for name in ["first", "second", "third"]}
    ids = list(range(30))
    costs = rng.integers(0, 6, 30).astype(float)
    contrasts = [("second", "first"), ("third", "first")]
    joint = bootstrap_policy_contrasts(scores, costs, ids, contrasts, resamples=80, family_size=2)
    for candidate, reference in contrasts:
        before = bootstrap_routing_cost(
            scores[reference], scores[candidate], costs, ids, [0.8], n_resamples=80
        )[0]
        row = joint[(joint.candidate == candidate) & (joint.confidence_level == 0.95)].iloc[0]
        np.testing.assert_allclose(
            [row.difference, row.lower, row.upper],
            [before.difference, before.lower, before.upper],
            atol=1e-12,
        )
    assert set(joint.confidence_level) == {0.95, 0.975}


def test_rejects_insufficient_family_size() -> None:
    with pytest.raises(ValueError, match="family size"):
        bootstrap_policy_contrasts(
            {"a": np.ones(3), "b": np.ones(3)},
            np.ones(3),
            [0, 1, 2],
            [("a", "b"), ("b", "a")],
            resamples=5,
            family_size=1,
        )
