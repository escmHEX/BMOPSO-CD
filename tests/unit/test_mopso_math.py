from __future__ import annotations

import numpy as np

from binary_mopso_cd.mopso import semantic_velocity_delta


def test_semantic_velocity_delta_uses_normalized_strategy_distance():
    current = np.asarray([1.0, 0.0])
    same = np.asarray([1.0, 0.0])
    orthogonal = np.asarray([0.0, 1.0])
    opposite = np.asarray([-1.0, 0.0])

    assert semantic_velocity_delta(current, same) == 0.0
    assert semantic_velocity_delta(current, orthogonal) == 0.5
    assert semantic_velocity_delta(current, opposite) == 1.0
