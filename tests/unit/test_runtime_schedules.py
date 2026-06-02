from __future__ import annotations

import pytest

from binary_mopso_cd.router import TASK_INFLUENCE, RouteTask, SemanticRouter
from binary_mopso_cd.utils import progress_ratio


def test_progress_ratio_stays_inside_unit_interval():
    assert progress_ratio(0, 0) == 0.0
    assert progress_ratio(0, 1) == 0.0
    assert progress_ratio(0, 2) == 0.0
    assert progress_ratio(1, 2) == 1.0
    assert progress_ratio(99, 100) == 1.0
    with pytest.raises(ValueError):
        progress_ratio(100, 100)


def test_router_influence_schedule_uses_zero_based_index(test_config):
    router = SemanticRouter(test_config)
    first = router.route(
        RouteTask("1", "test", TASK_INFLUENCE, {"iteration": 0, "iterations": 100})
    )
    last = router.route(
        RouteTask("2", "test", TASK_INFLUENCE, {"iteration": 99, "iterations": 100})
    )
    assert first.alg_params["temperature"] == pytest.approx(0.70)
    assert first.alg_params["top_p"] == pytest.approx(0.95)
    assert last.alg_params["temperature"] == pytest.approx(0.50)
    assert last.alg_params["top_p"] == pytest.approx(0.90)


def test_checkpoint_interval_validation_only_when_enabled(test_config):
    test_config.set("checkpoint.enabled", False)
    test_config.set("checkpoint.interval", 0)
    test_config.validate()
    test_config.set("checkpoint.enabled", True)
    with pytest.raises(ValueError):
        test_config.validate()
