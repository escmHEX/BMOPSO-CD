from __future__ import annotations

import pytest

from binary_mopso_cd.router import TASK_INFLUENCE, RouteTask, SemanticRouter
from binary_mopso_cd.settings import MOPSOSettings
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


def test_router_influence_schedule_accepts_total_generations_alias(test_config):
    router = SemanticRouter(test_config)
    last = router.route(
        RouteTask("1", "test", TASK_INFLUENCE, {"iteration": 99, "totalGenerations": 100})
    )

    assert last.alg_params["temperature"] == pytest.approx(0.50)
    assert last.alg_params["top_p"] == pytest.approx(0.90)


def test_checkpoint_interval_validation_only_when_enabled(test_config):
    test_config.set("checkpoint.enabled", False)
    test_config.set("checkpoint.interval", 0)
    test_config.validate()
    test_config.set("checkpoint.enabled", True)
    with pytest.raises(ValueError):
        test_config.validate()


def test_mopso_default_hyperparameters_match_strategy(test_config):
    settings = MOPSOSettings.from_config(test_config)
    assert settings.kcand == 7
    assert settings.alpha == 1.0
    assert settings.p_tur_max == 0.07
    assert settings.p_tur_min == 0.02
    assert settings.p_anchor_min == 0.05
    assert settings.p_anchor_max == 0.70


def test_generated_text_validation_threshold_must_be_cosine_range(test_config):
    test_config.set("generated_text_validation.tau_gen_min", 1.1)
    with pytest.raises(ValueError):
        test_config.validate()


def test_anchor_probability_schedule_validation(test_config):
    test_config.set("mopso.p_anchor_min", 0.8)
    test_config.set("mopso.p_anchor_max", 0.7)
    with pytest.raises(ValueError, match="p_anchor_min"):
        test_config.validate()
    test_config.set("mopso.p_anchor_min", 0.0)
    test_config.set("mopso.p_anchor_max", 1.1)
    with pytest.raises(ValueError, match="p_anchor_max"):
        test_config.validate()
