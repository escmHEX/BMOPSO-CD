from __future__ import annotations

from pathlib import Path

import pytest

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.router import TASK_INFLUENCE, RouteTask, SemanticRouter
from binary_mopso_cd.settings import MOPSOSettings, ParallelismSettings
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
    assert settings.archive_multiplier == 0.7
    assert settings.kcand == 6
    assert settings.alpha == 1.0
    assert settings.p_tur_max == 0.08
    assert settings.p_tur_min == 0.02
    assert settings.p_anchor_enabled is True
    assert settings.p_anchor_min == 0.05
    assert settings.p_anchor_max == 0.40
    assert settings.guided_trajectory_validation_enabled is True
    assert settings.guided_trajectory_relative_margin == pytest.approx(0.40)
    assert settings.guided_candidate_diagnostics_enabled is True
    assert settings.guided_candidate_rejection_count_reasons == {
        "empty_candidate": False,
        "duplicate_candidate_output": False,
        "literal_copy_current": False,
        "literal_copy_target": False,
        "too_many_words": False,
        "semantic_duplicate": False,
        "no_semantic_progress": False,
        "guided_trajectory_inconsistent": True,
    }


def test_parallelism_defaults_match_strategy(test_config):
    settings = ParallelismSettings.from_config(test_config)
    assert settings.enabled is True
    assert settings.particle_update_max_concurrent == 10
    assert settings.initial_text_generation_max_concurrent == 10


def test_monitor_is_enabled_by_default_in_repo_configs():
    for path in [None, Path("configs/minimal.yaml"), Path("configs/test.yaml")]:
        config = RuntimeConfig.load(path)
        assert config.get("monitor.enabled") is True


def test_parallelism_settings_validation(test_config):
    test_config.set("parallelism.enabled", "true")
    with pytest.raises(ValueError, match="parallelism.enabled"):
        test_config.validate()
    test_config.set("parallelism.enabled", True)
    test_config.set("parallelism.particle_update_max_concurrent", 0)
    with pytest.raises(ValueError, match="particle_update_max_concurrent"):
        test_config.validate()
    test_config.set("parallelism.particle_update_max_concurrent", 1)
    test_config.set("parallelism.initial_text_generation_max_concurrent", 0)
    with pytest.raises(ValueError, match="initial_text_generation_max_concurrent"):
        test_config.validate()


def test_anchor_enabled_must_be_boolean(test_config):
    test_config.set("mopso.p_anchor_enabled", "false")
    with pytest.raises(ValueError, match="p_anchor_enabled"):
        test_config.validate()


def test_guided_trajectory_validation_enabled_must_be_boolean(test_config):
    test_config.set("mopso.guided_trajectory_validation_enabled", "false")
    with pytest.raises(ValueError, match="guided_trajectory_validation_enabled"):
        test_config.validate()


def test_guided_trajectory_relative_margin_must_be_non_negative(test_config):
    test_config.set("mopso.guided_trajectory_relative_margin", -0.01)
    with pytest.raises(ValueError, match="guided_trajectory_relative_margin"):
        test_config.validate()


def test_guided_trajectory_relative_margin_is_configurable(test_config):
    test_config.set("mopso.guided_trajectory_relative_margin", 1.5)
    test_config.validate()

    settings = MOPSOSettings.from_config(test_config)

    assert settings.guided_trajectory_relative_margin == pytest.approx(1.5)


def test_guided_candidate_diagnostics_enabled_must_be_boolean(test_config):
    test_config.set("mopso.guided_candidate_diagnostics_enabled", "false")
    with pytest.raises(ValueError, match="guided_candidate_diagnostics_enabled"):
        test_config.validate()


def test_guided_candidate_diagnostics_enabled_is_configurable(test_config):
    test_config.set("mopso.guided_candidate_diagnostics_enabled", False)
    test_config.validate()

    settings = MOPSOSettings.from_config(test_config)

    assert settings.guided_candidate_diagnostics_enabled is False


def test_guided_candidate_rejection_count_reasons_must_be_mapping(test_config):
    test_config.set("mopso.guided_candidate_rejection_count_reasons", ["guided_trajectory_inconsistent"])
    with pytest.raises(ValueError, match="guided_candidate_rejection_count_reasons"):
        test_config.validate()


def test_guided_candidate_rejection_count_reasons_must_be_boolean(test_config):
    test_config.set("mopso.guided_candidate_rejection_count_reasons.no_semantic_progress", "true")
    with pytest.raises(ValueError, match="guided_candidate_rejection_count_reasons.no_semantic_progress"):
        test_config.validate()


def test_guided_candidate_rejection_count_reasons_rejects_unknown_reason(test_config):
    test_config.set("mopso.guided_candidate_rejection_count_reasons.unknown_reason", True)
    with pytest.raises(ValueError, match="unknown_reason"):
        test_config.validate()


def test_guided_candidate_rejection_count_reasons_are_configurable(test_config):
    test_config.set("mopso.guided_candidate_rejection_count_reasons.no_semantic_progress", True)
    test_config.set("mopso.guided_candidate_rejection_count_reasons.guided_trajectory_inconsistent", False)
    test_config.validate()

    settings = MOPSOSettings.from_config(test_config)

    assert settings.guided_candidate_rejection_count_reasons["no_semantic_progress"] is True
    assert settings.guided_candidate_rejection_count_reasons["guided_trajectory_inconsistent"] is False


def test_all_components_can_be_frozen(test_config):
    test_config.set("experiment.frozen_components", ["role", "topic", "action"])
    test_config.validate()


def test_dmax_must_not_exceed_active_components_when_any_remain(test_config):
    test_config.set("experiment.frozen_components", ["role", "topic"])
    test_config.set("mopso.dmax", 2)
    with pytest.raises(ValueError, match="dmax"):
        test_config.validate()


def test_archive_multiplier_accepts_fractional_positive_values(test_config):
    test_config.set("mopso.archive_multiplier", 0.25)
    test_config.validate()
    assert MOPSOSettings.from_config(test_config).archive_multiplier == 0.25


def test_archive_multiplier_must_be_positive(test_config):
    test_config.set("mopso.archive_multiplier", 0.0)
    with pytest.raises(ValueError, match="archive_multiplier"):
        test_config.validate()


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
