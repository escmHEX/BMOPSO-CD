from __future__ import annotations

import pytest

from binary_mopso_cd.cli import apply_args, build_parser


def test_cli_set_updates_scalar_value(test_config):
    parser = build_parser()
    args = parser.parse_args(["--reference-text", "reference", "--set", "experiment.n=5"])
    config = apply_args(test_config, args)
    assert config.n == 5


def test_cli_set_parses_boolean_value(test_config):
    parser = build_parser()
    args = parser.parse_args(["--reference-text", "reference", "--set", "parallelism.enabled=false"])
    config = apply_args(test_config, args)
    assert config.get("parallelism.enabled") is False


def test_cli_set_parses_list_value(test_config):
    parser = build_parser()
    args = parser.parse_args(
        ["--reference-text", "reference", "--set", 'experiment.frozen_components=["role","topic"]']
    )
    config = apply_args(test_config, args)
    assert config.get("experiment.frozen_components") == ["role", "topic"]


def test_cli_set_applies_multiple_overrides(test_config):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference-text",
            "reference",
            "--set",
            "experiment.n=5",
            "--set",
            "mopso.archive_multiplier=0.5",
        ]
    )
    config = apply_args(test_config, args)
    assert config.n == 5
    assert config.get("mopso.archive_multiplier") == 0.5


def test_cli_set_updates_phase_task_model_override(test_config):
    parser = build_parser()
    args = parser.parse_args(
        [
            "--reference-text",
            "reference",
            "--set",
            "router.phase_task_models.initialization.synthetic_text_generation=qwen3.5:2b",
        ]
    )
    config = apply_args(test_config, args)
    assert config.get("router.phase_task_models.initialization.synthetic_text_generation") == "qwen3.5:2b"


def test_cli_set_rejects_unknown_path(test_config):
    parser = build_parser()
    args = parser.parse_args(["--reference-text", "reference", "--set", "mopso.unknown=1"])
    with pytest.raises(ValueError, match="Unknown configuration path"):
        apply_args(test_config, args)


def test_cli_set_rejects_assignment_without_equals(test_config):
    parser = build_parser()
    args = parser.parse_args(["--reference-text", "reference", "--set", "experiment.n"])
    with pytest.raises(ValueError, match="PATH=VALUE"):
        apply_args(test_config, args)


def test_cli_rejects_removed_specific_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--reference-text", "reference", "--n", "5"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--reference-text", "reference", "--model", "llama-test"])
