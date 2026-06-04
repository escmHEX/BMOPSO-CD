from __future__ import annotations

from binary_mopso_cd.cli import apply_args, build_parser


def test_cli_seed_overrides_experiment_seed(test_config):
    parser = build_parser()
    args = parser.parse_args(["--reference-text", "reference", "--seed", "123"])
    config = apply_args(test_config, args)
    assert config.seed == 123


def test_cli_model_override_includes_central_anchor_selection(test_config):
    parser = build_parser()
    args = parser.parse_args(["--reference-text", "reference", "--model", "llama-test"])
    config = apply_args(test_config, args)

    assert config.get("router.task_models.central_anchor_selection") == "llama-test"
