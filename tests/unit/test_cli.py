from __future__ import annotations

from pathlib import Path

from binary_mopso_cd.cli import apply_args, build_compare_monitoring_parser, build_parser


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


def test_compare_monitoring_parser_uses_expected_defaults():
    parser = build_compare_monitoring_parser()
    args = parser.parse_args(["--inputs", "spec.json", "--outdir", "comparison"])

    assert args.inputs == Path("spec.json")
    assert args.outdir == Path("comparison")
    assert args.generation_column == "generation"
    assert args.text_column == "generated_text"
    assert args.run_column == "run"
