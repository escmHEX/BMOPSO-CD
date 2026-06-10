from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary MOPSO-CD semantic optimizer")
    parser.add_argument("--reference-text", required=True, help="Reference text for f1 fidelity.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="config_overrides",
        metavar="PATH=VALUE",
        help="Override an existing YAML config path using a YAML-typed value.",
    )
    return parser


def parse_config_override(assignment: str) -> tuple[str, object]:
    if "=" not in assignment:
        raise ValueError(f"Expected PATH=VALUE override, got {assignment!r}")
    path, raw_value = assignment.split("=", 1)
    path = path.strip()
    if not path:
        raise ValueError(f"Expected non-empty configuration path in override {assignment!r}")
    return path, yaml.safe_load(raw_value)


def apply_args(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    for assignment in args.config_overrides:
        path, value = parse_config_override(assignment)
        config.set_existing(path, value)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = RuntimeConfig.load(args.config)
    config = apply_args(config, args)
    outdirs = ExperimentRunner(config, args.reference_text).run_all()
    for outdir in outdirs:
        print(outdir)
    return 0
