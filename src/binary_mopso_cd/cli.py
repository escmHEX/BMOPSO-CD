from __future__ import annotations

import argparse
from pathlib import Path

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.external_monitoring import (
    ExternalMonitoringComparator,
    SpacyEntityLabeler,
    build_comparison_embedding_service,
    load_monitoring_specs,
)
from binary_mopso_cd.runner import ExperimentRunner
from binary_mopso_cd.utils import parse_bool_assignment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binary MOPSO-CD semantic optimizer")
    parser.add_argument("--reference-text", required=True, help="Reference text for f1 fidelity.")
    parser.add_argument("--n", type=int, default=None, help="Population size.")
    parser.add_argument("--iterations", type=int, default=None, help="Optimizer iterations.")
    parser.add_argument("--runs", type=int, default=None, help="Independent runs.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--model", default=None, help="Default Ollama model.")
    parser.add_argument("--bert-model", default=None, help="SBERT alias or model name.")
    parser.add_argument("--outdir-base", type=Path, default=None, help="Output base directory.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file.")
    parser.add_argument("--freeze-components", default=None, help="Comma-separated component names.")
    parser.add_argument("--enable-monitor", action="store_true", help="Enable observational monitor.")
    parser.add_argument("--disable-selection", action="store_true", help="Disable final selection module.")
    parser.add_argument("--router-heuristic", action="append", default=[], help="name=true|false override.")
    parser.add_argument("--task-model", action="append", default=[], help="task=model override.")
    parser.add_argument("--ppdb-source", type=Path, default=None, help="PPDB source file used to build the local index.")
    parser.add_argument("--ppdb-index", type=Path, default=None, help="PPDB SQLite index path.")
    parser.add_argument("--enable-checkpoint", action="store_true", help="Enable deferred optimizer checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=None, help="Checkpoint interval in generations.")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Alias for --checkpoint-every.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint to resume from.")
    return parser


def build_compare_monitoring_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute common observational monitoring metrics.")
    parser.add_argument("--inputs", type=Path, required=True, help="JSON spec with comparison inputs.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for comparison artifacts.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file used for model settings.")
    parser.add_argument("--bert-model", default=None, help="SBERT alias or model name.")
    parser.add_argument("--spacy-model", default=None, help="spaCy model for entity labels.")
    parser.add_argument("--kmeans-clusters", type=int, default=3, help="KMeans clusters for global inertia.")
    parser.add_argument("--generation-column", default="generation", help="Input generation column.")
    parser.add_argument("--text-column", default="generated_text", help="Input generated text column.")
    parser.add_argument("--run-column", default="run", help="Input run column.")
    return parser


def apply_args(config: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    if args.n is not None:
        config.set("experiment.n", args.n)
    if args.iterations is not None:
        config.set("experiment.iterations", args.iterations)
    if args.runs is not None:
        config.set("experiment.runs", args.runs)
    if args.seed is not None:
        config.set("experiment.seed", args.seed)
    if args.model:
        config.set("ollama.default_model", args.model)
        for task in [
            "semantic_anchor_extraction",
            "central_anchor_selection",
            "semantic_pool_generation",
            "semantic_pool_expansion",
            "semantic_component_influence_candidates",
            "synthetic_text_generation",
        ]:
            config.set(f"router.task_models.{task}", args.model)
    if args.bert_model:
        config.set("models.sbert.default", args.bert_model)
    if args.outdir_base:
        config.set("runtime.outdir_base", str(args.outdir_base))
    if args.freeze_components is not None:
        components = [item.strip() for item in args.freeze_components.split(",") if item.strip()]
        config.set("experiment.frozen_components", components)
    if args.enable_monitor:
        config.set("monitor.enabled", True)
    if args.disable_selection:
        config.set("selection.enabled", False)
    for assignment in args.router_heuristic:
        name, enabled = parse_bool_assignment(assignment)
        config.set(f"router.heuristics.{name}", enabled)
    for assignment in args.task_model:
        if "=" not in assignment:
            raise ValueError(f"Expected task=model assignment, got {assignment!r}")
        task, model = assignment.split("=", 1)
        config.set(f"router.task_models.{task.strip()}", model.strip())
    if args.ppdb_source is not None:
        config.set("models.ppdb.source_path", str(args.ppdb_source))
    if args.ppdb_index is not None:
        config.set("models.ppdb.index_path", str(args.ppdb_index))
    if args.enable_checkpoint:
        config.set("checkpoint.enabled", True)
    checkpoint_interval = args.checkpoint_interval if args.checkpoint_interval is not None else args.checkpoint_every
    if checkpoint_interval is not None:
        config.set("checkpoint.enabled", True)
        config.set("checkpoint.interval", checkpoint_interval)
    if args.resume_from is not None:
        config.set("runtime.resume_from", str(args.resume_from))
    config.validate()
    return config


def run_compare_monitoring(args: argparse.Namespace) -> int:
    config = RuntimeConfig.load(args.config)
    if args.bert_model:
        config.set("models.sbert.default", args.bert_model)
    args.outdir.mkdir(parents=True, exist_ok=True)
    embedding_service = build_comparison_embedding_service(config, args.outdir)
    labeler = SpacyEntityLabeler(str(args.spacy_model or config.get("models.spacy.model", "en_core_web_sm")))
    comparator = ExternalMonitoringComparator(
        embedding_service=embedding_service,
        entity_labeler=labeler,
        kmeans_clusters=args.kmeans_clusters,
        generation_column=args.generation_column,
        text_column=args.text_column,
        run_column=args.run_column,
    )
    output = comparator.compare(load_monitoring_specs(args.inputs), args.outdir)
    embedding_service.cache.save()
    print(output.metrics_csv)
    print(output.global_inertia_svg)
    print(output.entity_entropy_svg)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        current_argv = list(argv)
    else:
        import sys

        current_argv = sys.argv[1:]
    if current_argv and current_argv[0] == "compare-monitoring":
        parser = build_compare_monitoring_parser()
        args = parser.parse_args(current_argv[1:])
        return run_compare_monitoring(args)
    parser = build_parser()
    args = parser.parse_args(current_argv)
    config = RuntimeConfig.load(args.config)
    config = apply_args(config, args)
    outdirs = ExperimentRunner(config, args.reference_text).run_all()
    for outdir in outdirs:
        print(outdir)
    return 0
