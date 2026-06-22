from __future__ import annotations

import csv
import json
from pathlib import Path

from binary_mopso_cd.runner import ExperimentRunner


REFERENCE = "Evacuation orders remain in effect for Zone A until further notice."


def run_with_config(config, tmp_path: Path, name: str) -> Path:
    config.set("runtime.outdir_base", str(tmp_path / name))
    outdirs = ExperimentRunner(config, REFERENCE).run_all()
    assert len(outdirs) == 1
    return outdirs[0]


def test_reduced_initialization_run_uses_real_services(test_config, tmp_path):
    test_config.set("experiment.n", 1)
    test_config.set("experiment.iterations", 0)
    outdir = run_with_config(test_config, tmp_path, "init")
    assert (outdir / "reference.txt").exists()
    assert (outdir / "data_initial_population.json").exists()
    assert not (outdir / "checkpoints").exists()
    runtime_log = (outdir / "runtime.log").read_text(encoding="utf-8")
    assert "archive_updates=1" in runtime_log
    assert "archive_prunes=0" in runtime_log


def test_reduced_optimization_run_uses_real_services(test_config, tmp_path):
    test_config.set("experiment.n", 2)
    test_config.set("experiment.iterations", 1)
    outdir = run_with_config(test_config, tmp_path, "opt")
    assert (outdir / "population_evaluated.json").exists()
    assert (outdir / "pareto_front.json").exists()
    assert (outdir / "llm_calls.jsonl").exists()
    assert (outdir / "cost_metrics.json").exists()
    assert (outdir / "runtime.log").exists()
    cost_metrics = json.loads((outdir / "cost_metrics.json").read_text(encoding="utf-8"))
    assert "wall_clock_seconds" in cost_metrics
    assert "llm_calls_total" in cost_metrics
    runtime_txt = (outdir / "runtime.txt").read_text(encoding="utf-8")
    assert "total_sec:" in runtime_txt
    with (outdir / "evolucion_metricas.csv").open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "modified_count" in row
    assert "hypervolume" in row
    assert "spread" in row
    assert "archive_update_count" in row
    assert "archive_prune_count" in row
    assert int(row["archive_update_count"]) >= 1
    assert int(row["archive_prune_count"]) >= 0


def test_reduced_monitor_run_observes_by_default_without_decision_feedback(test_config, tmp_path):
    test_config.set("experiment.n", 2)
    test_config.set("experiment.iterations", 1)
    test_config.set("mopso.p_anchor_enabled", False)
    assert test_config.get("monitor.enabled") is True
    outdir = run_with_config(test_config, tmp_path, "monitor")
    assert (outdir / "monitor_metrics.csv").exists()
    with (outdir / "monitor_metrics.csv").open("r", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert "kmeans_inertia" in row
    assert "entity_entropy" in row


def test_reduced_checkpoint_resume(test_config, tmp_path):
    test_config.set("experiment.n", 2)
    test_config.set("experiment.iterations", 1)
    test_config.set("checkpoint.enabled", True)
    test_config.set("checkpoint.interval", 1)
    first = run_with_config(test_config, tmp_path, "checkpoint_first")
    checkpoint = first / "checkpoints" / "generation_0001.json"
    assert checkpoint.exists()
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "archive_stats" in checkpoint_payload

    test_config.set("runtime.resume_from", str(checkpoint))
    test_config.set("experiment.iterations", 2)
    resumed = run_with_config(test_config, tmp_path, "checkpoint_resumed")
    assert (resumed / "checkpoints" / "generation_0002.json").exists()
