from __future__ import annotations

import math

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.metrics import archive_metrics, calculate_hypervolume, calculate_spread, normalized_objective_point
from binary_mopso_cd.progress import ProgressLogger


def solution(f1: float, f2: float, idx: int) -> Solution:
    return Solution(
        SemanticVector({"role": f"role {idx}", "topic": f"topic {idx}", "action": f"action {idx}"}),
        f"prompt {idx}",
        f"text {idx}",
        Objectives(f1, f2),
    )


def test_hypervolume_and_spread_follow_local_convention():
    points = [(0.2, 0.9), (0.5, 0.6), (0.9, 0.2)]
    distances = [math.dist(points[0], points[1]), math.dist(points[1], points[2])]
    mean_distance = sum(distances) / len(distances)
    expected_spread = sum(abs(distance - mean_distance) for distance in distances) / (
        len(distances) * mean_distance
    )

    assert calculate_hypervolume(points) == 0.44
    assert math.isclose(calculate_spread(points), expected_spread)


def test_archive_metrics_normalize_fidelity_and_cosine_distance_diversity():
    metrics = archive_metrics([solution(-1.0, 1.5, 1), solution(1.0, 0.5, 2)])

    assert normalized_objective_point(solution(-1.0, 1.5, 1)) == (0.0, 0.75)
    assert metrics["hypervolume"] == 0.25
    assert metrics["spread"] is None


def test_progress_logger_writes_generation_line(test_config, tmp_path):
    test_config.set("logging.console", False)
    logger = ProgressLogger(test_config, tmp_path, run_index=1, total_runs=1)

    logger.generation(
        1,
        2,
        modified_count=3,
        population_size=4,
        archive_size=5,
        hypervolume=0.25,
        spread=None,
        archive_update_count=2,
        archive_prune_count=1,
    )
    logger.close()

    text = (tmp_path / "runtime.log").read_text(encoding="utf-8")
    assert "generation 1/2" in text
    assert "modified=3/4" in text
    assert "archive=5" in text
    assert "hv=0.250000" in text
    assert "spread=NA" in text
    assert "archive_updates=2" in text
    assert "archive_prunes=1" in text


def test_progress_logger_writes_archive_counts_on_finish(test_config, tmp_path):
    test_config.set("logging.console", False)
    logger = ProgressLogger(test_config, tmp_path, run_index=1, total_runs=1)

    logger.finish(tmp_path, archive_update_count=3, archive_prune_count=2)
    logger.close()

    text = (tmp_path / "runtime.log").read_text(encoding="utf-8")
    assert "finished" in text
    assert "archive_updates=3" in text
    assert "archive_prunes=2" in text


def test_progress_logger_writes_stage_and_generation_start(test_config, tmp_path):
    test_config.set("logging.console", False)
    logger = ProgressLogger(test_config, tmp_path, run_index=1, total_runs=1)

    logger.stage(3, 6, "Construyendo poblacion inicial")
    logger.generation_start(2, 5)
    logger.close()

    text = (tmp_path / "runtime.log").read_text(encoding="utf-8")
    assert "3/6 Construyendo poblacion inicial" in text
    assert "run 1/1 | generation 2/5 started" in text


def test_progress_logger_records_errors(test_config, tmp_path):
    test_config.set("logging.console", False)
    logger = ProgressLogger(test_config, tmp_path, run_index=1, total_runs=1)

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.exception("run failed")
    logger.close()

    text = (tmp_path / "runtime.log").read_text(encoding="utf-8")
    assert "ERROR" in text
    assert "run failed" in text
    assert "RuntimeError: boom" in text
