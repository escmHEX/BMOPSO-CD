from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from binary_mopso_cd.config import RuntimeConfig


class ProgressLogger:
    def __init__(self, config: RuntimeConfig, outdir: Path, run_index: int, total_runs: int):
        self.enabled = bool(config.get("logging.enabled", True))
        self.run_index = run_index
        self.total_runs = total_runs
        self.started = time.perf_counter()
        self.logger = logging.getLogger(f"binary_mopso_cd.progress.{id(self)}")
        self.logger.handlers.clear()
        self.logger.propagate = False
        level_name = str(config.get("logging.level", "INFO")).upper()
        self.logger.setLevel(getattr(logging, level_name, logging.INFO))
        if not self.enabled:
            self.logger.disabled = True
            return
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        if bool(config.get("logging.console", True)):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        configured_file = config.get("logging.file", "runtime.log")
        log_file = "" if configured_file is None else str(configured_file)
        if log_file:
            file_handler = logging.FileHandler(outdir / log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def start(self, n: int, iterations: int, outdir: Path) -> None:
        self.info(
            "run %s/%s started | n=%s | iterations=%s | outdir=%s",
            self.run_index,
            self.total_runs,
            n,
            iterations,
            outdir,
        )

    def stage(self, index: int, total: int, label: str) -> None:
        self.info("%s/%s %s", index, total, label)

    def generation_start(self, generation: int, total_generations: int) -> None:
        self.info(
            "run %s/%s | generation %s/%s started | elapsed=%s",
            self.run_index,
            self.total_runs,
            generation,
            total_generations,
            format_elapsed(time.perf_counter() - self.started),
        )

    def generation(
        self,
        generation: int,
        total_generations: int,
        modified_count: int,
        population_size: int,
        archive_size: int,
        hypervolume: float | None,
        guided_candidate_rejections: int,
        archive_update_count: int,
        archive_prune_count: int,
    ) -> None:
        self.info(
            (
                "run %s/%s | generation %s/%s | modified=%s/%s | archive=%s | hv=%s "
                "| guided_candidate_rejections=%s | archive_updates=%s | archive_prunes=%s | elapsed=%s"
            ),
            self.run_index,
            self.total_runs,
            generation,
            total_generations,
            modified_count,
            population_size,
            archive_size,
            format_optional_float(hypervolume),
            guided_candidate_rejections,
            archive_update_count,
            archive_prune_count,
            format_elapsed(time.perf_counter() - self.started),
        )

    def finish(
        self,
        outdir: Path,
        archive_update_count: int,
        archive_prune_count: int,
    ) -> None:
        self.info(
            (
                "run %s/%s finished | archive_updates=%s | archive_prunes=%s "
                "| elapsed=%s | outdir=%s"
            ),
            self.run_index,
            self.total_runs,
            archive_update_count,
            archive_prune_count,
            format_elapsed(time.perf_counter() - self.started),
            outdir,
        )

    def exception(self, message: str, *args: Any) -> None:
        if self.enabled:
            self.logger.exception(message, *args)
            self.flush()

    def info(self, message: str, *args: Any) -> None:
        if self.enabled:
            self.logger.info(message, *args)
            self.flush()

    def flush(self) -> None:
        for handler in self.logger.handlers:
            handler.flush()

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)


def format_optional_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6f}"


def format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
