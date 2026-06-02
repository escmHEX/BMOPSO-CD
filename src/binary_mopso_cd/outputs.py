from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from binary_mopso_cd.entities import Solution, solution_to_dict
from binary_mopso_cd.utils import ensure_dir, timestamp_id


def create_run_dir(base_dir: Path, run_index: int | None = None) -> Path:
    root = ensure_dir(base_dir)
    name = timestamp_id() if run_index is None else f"{timestamp_id()}_run{run_index}"
    return ensure_dir(root / name)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def write_solutions(path: Path, solutions: list[Solution]) -> None:
    write_json(path, [solution_to_dict(solution) for solution in solutions])


class RuntimeTimer:
    def __init__(self):
        self.started = time.perf_counter()

    def write(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload = {"runtime_seconds": time.perf_counter() - self.started}
        if extra:
            payload.update(extra)
        lines = [f"{key}: {value}" for key, value in payload.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

