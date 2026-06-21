from __future__ import annotations

import json
import time
from collections import defaultdict
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

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def write(self, path: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        elapsed = self.elapsed_seconds()
        payload = {"runtime_seconds": elapsed, "total_sec": elapsed}
        if extra:
            payload.update(extra)
        lines = [f"{key}: {value}" for key, value in payload.items()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return payload


def _int_value(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return int(value)
    return 0


def summarize_llm_costs(llm_log_path: Path, wall_clock_seconds: float) -> dict[str, Any]:
    calls_by_task_model: defaultdict[str, int] = defaultdict(int)
    input_tokens_by_task_model: defaultdict[str, int] = defaultdict(int)
    output_tokens_by_task_model: defaultdict[str, int] = defaultdict(int)
    if llm_log_path.exists():
        with llm_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = f"{row.get('semantic_task', '')}::{row.get('model', '')}"
                input_tokens = _int_value(row, "input_tokens", "promptEvalCount", "prompt_eval_count")
                output_tokens = _int_value(row, "output_tokens", "evalCount", "eval_count")
                calls_by_task_model[key] += 1
                input_tokens_by_task_model[key] += input_tokens
                output_tokens_by_task_model[key] += output_tokens
    total_input = sum(input_tokens_by_task_model.values())
    total_output = sum(output_tokens_by_task_model.values())
    return {
        "wall_clock_seconds": wall_clock_seconds,
        "llm_calls_total": sum(calls_by_task_model.values()),
        "input_tokens_total": total_input,
        "output_tokens_total": total_output,
        "tokens_total": total_input + total_output,
        "calls_by_task_model": dict(calls_by_task_model),
        "input_tokens_by_task_model": dict(input_tokens_by_task_model),
        "output_tokens_by_task_model": dict(output_tokens_by_task_model),
    }


def write_cost_metrics(path: Path, llm_log_path: Path, wall_clock_seconds: float) -> dict[str, Any]:
    payload = summarize_llm_costs(llm_log_path, wall_clock_seconds)
    write_json(path, payload)
    return payload
