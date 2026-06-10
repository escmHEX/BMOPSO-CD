from __future__ import annotations

import base64
import hashlib
import json
import math
import pickle
import re
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Iterable

import numpy as np


_SPACE_RE = re.compile(r"\s+")


def canonical_text(text: str) -> str:
    value = _SPACE_RE.sub(" ", str(text).strip()).lower()
    return value.strip("\"'` ")


def word_count(text: str) -> int:
    return len([part for part in _SPACE_RE.split(text.strip()) if part])


def stable_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def progress_ratio(index: int, total: int) -> float:
    if total <= 1:
        if index < 0:
            raise ValueError("progress index must be non-negative")
        return 0.0
    if index < 0 or index > total - 1:
        raise ValueError(f"progress index {index} is outside [0, {total - 1}]")
    return float(index / (total - 1))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def rng_to_text(rng: Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def rng_from_text(seed: int, state_text: str | None) -> Random:
    rng = Random(seed)
    if state_text:
        rng.setstate(pickle.loads(base64.b64decode(state_text.encode("ascii"))))
    return rng


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = canonical_text(value)
        if key and key not in seen:
            seen.add(key)
            result.append(_SPACE_RE.sub(" ", value.strip()))
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number
