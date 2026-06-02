from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


@dataclass(slots=True)
class RuntimeConfig:
    data: dict[str, Any]

    @classmethod
    def load(cls, config_path: Path | None = None) -> "RuntimeConfig":
        default = load_yaml(DEFAULT_CONFIG_PATH)
        if config_path is None or config_path == DEFAULT_CONFIG_PATH:
            return cls(default)
        return cls(deep_merge(default, load_yaml(config_path)))

    def get(self, path: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        current = self.data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(self.data)

    @property
    def n(self) -> int:
        return int(self.get("experiment.n"))

    @property
    def iterations(self) -> int:
        return int(self.get("experiment.iterations"))

    @property
    def runs(self) -> int:
        return int(self.get("experiment.runs"))

    @property
    def seed(self) -> int:
        return int(self.get("experiment.seed"))

    @property
    def components(self) -> list[str]:
        return list(self.get("experiment.components", ["role", "topic", "action"]))

    @property
    def frozen_components(self) -> set[str]:
        return set(self.get("experiment.frozen_components", []))

    def validate(self) -> None:
        if self.n <= 0:
            raise ValueError("experiment.n must be positive")
        if self.iterations < 0:
            raise ValueError("experiment.iterations must be non-negative")
        if self.runs <= 0:
            raise ValueError("experiment.runs must be positive")
        unknown = self.frozen_components.difference(self.components)
        if unknown:
            raise ValueError(f"Frozen components are not defined components: {sorted(unknown)}")
        if len(self.frozen_components) >= len(self.components):
            raise ValueError("At least one component must remain active")
        if bool(self.get("ollama.speculative_decoding_enabled", False)):
            raise NotImplementedError(
                "Speculative decoding is intentionally blocked until Ollama exposes "
                "a concrete supported option for this project."
            )

