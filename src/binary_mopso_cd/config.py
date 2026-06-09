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
        configured = list(self.get("semantic_components.order", self.get("experiment.components", ["role", "topic", "action"])))
        return configured

    @property
    def frozen_components(self) -> set[str]:
        return set(self.get("experiment.frozen_components", []))

    @property
    def checkpoint_enabled(self) -> bool:
        return bool(self.get("checkpoint.enabled", False))

    @property
    def checkpoint_interval(self) -> int:
        return int(self.get("checkpoint.interval", 1))

    def component_rule(self, component: str) -> dict[str, Any]:
        rule = self.get(f"semantic_components.rules.{component}", {})
        if not isinstance(rule, dict):
            raise ValueError(f"semantic_components.rules.{component} must be a mapping")
        return dict(rule)

    def component_max_words(self, component: str) -> int:
        return int(self.component_rule(component).get("max_words", 8))

    def component_alpha(self, component: str) -> float:
        return float(self.component_rule(component).get("alpha", 1.0))

    @property
    def expansion_order(self) -> list[str]:
        return list(self.get("semantic_components.expansion_order", ["role", "action", "topic"]))

    def validate(self) -> None:
        if self.n <= 0:
            raise ValueError("experiment.n must be positive")
        if self.iterations < 0:
            raise ValueError("experiment.iterations must be non-negative")
        if self.runs <= 0:
            raise ValueError("experiment.runs must be positive")
        if not self.components:
            raise ValueError("semantic_components.order must contain at least one component")
        duplicated = {component for component in self.components if self.components.count(component) > 1}
        if duplicated:
            raise ValueError(f"semantic_components.order contains duplicates: {sorted(duplicated)}")
        expansion_order = self.expansion_order
        unknown_expansion = set(expansion_order).difference(self.components)
        if unknown_expansion:
            raise ValueError(f"semantic_components.expansion_order has unknown components: {sorted(unknown_expansion)}")
        configured_components = list(self.get("experiment.components", self.components))
        if configured_components != self.components:
            self.set("experiment.components", list(self.components))
        unknown = self.frozen_components.difference(self.components)
        if unknown:
            raise ValueError(f"Frozen components are not defined components: {sorted(unknown)}")
        if len(self.frozen_components) >= len(self.components):
            raise ValueError("At least one component must remain active")
        active_count = len(self.components) - len(self.frozen_components)
        dmax = int(self.get("mopso.dmax", 1))
        if dmax <= 0:
            raise ValueError("mopso.dmax must be positive")
        if dmax > active_count:
            raise ValueError("mopso.dmax must not exceed the number of active components")
        if int(self.get("mopso.kcand", 7)) <= 0:
            raise ValueError("mopso.kcand must be positive")
        if int(self.get("mopso.leader_tournament_size", 3)) <= 0:
            raise ValueError("mopso.leader_tournament_size must be positive")
        if float(self.get("mopso.archive_multiplier", 2)) <= 0:
            raise ValueError("mopso.archive_multiplier must be positive")
        for path in [
            "mopso.omega_max",
            "mopso.omega_min",
            "mopso.c1",
            "mopso.c2",
            "mopso.vmax",
            "mopso.alpha",
            "mopso.p_tur_max",
            "mopso.p_tur_min",
            "mopso.p_anchor_min",
            "mopso.p_anchor_max",
            "mopso.tau_dup",
            "mopso.tau_tur_min",
            "mopso.tau_tur_max",
        ]:
            value = float(self.get(path))
            if value < 0:
                raise ValueError(f"{path} must be non-negative")
        if float(self.get("mopso.p_tur_min")) > float(self.get("mopso.p_tur_max")):
            raise ValueError("mopso.p_tur_min must not exceed mopso.p_tur_max")
        p_anchor_enabled = self.get("mopso.p_anchor_enabled", False)
        if not isinstance(p_anchor_enabled, bool):
            raise ValueError("mopso.p_anchor_enabled must be boolean")
        p_anchor_min = float(self.get("mopso.p_anchor_min", 0.05))
        p_anchor_max = float(self.get("mopso.p_anchor_max", 0.70))
        if p_anchor_min > p_anchor_max:
            raise ValueError("mopso.p_anchor_min must not exceed mopso.p_anchor_max")
        if p_anchor_max > 1.0:
            raise ValueError("mopso.p_anchor_max must be in [0, 1]")
        if float(self.get("mopso.tau_tur_min")) > float(self.get("mopso.tau_tur_max")):
            raise ValueError("mopso.tau_tur_min must not exceed mopso.tau_tur_max")
        if int(self.get("mopso.k_retry", 0)) != 0:
            raise ValueError("mopso.k_retry must remain 0 for the specified strategy")
        parallelism_enabled = self.get("parallelism.enabled", True)
        if not isinstance(parallelism_enabled, bool):
            raise ValueError("parallelism.enabled must be boolean")
        for path in [
            "parallelism.particle_update_max_concurrent",
            "parallelism.initial_text_generation_max_concurrent",
        ]:
            if int(self.get(path, 10)) <= 0:
                raise ValueError(f"{path} must be positive")
        tau_gen_min = float(self.get("generated_text_validation.tau_gen_min", 0.05))
        if tau_gen_min < -1.0 or tau_gen_min > 1.0:
            raise ValueError("generated_text_validation.tau_gen_min must be in [-1, 1]")
        if int(self.get("initialization.candidate_multiplier", 4)) <= 0:
            raise ValueError("initialization.candidate_multiplier must be positive")
        if int(self.get("initialization.prompt_reduction_multiplier", 2)) <= 0:
            raise ValueError("initialization.prompt_reduction_multiplier must be positive")
        if int(self.get("initialization.min_product_multiplier", 3)) <= 0:
            raise ValueError("initialization.min_product_multiplier must be positive")
        for component in self.components:
            if self.component_max_words(component) <= 0:
                raise ValueError(f"semantic_components.rules.{component}.max_words must be positive")
            if self.component_alpha(component) <= 0:
                raise ValueError(f"semantic_components.rules.{component}.alpha must be positive")
        if bool(self.checkpoint_enabled) and self.checkpoint_interval <= 0:
            raise ValueError("checkpoint.interval must be positive when checkpoint.enabled is true")
        if int(self.get("models.sbert.batch_size", 64)) <= 0:
            raise ValueError("models.sbert.batch_size must be positive")
        if int(self.get("models.distilbert.top_k_multiplier", 3)) <= 0:
            raise ValueError("models.distilbert.top_k_multiplier must be positive")
        if bool(self.get("models.ppdb.enabled", True)):
            if not str(self.get("models.ppdb.index_path", "")).strip():
                raise ValueError("models.ppdb.index_path must be configured when PPDB is enabled")
        if int(self.get("selection.k", 5)) <= 0:
            raise ValueError("selection.k must be positive")
        lambda_mmr = float(self.get("selection.lambda_mmr", 0.35))
        if lambda_mmr < 0 or lambda_mmr > 1:
            raise ValueError("selection.lambda_mmr must be in [0, 1]")
        if float(self.get("selection.tau_min", 0.20)) > float(self.get("selection.tau_max", 0.94)):
            raise ValueError("selection.tau_min must not exceed selection.tau_max")
        if str(self.get("logging.level", "INFO")).upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("logging.level must be a valid Python logging level")
        if bool(self.get("ollama.speculative_decoding_enabled", False)):
            raise NotImplementedError(
                "Speculative decoding is intentionally blocked until Ollama exposes "
                "a concrete supported option for this project."
            )
