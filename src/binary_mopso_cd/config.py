from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DISABLED_THINKING_STRINGS = {"", "false", "0", "no", "none", "null"}
ENABLED_THINKING_STRINGS = {"true", "1", "yes"}
OLLAMA_THINKING_LEVELS = {"low", "medium", "high"}


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

    def has_path(self, path: str) -> bool:
        if not path or any(not part for part in path.split(".")):
            return False
        current: Any = self.data
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        current = self.data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def set_existing(self, path: str, value: Any) -> None:
        if not self.has_path(path):
            raise ValueError(f"Unknown configuration path: {path}")
        self.set(path, value)

    def ollama_model_options(self) -> list[str]:
        raw_options = self.get("ollama.model_options", [])
        if not isinstance(raw_options, list):
            raise ValueError("ollama.model_options must be a list")
        return [str(model).strip() for model in raw_options if str(model).strip()]

    def ollama_model_capabilities(self) -> dict[str, dict[str, Any]]:
        raw_capabilities = self.get("ollama.model_capabilities", {})
        if not isinstance(raw_capabilities, dict):
            raise ValueError("ollama.model_capabilities must be a mapping")
        capabilities: dict[str, dict[str, Any]] = {}
        for model, values in raw_capabilities.items():
            if not isinstance(values, dict):
                raise ValueError(f"ollama.model_capabilities.{model} must be a mapping")
            capabilities[str(model)] = dict(values)
        return capabilities

    def ollama_model_profiles(self) -> dict[str, dict[str, Any]]:
        raw_profiles = self.get("ollama.model_profiles", {})
        if not isinstance(raw_profiles, dict):
            raise ValueError("ollama.model_profiles must be a mapping")
        profiles: dict[str, dict[str, Any]] = {}
        for model, values in raw_profiles.items():
            if not isinstance(values, dict):
                raise ValueError(f"ollama.model_profiles.{model} must be a mapping")
            profiles[str(model)] = deepcopy(values)
        return profiles

    def model_supports_thinking(self, model: str) -> bool:
        capabilities = self.ollama_model_capabilities()
        if model not in capabilities:
            raise ValueError(f"Model {model!r} is not declared in ollama.model_capabilities")
        return bool(capabilities[model].get("thinking", False))

    def resolved_task_model(self, semantic_task: str, operation_context: str | None = None) -> str:
        if operation_context:
            phase_model = self.get(f"router.phase_task_models.{operation_context}.{semantic_task}")
            if phase_model is not None:
                return str(phase_model)
        return str(self.get(f"router.task_models.{semantic_task}") or self.get("ollama.default_model"))

    def resolved_task_thinking(self, semantic_task: str, llm_params_thinking: Any = None) -> bool | str | None:
        task_thinking = self.get(f"router.task_thinking.{semantic_task}")
        if task_thinking is not None:
            return task_thinking
        if llm_params_thinking is not None:
            return llm_params_thinking
        return self.get("ollama.think", False)

    def _thinking_enabled(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() not in DISABLED_THINKING_STRINGS
        return bool(value)

    def _validate_thinking_value(self, value: Any, path: str) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in DISABLED_THINKING_STRINGS | ENABLED_THINKING_STRINGS | OLLAMA_THINKING_LEVELS:
                return
        raise ValueError(f"{path} must be false, true, null, low, medium, or high")

    def _thinking_entries(self, value: Any, path: str) -> list[tuple[str, Any]]:
        if isinstance(value, dict):
            entries: list[tuple[str, Any]] = []
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key == "thinking":
                    entries.append((child_path, child))
                else:
                    entries.extend(self._thinking_entries(child, child_path))
            return entries
        if isinstance(value, list):
            entries: list[tuple[str, Any]] = []
            for index, child in enumerate(value):
                entries.extend(self._thinking_entries(child, f"{path}[{index}]"))
            return entries
        return []

    def _thinking_values(self, value: Any) -> list[Any]:
        return [entry_value for _, entry_value in self._thinking_entries(value, "")]

    def _task_llm_params_thinking_entries(self, semantic_task: str) -> list[tuple[str, Any]]:
        return self._thinking_entries(
            self.get(f"router.llm_params.{semantic_task}", {}),
            f"router.llm_params.{semantic_task}",
        )

    def _task_llm_params_thinking(self, semantic_task: str) -> Any:
        values = [value for _, value in self._task_llm_params_thinking_entries(semantic_task)]
        if any(self._thinking_enabled(value) for value in values):
            return True
        if values:
            return False
        return self.get("ollama.think", False)

    def _phase_contexts_for_task(self, semantic_task: str) -> list[str | None]:
        contexts: list[str | None] = [None]
        raw_phase_task_models = self.get("router.phase_task_models", {})
        if isinstance(raw_phase_task_models, dict):
            for context, task_models in raw_phase_task_models.items():
                if isinstance(task_models, dict) and semantic_task in task_models:
                    contexts.append(str(context))
        return contexts

    def validate_task_thinking_support(self) -> None:
        task_models = self.get("router.task_models", {})
        if not isinstance(task_models, dict):
            raise ValueError("router.task_models must be a mapping")
        task_thinking = self.get("router.task_thinking", {})
        if not isinstance(task_thinking, dict):
            raise ValueError("router.task_thinking must be a mapping")
        self._validate_thinking_value(self.get("ollama.think", False), "ollama.think")
        for semantic_task, thinking_value in task_thinking.items():
            self._validate_thinking_value(thinking_value, f"router.task_thinking.{semantic_task}")
        for thinking_path, thinking_value in self._thinking_entries(self.get("router.llm_params", {}), "router.llm_params"):
            self._validate_thinking_value(thinking_value, thinking_path)
        for semantic_task in task_models:
            thinking = self.resolved_task_thinking(str(semantic_task), self._task_llm_params_thinking(str(semantic_task)))
            if not self._thinking_enabled(thinking):
                continue
            for context in self._phase_contexts_for_task(str(semantic_task)):
                model = self.resolved_task_model(str(semantic_task), context)
                try:
                    supports_thinking = self.model_supports_thinking(model)
                except ValueError as exc:
                    raise ValueError(f"{model} for {semantic_task} is not declared in ollama.model_capabilities") from exc
                if not supports_thinking:
                    context_label = f" in {context}" if context else ""
                    raise ValueError(f"{model} for {semantic_task}{context_label} does not support thinking")

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
        active_count = len(self.components) - len(self.frozen_components)
        dmax = int(self.get("mopso.dmax", 1))
        if dmax <= 0:
            raise ValueError("mopso.dmax must be positive")
        if active_count > 0 and dmax > active_count:
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
        self.ollama_model_capabilities()
        self.ollama_model_profiles()
        if bool(self.get("ollama.speculative_decoding_enabled", False)):
            raise NotImplementedError(
                "Speculative decoding is intentionally blocked until Ollama exposes "
                "a concrete supported option for this project."
            )
        self.validate_task_thinking_support()
