from __future__ import annotations

from dataclasses import dataclass

from binary_mopso_cd.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class ComponentSettings:
    order: list[str]
    frozen: set[str]
    expansion_order: list[str]
    max_words: dict[str, int]
    alpha: dict[str, float]

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "ComponentSettings":
        return cls(
            order=config.components,
            frozen=config.frozen_components,
            expansion_order=config.expansion_order
            + [component for component in config.components if component not in config.expansion_order],
            max_words={component: config.component_max_words(component) for component in config.components},
            alpha={component: config.component_alpha(component) for component in config.components},
        )

    @property
    def active(self) -> list[str]:
        return [component for component in self.order if component not in self.frozen]


@dataclass(frozen=True, slots=True)
class InitializationSettings:
    candidate_multiplier: int
    min_product_multiplier: int
    prompt_reduction_multiplier: int
    generated_sentences_max: int

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "InitializationSettings":
        return cls(
            candidate_multiplier=int(config.get("initialization.candidate_multiplier", 4)),
            min_product_multiplier=int(config.get("initialization.min_product_multiplier", 3)),
            prompt_reduction_multiplier=int(config.get("initialization.prompt_reduction_multiplier", 2)),
            generated_sentences_max=int(config.get("initialization.generated_sentences_max", 4)),
        )


@dataclass(frozen=True, slots=True)
class ParallelismSettings:
    enabled: bool
    particle_update_max_concurrent: int
    initial_text_generation_max_concurrent: int

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "ParallelismSettings":
        return cls(
            enabled=config.get("parallelism.enabled", True) is True,
            particle_update_max_concurrent=int(config.get("parallelism.particle_update_max_concurrent", 10)),
            initial_text_generation_max_concurrent=int(
                config.get("parallelism.initial_text_generation_max_concurrent", 10)
            ),
        )


@dataclass(frozen=True, slots=True)
class MOPSOSettings:
    archive_multiplier: float
    leader_tournament_size: int
    dmax: int
    kcand: int
    candidate_min_words: int
    candidate_max_words: int
    omega_max: float
    omega_min: float
    c1: float
    c2: float
    vmax: float
    alpha: float
    p_tur_max: float
    p_tur_min: float
    p_anchor_enabled: bool
    p_anchor_min: float
    p_anchor_max: float
    tau_dup: float
    tau_tur_min: float
    tau_tur_max: float
    utility_weights: dict[str, float]

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "MOPSOSettings":
        return cls(
            archive_multiplier=float(config.get("mopso.archive_multiplier", 2)),
            leader_tournament_size=int(config.get("mopso.leader_tournament_size", 3)),
            dmax=int(config.get("mopso.dmax", 1)),
            kcand=int(config.get("mopso.kcand", 7)),
            candidate_min_words=int(config.get("mopso.candidate_min_words", 2)),
            candidate_max_words=int(config.get("mopso.candidate_max_words", 8)),
            omega_max=float(config.get("mopso.omega_max", 0.9)),
            omega_min=float(config.get("mopso.omega_min", 0.4)),
            c1=float(config.get("mopso.c1", 1.5)),
            c2=float(config.get("mopso.c2", 1.5)),
            vmax=float(config.get("mopso.vmax", 4.0)),
            alpha=float(config.get("mopso.alpha", 1.0)),
            p_tur_max=float(config.get("mopso.p_tur_max", 0.07)),
            p_tur_min=float(config.get("mopso.p_tur_min", 0.02)),
            p_anchor_enabled=config.get("mopso.p_anchor_enabled", False) is True,
            p_anchor_min=float(config.get("mopso.p_anchor_min", 0.05)),
            p_anchor_max=float(config.get("mopso.p_anchor_max", 0.70)),
            tau_dup=float(config.get("mopso.tau_dup", 0.92)),
            tau_tur_min=float(config.get("mopso.tau_tur_min", 0.65)),
            tau_tur_max=float(config.get("mopso.tau_tur_max", 0.90)),
            utility_weights=dict(config.get("mopso.utility_weights", {"f1": 0.5, "f2": 0.5})),
        )


@dataclass(frozen=True, slots=True)
class CheckpointSettings:
    enabled: bool
    interval: int
    directory: str

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> "CheckpointSettings":
        return cls(
            enabled=config.checkpoint_enabled,
            interval=config.checkpoint_interval,
            directory=str(config.get("checkpoint.directory", "checkpoints")),
        )
