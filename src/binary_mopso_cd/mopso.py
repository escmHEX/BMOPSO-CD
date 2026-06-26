from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any
from uuid import uuid4

import numpy as np

from binary_mopso_cd.async_utils import run_async, run_limited
from binary_mopso_cd.checkpoint import CheckpointManager
from binary_mopso_cd.component_memory import ComponentMemoryIndex
from binary_mopso_cd.component_specs import component_spec
from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.entities import Objectives, SemanticVector, Solution, solution_to_dict
from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.generated_text_validation import validate_generated_text
from binary_mopso_cd.llm_prompts import build_anchored_text_generation_user_prompt
from binary_mopso_cd.metrics import archive_metrics
from binary_mopso_cd.monitor import ObservationalMonitor
from binary_mopso_cd.objectives import evaluate_solutions, semantic_fidelity_scores
from binary_mopso_cd.router import (
    TASK_INFLUENCE,
    TASK_PROMPT_RENDERING,
    TASK_SYNTHETIC_TEXT,
    TASK_WORD_REPLACEMENT,
    RouteTask,
    SemanticRouter,
)
from binary_mopso_cd.services.embedding import EmbeddingService
from binary_mopso_cd.settings import CheckpointSettings, ComponentSettings, MOPSOSettings, ParallelismSettings
from binary_mopso_cd.utils import canonical_text, progress_ratio, rng_to_text, stable_digest, word_count


SolutionSignature = tuple[tuple[str, str], ...]
GUIDED_MOVES = {"cognitive", "social"}
GUIDED_TRAJECTORY_EPSILON = 1e-8
REASON_EMPTY_CANDIDATE = "empty_candidate"
REASON_DUPLICATE_CANDIDATE_OUTPUT = "duplicate_candidate_output"
REASON_LITERAL_COPY_CURRENT = "literal_copy_current"
REASON_LITERAL_COPY_TARGET = "literal_copy_target"
REASON_TOO_MANY_WORDS = "too_many_words"
REASON_SEMANTIC_DUPLICATE = "semantic_duplicate"
REASON_NO_SEMANTIC_PROGRESS = "no_semantic_progress"
REASON_GUIDED_TRAJECTORY_INCONSISTENT = "guided_trajectory_inconsistent"


def dominates(left: Objectives, right: Objectives) -> bool:
    return left.f1 >= right.f1 and left.f2 >= right.f2 and (left.f1 > right.f1 or left.f2 > right.f2)


def utility(objectives: Objectives, weights: dict[str, float] | None = None) -> float:
    weights = weights or {"f1": 0.5, "f2": 0.5}
    f1_norm = (objectives.f1 + 1.0) / 2.0
    f2_norm = objectives.f2 / 2.0
    return float(weights.get("f1", 0.5) * f1_norm + weights.get("f2", 0.5) * f2_norm)


def semantic_velocity_delta(left_embedding: np.ndarray, right_embedding: np.ndarray) -> float:
    return (1.0 - float(left_embedding @ right_embedding)) / 2.0


def guided_trajectory_consistency_mask(
    current_target_similarity: float,
    candidate_current_similarities: np.ndarray,
    candidate_target_similarities: np.ndarray,
    relative_margin: float,
) -> np.ndarray:
    direct_path = float(np.arccos(np.clip(current_target_similarity, -1.0, 1.0)))
    candidate_path = np.arccos(np.clip(candidate_current_similarities, -1.0, 1.0)) + np.arccos(
        np.clip(candidate_target_similarities, -1.0, 1.0)
    )
    return candidate_path <= (1.0 + relative_margin) * direct_path + GUIDED_TRAJECTORY_EPSILON


def crowding_distance(solutions: list[Solution]) -> dict[str, float]:
    distances = {solution.solution_id: 0.0 for solution in solutions}
    if len(solutions) <= 2:
        return {solution.solution_id: math.inf for solution in solutions}
    for objective_name in ["f1", "f2"]:
        ordered = sorted(solutions, key=lambda item: getattr(item.objectives, objective_name))
        distances[ordered[0].solution_id] = math.inf
        distances[ordered[-1].solution_id] = math.inf
        min_value = getattr(ordered[0].objectives, objective_name)
        max_value = getattr(ordered[-1].objectives, objective_name)
        if max_value == min_value:
            continue
        for idx in range(1, len(ordered) - 1):
            previous_value = getattr(ordered[idx - 1].objectives, objective_name)
            next_value = getattr(ordered[idx + 1].objectives, objective_name)
            if not math.isinf(distances[ordered[idx].solution_id]):
                distances[ordered[idx].solution_id] += (next_value - previous_value) / (max_value - min_value)
    return distances


def non_dominated(solutions: list[Solution]) -> list[Solution]:
    result: list[Solution] = []
    for idx, candidate in enumerate(solutions):
        if candidate.objectives is None:
            continue
        dominated = False
        for other_idx, other in enumerate(solutions):
            if idx == other_idx or other.objectives is None:
                continue
            if dominates(other.objectives, candidate.objectives):
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return result


def deduplicate_solutions_by_signature(solutions: list[Solution]) -> list[Solution]:
    first_by_signature: dict[SolutionSignature, Solution] = {}
    for solution in solutions:
        signature = solution.vector.signature()
        if signature not in first_by_signature:
            first_by_signature[signature] = solution
    return list(first_by_signature.values())


def copy_evaluation(source: Solution, target: Solution) -> None:
    target.objectives = None if source.objectives is None else Objectives(source.objectives.f1, source.objectives.f2)
    target.embedding = None if source.embedding is None else list(source.embedding)


def evaluate_unique_solutions_by_signature(
    solutions: list[Solution],
    reference_text: str,
    embedding_service: EmbeddingService,
) -> list[Solution]:
    unique_solutions = deduplicate_solutions_by_signature(solutions)
    evaluate_solutions(unique_solutions, reference_text, embedding_service)
    evaluated_by_signature = {solution.vector.signature(): solution for solution in unique_solutions}
    for solution in solutions:
        copy_evaluation(evaluated_by_signature[solution.vector.signature()], solution)
    return solutions


@dataclass
class ExternalArchive:
    max_size: int
    rng: Random
    solutions: list[Solution] = field(default_factory=list)
    update_count: int = 0
    prune_count: int = 0

    def update(self, candidates: list[Solution]) -> list[Solution]:
        previous_signatures = self._signature_state()
        merged = deduplicate_solutions_by_signature(self.solutions + [candidate.clone() for candidate in candidates])
        self.solutions = non_dominated(merged)
        pruned = len(self.solutions) > self.max_size
        if pruned:
            self.prune_count += 1
        while len(self.solutions) > self.max_size:
            distances = crowding_distance(self.solutions)
            finite_values = [value for value in distances.values() if not math.isinf(value)]
            min_distance = min(finite_values) if finite_values else min(distances.values())
            tied = [solution for solution in self.solutions if distances[solution.solution_id] == min_distance]
            remove = self.rng.choice(tied)
            self.solutions = [solution for solution in self.solutions if solution.solution_id != remove.solution_id]
        if self._signature_state() != previous_signatures:
            self.update_count += 1
        return self.solutions

    def _signature_state(self) -> tuple[SolutionSignature, ...]:
        return tuple(solution.vector.signature() for solution in self.solutions)

    def select_leader(self, tournament_size: int) -> Solution:
        if not self.solutions:
            raise RuntimeError("Cannot select leader from an empty archive")
        q_eff = min(tournament_size, len(self.solutions))
        participants = self.rng.sample(self.solutions, q_eff)
        distances = crowding_distance(self.solutions)
        best_distance = max(distances[p.solution_id] for p in participants)
        tied = [p for p in participants if distances[p.solution_id] == best_distance]
        return self.rng.choice(tied)


def archive_capacity(population_size: int, multiplier: float) -> int:
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if multiplier <= 0:
        raise ValueError("archive multiplier must be positive")
    return max(1, int(math.ceil(multiplier * population_size)))


class PBestUpdater:
    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def choose(self, current: Solution, previous: Solution) -> Solution:
        if current.objectives is None or previous.objectives is None:
            raise ValueError("pbest update requires evaluated solutions")
        if dominates(current.objectives, previous.objectives):
            return current.clone()
        if dominates(previous.objectives, current.objectives):
            return previous
        if utility(current.objectives, self.weights) > utility(previous.objectives, self.weights):
            return current.clone()
        return previous


@dataclass(frozen=True, slots=True)
class ParticleUpdateJob:
    index: int
    particle: Solution
    pbest: Solution
    leader: Solution
    generation: int


@dataclass(frozen=True, slots=True)
class ParticleUpdateResult:
    index: int
    solution: Solution
    error: str | None = None
    optimization_rejections: tuple[dict[str, Any], ...] = ()
    guided_candidate_diagnostics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PopulationUpdateResult:
    population: list[Solution]
    guided_candidate_rejections: int = 0


@dataclass(frozen=True, slots=True)
class MovementContext:
    effective_mode: str
    target_position: dict[str, str]
    used_inertia: bool = False


def particle_update_seed(base_seed: int, generation: int, index: int) -> int:
    digest = stable_digest({"seed": base_seed, "generation": generation, "particle_index": index})
    return int(digest[:16], 16)


class BinaryMOPSOCDEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        router: SemanticRouter,
        executor: SemanticTaskExecutor,
        rng: Random,
        outdir: Path,
        reference_text: str,
        central_anchors: list[str] | None = None,
        progress_logger: Any | None = None,
        semantic_anchors: dict[str, list[str]] | None = None,
    ):
        self.config = config
        self.router = router
        self.executor = executor
        self.rng = rng
        self.outdir = outdir
        self.reference_text = reference_text
        self.central_anchors = list(central_anchors or [])
        self.semantic_anchors = {key: list(values) for key, values in (semantic_anchors or {}).items()}
        self.progress_logger = progress_logger
        self.components = ComponentSettings.from_config(config)
        self.mopso = MOPSOSettings.from_config(config)
        self.parallelism = ParallelismSettings.from_config(config)
        self.tau_gen_min = float(config.get("generated_text_validation.tau_gen_min", 0.05))
        checkpoint = CheckpointSettings.from_config(config)
        self.archive = ExternalArchive(max_size=archive_capacity(config.n, self.mopso.archive_multiplier), rng=rng)
        self.pbest_updater = PBestUpdater(self.mopso.utility_weights)
        self.component_memory = ComponentMemoryIndex(self.components.order, executor.embedding_service)
        self.monitor = ObservationalMonitor(
            enabled=bool(config.get("monitor.enabled", True)),
            spacy_model=str(config.get("models.spacy.model", "en_core_web_sm")),
            embedding_service=executor.embedding_service,
        )
        self.checkpoints = CheckpointManager(
            enabled=checkpoint.enabled,
            outdir=outdir,
            directory_name=checkpoint.directory,
            interval=checkpoint.interval,
        )

    def run(
        self,
        initial_population: list[Solution],
        start_generation: int = 0,
        pbest_state: list[Solution] | None = None,
        archive_state: list[Solution] | None = None,
        archive_stats: dict[str, int] | None = None,
        component_memory: dict[str, list[str]] | None = None,
    ) -> tuple[list[Solution], ExternalArchive]:
        population = [solution.clone(keep_id=True) for solution in initial_population]
        pbest = [solution.clone() for solution in (pbest_state or population)]
        if archive_state is not None:
            if archive_stats is None:
                raise ValueError("Checkpoint is missing archive_stats; cannot resume archive metrics accurately")
            self.archive.solutions = [solution.clone() for solution in archive_state]
            self.archive.update_count = archive_stats["update_count"]
            self.archive.prune_count = archive_stats["prune_count"]
        else:
            self.archive.update(population)
        if component_memory is not None:
            self.component_memory = ComponentMemoryIndex.from_snapshot(
                self.components.order,
                self.executor.embedding_service,
                component_memory,
            )
        else:
            self.component_memory.add_solutions(population)
        metrics_rows: list[dict[str, Any]] = []
        monitor_rows: list[dict[str, Any]] = []
        if not self.components.active:
            try:
                self._write_metrics(metrics_rows)
                self._write_monitor_metrics(monitor_rows)
                return population, self.archive
            finally:
                self.checkpoints.close()
        try:
            for generation in range(start_generation + 1, self.config.iterations + 1):
                if self.progress_logger is not None:
                    self.progress_logger.generation_start(generation, self.config.iterations)
                population_update = self._update_population(population, pbest, generation)
                next_population = population_update.population
                modified_count = sum(1 for solution in next_population if solution.changed)
                comparison_batch = next_population + pbest + self.archive.solutions
                evaluate_unique_solutions_by_signature(
                    comparison_batch,
                    self.reference_text,
                    self.executor.embedding_service,
                )
                next_population = comparison_batch[: len(next_population)]
                pbest_candidates = comparison_batch[len(next_population) : len(next_population) + len(pbest)]
                archive_revalued = comparison_batch[len(next_population) + len(pbest) :]
                self.archive.solutions = archive_revalued
                pbest = [
                    self.pbest_updater.choose(current, previous)
                    for current, previous in zip(next_population, pbest_candidates, strict=True)
                ]
                self.archive.update(next_population)
                self.component_memory.add_solutions(next_population)
                population = next_population
                row = self._generation_metrics(
                    generation,
                    population,
                    modified_count,
                    population_update.guided_candidate_rejections,
                )
                metrics_rows.append(row)
                if self.progress_logger is not None:
                    self.progress_logger.generation(
                        generation,
                        self.config.iterations,
                        modified_count,
                        len(population),
                        len(self.archive.solutions),
                        row["hypervolume"],
                        row["guided_candidate_rejections"],
                        row["archive_update_count"],
                        row["archive_prune_count"],
                    )
                monitor_result = self.monitor.observe(generation, population)
                if self.monitor.enabled:
                    monitor_rows.append(
                        {
                            **monitor_result.metrics,
                            "monitor_overhead_seconds": monitor_result.overhead_seconds,
                        }
                    )
                self._write_archive_history(generation)
                self.checkpoints.submit(generation, self._checkpoint_payload(generation, population, pbest, metrics_rows))
        finally:
            self.checkpoints.close()
        self._write_metrics(metrics_rows)
        self._write_monitor_metrics(monitor_rows)
        return population, self.archive

    def _update_population(self, population: list[Solution], pbest: list[Solution], generation: int) -> PopulationUpdateResult:
        if not self.parallelism.enabled or self.parallelism.particle_update_max_concurrent <= 1:
            next_population = []
            guided_diagnostics: list[dict[str, Any]] = []
            for index, particle in enumerate(population):
                leader = self.archive.select_leader(self.mopso.leader_tournament_size)
                updated = self._update_particle(
                    particle,
                    pbest[index],
                    leader,
                    generation,
                    guided_diagnostics_sink=guided_diagnostics,
                )
                next_population.append(updated)
            self._write_guided_candidate_diagnostics(guided_diagnostics)
            return PopulationUpdateResult(
                next_population,
                self._guided_candidate_rejection_count(guided_diagnostics),
            )
        leaders = [self.archive.select_leader(self.mopso.leader_tournament_size) for _particle in population]
        jobs = [
            ParticleUpdateJob(
                index=index,
                particle=particle.clone(keep_id=True),
                pbest=pbest[index].clone(),
                leader=leaders[index].clone(),
                generation=generation,
            )
            for index, particle in enumerate(population)
        ]
        results = run_async(
            run_limited(
                jobs,
                self.parallelism.particle_update_max_concurrent,
                self._update_particle_job_async,
            )
        )
        results.sort(key=lambda result: result.index)
        guided_diagnostics = [row for result in results for row in result.guided_candidate_diagnostics]
        self._write_optimization_rejections(
            [row for result in results for row in result.optimization_rejections]
        )
        self._write_guided_candidate_diagnostics(guided_diagnostics)
        self._write_particle_update_errors(
            [
                {
                    "phase": "optimization",
                    "generation": result.solution.generation or generation,
                    "index": result.index,
                    "solution_id": result.solution.solution_id,
                    "error": result.error,
                }
                for result in results
                if result.error is not None
            ]
        )
        return PopulationUpdateResult(
            [result.solution for result in results],
            self._guided_candidate_rejection_count(guided_diagnostics),
        )

    async def _update_particle_job_async(self, job: ParticleUpdateJob) -> ParticleUpdateResult:
        rng = Random(particle_update_seed(self.config.seed, job.generation, job.index))
        rejections: list[dict[str, Any]] = []
        guided_diagnostics: list[dict[str, Any]] = []
        try:
            solution = await self._update_particle_async(
                job.particle,
                job.pbest,
                job.leader,
                job.generation,
                rng=rng,
                rejection_sink=rejections,
                guided_diagnostics_sink=guided_diagnostics,
            )
            return ParticleUpdateResult(
                job.index,
                solution,
                optimization_rejections=tuple(rejections),
                guided_candidate_diagnostics=tuple(guided_diagnostics),
            )
        except Exception as exc:
            fallback = job.particle.clone(keep_id=True)
            fallback.changed = False
            fallback.generation = job.generation
            return ParticleUpdateResult(
                job.index,
                fallback,
                error=str(exc),
                optimization_rejections=tuple(rejections),
                guided_candidate_diagnostics=tuple(guided_diagnostics),
            )

    def _update_particle(
        self,
        particle: Solution,
        pbest: Solution,
        leader: Solution,
        generation: int,
        rng: Random | None = None,
        rejection_sink: list[dict[str, Any]] | None = None,
        guided_diagnostics_sink: list[dict[str, Any]] | None = None,
    ) -> Solution:
        rng = rng or self.rng
        updated = particle.clone(keep_id=True)
        updated.changed = False
        active_components = self.components.active
        candidates: list[tuple[str, str, float]] = []
        schedule_index = generation - 1
        rho = progress_ratio(schedule_index, self.config.iterations)
        omega = self.mopso.omega_max - (self.mopso.omega_max - self.mopso.omega_min) * rho
        p_tur = self.mopso.p_tur_max - (self.mopso.p_tur_max - self.mopso.p_tur_min) * rho
        for component in active_components:
            current = particle.vector.components[component]
            pbest_value = pbest.vector.components[component]
            leader_value = leader.vector.components[component]
            embeddings = self.executor.embedding_service.encode([current, pbest_value, leader_value], text_type="component")
            delta_p = semantic_velocity_delta(embeddings[0], embeddings[1])
            delta_l = semantic_velocity_delta(embeddings[0], embeddings[2])
            r1 = rng.random()
            r2 = rng.random()
            previous_velocity = float(particle.velocity.get(component, 0.0))
            s_in_velocity = omega * previous_velocity
            s_in_weight = omega * abs(previous_velocity)
            s_cog = self.mopso.c1 * r1 * delta_p
            s_soc = self.mopso.c2 * r2 * delta_l
            raw_velocity = s_in_velocity + s_cog + s_soc
            velocity = max(-self.mopso.vmax, min(self.mopso.vmax, raw_velocity))
            updated.velocity[component] = velocity
            q_pso = abs(math.tanh(self.mopso.alpha * velocity))
            q_eff = 1.0 - (1.0 - q_pso) * (1.0 - p_tur)
            roll = rng.random()
            mode = None
            if roll < p_tur:
                mode = "turbulence"
            elif rng.random() < q_pso:
                mode = self._guided_mode(component, s_in_weight, s_cog, s_soc, particle, generation, rng)
            if mode:
                candidates.append((component, mode, q_eff))
        max_changes = min(self.mopso.dmax, len(candidates))
        if len(candidates) > max_changes:
            candidates = weighted_sample_without_replacement(candidates, max_changes, rng)
        for component, mode, _weight in candidates:
            context = self._movement_context(component, mode, updated, pbest, leader, generation)
            if context is None:
                continue
            replacement = self._candidate_for_mode(
                component,
                mode,
                updated,
                pbest,
                leader,
                generation,
                rng,
                guided_diagnostics_sink,
            )
            if context.used_inertia:
                updated.last_guided_move.pop(component, None)
            if replacement:
                updated.vector.components[component] = replacement
                updated.changed = True
                if context.effective_mode in GUIDED_MOVES and not context.used_inertia:
                    updated.last_guided_move[component] = self._guided_move_memory(
                        context.effective_mode,
                        context.target_position,
                        replacement,
                        generation,
                    )
        for component in self.components.frozen:
            updated.vector.components[component] = updated.initial_components.get(component, particle.vector.components[component])
            updated.velocity[component] = particle.velocity.get(component, 0.0)
        if updated.changed:
            proposed_prompt = self._render_prompt(updated.vector)
            used_central_anchors, anchor_probability = self._central_anchor_usage(generation, rng=rng)
            user_prompt_override = (
                build_anchored_text_generation_user_prompt(proposed_prompt, self.central_anchors)
                if used_central_anchors
                else None
            )
            proposed_text = self._generate_text(proposed_prompt, user_prompt_override=user_prompt_override)
            validation = validate_generated_text(proposed_text, self.reference_text)
            f1: float | None = None
            if validation.valid:
                f1 = self._generated_text_fidelity(proposed_text)
                validation = validate_generated_text(
                    proposed_text,
                    self.reference_text,
                    f1=f1,
                    tau_gen_min=self.tau_gen_min,
                )
            if not validation.valid:
                rejection = {
                    "phase": "optimization",
                    "generation": generation,
                    "solution_id": particle.solution_id,
                    "reason": validation.reason,
                    "f1": f1,
                    "text": proposed_text,
                    "used_central_anchors": used_central_anchors,
                    "anchor_inclusion_probability": anchor_probability,
                }
                if rejection_sink is None:
                    self._write_optimization_rejection(rejection)
                else:
                    rejection_sink.append(rejection)
                restored = particle.clone(keep_id=True)
                restored.velocity = dict(updated.velocity)
                restored.changed = False
                return restored
            updated.prompt = proposed_prompt
            updated.generated_text = proposed_text
            updated.generation = generation
            updated.metadata["used_central_anchors"] = used_central_anchors
            updated.metadata["anchor_inclusion_probability"] = anchor_probability
        return updated

    async def _update_particle_async(
        self,
        particle: Solution,
        pbest: Solution,
        leader: Solution,
        generation: int,
        rng: Random,
        rejection_sink: list[dict[str, Any]],
        guided_diagnostics_sink: list[dict[str, Any]] | None = None,
    ) -> Solution:
        updated = particle.clone(keep_id=True)
        updated.changed = False
        active_components = self.components.active
        candidates: list[tuple[str, str, float]] = []
        schedule_index = generation - 1
        rho = progress_ratio(schedule_index, self.config.iterations)
        omega = self.mopso.omega_max - (self.mopso.omega_max - self.mopso.omega_min) * rho
        p_tur = self.mopso.p_tur_max - (self.mopso.p_tur_max - self.mopso.p_tur_min) * rho
        for component in active_components:
            current = particle.vector.components[component]
            pbest_value = pbest.vector.components[component]
            leader_value = leader.vector.components[component]
            embeddings = self.executor.embedding_service.encode([current, pbest_value, leader_value], text_type="component")
            delta_p = semantic_velocity_delta(embeddings[0], embeddings[1])
            delta_l = semantic_velocity_delta(embeddings[0], embeddings[2])
            r1 = rng.random()
            r2 = rng.random()
            previous_velocity = float(particle.velocity.get(component, 0.0))
            s_in_velocity = omega * previous_velocity
            s_in_weight = omega * abs(previous_velocity)
            s_cog = self.mopso.c1 * r1 * delta_p
            s_soc = self.mopso.c2 * r2 * delta_l
            raw_velocity = s_in_velocity + s_cog + s_soc
            velocity = max(-self.mopso.vmax, min(self.mopso.vmax, raw_velocity))
            updated.velocity[component] = velocity
            q_pso = abs(math.tanh(self.mopso.alpha * velocity))
            q_eff = 1.0 - (1.0 - q_pso) * (1.0 - p_tur)
            roll = rng.random()
            mode = None
            if roll < p_tur:
                mode = "turbulence"
            elif rng.random() < q_pso:
                mode = self._guided_mode(component, s_in_weight, s_cog, s_soc, particle, generation, rng)
            if mode:
                candidates.append((component, mode, q_eff))
        max_changes = min(self.mopso.dmax, len(candidates))
        if len(candidates) > max_changes:
            candidates = weighted_sample_without_replacement(candidates, max_changes, rng)
        for component, mode, _weight in candidates:
            context = self._movement_context(component, mode, updated, pbest, leader, generation)
            if context is None:
                continue
            replacement = await self._candidate_for_mode_async(
                component,
                mode,
                updated,
                pbest,
                leader,
                generation,
                rng,
                guided_diagnostics_sink,
            )
            if context.used_inertia:
                updated.last_guided_move.pop(component, None)
            if replacement:
                updated.vector.components[component] = replacement
                updated.changed = True
                if context.effective_mode in GUIDED_MOVES and not context.used_inertia:
                    updated.last_guided_move[component] = self._guided_move_memory(
                        context.effective_mode,
                        context.target_position,
                        replacement,
                        generation,
                    )
        for component in self.components.frozen:
            updated.vector.components[component] = updated.initial_components.get(component, particle.vector.components[component])
            updated.velocity[component] = particle.velocity.get(component, 0.0)
        if updated.changed:
            proposed_prompt = await self._render_prompt_async(updated.vector)
            used_central_anchors, anchor_probability = self._central_anchor_usage(generation, rng=rng)
            user_prompt_override = (
                build_anchored_text_generation_user_prompt(proposed_prompt, self.central_anchors)
                if used_central_anchors
                else None
            )
            proposed_text = await self._generate_text_async(proposed_prompt, user_prompt_override=user_prompt_override)
            validation = validate_generated_text(proposed_text, self.reference_text)
            f1: float | None = None
            if validation.valid:
                f1 = self._generated_text_fidelity(proposed_text)
                validation = validate_generated_text(
                    proposed_text,
                    self.reference_text,
                    f1=f1,
                    tau_gen_min=self.tau_gen_min,
                )
            if not validation.valid:
                rejection_sink.append(
                    {
                        "phase": "optimization",
                        "generation": generation,
                        "solution_id": particle.solution_id,
                        "reason": validation.reason,
                        "f1": f1,
                        "text": proposed_text,
                        "used_central_anchors": used_central_anchors,
                        "anchor_inclusion_probability": anchor_probability,
                    }
                )
                restored = particle.clone(keep_id=True)
                restored.velocity = dict(updated.velocity)
                restored.changed = False
                return restored
            updated.prompt = proposed_prompt
            updated.generated_text = proposed_text
            updated.generation = generation
            updated.metadata["used_central_anchors"] = used_central_anchors
            updated.metadata["anchor_inclusion_probability"] = anchor_probability
        return updated

    def _central_anchor_usage(self, generation: int, rng: Random | None = None) -> tuple[bool, float | None]:
        if not self.mopso.p_anchor_enabled or not self.central_anchors:
            return False, None
        probability = self._anchor_inclusion_probability(generation)
        return (rng or self.rng).random() < probability, probability

    def _anchor_inclusion_probability(self, generation: int) -> float:
        rho = progress_ratio(generation - 1, self.config.iterations)
        return self.mopso.p_anchor_min + (self.mopso.p_anchor_max - self.mopso.p_anchor_min) * rho

    def _guided_mode(
        self,
        component: str,
        inertia_weight: float,
        cognitive_weight: float,
        social_weight: float,
        particle: Solution,
        generation: int,
        rng: Random | None = None,
    ) -> str | None:
        rng = rng or self.rng
        weights = {
            "inertia": inertia_weight if self._valid_inertia_memory(component, particle, generation) is not None else 0.0,
            "cognitive": cognitive_weight,
            "social": social_weight,
        }
        total = sum(weights.values())
        if total <= 0:
            return None
        roll = rng.random() * total
        cumulative = 0.0
        for name, weight in weights.items():
            cumulative += weight
            if roll <= cumulative:
                return name
        return "social"

    def _candidate_for_mode(
        self,
        component: str,
        mode: str,
        particle: Solution,
        pbest: Solution,
        leader: Solution,
        generation: int,
        rng: Random | None = None,
        guided_diagnostics_sink: list[dict[str, Any]] | None = None,
    ) -> str | None:
        current = particle.vector.components[component]
        context = self._movement_context(component, mode, particle, pbest, leader, generation)
        if context is None:
            return None
        if context.effective_mode == "turbulence":
            return self._turbulence_candidate(component, current, rng=rng)
        target = context.target_position[component]
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_INFLUENCE,
            self._influence_task_params(component, current, target, particle, generation),
        )
        raw_candidates = list(self.executor.execute(self.router.route(route)))
        return self._select_guided_candidate(
            component,
            current,
            target,
            raw_candidates,
            diagnostic_context=self._guided_candidate_diagnostic_context(
                component,
                mode,
                context,
                current,
                target,
                particle,
                generation,
                guided_diagnostics_sink,
            ),
            diagnostics_sink=guided_diagnostics_sink,
        )

    async def _candidate_for_mode_async(
        self,
        component: str,
        mode: str,
        particle: Solution,
        pbest: Solution,
        leader: Solution,
        generation: int,
        rng: Random,
        guided_diagnostics_sink: list[dict[str, Any]] | None = None,
    ) -> str | None:
        current = particle.vector.components[component]
        context = self._movement_context(component, mode, particle, pbest, leader, generation)
        if context is None:
            return None
        if context.effective_mode == "turbulence":
            return self._turbulence_candidate(component, current, rng=rng)
        target = context.target_position[component]
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_INFLUENCE,
            self._influence_task_params(component, current, target, particle, generation),
        )
        routed = self.router.route(route)
        if hasattr(self.executor, "execute_async"):
            raw_candidates = list(await self.executor.execute_async(routed))
        else:
            raw_candidates = list(self.executor.execute(routed))
        return self._select_guided_candidate(
            component,
            current,
            target,
            raw_candidates,
            diagnostic_context=self._guided_candidate_diagnostic_context(
                component,
                mode,
                context,
                current,
                target,
                particle,
                generation,
                guided_diagnostics_sink,
            ),
            diagnostics_sink=guided_diagnostics_sink,
        )

    def _guided_candidate_diagnostic_context(
        self,
        component: str,
        mode: str,
        context: MovementContext,
        current: str,
        target: str,
        particle: Solution,
        generation: int,
        diagnostics_sink: list[dict[str, Any]] | None,
    ) -> dict[str, Any] | None:
        if diagnostics_sink is None:
            return None
        return {
            "phase": "optimization",
            "generation": generation,
            "solution_id": particle.solution_id,
            "component": component,
            "mode": mode,
            "effective_mode": context.effective_mode,
            "used_inertia": context.used_inertia,
            "current": current,
            "target": target,
        }

    def _effective_candidate_mode(
        self,
        component: str,
        mode: str,
        particle: Solution,
        generation: int,
    ) -> str | None:
        context = self._movement_context(component, mode, particle, particle, particle, generation)
        return None if context is None else context.effective_mode

    def _movement_context(
        self,
        component: str,
        mode: str,
        particle: Solution,
        pbest: Solution,
        leader: Solution,
        generation: int,
    ) -> MovementContext | None:
        if mode == "turbulence":
            return MovementContext("turbulence", {})
        if mode == "cognitive":
            return MovementContext("cognitive", dict(pbest.vector.components))
        if mode == "social":
            return MovementContext("social", dict(leader.vector.components))
        if mode != "inertia":
            return None
        memory = self._valid_inertia_memory(component, particle, generation)
        if memory is None:
            return None
        return MovementContext(memory["movement"], dict(memory["target_position"]), used_inertia=True)

    def _valid_inertia_memory(self, component: str, particle: Solution, generation: int) -> dict[str, Any] | None:
        memory = particle.last_guided_move.get(component)
        if not isinstance(memory, dict):
            return None
        movement = str(memory.get("movement", "")).strip().lower()
        target_position = memory.get("target_position")
        if movement not in GUIDED_MOVES or not isinstance(target_position, dict):
            return None
        if component not in target_position:
            return None
        if int(memory.get("iteration", -1)) != generation - 1:
            return None
        return {
            "movement": movement,
            "target_position": dict(target_position),
            "accepted_component": str(memory.get("accepted_component", "")),
            "iteration": int(memory["iteration"]),
        }

    @staticmethod
    def _guided_move_memory(
        movement: str,
        target_position: dict[str, str],
        accepted_component: str,
        generation: int,
    ) -> dict[str, Any]:
        return {
            "movement": movement,
            "target_position": dict(target_position),
            "accepted_component": accepted_component,
            "iteration": generation,
        }

    def _influence_task_params(
        self,
        component: str,
        current: str,
        target: str,
        particle: Solution,
        generation: int,
    ) -> dict[str, Any]:
        spec = component_spec(component)
        other_components = {
            name: value
            for name, value in particle.vector.components.items()
            if name != component
        }
        return {
            "numCandidates": self.mopso.kcand,
            "componentName": spec.name,
            "componentDefinition": spec.definition,
            "currentComponent": current,
            "targetComponent": target,
            "otherComponents": other_components,
            "referenceText": self.reference_text,
            "componentAdditionalInstruction": spec.influence_instruction,
            "iteration": generation - 1,
            "totalGenerations": self.config.iterations,
            "component": component,
            "current": current,
            "target": target,
            "reference_text": self.reference_text,
            "max_candidates": self.mopso.kcand,
            "iterations": self.config.iterations,
        }

    def _turbulence_candidate(self, component: str, current: str, rng: Random | None = None) -> str | None:
        rng = rng or self.rng
        spec = component_spec(component)
        units = self.executor.turbulence_provider.modifiable_units(current, spec.preferred_turbulence_pos)
        if not units:
            return None
        selected = rng.choice(units)
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_WORD_REPLACEMENT,
            {
                "component": current,
                "componentType": spec.name,
                "targetWord": selected["text"],
                "targetLemma": selected["lemma"],
                "targetPos": selected["pos"],
                "targetSpan": selected["span"],
                "targetWordLeftTokens": selected["leftTokens"],
                "targetWordRightTokens": selected["rightTokens"],
                "maxVariants": self.mopso.kcand,
                "text": current,
                "tokens": selected["tokens"],
                "target_index": selected["target_index"],
                "target_lemma": selected["lemma"],
                "target_pos": selected["pos"],
                "max_variants": self.mopso.kcand,
            },
        )
        raw_candidates = list(self.executor.execute(self.router.route(route)))
        return self._select_turbulence_candidate(component, current, raw_candidates)

    def _select_guided_candidate(
        self,
        component: str,
        current: str,
        target: str,
        raw_candidates: list[str],
        diagnostic_context: dict[str, Any] | None = None,
        diagnostics_sink: list[dict[str, Any]] | None = None,
    ) -> str | None:
        if (
            diagnostic_context is not None
            and diagnostics_sink is not None
        ):
            return self._select_guided_candidate_with_diagnostics(
                component,
                current,
                target,
                raw_candidates,
                diagnostic_context,
                diagnostics_sink,
            )
        candidates = self._basic_candidates(component, current, raw_candidates, forbidden=target)
        if not candidates:
            return None
        candidate_embeddings = self.executor.embedding_service.encode(candidates, text_type="component")
        duplicate_sims = self.component_memory.max_similarity(component, candidate_embeddings)
        target_embeddings = self.executor.embedding_service.encode([current, target], text_type="component")
        before = float(target_embeddings[0] @ target_embeddings[1])
        target_sims = candidate_embeddings @ target_embeddings[1]
        valid_mask = (duplicate_sims < self.mopso.tau_dup) & (target_sims > before)
        if self.mopso.guided_trajectory_validation_enabled:
            current_sims = candidate_embeddings @ target_embeddings[0]
            valid_mask &= guided_trajectory_consistency_mask(
                before,
                current_sims,
                target_sims,
                self.mopso.guided_trajectory_relative_margin,
            )
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            return None
        best_idx = int(valid_indices[np.argmax(target_sims[valid_indices])])
        return candidates[best_idx]

    def _select_guided_candidate_with_diagnostics(
        self,
        component: str,
        current: str,
        target: str,
        raw_candidates: list[str],
        diagnostic_context: dict[str, Any],
        diagnostics_sink: list[dict[str, Any]],
    ) -> str | None:
        candidates, rejected_candidates = self._basic_candidates_with_diagnostics(
            component,
            current,
            raw_candidates,
            forbidden=target,
        )
        accepted_candidate: str | None = None
        if candidates:
            candidate_embeddings = self.executor.embedding_service.encode(candidates, text_type="component")
            duplicate_sims = self.component_memory.max_similarity(component, candidate_embeddings)
            target_embeddings = self.executor.embedding_service.encode([current, target], text_type="component")
            before = float(target_embeddings[0] @ target_embeddings[1])
            target_sims = candidate_embeddings @ target_embeddings[1]
            duplicate_mask = duplicate_sims < self.mopso.tau_dup
            progress_mask = target_sims > before
            trajectory_mask = np.ones(len(candidates), dtype=bool)
            if self.mopso.guided_trajectory_validation_enabled:
                current_sims = candidate_embeddings @ target_embeddings[0]
                trajectory_mask = guided_trajectory_consistency_mask(
                    before,
                    current_sims,
                    target_sims,
                    self.mopso.guided_trajectory_relative_margin,
                )
            valid_mask = duplicate_mask & progress_mask & trajectory_mask
            for index, candidate in enumerate(candidates):
                reason = None
                if not bool(duplicate_mask[index]):
                    reason = REASON_SEMANTIC_DUPLICATE
                elif not bool(progress_mask[index]):
                    reason = REASON_NO_SEMANTIC_PROGRESS
                elif self.mopso.guided_trajectory_validation_enabled and not bool(trajectory_mask[index]):
                    reason = REASON_GUIDED_TRAJECTORY_INCONSISTENT
                if reason is not None:
                    rejected_candidates.append({"candidate": candidate, "reason": reason})
            valid_indices = np.where(valid_mask)[0]
            if valid_indices.size > 0:
                best_idx = int(valid_indices[np.argmax(target_sims[valid_indices])])
                accepted_candidate = candidates[best_idx]
        self._append_guided_candidate_diagnostics(
            component,
            raw_candidates,
            accepted_candidate,
            rejected_candidates,
            diagnostic_context,
            diagnostics_sink,
        )
        return accepted_candidate

    def _select_turbulence_candidate(self, component: str, current: str, raw_candidates: list[str]) -> str | None:
        candidates = self._basic_candidates(component, current, raw_candidates, forbidden=None)
        if not candidates:
            return None
        candidate_embeddings = self.executor.embedding_service.encode(candidates, text_type="component")
        duplicate_sims = self.component_memory.max_similarity(component, candidate_embeddings)
        current_embedding = self.executor.embedding_service.encode([current], text_type="component")[0]
        current_sims = candidate_embeddings @ current_embedding
        valid_indices = np.where(
            (duplicate_sims < self.mopso.tau_dup)
            & (current_sims >= self.mopso.tau_tur_min)
            & (current_sims <= self.mopso.tau_tur_max)
        )[0]
        if valid_indices.size == 0:
            return None
        best_idx = int(valid_indices[np.argmax(current_sims[valid_indices])])
        return candidates[best_idx]

    def _basic_candidates(
        self,
        component: str,
        current: str,
        raw_candidates: list[str],
        forbidden: str | None,
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        current_key = canonical_text(current)
        forbidden_key = canonical_text(forbidden) if forbidden is not None else None
        max_words = self.components.max_words[component]
        for candidate in raw_candidates:
            text = str(candidate).strip()
            key = canonical_text(text)
            if not key or key in seen or key == current_key:
                continue
            if forbidden_key is not None and key == forbidden_key:
                continue
            if word_count(text) > max_words:
                continue
            seen.add(key)
            result.append(text)
        return result

    def _basic_candidates_with_diagnostics(
        self,
        component: str,
        current: str,
        raw_candidates: list[str],
        forbidden: str | None,
    ) -> tuple[list[str], list[dict[str, str]]]:
        result: list[str] = []
        rejected: list[dict[str, str]] = []
        seen: set[str] = set()
        current_key = canonical_text(current)
        forbidden_key = canonical_text(forbidden) if forbidden is not None else None
        max_words = self.components.max_words[component]
        for candidate in raw_candidates:
            raw_text = str(candidate)
            text = raw_text.strip()
            key = canonical_text(text)
            reason = None
            if not key:
                reason = REASON_EMPTY_CANDIDATE
            elif key in seen:
                reason = REASON_DUPLICATE_CANDIDATE_OUTPUT
            elif key == current_key:
                reason = REASON_LITERAL_COPY_CURRENT
            elif forbidden_key is not None and key == forbidden_key:
                reason = REASON_LITERAL_COPY_TARGET
            elif word_count(text) > max_words:
                reason = REASON_TOO_MANY_WORDS
            if reason is not None:
                rejected.append({"candidate": raw_text, "reason": reason})
                continue
            seen.add(key)
            result.append(text)
        return result, rejected

    @staticmethod
    def _append_guided_candidate_diagnostics(
        component: str,
        raw_candidates: list[str],
        accepted_candidate: str | None,
        rejected_candidates: list[dict[str, str]],
        diagnostic_context: dict[str, Any],
        diagnostics_sink: list[dict[str, Any]],
    ) -> None:
        rejected_count_by_reason: dict[str, int] = {}
        for item in rejected_candidates:
            reason = item["reason"]
            rejected_count_by_reason[reason] = rejected_count_by_reason.get(reason, 0) + 1
        row = dict(diagnostic_context)
        row["component"] = component
        row["raw_candidate_count"] = len(raw_candidates)
        row["accepted_candidate"] = accepted_candidate
        row["rejected_count_by_reason"] = rejected_count_by_reason
        row["candidates"] = rejected_candidates
        diagnostics_sink.append(row)

    def _render_prompt(self, vector: SemanticVector) -> str:
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_PROMPT_RENDERING,
            {"components": vector.components, "domain": self.config.get("experiment.domain")},
        )
        return str(self.executor.execute(self.router.route(route)))

    async def _render_prompt_async(self, vector: SemanticVector) -> str:
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_PROMPT_RENDERING,
            {"components": vector.components, "domain": self.config.get("experiment.domain")},
        )
        routed = self.router.route(route)
        if hasattr(self.executor, "execute_async"):
            return str(await self.executor.execute_async(routed))
        return str(self.executor.execute(routed))

    def _generate_text(self, prompt: str, user_prompt_override: str | None = None) -> str:
        task_params: dict[str, Any] = {"prompt": prompt, "reference_text": self.reference_text}
        if user_prompt_override is not None:
            task_params["userPromptOverride"] = user_prompt_override
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_SYNTHETIC_TEXT,
            task_params,
        )
        return str(self.executor.execute(self.router.route(route))).strip()

    async def _generate_text_async(self, prompt: str, user_prompt_override: str | None = None) -> str:
        task_params: dict[str, Any] = {"prompt": prompt, "reference_text": self.reference_text}
        if user_prompt_override is not None:
            task_params["userPromptOverride"] = user_prompt_override
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_SYNTHETIC_TEXT,
            task_params,
        )
        routed = self.router.route(route)
        if hasattr(self.executor, "execute_async"):
            return str(await self.executor.execute_async(routed)).strip()
        return str(self.executor.execute(routed)).strip()

    def _generated_text_fidelity(self, text: str) -> float:
        scores = semantic_fidelity_scores([text], self.reference_text, self.executor.embedding_service)
        return float(scores[0]) if scores.size else 0.0

    def _write_optimization_rejection(self, row: dict[str, Any]) -> None:
        self._write_optimization_rejections([row])

    def _write_optimization_rejections(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.outdir / "optimization_rejections.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_guided_candidate_diagnostics(self, rows: list[dict[str, Any]]) -> None:
        if not rows or not self.mopso.guided_candidate_diagnostics_enabled:
            return
        path = self.outdir / "guided_candidate_diagnostics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _guided_candidate_rejection_count(rows: list[dict[str, Any]]) -> int:
        total = 0
        for row in rows:
            counts = row.get("rejected_count_by_reason", {})
            if isinstance(counts, dict):
                total += sum(int(value) for value in counts.values())
        return total

    def _write_particle_update_errors(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.outdir / "particle_update_errors.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _generation_metrics(
        self,
        generation: int,
        population: list[Solution],
        modified_count: int,
        guided_candidate_rejections: int,
    ) -> dict[str, Any]:
        f1 = [solution.objectives.f1 for solution in population if solution.objectives]
        f2 = [solution.objectives.f2 for solution in population if solution.objectives]
        mo_metrics = archive_metrics(self.archive.solutions)
        return {
            "generation": generation,
            "modified_count": modified_count,
            "mean_f1": float(np.mean(f1)) if f1 else 0.0,
            "max_f1": float(np.max(f1)) if f1 else 0.0,
            "mean_f2": float(np.mean(f2)) if f2 else 0.0,
            "max_f2": float(np.max(f2)) if f2 else 0.0,
            "archive_size": len(self.archive.solutions),
            "archive_update_count": self.archive.update_count,
            "archive_prune_count": self.archive.prune_count,
            "generated_with_central_anchors": sum(
                1 for solution in population if solution.changed and solution.metadata.get("used_central_anchors") is True
            ),
            "generated_without_central_anchors": sum(
                1 for solution in population if solution.changed and solution.metadata.get("used_central_anchors") is not True
            ),
            "hypervolume": mo_metrics["hypervolume"],
            "guided_candidate_rejections": guided_candidate_rejections,
        }

    def _write_archive_history(self, generation: int) -> None:
        path = self.outdir / "archive_history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "generation": generation,
                        "archive": [solution_to_dict(solution) for solution in self.archive.solutions],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _write_metrics(self, rows: list[dict[str, Any]]) -> None:
        import pandas as pd

        pd.DataFrame(rows).to_csv(self.outdir / "evolucion_metricas.csv", index=False)

    def _write_monitor_metrics(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        import pandas as pd

        pd.DataFrame(rows).to_csv(self.outdir / "monitor_metrics.csv", index=False)

    def _checkpoint_payload(
        self,
        generation: int,
        population: list[Solution],
        pbest: list[Solution],
        metrics_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "generation": generation,
            "population": [solution_to_dict(solution) for solution in population],
            "pbest": [solution_to_dict(solution) for solution in pbest],
            "archive": [solution_to_dict(solution) for solution in self.archive.solutions],
            "archive_stats": {
                "update_count": self.archive.update_count,
                "prune_count": self.archive.prune_count,
            },
            "component_memory": self.component_memory.to_snapshot(),
            "rng_state": rng_to_text(self.rng),
            "metrics": metrics_rows,
            "embedding_cache_file": str(self.config.get("runtime.embedding_cache_file", "embedding_cache.json")),
            "embedding_cache_size": self.executor.embedding_service.cache.size,
            "llm_log_file": "llm_calls.jsonl",
            "semantic_anchors": {key: list(values) for key, values in self.semantic_anchors.items()},
            "central_anchors": list(self.central_anchors),
        }


def weighted_sample_without_replacement(
    candidates: list[tuple[str, str, float]],
    count: int,
    rng: Random,
) -> list[tuple[str, str, float]]:
    pool = list(candidates)
    selected: list[tuple[str, str, float]] = []
    while pool and len(selected) < count:
        total = sum(max(item[2], 0.0) for item in pool)
        if total <= 0.0:
            choice = rng.randrange(len(pool))
        else:
            roll = rng.random() * total
            cumulative = 0.0
            choice = len(pool) - 1
            for idx, item in enumerate(pool):
                cumulative += max(item[2], 0.0)
                if roll <= cumulative:
                    choice = idx
                    break
        selected.append(pool.pop(choice))
    return selected
