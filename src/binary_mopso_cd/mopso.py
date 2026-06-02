from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any
from uuid import uuid4

import numpy as np

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.entities import Objectives, SemanticVector, Solution, solution_to_dict
from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.monitor import ObservationalMonitor
from binary_mopso_cd.objectives import evaluate_solutions
from binary_mopso_cd.router import (
    TASK_INFLUENCE,
    TASK_PROMPT_RENDERING,
    TASK_SYNTHETIC_TEXT,
    TASK_WORD_REPLACEMENT,
    RouteTask,
    SemanticRouter,
)
from binary_mopso_cd.services.turbulence import tokenize_component
from binary_mopso_cd.utils import canonical_text, rng_to_text, word_count


def dominates(left: Objectives, right: Objectives) -> bool:
    return left.f1 >= right.f1 and left.f2 >= right.f2 and (left.f1 > right.f1 or left.f2 > right.f2)


def utility(objectives: Objectives, weights: dict[str, float] | None = None) -> float:
    weights = weights or {"f1": 0.5, "f2": 0.5}
    f1_norm = (objectives.f1 + 1.0) / 2.0
    f2_norm = objectives.f2 / 2.0
    return float(weights.get("f1", 0.5) * f1_norm + weights.get("f2", 0.5) * f2_norm)


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


@dataclass
class ExternalArchive:
    max_size: int
    rng: Random
    solutions: list[Solution] = field(default_factory=list)

    def update(self, candidates: list[Solution]) -> list[Solution]:
        merged = self._deduplicate(self.solutions + [candidate.clone() for candidate in candidates])
        self.solutions = non_dominated(merged)
        while len(self.solutions) > self.max_size:
            distances = crowding_distance(self.solutions)
            finite_values = [value for value in distances.values() if not math.isinf(value)]
            min_distance = min(finite_values) if finite_values else min(distances.values())
            tied = [solution for solution in self.solutions if distances[solution.solution_id] == min_distance]
            remove = self.rng.choice(tied)
            self.solutions = [solution for solution in self.solutions if solution.solution_id != remove.solution_id]
        return self.solutions

    def select_leader(self, tournament_size: int) -> Solution:
        if not self.solutions:
            raise RuntimeError("Cannot select leader from an empty archive")
        q_eff = min(tournament_size, len(self.solutions))
        participants = self.rng.sample(self.solutions, q_eff)
        distances = crowding_distance(self.solutions)
        best_distance = max(distances[p.solution_id] for p in participants)
        tied = [p for p in participants if distances[p.solution_id] == best_distance]
        return self.rng.choice(tied)

    def _deduplicate(self, candidates: list[Solution]) -> list[Solution]:
        best_by_key: dict[tuple[tuple[str, str], ...], Solution] = {}
        for solution in candidates:
            key = solution.vector.signature()
            previous = best_by_key.get(key)
            if previous is None:
                best_by_key[key] = solution
                continue
            if solution.objectives and previous.objectives and utility(solution.objectives) > utility(previous.objectives):
                best_by_key[key] = solution
        return list(best_by_key.values())


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


class BinaryMOPSOCDEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        router: SemanticRouter,
        executor: SemanticTaskExecutor,
        rng: Random,
        outdir: Path,
        reference_text: str,
    ):
        self.config = config
        self.router = router
        self.executor = executor
        self.rng = rng
        self.outdir = outdir
        self.reference_text = reference_text
        self.archive = ExternalArchive(max_size=int(config.get("mopso.archive_multiplier", 2)) * config.n, rng=rng)
        self.pbest_updater = PBestUpdater(dict(config.get("mopso.utility_weights", {"f1": 0.5, "f2": 0.5})))
        self.component_memory: dict[str, list[str]] = {component: [] for component in config.components}
        self.monitor = ObservationalMonitor(
            enabled=bool(config.get("monitor.enabled", False)),
            spacy_model=str(config.get("models.spacy.model", "en_core_web_sm")),
            kmeans_clusters=int(config.get("monitor.kmeans_clusters", 3)),
        )

    def run(
        self,
        initial_population: list[Solution],
        start_generation: int = 0,
        pbest_state: list[Solution] | None = None,
        archive_state: list[Solution] | None = None,
        component_memory: dict[str, list[str]] | None = None,
    ) -> tuple[list[Solution], ExternalArchive]:
        population = [solution.clone(keep_id=True) for solution in initial_population]
        pbest = [solution.clone() for solution in (pbest_state or population)]
        if archive_state is not None:
            self.archive.solutions = [solution.clone() for solution in archive_state]
        else:
            self.archive.update(population)
        if component_memory is not None:
            self.component_memory = {key: list(values) for key, values in component_memory.items()}
        else:
            self._remember_components(population)
        metrics_rows: list[dict[str, Any]] = []
        monitor_rows: list[dict[str, Any]] = []
        for generation in range(start_generation + 1, self.config.iterations + 1):
            next_population = []
            for index, particle in enumerate(population):
                leader = self.archive.select_leader(int(self.config.get("mopso.leader_tournament_size", 3)))
                updated = self._update_particle(particle, pbest[index], leader, generation)
                next_population.append(updated)
            comparison_batch = next_population + pbest + self.archive.solutions
            evaluate_solutions(comparison_batch, self.reference_text, self.executor.embedding_service)
            next_population = comparison_batch[: len(next_population)]
            pbest_candidates = comparison_batch[len(next_population) : len(next_population) + len(pbest)]
            archive_revalued = comparison_batch[len(next_population) + len(pbest) :]
            self.archive.solutions = archive_revalued
            pbest = [self.pbest_updater.choose(current, previous) for current, previous in zip(next_population, pbest_candidates, strict=True)]
            self.archive.update(next_population)
            self._remember_components(next_population)
            population = next_population
            metrics_rows.append(self._generation_metrics(generation, population))
            monitor_result = self.monitor.observe(generation, population)
            if self.monitor.enabled:
                monitor_rows.append(
                    {
                        **monitor_result.metrics,
                        "monitor_overhead_seconds": monitor_result.overhead_seconds,
                    }
                )
            self._write_archive_history(generation)
            if generation % int(self.config.get("runtime.checkpoint_every", 1)) == 0:
                self._save_checkpoint(generation, population, pbest, metrics_rows)
            self.executor.save_caches()
        self._write_metrics(metrics_rows)
        self._write_monitor_metrics(monitor_rows)
        return population, self.archive

    def _update_particle(self, particle: Solution, pbest: Solution, leader: Solution, generation: int) -> Solution:
        updated = particle.clone(keep_id=True)
        updated.changed = False
        active_components = [name for name in self.config.components if name not in self.config.frozen_components]
        candidates: list[tuple[str, str, str]] = []
        rho = 0.0 if self.config.iterations <= 1 else generation / max(self.config.iterations - 1, 1)
        omega = float(self.config.get("mopso.omega_max", 0.9)) - (
            float(self.config.get("mopso.omega_max", 0.9)) - float(self.config.get("mopso.omega_min", 0.4))
        ) * rho
        p_tur = float(self.config.get("mopso.p_tur_max", 0.05)) - (
            float(self.config.get("mopso.p_tur_max", 0.05)) - float(self.config.get("mopso.p_tur_min", 0.01))
        ) * rho
        for component in active_components:
            current = particle.vector.components[component]
            pbest_value = pbest.vector.components[component]
            leader_value = leader.vector.components[component]
            delta_p = 1.0 - self.executor.embedding_service.similarity(current, pbest_value, "component")
            delta_l = 1.0 - self.executor.embedding_service.similarity(current, leader_value, "component")
            r1 = self.rng.random()
            r2 = self.rng.random()
            previous_velocity = float(particle.velocity.get(component, 0.0))
            s_in_velocity = omega * previous_velocity
            s_in_weight = omega * abs(previous_velocity)
            s_cog = float(self.config.get("mopso.c1", 1.5)) * r1 * delta_p
            s_soc = float(self.config.get("mopso.c2", 1.5)) * r2 * delta_l
            raw_velocity = s_in_velocity + s_cog + s_soc
            vmax = float(self.config.get("mopso.vmax", 4.0))
            velocity = max(-vmax, min(vmax, raw_velocity))
            updated.velocity[component] = velocity
            q_pso = abs(math.tanh(float(self.config.get("mopso.alpha", 0.5)) * velocity))
            roll = self.rng.random()
            mode = None
            if roll < p_tur:
                mode = "turbulence"
            elif self.rng.random() < q_pso:
                mode = self._guided_mode(component, s_in_weight, s_cog, s_soc, particle)
            if mode:
                candidates.append((component, mode, ""))
        max_changes = min(int(self.config.get("mopso.dmax", 1)), len(candidates))
        if len(candidates) > max_changes:
            candidates = self.rng.sample(candidates, max_changes)
        for component, mode, _ in candidates:
            replacement = self._candidate_for_mode(component, mode, updated, pbest, leader, generation)
            if replacement:
                updated.vector.components[component] = replacement
                updated.changed = True
                if mode in {"cognitive", "social"}:
                    updated.last_guided_move[component] = replacement
        for component in self.config.frozen_components:
            updated.vector.components[component] = updated.initial_components.get(component, particle.vector.components[component])
            updated.velocity[component] = particle.velocity.get(component, 0.0)
        if updated.changed:
            updated.prompt = self._render_prompt(updated.vector)
            updated.generated_text = self._generate_text(updated.prompt)
            updated.generation = generation
        return updated

    def _guided_mode(
        self,
        component: str,
        inertia_weight: float,
        cognitive_weight: float,
        social_weight: float,
        particle: Solution,
    ) -> str | None:
        weights = {
            "inertia": inertia_weight if component in particle.last_guided_move else 0.0,
            "cognitive": cognitive_weight,
            "social": social_weight,
        }
        total = sum(weights.values())
        if total <= 0:
            return None
        roll = self.rng.random() * total
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
    ) -> str | None:
        current = particle.vector.components[component]
        if mode == "inertia":
            candidate = particle.last_guided_move.get(component)
            return candidate if candidate and self._valid_candidate(component, current, candidate, None, "inertia") else None
        if mode == "turbulence":
            return self._turbulence_candidate(component, current)
        target = pbest.vector.components[component] if mode == "cognitive" else leader.vector.components[component]
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_INFLUENCE,
            {
                "component": component,
                "current": current,
                "target": target,
                "reference_text": self.reference_text,
                "max_candidates": int(self.config.get("mopso.kcand", 5)),
                "iteration": generation,
                "iterations": self.config.iterations,
            },
        )
        raw_candidates = list(self.executor.execute(self.router.route(route)))
        valid = [value for value in raw_candidates if self._valid_candidate(component, current, value, target, mode)]
        if not valid:
            return None
        return max(valid, key=lambda value: self.executor.embedding_service.similarity(value, target, "component"))

    def _turbulence_candidate(self, component: str, current: str) -> str | None:
        tokens = tokenize_component(current)
        if not tokens:
            return None
        target_index = self.rng.randrange(len(tokens))
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_WORD_REPLACEMENT,
            {
                "text": current,
                "tokens": tokens,
                "target_index": target_index,
                "max_variants": int(self.config.get("mopso.kcand", 5)),
            },
        )
        raw_candidates = list(self.executor.execute(self.router.route(route)))
        valid = [value for value in raw_candidates if self._valid_candidate(component, current, value, None, "turbulence")]
        if not valid:
            return None
        return max(valid, key=lambda value: self.executor.embedding_service.similarity(value, current, "component"))

    def _valid_candidate(
        self,
        component: str,
        current: str,
        candidate: str,
        target: str | None,
        mode: str,
    ) -> bool:
        normalized = canonical_text(candidate)
        if not normalized or normalized == canonical_text(current):
            return False
        if word_count(normalized) < 2 or word_count(normalized) > 8:
            return False
        if target is not None and normalized == canonical_text(target):
            return False
        memory = self.component_memory.get(component, [])
        if memory:
            similarities = [
                self.executor.embedding_service.similarity(candidate, remembered, "component") for remembered in memory
            ]
            if max(similarities) >= float(self.config.get("mopso.tau_dup", 0.92)):
                return False
        if target is not None:
            before = self.executor.embedding_service.similarity(current, target, "component")
            after = self.executor.embedding_service.similarity(candidate, target, "component")
            return after > before
        if mode == "turbulence":
            sim = self.executor.embedding_service.similarity(candidate, current, "component")
            return float(self.config.get("mopso.tau_tur_min", 0.65)) <= sim <= float(
                self.config.get("mopso.tau_tur_max", 0.90)
            )
        return True

    def _render_prompt(self, vector: SemanticVector) -> str:
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_PROMPT_RENDERING,
            {"components": vector.components, "domain": self.config.get("experiment.domain")},
        )
        return str(self.executor.execute(self.router.route(route)))

    def _generate_text(self, prompt: str) -> str:
        route = RouteTask(
            uuid4().hex,
            "optimization",
            TASK_SYNTHETIC_TEXT,
            {"prompt": prompt, "reference_text": self.reference_text},
        )
        return str(self.executor.execute(self.router.route(route))).strip()

    def _remember_components(self, solutions: list[Solution]) -> None:
        for solution in solutions:
            for component, value in solution.vector.components.items():
                key = canonical_text(value)
                if key and all(canonical_text(existing) != key for existing in self.component_memory[component]):
                    self.component_memory[component].append(value)

    def _generation_metrics(self, generation: int, population: list[Solution]) -> dict[str, Any]:
        f1 = [solution.objectives.f1 for solution in population if solution.objectives]
        f2 = [solution.objectives.f2 for solution in population if solution.objectives]
        return {
            "generation": generation,
            "mean_f1": float(np.mean(f1)) if f1 else 0.0,
            "max_f1": float(np.max(f1)) if f1 else 0.0,
            "mean_f2": float(np.mean(f2)) if f2 else 0.0,
            "max_f2": float(np.max(f2)) if f2 else 0.0,
            "archive_size": len(self.archive.solutions),
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

    def _save_checkpoint(
        self,
        generation: int,
        population: list[Solution],
        pbest: list[Solution],
        metrics_rows: list[dict[str, Any]],
    ) -> None:
        checkpoint_dir = self.outdir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "generation": generation,
            "population": [solution_to_dict(solution) for solution in population],
            "pbest": [solution_to_dict(solution) for solution in pbest],
            "archive": [solution_to_dict(solution) for solution in self.archive.solutions],
            "component_memory": self.component_memory,
            "rng_state": rng_to_text(self.rng),
            "metrics": metrics_rows,
            "embedding_cache": self.executor.embedding_service.cache.to_dict(),
            "llm_log_file": "llm_calls.jsonl",
        }
        with (checkpoint_dir / f"generation_{generation:04d}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
