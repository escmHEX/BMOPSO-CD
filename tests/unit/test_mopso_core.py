from __future__ import annotations

import asyncio
import json
import math
from random import Random

import numpy as np
import pytest

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution, solution_from_dict, solution_to_dict
from binary_mopso_cd.mopso import (
    BinaryMOPSOCDEngine,
    ExternalArchive,
    ParticleUpdateResult,
    archive_capacity,
    crowding_distance,
    dominates,
    evaluate_unique_solutions_by_signature,
    particle_update_seed,
    utility,
)
from binary_mopso_cd.router import TASK_SYNTHETIC_TEXT
from binary_mopso_cd.settings import ComponentSettings


def make_solution(f1: float, f2: float, idx: int) -> Solution:
    return Solution(
        SemanticVector({"role": f"role {idx}", "topic": f"topic {idx}", "action": f"action {idx}"}),
        f"prompt {idx}",
        f"text {idx}",
        Objectives(f1, f2),
    )


class StubEmbeddingService:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.cache = StubEmbeddingCache()

    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


class StubEmbeddingCache:
    size = 0


class PassthroughRouter:
    def route(self, task):
        return task


class StubTurbulenceProvider:
    def __init__(self):
        self.preferred_pos: tuple[str, ...] | None = None

    def modifiable_units(self, _text: str, preferred_pos: tuple[str, ...] = ()) -> list[dict]:
        self.preferred_pos = preferred_pos
        return [
            {
                "text": "safety",
                "lemma": "safety",
                "pos": "NOUN",
                "span": [7, 13],
                "leftTokens": 1,
                "rightTokens": 1,
                "target_index": 1,
                "tokens": ["public", "safety", "alert"],
            }
        ]


class StubExecutor:
    def __init__(self):
        self.embedding_service = StubEmbeddingService({})
        self.turbulence_provider = StubTurbulenceProvider()
        self.executed_task = None

    def execute(self, task):
        self.executed_task = task
        if task.semantic_task == TASK_SYNTHETIC_TEXT:
            return "public emergency alert"
        return ["public emergency alert"]


class GuidedMoveEmbeddingService:
    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=float)

    def _vector(self, text: str) -> list[float]:
        if text in {
            "pbest topic",
            "leader topic",
            "leader aligned topic",
            "updated topic",
            "async aligned topic",
        }:
            return [0.0, 1.0]
        return [1.0, 0.0]


class GuidedMoveExecutor:
    def __init__(self, candidates: list[str] | None = None):
        self.embedding_service = GuidedMoveEmbeddingService()
        self.turbulence_provider = StubTurbulenceProvider()
        self.candidates = list(candidates or [])
        self.executed_task = None
        self.tasks = []

    def execute(self, task):
        self.executed_task = task
        self.tasks.append(task)
        if task.semantic_task == TASK_SYNTHETIC_TEXT:
            return "valid generated text"
        return list(self.candidates)


class AsyncGuidedMoveExecutor(GuidedMoveExecutor):
    async def execute_async(self, task):
        return self.execute(task)


def topic_only_components() -> ComponentSettings:
    return ComponentSettings(
        order=["topic"],
        frozen=set(),
        expansion_order=["topic"],
        max_words={"topic": 8},
        alpha={"topic": 1.0},
    )


def test_dominance_and_utility():
    assert dominates(Objectives(0.8, 0.5), Objectives(0.7, 0.5))
    assert not dominates(Objectives(0.8, 0.4), Objectives(0.7, 0.5))
    assert utility(Objectives(1.0, 2.0)) == 1.0


def test_crowding_distance_boundaries_are_infinite():
    solutions = [make_solution(0.1, 0.1, 1), make_solution(0.5, 0.5, 2), make_solution(0.9, 0.9, 3)]
    distances = crowding_distance(solutions)
    assert sum(1 for value in distances.values() if math.isinf(value)) >= 2


def test_external_archive_keeps_non_dominated_and_prunes():
    archive = ExternalArchive(max_size=2, rng=Random(1))
    archive.update([make_solution(0.8, 0.1, 1), make_solution(0.1, 0.8, 2), make_solution(0.5, 0.5, 3)])
    assert len(archive.solutions) == 2


def test_external_archive_counts_only_real_signature_changes():
    archive = ExternalArchive(max_size=4, rng=Random(1))
    archive.update([make_solution(0.8, 0.8, 1)])

    assert archive.update_count == 1
    assert archive.prune_count == 0

    archive.update([make_solution(0.1, 0.1, 2)])
    duplicate = Solution(
        SemanticVector({"role": " role 1 ", "topic": "TOPIC 1", "action": "action 1"}),
        "duplicate prompt",
        "duplicate text",
        Objectives(1.0, 1.0),
    )
    archive.update([duplicate])

    assert archive.update_count == 1
    assert archive.prune_count == 0


def test_external_archive_counts_prune_event_once_for_multiple_removals():
    archive = ExternalArchive(max_size=1, rng=Random(1))
    archive.update([make_solution(0.1, 0.9, 1), make_solution(0.5, 0.5, 2), make_solution(0.9, 0.1, 3)])

    assert len(archive.solutions) == 1
    assert archive.update_count == 1
    assert archive.prune_count == 1


def test_archive_capacity_supports_fractional_multipliers():
    assert archive_capacity(100, 0.5) == 50
    assert archive_capacity(100, 0.25) == 25
    assert archive_capacity(3, 0.5) == 2
    assert archive_capacity(3, 0.25) == 1


def test_mopso_uses_fractional_archive_multiplier(test_config, tmp_path):
    test_config.set("experiment.n", 4)
    test_config.set("mopso.archive_multiplier", 0.5)
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        StubExecutor(),
        Random(1),
        tmp_path,
        "reference",
    )

    assert engine.archive.max_size == 2


def test_particle_update_seed_is_deterministic_and_particle_specific():
    assert particle_update_seed(7, 3, 1) == particle_update_seed(7, 3, 1)
    assert particle_update_seed(7, 3, 1) != particle_update_seed(7, 3, 2)


def test_parallel_update_preserves_particle_order_and_logs_worker_errors(test_config, tmp_path):
    test_config.set("experiment.n", 2)
    test_config.set("parallelism.enabled", True)
    test_config.set("parallelism.particle_update_max_concurrent", 2)
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        StubExecutor(),
        Random(1),
        tmp_path,
        "reference",
    )
    leader = make_solution(0.5, 0.5, 99)
    engine.archive.solutions = [leader]
    population = [make_solution(0.1, 0.1, 1), make_solution(0.2, 0.2, 2)]
    pbest = [solution.clone() for solution in population]

    async def synthetic_update(job):
        import asyncio

        await asyncio.sleep(0.01 if job.index == 0 else 0.0)
        solution = job.particle.clone(keep_id=True)
        solution.generated_text = f"updated {job.index}"
        if job.index == 1:
            return ParticleUpdateResult(job.index, solution, error="worker failed")
        return ParticleUpdateResult(
            job.index,
            solution,
            guided_candidate_diagnostics=(
                {
                    "phase": "optimization",
                    "generation": job.generation,
                    "solution_id": solution.solution_id,
                    "component": "topic",
                    "mode": "cognitive",
                    "effective_mode": "cognitive",
                    "used_inertia": False,
                    "current": "topic 1",
                    "target": "topic 1",
                    "raw_candidate_count": 0,
                    "accepted_candidate": None,
                    "rejected_count_by_reason": {},
                    "candidates": [],
                },
            ),
        )

    engine._update_particle_job_async = synthetic_update

    updated = engine._update_population(population, pbest, generation=1)

    assert [solution.generated_text for solution in updated] == ["updated 0", "updated 1"]
    error_log = (tmp_path / "particle_update_errors.jsonl").read_text(encoding="utf-8")
    assert "worker failed" in error_log
    diagnostics = [
        json.loads(line)
        for line in (tmp_path / "guided_candidate_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert diagnostics[0]["solution_id"] == population[0].solution_id


def test_external_archive_deduplicates_by_normalized_signature():
    archive = ExternalArchive(max_size=4, rng=Random(1))
    first = Solution(
        SemanticVector({"role": "Role A", "topic": "Topic A", "action": "Action A"}),
        "prompt first",
        "text first",
        Objectives(0.1, 0.1),
    )
    duplicate = Solution(
        SemanticVector({"role": " role a ", "topic": "topic a", "action": "action a"}),
        "prompt duplicate",
        "text duplicate",
        Objectives(1.0, 1.0),
    )

    archive.update([first, duplicate])

    assert len(archive.solutions) == 1
    assert archive.solutions[0].generated_text == "text first"


def test_unique_signature_evaluation_deduplicates_before_f2_and_propagates_results():
    service = StubEmbeddingService(
        {
            "reference": [1.0, 0.0],
            "text a": [1.0, 0.0],
            "text duplicate": [-1.0, 0.0],
            "text b": [0.0, 1.0],
            "text c": [-1.0, 0.0],
        }
    )
    first = Solution(
        SemanticVector({"role": "Role A", "topic": "Topic A", "action": "Action A"}),
        "prompt a",
        "text a",
    )
    duplicate = Solution(
        SemanticVector({"role": " role a ", "topic": "topic a", "action": "action a"}),
        "prompt duplicate",
        "text duplicate",
    )
    second = Solution(
        SemanticVector({"role": "Role B", "topic": "Topic B", "action": "Action B"}),
        "prompt b",
        "text b",
    )
    third = Solution(
        SemanticVector({"role": "Role C", "topic": "Topic C", "action": "Action C"}),
        "prompt c",
        "text c",
    )

    evaluate_unique_solutions_by_signature([first, duplicate, second, third], "reference", service)

    assert first.objectives is not None
    assert duplicate.objectives is not None
    assert second.objectives is not None
    assert third.objectives is not None
    assert first.objectives.f1 == pytest.approx(1.0)
    assert duplicate.objectives.f1 == pytest.approx(first.objectives.f1)
    assert duplicate.objectives.f2 == pytest.approx(first.objectives.f2)
    assert first.objectives.f2 == pytest.approx(1.5)
    assert second.objectives.f2 == pytest.approx(1.0)
    assert third.objectives.f2 == pytest.approx(1.5)
    assert duplicate.embedding == first.embedding


def test_mopso_influence_params_use_component_spec_and_strategy_fields(test_config, tmp_path):
    executor = StubExecutor()
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "Flooded roads near the bridge need urgent support.",
    )
    particle = Solution(
        SemanticVector({"role": "local official", "topic": "flooded roads", "action": "request aid"}),
        "prompt",
        "text",
    )

    params = engine._influence_task_params(
        "action",
        "request aid",
        "warn residents about flooding",
        particle,
        generation=1,
    )

    assert params["numCandidates"] == 6
    assert params["componentName"] == "action"
    assert params["componentDefinition"] == (
        "Communicative intent or discourse operation that indicates how the message communicates information. "
        "It must be a verb phrase, not a resource, program, service, or support type."
    )
    assert params["currentComponent"] == "request aid"
    assert params["targetComponent"] == "warn residents about flooding"
    assert params["otherComponents"] == {"role": "local official", "topic": "flooded roads"}
    assert params["referenceText"] == "Flooded roads near the bridge need urgent support."
    assert params["componentAdditionalInstruction"].startswith("Keep each candidate as a communicative verb phrase.")
    assert params["iteration"] == 0
    assert params["totalGenerations"] == test_config.iterations


def test_mopso_turbulence_params_include_selected_unit_contract(test_config, tmp_path):
    executor = StubExecutor()
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine._select_turbulence_candidate = lambda _component, _current, raw_candidates: raw_candidates[0]

    result = engine._turbulence_candidate("topic", "public safety alert")

    assert result == "public emergency alert"
    assert executor.turbulence_provider.preferred_pos == ("NOUN", "PROPN", "ADJ")
    params = executor.executed_task.task_params
    assert params["component"] == "public safety alert"
    assert params["componentType"] == "topic"
    assert params["targetWord"] == "safety"
    assert params["targetLemma"] == "safety"
    assert params["targetPos"] == "NOUN"
    assert params["targetSpan"] == [7, 13]
    assert params["targetWordLeftTokens"] == 1
    assert params["targetWordRightTokens"] == 1
    assert params["maxVariants"] == 6
    assert params["target_index"] == 1


def test_mopso_guided_candidate_rejects_excessive_angular_trajectory(test_config, tmp_path):
    candidate_x = math.cos(math.radians(80.0))
    candidate_y = 0.5
    candidate_z = math.sqrt(1.0 - candidate_x**2 - candidate_y**2)
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0],
            "wide candidate": [candidate_x, candidate_y, candidate_z],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )

    result = engine._select_guided_candidate("topic", "current topic", "target topic", ["wide candidate"])

    assert result is None


def test_mopso_guided_candidate_can_disable_angular_trajectory_validation(test_config, tmp_path):
    test_config.set("mopso.guided_trajectory_validation_enabled", False)
    candidate_x = math.cos(math.radians(80.0))
    candidate_y = 0.5
    candidate_z = math.sqrt(1.0 - candidate_x**2 - candidate_y**2)
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0],
            "wide candidate": [candidate_x, candidate_y, candidate_z],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )

    result = engine._select_guided_candidate("topic", "current topic", "target topic", ["wide candidate"])

    assert result == "wide candidate"


def test_mopso_guided_candidate_uses_configurable_trajectory_margin(test_config, tmp_path):
    test_config.set("mopso.guided_trajectory_relative_margin", 1.5)
    candidate_x = math.cos(math.radians(80.0))
    candidate_y = 0.5
    candidate_z = math.sqrt(1.0 - candidate_x**2 - candidate_y**2)
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0],
            "wide candidate": [candidate_x, candidate_y, candidate_z],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )

    result = engine._select_guided_candidate("topic", "current topic", "target topic", ["wide candidate"])

    assert result == "wide candidate"


def test_mopso_guided_candidate_diagnostics_report_basic_and_duplicate_reasons(test_config, tmp_path):
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [0.0, 1.0, 0.0],
            "valid topic": [0.2, math.sqrt(0.96), 0.0],
            "memory topic": [0.6, 0.8, 0.0],
            "semantic duplicate topic": [0.6, 0.8, 0.0],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine.component_memory.add_components({"topic": ["memory topic"]})
    diagnostics = []

    result = engine._select_guided_candidate(
        "topic",
        "current topic",
        "target topic",
        [
            "",
            "current topic",
            "target topic",
            "one two three four five six seven eight nine",
            "valid topic",
            " valid topic ",
            "semantic duplicate topic",
        ],
        diagnostic_context={
            "phase": "optimization",
            "generation": 1,
            "solution_id": "solution-1",
            "mode": "cognitive",
            "effective_mode": "cognitive",
            "used_inertia": False,
        },
        diagnostics_sink=diagnostics,
    )

    assert result == "valid topic"
    row = diagnostics[0]
    assert row["raw_candidate_count"] == 7
    assert row["accepted_candidate"] == "valid topic"
    assert row["rejected_count_by_reason"] == {
        "empty_candidate": 1,
        "duplicate_candidate_output": 1,
        "literal_copy_current": 1,
        "literal_copy_target": 1,
        "too_many_words": 1,
        "semantic_duplicate": 1,
    }
    assert row["candidates"][0] == {"candidate": "", "reason": "empty_candidate"}
    assert any(
        item == {"candidate": " valid topic ", "reason": "duplicate_candidate_output"}
        for item in row["candidates"]
    )


def test_mopso_guided_candidate_diagnostics_report_no_semantic_progress(test_config, tmp_path):
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [0.0, 1.0, 0.0],
            "away topic": [1.0, 0.0, 0.0],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    diagnostics = []

    result = engine._select_guided_candidate(
        "topic",
        "current topic",
        "target topic",
        ["away topic"],
        diagnostic_context={
            "phase": "optimization",
            "generation": 1,
            "solution_id": "solution-1",
            "mode": "cognitive",
            "effective_mode": "cognitive",
            "used_inertia": False,
        },
        diagnostics_sink=diagnostics,
    )

    assert result is None
    assert diagnostics[0]["rejected_count_by_reason"] == {"no_semantic_progress": 1}
    assert diagnostics[0]["candidates"] == [{"candidate": "away topic", "reason": "no_semantic_progress"}]


def test_mopso_guided_candidate_diagnostics_report_trajectory_inconsistency(test_config, tmp_path):
    candidate_x = math.cos(math.radians(80.0))
    candidate_y = 0.5
    candidate_z = math.sqrt(1.0 - candidate_x**2 - candidate_y**2)
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [math.cos(math.radians(60.0)), math.sin(math.radians(60.0)), 0.0],
            "wide candidate": [candidate_x, candidate_y, candidate_z],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    diagnostics = []

    result = engine._select_guided_candidate(
        "topic",
        "current topic",
        "target topic",
        ["wide candidate"],
        diagnostic_context={
            "phase": "optimization",
            "generation": 1,
            "solution_id": "solution-1",
            "mode": "cognitive",
            "effective_mode": "cognitive",
            "used_inertia": False,
        },
        diagnostics_sink=diagnostics,
    )

    assert result is None
    assert diagnostics[0]["rejected_count_by_reason"] == {"guided_trajectory_inconsistent": 1}
    assert diagnostics[0]["candidates"] == [
        {"candidate": "wide candidate", "reason": "guided_trajectory_inconsistent"}
    ]


def test_mopso_guided_candidate_diagnostics_preserve_best_valid_selection(test_config, tmp_path):
    executor = StubExecutor()
    executor.embedding_service = StubEmbeddingService(
        {
            "current topic": [1.0, 0.0, 0.0],
            "target topic": [0.0, 1.0, 0.0],
            "weaker topic": [0.8, 0.6, 0.0],
            "stronger topic": [0.2, math.sqrt(0.96), 0.0],
        }
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    diagnostics = []

    result = engine._select_guided_candidate(
        "topic",
        "current topic",
        "target topic",
        ["weaker topic", "stronger topic"],
        diagnostic_context={
            "phase": "optimization",
            "generation": 1,
            "solution_id": "solution-1",
            "mode": "cognitive",
            "effective_mode": "cognitive",
            "used_inertia": False,
        },
        diagnostics_sink=diagnostics,
    )

    assert result == "stronger topic"
    assert diagnostics[0]["accepted_candidate"] == "stronger topic"
    assert diagnostics[0]["rejected_count_by_reason"] == {}


def test_mopso_update_writes_guided_candidate_diagnostics(test_config, tmp_path):
    test_config.set("parallelism.enabled", False)
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    executor = GuidedMoveExecutor(["updated topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "cognitive"
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generate_text = lambda _prompt, **_kwargs: "valid generated text"
    engine._generated_text_fidelity = lambda _text: 1.0
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text", Objectives(0.6, 0.6))
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text", Objectives(0.7, 0.7))
    engine.archive.solutions = [leader]

    updated = engine._update_population([particle], [pbest], generation=1)

    rows = [
        json.loads(line)
        for line in (tmp_path / "guided_candidate_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert updated[0].vector.components["topic"] == "updated topic"
    assert row["phase"] == "optimization"
    assert row["generation"] == 1
    assert row["solution_id"] == particle.solution_id
    assert row["component"] == "topic"
    assert row["mode"] == "cognitive"
    assert row["effective_mode"] == "cognitive"
    assert row["used_inertia"] is False
    assert row["current"] == "current topic"
    assert row["target"] == "pbest topic"
    assert row["raw_candidate_count"] == 1
    assert row["accepted_candidate"] == "updated topic"
    assert row["rejected_count_by_reason"] == {}
    assert row["candidates"] == []


def test_mopso_update_skips_guided_candidate_diagnostics_when_disabled(test_config, tmp_path):
    test_config.set("parallelism.enabled", False)
    test_config.set("mopso.guided_candidate_diagnostics_enabled", False)
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    executor = GuidedMoveExecutor(["updated topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "cognitive"
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generate_text = lambda _prompt, **_kwargs: "valid generated text"
    engine._generated_text_fidelity = lambda _text: 1.0
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text", Objectives(0.6, 0.6))
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text", Objectives(0.7, 0.7))
    engine.archive.solutions = [leader]

    engine._update_population([particle], [pbest], generation=1)

    assert not (tmp_path / "guided_candidate_diagnostics.jsonl").exists()


def test_mopso_text_generation_uses_base_prompt_and_checkpoint_persists_anchors(test_config, tmp_path):
    executor = StubExecutor()
    central_anchors = ["bridge", "flooded roads", "request support", "urgent"]
    semantic_anchors = {"entities": ["bridge"], "topics": ["flooded roads"]}
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
        central_anchors,
        semantic_anchors=semantic_anchors,
    )

    engine._generate_text("prompt text")
    engine.archive.update_count = 2
    engine.archive.prune_count = 1
    payload = engine._checkpoint_payload(1, [], [], [])

    assert "centralAnchors" not in executor.executed_task.task_params
    assert "userPromptOverride" not in executor.executed_task.task_params
    assert payload["central_anchors"] == central_anchors
    assert payload["semantic_anchors"] == semantic_anchors
    assert payload["archive_stats"] == {"update_count": 2, "prune_count": 1}


def test_mopso_restores_archive_stats_from_checkpoint_state(test_config, tmp_path):
    test_config.set("experiment.frozen_components", ["role", "topic", "action"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        StubExecutor(),
        Random(1),
        tmp_path,
        "reference",
    )
    archived = make_solution(0.5, 0.5, 1)

    _population, archive = engine.run(
        [archived],
        archive_state=[archived],
        archive_stats={"update_count": 4, "prune_count": 2},
        component_memory={},
    )

    assert archive.update_count == 4
    assert archive.prune_count == 2


def test_mopso_anchor_inclusion_probability_matches_strategy_schedule(test_config, tmp_path):
    test_config.set("experiment.iterations", 3)
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        StubExecutor(),
        Random(1),
        tmp_path,
        "reference",
        ["bridge", "flooded roads", "request support"],
    )

    assert engine._anchor_inclusion_probability(1) == pytest.approx(0.05)
    assert engine._anchor_inclusion_probability(3) == pytest.approx(0.40)


def test_mopso_update_uses_anchor_override_when_probability_event_occurs(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    test_config.set("mopso.p_anchor_enabled", True)
    test_config.set("mopso.p_anchor_min", 1.0)
    test_config.set("mopso.p_anchor_max", 1.0)
    executor = GuidedMoveExecutor()
    central_anchors = ["bridge", "flooded roads", "request support"]
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
        central_anchors,
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "cognitive"
    engine._candidate_for_mode = lambda _component, mode, *_args: "updated topic" if mode == "cognitive" else None
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generated_text_fidelity = lambda _text: 1.0
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    updated = engine._update_particle(particle, pbest, leader, generation=1)

    synthetic_task = [task for task in executor.tasks if task.semantic_task == TASK_SYNTHETIC_TEXT][0]
    assert "userPromptOverride" in synthetic_task.task_params
    assert "Reference-specific anchors:" in synthetic_task.task_params["userPromptOverride"]
    assert "Instruction:" in synthetic_task.task_params["userPromptOverride"]
    assert "centralAnchors" not in synthetic_task.task_params
    assert updated.metadata["used_central_anchors"] is True
    assert updated.metadata["anchor_inclusion_probability"] == pytest.approx(1.0)


def test_mopso_anchor_disabled_uses_base_prompt_without_probability_eval(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    test_config.set("mopso.p_anchor_enabled", False)
    test_config.set("mopso.p_anchor_min", 1.0)
    test_config.set("mopso.p_anchor_max", 1.0)
    executor = GuidedMoveExecutor()
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
        ["bridge", "flooded roads", "request support"],
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "cognitive"
    engine._candidate_for_mode = lambda _component, mode, *_args: "updated topic" if mode == "cognitive" else None
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generated_text_fidelity = lambda _text: 1.0

    def fail_probability(_generation):
        raise AssertionError("anchor probability should not be evaluated")

    engine._anchor_inclusion_probability = fail_probability
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    updated = engine._update_particle(particle, pbest, leader, generation=1)

    synthetic_task = [task for task in executor.tasks if task.semantic_task == TASK_SYNTHETIC_TEXT][0]
    assert "userPromptOverride" not in synthetic_task.task_params
    assert updated.metadata["used_central_anchors"] is False
    assert updated.metadata["anchor_inclusion_probability"] is None


def test_mopso_update_uses_base_prompt_when_anchor_event_does_not_occur(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    test_config.set("mopso.p_anchor_enabled", True)
    test_config.set("mopso.p_anchor_min", 0.0)
    test_config.set("mopso.p_anchor_max", 0.0)
    executor = GuidedMoveExecutor()
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
        ["bridge", "flooded roads", "request support"],
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "cognitive"
    engine._candidate_for_mode = lambda _component, mode, *_args: "updated topic" if mode == "cognitive" else None
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generated_text_fidelity = lambda _text: 1.0
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    updated = engine._update_particle(particle, pbest, leader, generation=1)

    synthetic_task = [task for task in executor.tasks if task.semantic_task == TASK_SYNTHETIC_TEXT][0]
    assert "userPromptOverride" not in synthetic_task.task_params
    assert updated.metadata["used_central_anchors"] is False
    assert updated.metadata["anchor_inclusion_probability"] == pytest.approx(0.0)


def test_mopso_unchanged_particle_does_not_generate_text_or_evaluate_anchor_probability(test_config, tmp_path):
    executor = GuidedMoveExecutor()
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
        ["bridge", "flooded roads", "request support"],
    )
    engine.components = topic_only_components()
    engine._candidate_for_mode = lambda *_args: None
    engine._anchor_inclusion_probability = lambda _generation: (_ for _ in ()).throw(AssertionError("not expected"))
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        Objectives(0.5, 0.5),
        velocity={"topic": 0.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )

    updated = engine._update_particle(particle, particle.clone(), particle.clone(), generation=1)

    assert not updated.changed
    assert all(task.semantic_task != TASK_SYNTHETIC_TEXT for task in executor.tasks)


def test_mopso_stores_guided_movement_memory_after_cognitive_acceptance(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    executor = GuidedMoveExecutor()
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "cognitive"
    engine._candidate_for_mode = lambda _component, mode, *_args: "updated topic" if mode == "cognitive" else None
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generate_text = lambda _prompt, **_kwargs: "valid generated text"
    engine._generated_text_fidelity = lambda _text: 1.0
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    updated = engine._update_particle(particle, pbest, leader, generation=1)

    assert updated.changed
    assert updated.vector.components["topic"] == "updated topic"
    assert updated.last_guided_move["topic"] == {
        "movement": "cognitive",
        "target_position": {"topic": "pbest topic"},
        "accepted_component": "updated topic",
        "iteration": 1,
    }


def test_solution_clone_and_serialization_deep_copy_inertia_memory():
    solution = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        last_guided_move={
            "topic": {
                "movement": "social",
                "target_position": {"topic": "leader topic"},
                "accepted_component": "leader aligned topic",
                "iteration": 1,
            }
        },
    )

    clone = solution.clone()
    clone.last_guided_move["topic"]["target_position"]["topic"] = "mutated target"
    payload = solution_to_dict(solution)
    payload["last_guided_move"]["topic"]["target_position"]["topic"] = "payload target"
    restored = solution_from_dict(solution_to_dict(solution))
    restored.last_guided_move["topic"]["target_position"]["topic"] = "restored target"

    assert solution.last_guided_move["topic"]["target_position"]["topic"] == "leader topic"


def test_mopso_inertia_repeats_last_social_move_toward_previous_target(test_config, tmp_path):
    test_config.set("experiment.iterations", 2)
    executor = GuidedMoveExecutor(["leader aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        last_guided_move={
            "topic": {
                "movement": "social",
                "target_position": {"topic": "leader topic"},
                "accepted_component": "previous accepted topic",
                "iteration": 1,
            }
        },
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "new leader topic"}), "prompt", "text")

    result = engine._candidate_for_mode("topic", "inertia", particle, pbest, leader, generation=2)

    assert result == "leader aligned topic"
    assert executor.executed_task.task_params["targetComponent"] == "leader topic"


def test_mopso_inertia_guided_candidate_diagnostics_use_resolved_target(test_config, tmp_path):
    test_config.set("experiment.iterations", 2)
    executor = GuidedMoveExecutor(["leader aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        last_guided_move={
            "topic": {
                "movement": "social",
                "target_position": {"topic": "leader topic"},
                "accepted_component": "previous accepted topic",
                "iteration": 1,
            }
        },
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "new leader topic"}), "prompt", "text")
    diagnostics = []

    result = engine._candidate_for_mode(
        "topic",
        "inertia",
        particle,
        pbest,
        leader,
        generation=2,
        guided_diagnostics_sink=diagnostics,
    )

    assert result == "leader aligned topic"
    row = diagnostics[0]
    assert row["mode"] == "inertia"
    assert row["effective_mode"] == "social"
    assert row["used_inertia"] is True
    assert row["target"] == "leader topic"
    assert row["accepted_candidate"] == "leader aligned topic"


def test_mopso_inertia_consumes_memory_without_creating_new_chain(test_config, tmp_path):
    test_config.set("experiment.iterations", 2)
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    executor = GuidedMoveExecutor(["leader aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "inertia"
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generate_text = lambda _prompt, **_kwargs: "valid generated text"
    engine._generated_text_fidelity = lambda _text: 1.0
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        last_guided_move={
            "topic": {
                "movement": "social",
                "target_position": {"topic": "leader topic"},
                "accepted_component": "previous accepted topic",
                "iteration": 1,
            }
        },
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "new leader topic"}), "prompt", "text")

    updated = engine._update_particle(particle, pbest, leader, generation=2)

    assert updated.changed
    assert updated.vector.components["topic"] == "leader aligned topic"
    assert "topic" not in updated.last_guided_move


def test_mopso_rejected_inertia_text_restores_consumed_memory(test_config, tmp_path):
    test_config.set("experiment.iterations", 2)
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
    executor = GuidedMoveExecutor(["leader aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    engine.components = topic_only_components()
    engine._guided_mode = lambda *_args: "inertia"
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generate_text = lambda _prompt, **_kwargs: "not a feasible request"
    memory = {
        "topic": {
            "movement": "social",
            "target_position": {"topic": "leader topic"},
            "accepted_component": "previous accepted topic",
            "iteration": 1,
        }
    }
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"topic": 4.0},
        initial_components={"topic": "current topic"},
        last_guided_move=memory,
        changed=False,
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "new leader topic"}), "prompt", "text")

    updated = engine._update_particle(particle, pbest, leader, generation=2)

    assert not updated.changed
    assert updated.vector.components == particle.vector.components
    assert updated.last_guided_move == memory
    assert updated.velocity["topic"] != particle.velocity["topic"]
    assert executor.executed_task.task_params["targetComponent"] == "leader topic"
    assert (tmp_path / "optimization_rejections.jsonl").exists()


def test_mopso_async_inertia_uses_previous_target(test_config, tmp_path):
    test_config.set("experiment.iterations", 2)
    executor = AsyncGuidedMoveExecutor(["async aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        last_guided_move={
            "topic": {
                "movement": "social",
                "target_position": {"topic": "leader topic"},
                "accepted_component": "previous accepted topic",
                "iteration": 1,
            }
        },
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "new leader topic"}), "prompt", "text")

    result = asyncio.run(engine._candidate_for_mode_async("topic", "inertia", particle, pbest, leader, 2, Random(1)))

    assert result == "async aligned topic"
    assert executor.executed_task.task_params["targetComponent"] == "leader topic"


def test_mopso_expired_inertia_memory_does_not_enable_inertia(test_config, tmp_path):
    test_config.set("experiment.iterations", 3)
    executor = GuidedMoveExecutor(["leader aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        last_guided_move={
            "topic": {
                "movement": "social",
                "target_position": {"topic": "leader topic"},
                "accepted_component": "previous accepted topic",
                "iteration": 1,
            }
        },
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    assert engine._guided_mode("topic", 10.0, 0.0, 0.0, particle, generation=3) is None
    assert engine._candidate_for_mode("topic", "inertia", particle, pbest, leader, generation=3) is None
    assert executor.executed_task is None


def test_mopso_legacy_textual_last_guided_move_does_not_enable_inertia(test_config, tmp_path):
    executor = GuidedMoveExecutor(["leader aligned topic"])
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        executor,
        Random(1),
        tmp_path,
        "reference",
    )
    particle = Solution(
        SemanticVector({"topic": "current topic"}),
        "prompt",
        "text",
        last_guided_move={"topic": "SMB Owner"},
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    assert engine._guided_mode("topic", 10.0, 0.0, 0.0, particle, generation=2) is None
    assert engine._candidate_for_mode("topic", "inertia", particle, pbest, leader, generation=2) is None
    assert executor.executed_task is None
