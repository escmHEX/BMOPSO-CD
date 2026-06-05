from __future__ import annotations

import math
from random import Random

import numpy as np
import pytest

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.mopso import (
    BinaryMOPSOCDEngine,
    ExternalArchive,
    crowding_distance,
    dominates,
    evaluate_unique_solutions_by_signature,
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
        if text in {"pbest topic", "leader topic", "leader aligned topic", "updated topic"}:
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

    assert params["numCandidates"] == 7
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
    assert params["maxVariants"] == 7
    assert params["target_index"] == 1


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
    payload = engine._checkpoint_payload(1, [], [], [])

    assert "centralAnchors" not in executor.executed_task.task_params
    assert "userPromptOverride" not in executor.executed_task.task_params
    assert payload["central_anchors"] == central_anchors
    assert payload["semantic_anchors"] == semantic_anchors


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
    assert engine._anchor_inclusion_probability(3) == pytest.approx(0.70)


def test_mopso_update_uses_anchor_override_when_probability_event_occurs(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
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


def test_mopso_update_uses_base_prompt_when_anchor_event_does_not_occur(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 0.0)
    test_config.set("mopso.p_tur_min", 0.0)
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


def test_mopso_stores_guided_movement_type_after_cognitive_acceptance(test_config, tmp_path):
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
    assert updated.last_guided_move["topic"] == "cognitive"


def test_mopso_inertia_repeats_last_social_move_toward_leader(test_config, tmp_path):
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
        last_guided_move={"topic": "social"},
    )
    pbest = Solution(SemanticVector({"topic": "pbest topic"}), "prompt", "text")
    leader = Solution(SemanticVector({"topic": "leader topic"}), "prompt", "text")

    result = engine._candidate_for_mode("topic", "inertia", particle, pbest, leader, generation=1)

    assert result == "leader aligned topic"
    assert executor.executed_task.task_params["targetComponent"] == "leader topic"


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

    assert engine._guided_mode("topic", 10.0, 0.0, 0.0, particle) is None
    assert engine._candidate_for_mode("topic", "inertia", particle, pbest, leader, generation=1) is None
    assert executor.executed_task is None
