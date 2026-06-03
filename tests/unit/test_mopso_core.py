from __future__ import annotations

import math
from random import Random

import numpy as np
import pytest

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.mopso import (
    ExternalArchive,
    crowding_distance,
    dominates,
    evaluate_unique_solutions_by_signature,
    utility,
)


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

    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


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
