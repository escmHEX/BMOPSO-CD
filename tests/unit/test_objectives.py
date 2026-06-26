from __future__ import annotations

import numpy as np

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.objectives import (
    evaluate_solutions,
    recompute_peer_set_f2,
    recompute_pbest_f2_against_population,
    semantic_diversity_scores,
)


class StubEmbeddingService:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


def test_f2_uses_vectorized_cosine_distance_matrix():
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    scores = semantic_diversity_scores(embeddings)
    assert np.allclose(scores, [0.5, 1.0, 0.5])


def test_evaluate_solutions_sets_f1_f2(real_embedding_service):
    service = real_embedding_service
    solutions = [
        Solution(SemanticVector({"role": "a", "topic": "b", "action": "c"}), "p1", "text one"),
        Solution(SemanticVector({"role": "d", "topic": "e", "action": "f"}), "p2", "text two"),
    ]
    evaluate_solutions(solutions, "reference", service)
    assert all(solution.objectives is not None for solution in solutions)
    assert all(solution.embedding for solution in solutions)


def test_recompute_peer_set_f2_preserves_f1_and_counts_duplicate_occurrences():
    service = StubEmbeddingService(
        {
            "text a": [1.0, 0.0],
            "text b": [0.0, 1.0],
            "text duplicate": [1.0, 0.0],
        }
    )
    solutions = [
        Solution(SemanticVector({"topic": "a"}), "p1", "text a", Objectives(0.1, 9.0)),
        Solution(SemanticVector({"topic": "b"}), "p2", "text b", Objectives(0.2, 9.0)),
        Solution(SemanticVector({"topic": "c"}), "p3", "text duplicate", Objectives(0.3, 9.0)),
    ]

    recompute_peer_set_f2(solutions, service)

    assert [solution.objectives.f1 for solution in solutions] == [0.1, 0.2, 0.3]
    assert [solution.objectives.f2 for solution in solutions] == [0.5, 1.0, 0.5]
    assert solutions[0].embedding == [1.0, 0.0]


def test_recompute_peer_set_f2_uses_zero_for_single_solution():
    service = StubEmbeddingService({"solo": [1.0, 0.0]})
    solution = Solution(SemanticVector({"topic": "solo"}), "prompt", "solo", Objectives(0.7, 9.0))

    recompute_peer_set_f2([solution], service)

    assert solution.objectives == Objectives(0.7, 0.0)


def test_recompute_pbest_f2_uses_population_without_matching_index():
    service = StubEmbeddingService(
        {
            "current zero": [1.0, 0.0],
            "current one": [0.0, 1.0],
            "pbest zero": [0.0, 1.0],
            "pbest one": [1.0, 0.0],
        }
    )
    population = [
        Solution(SemanticVector({"topic": "current zero"}), "p0", "current zero", Objectives(0.5, 9.0)),
        Solution(SemanticVector({"topic": "current one"}), "p1", "current one", Objectives(0.5, 9.0)),
    ]
    pbest = [
        Solution(SemanticVector({"topic": "pbest zero"}), "pb0", "pbest zero", Objectives(0.1, 9.0)),
        Solution(SemanticVector({"topic": "pbest one"}), "pb1", "pbest one", Objectives(0.2, 9.0)),
    ]

    recompute_pbest_f2_against_population(pbest, population, service)

    assert [solution.objectives.f1 for solution in pbest] == [0.1, 0.2]
    assert [solution.objectives.f2 for solution in pbest] == [0.0, 0.0]
