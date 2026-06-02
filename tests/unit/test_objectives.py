from __future__ import annotations

import numpy as np

from binary_mopso_cd.entities import SemanticVector, Solution
from binary_mopso_cd.objectives import evaluate_solutions, semantic_diversity_scores


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
