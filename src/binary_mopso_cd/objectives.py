from __future__ import annotations

import numpy as np

from binary_mopso_cd.entities import Objectives, Solution
from binary_mopso_cd.services.embedding import EmbeddingService
from binary_mopso_cd.utils import cosine_matrix


def semantic_diversity_scores(embeddings: np.ndarray) -> np.ndarray:
    n = embeddings.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=float)
    similarities = cosine_matrix(embeddings)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances.sum(axis=1) / (n - 1)


def evaluate_solutions(
    solutions: list[Solution],
    reference_text: str,
    embedding_service: EmbeddingService,
) -> list[Solution]:
    if not solutions:
        return []
    generated_texts = [solution.generated_text for solution in solutions]
    generated_embeddings = embedding_service.encode(generated_texts, text_type="generated_text")
    reference_embedding = embedding_service.encode([reference_text], text_type="reference_text")[0]
    f1_values = generated_embeddings @ reference_embedding
    f2_values = semantic_diversity_scores(generated_embeddings)
    for solution, embedding, f1, f2 in zip(solutions, generated_embeddings, f1_values, f2_values, strict=True):
        solution.objectives = Objectives(float(f1), float(f2))
        solution.embedding = [float(x) for x in embedding]
    return solutions

