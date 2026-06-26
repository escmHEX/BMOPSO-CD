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


def recompute_peer_set_f2(
    solutions: list[Solution],
    embedding_service: EmbeddingService,
) -> list[Solution]:
    if not solutions:
        return solutions
    embeddings = _generated_text_embeddings(solutions, embedding_service)
    f2_values = semantic_diversity_scores(embeddings)
    _assign_f2_values(solutions, embeddings, f2_values)
    return solutions


def recompute_pbest_f2_against_population(
    pbest: list[Solution],
    population: list[Solution],
    embedding_service: EmbeddingService,
) -> list[Solution]:
    if len(pbest) != len(population):
        raise ValueError("pbest and population must have the same length")
    if not pbest:
        return pbest
    pbest_embeddings = _generated_text_embeddings(pbest, embedding_service)
    population_embeddings = _generated_text_embeddings(population, embedding_service)
    if len(population) <= 1:
        f2_values = np.zeros(len(pbest), dtype=float)
    else:
        distances = 1.0 - _cosine_matrix_between(pbest_embeddings, population_embeddings)
        np.fill_diagonal(distances, 0.0)
        f2_values = distances.sum(axis=1) / (len(population) - 1)
    _assign_f2_values(pbest, pbest_embeddings, f2_values)
    return pbest


def semantic_fidelity_scores(
    generated_texts: list[str],
    reference_text: str,
    embedding_service: EmbeddingService,
) -> np.ndarray:
    if not generated_texts:
        return np.zeros(0, dtype=float)
    generated_embeddings = embedding_service.encode(generated_texts, text_type="generated_text")
    reference_embedding = embedding_service.encode([reference_text], text_type="reference_text")[0]
    return generated_embeddings @ reference_embedding


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


def _generated_text_embeddings(solutions: list[Solution], embedding_service: EmbeddingService) -> np.ndarray:
    return embedding_service.encode([solution.generated_text for solution in solutions], text_type="generated_text")


def _cosine_matrix_between(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_matrix = np.asarray(left, dtype=float)
    right_matrix = np.asarray(right, dtype=float)
    left_norms = np.linalg.norm(left_matrix, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_matrix, axis=1, keepdims=True)
    left_norms[left_norms == 0.0] = 1.0
    right_norms[right_norms == 0.0] = 1.0
    return (left_matrix / left_norms) @ (right_matrix / right_norms).T


def _assign_f2_values(solutions: list[Solution], embeddings: np.ndarray, f2_values: np.ndarray) -> None:
    for solution, embedding, f2 in zip(solutions, embeddings, f2_values, strict=True):
        if solution.objectives is None:
            raise ValueError("contextual f2 recomputation requires existing f1 objectives")
        solution.objectives = Objectives(solution.objectives.f1, float(f2))
        solution.embedding = [float(x) for x in embedding]
