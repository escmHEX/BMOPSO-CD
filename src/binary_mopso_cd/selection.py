from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from binary_mopso_cd.entities import Solution
from binary_mopso_cd.services.embedding import EmbeddingService


@dataclass(slots=True)
class RankedSolution:
    solution: Solution
    topsis_score: float
    selected: bool = False


def entropy_weights(matrix: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    n, m = matrix.shape
    if n <= 1:
        return np.ones(m) / m
    column_shift = np.maximum(0.0, -matrix.min(axis=0, keepdims=True)) + epsilon
    shifted = matrix + column_shift
    sums = shifted.sum(axis=0, keepdims=True)
    probabilities = shifted / sums
    entropies = -(probabilities * np.log(probabilities)).sum(axis=0) / math.log(n)
    divergence = 1.0 - entropies
    total = divergence.sum()
    if total <= 0:
        return np.ones(m) / m
    return divergence / total


def topsis_scores(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=0, keepdims=True)
    denom[denom == 0.0] = 1.0
    weighted = (matrix / denom) * weights
    ideal = weighted.max(axis=0)
    anti_ideal = weighted.min(axis=0)
    d_pos = np.linalg.norm(weighted - ideal, axis=1)
    d_neg = np.linalg.norm(weighted - anti_ideal, axis=1)
    total = d_pos + d_neg
    total[total == 0.0] = 1.0
    return d_neg / total


def rank_solutions(solutions: list[Solution], tau_min: float, tau_max: float, epsilon: float) -> list[RankedSolution]:
    filtered = [
        solution
        for solution in solutions
        if solution.objectives is not None and tau_min <= solution.objectives.f1 <= tau_max
    ]
    if not filtered:
        return []
    matrix = np.asarray([[solution.objectives.f1, solution.objectives.f2] for solution in filtered], dtype=float)
    weights = entropy_weights(matrix, epsilon=epsilon)
    scores = topsis_scores(matrix, weights)
    ranked = [RankedSolution(solution, float(score)) for solution, score in zip(filtered, scores, strict=True)]
    ranked.sort(key=lambda item: item.topsis_score, reverse=True)
    return ranked


def mmr_select(
    ranked: list[RankedSolution],
    embedding_service: EmbeddingService,
    k: int,
    lambda_mmr: float,
) -> list[RankedSolution]:
    if not ranked or k <= 0:
        return []
    selected: list[RankedSolution] = [ranked[0]]
    remaining = ranked[1:]
    selected[0].selected = True
    while remaining and len(selected) < k:
        candidate_texts = [item.solution.generated_text for item in remaining]
        selected_texts = [item.solution.generated_text for item in selected]
        candidate_embeddings = embedding_service.encode(candidate_texts, text_type="generated_text")
        selected_embeddings = embedding_service.encode(selected_texts, text_type="generated_text")
        sims = np.maximum(candidate_embeddings @ selected_embeddings.T, 0.0)
        redundancy = sims.max(axis=1)
        mmr_scores = [
            lambda_mmr * item.topsis_score - (1.0 - lambda_mmr) * float(redundancy[idx])
            for idx, item in enumerate(remaining)
        ]
        best_idx = int(np.argmax(mmr_scores))
        winner = remaining.pop(best_idx)
        winner.selected = True
        selected.append(winner)
    return selected
