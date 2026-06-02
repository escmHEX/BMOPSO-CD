from __future__ import annotations

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.selection import mmr_select, rank_solutions
from binary_mopso_cd.services.embedding import fake_embedding_service


def make_solution(f1: float, f2: float, idx: int) -> Solution:
    return Solution(
        SemanticVector({"role": f"role {idx}", "topic": f"topic {idx}", "action": f"action {idx}"}),
        f"prompt {idx}",
        f"generated text {idx}",
        Objectives(f1, f2),
    )


def test_selection_filters_ranks_and_mmr_selects():
    ranked = rank_solutions(
        [make_solution(0.1, 1.0, 1), make_solution(0.5, 0.5, 2), make_solution(0.8, 0.8, 3)],
        tau_min=0.2,
        tau_max=0.94,
        epsilon=1e-4,
    )
    selected = mmr_select(ranked, fake_embedding_service(), k=2, lambda_mmr=0.35)
    assert len(ranked) == 2
    assert len(selected) == 2
    assert all(item.selected for item in selected)

