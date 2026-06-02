from __future__ import annotations

import math
from random import Random

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.mopso import ExternalArchive, crowding_distance, dominates, utility


def make_solution(f1: float, f2: float, idx: int) -> Solution:
    return Solution(
        SemanticVector({"role": f"role {idx}", "topic": f"topic {idx}", "action": f"action {idx}"}),
        f"prompt {idx}",
        f"text {idx}",
        Objectives(f1, f2),
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

