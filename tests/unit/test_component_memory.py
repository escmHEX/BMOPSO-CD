from __future__ import annotations

from binary_mopso_cd.component_memory import ComponentMemoryIndex
from binary_mopso_cd.entities import SemanticVector, Solution


def test_component_memory_indexes_batches_by_component(real_embedding_service):
    index = ComponentMemoryIndex(["role", "topic"], real_embedding_service)
    solutions = [
        Solution(SemanticVector({"role": "local official", "topic": "flood alert"}), "p1", "g1"),
        Solution(SemanticVector({"role": "local official", "topic": "road closure"}), "p2", "g2"),
    ]
    index.add_solutions(solutions)
    assert len(index.texts["role"]) == 1
    assert len(index.texts["topic"]) == 2
    candidates = real_embedding_service.encode(["local official"], text_type="component")
    sims = index.max_similarity("role", candidates)
    assert sims.shape == (1,)
    assert sims[0] > 0.99
