from __future__ import annotations

from random import Random

from binary_mopso_cd.initialization import InitialPopulationBuilder


def test_candidate_sampling_never_exceeds_four_n(test_config):
    test_config.set("experiment.n", 2)
    builder = InitialPopulationBuilder(test_config, router=None, executor=None, rng=Random(1))
    pools = {
        "role": [f"role {idx}" for idx in range(20)],
        "topic": [f"topic {idx}" for idx in range(20)],
        "action": [f"action {idx}" for idx in range(20)],
    }
    vectors = builder._candidate_vectors(pools)
    assert len(vectors) == 4 * test_config.n
