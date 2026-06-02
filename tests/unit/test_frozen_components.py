from __future__ import annotations

from random import Random

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.mopso import BinaryMOPSOCDEngine, ExternalArchive
from binary_mopso_cd.router import SemanticRouter


def test_frozen_component_remains_initial(test_config, tmp_path):
    test_config.set("experiment.frozen_components", ["role"])
    router = SemanticRouter(test_config)
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path)
    rng = Random(3)
    particle = Solution(
        SemanticVector({"role": "fixed role", "topic": "storm topic", "action": "warn residents"}),
        "prompt",
        "generated",
        Objectives(0.5, 0.5),
        initial_components={"role": "fixed role", "topic": "storm topic", "action": "warn residents"},
        velocity={"role": 2.0, "topic": 2.0, "action": 2.0},
    )
    pbest = particle.clone()
    leader = particle.clone()
    engine = BinaryMOPSOCDEngine(test_config, router, executor, rng, tmp_path, "reference")
    engine.archive = ExternalArchive(4, rng, [leader])
    updated = engine._update_particle(particle, pbest, leader, generation=1)
    assert updated.vector.components["role"] == "fixed role"
    assert updated.velocity["role"] == 2.0

