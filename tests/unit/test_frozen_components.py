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


def test_all_frozen_components_return_initial_population_without_updates(test_config, tmp_path, monkeypatch):
    test_config.set("experiment.frozen_components", ["role", "topic", "action"])
    test_config.set("experiment.iterations", 3)
    router = SemanticRouter(test_config)
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path)
    rng = Random(3)
    initial = Solution(
        SemanticVector({"role": "fixed role", "topic": "storm topic", "action": "warn residents"}),
        "prompt",
        "generated",
        Objectives(0.5, 0.5),
        initial_components={"role": "fixed role", "topic": "storm topic", "action": "warn residents"},
        velocity={"role": 2.0, "topic": 2.0, "action": 2.0},
        changed=False,
    )
    engine = BinaryMOPSOCDEngine(test_config, router, executor, rng, tmp_path, "reference")

    def fail_update(*_args, **_kwargs):
        raise AssertionError("particle updates should be skipped")

    monkeypatch.setattr(engine, "_update_population", fail_update)

    population, archive = engine.run([initial])

    assert [solution.vector.components for solution in population] == [initial.vector.components]
    assert [solution.generated_text for solution in population] == [initial.generated_text]
    assert [solution.vector.components for solution in archive.solutions] == [initial.vector.components]
    assert (tmp_path / "evolucion_metricas.csv").exists()
