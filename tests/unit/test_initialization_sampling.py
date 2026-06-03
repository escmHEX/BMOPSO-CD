from __future__ import annotations

from random import Random

from binary_mopso_cd.initialization import InitialPopulationBuilder
from binary_mopso_cd.router import TASK_POOL_EXPANSION


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


def test_candidate_sampling_balances_component_frequencies(test_config):
    test_config.set("experiment.n", 2)
    builder = InitialPopulationBuilder(test_config, router=None, executor=None, rng=Random(1))
    pools = {
        "role": [f"role {idx}" for idx in range(4)],
        "topic": [f"topic {idx}" for idx in range(4)],
        "action": [f"action {idx}" for idx in range(4)],
    }

    vectors = builder._candidate_vectors(pools)

    assert len(vectors) == 4 * test_config.n
    for component in ["role", "topic", "action"]:
        counts = {value: 0 for value in pools[component]}
        for vector in vectors:
            counts[vector.components[component]] += 1
        assert max(counts.values()) - min(counts.values()) <= 1


class PassthroughRouter:
    def route(self, task):
        return task


class RecordingExecutor:
    def __init__(self):
        self.tasks = []

    def execute(self, task):
        self.tasks.append(task)
        return ["warn residents"]


def test_build_pool_passes_strategy_component_context(test_config):
    executor = RecordingExecutor()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    anchors = {"entities": ["bridge"], "topics": ["flooded roads"], "actions": ["request support"]}

    builder._build_pool(
        "action",
        1,
        "Flooded roads near the bridge need urgent support.",
        anchors,
        central_anchor_count=3,
        domain="social media messages related to crises and emergencies",
        existing=["request help"],
        task_name=TASK_POOL_EXPANSION,
    )

    params = executor.tasks[0].task_params
    assert params["component"] == "action"
    assert params["component_type"] == "actions"
    assert params["required_new_items"] == 1
    assert params["existing"] == ["request help"]
    assert params["anchors"] == anchors
    assert params["domain"] == "social media messages related to crises and emergencies"
    assert "Return communicative intents" in params["component_additional_instruction"]
