from __future__ import annotations

from random import Random

import numpy as np

from binary_mopso_cd.initialization import InitialPopulationBuilder
from binary_mopso_cd.entities import SemanticVector
from binary_mopso_cd.router import (
    TASK_ANCHORS,
    TASK_CENTRAL_ANCHOR_SELECTION,
    TASK_POOL_EXPANSION,
    TASK_SYNTHETIC_TEXT,
)


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
        self.embedding_service = ConstantEmbeddingService()
        self.outdir = None

    def execute(self, task):
        self.tasks.append(task)
        if task.semantic_task == TASK_ANCHORS:
            return {
                "entities": ["bridge"],
                "topics": ["flooded roads"],
                "actions": ["request support"],
                "constraints": ["urgent"],
            }
        if task.semantic_task == TASK_CENTRAL_ANCHOR_SELECTION:
            return ["bridge", "flooded roads", "request support", "urgent"]
        return ["warn residents"]


class ConstantEmbeddingService:
    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _text in texts], dtype=float)


class TextRecordingExecutor:
    def __init__(self, texts: list[str]):
        self.tasks = []
        self.texts = list(texts)
        self.embedding_service = ConstantEmbeddingService()
        self.outdir = None

    def execute(self, task):
        self.tasks.append(task)
        return self.texts.pop(0)


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


def test_reference_context_selects_central_anchors_once_after_extraction(test_config):
    executor = RecordingExecutor()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))

    context = builder.build_reference_context("Flooded roads near the bridge need urgent support.")

    assert [task.semantic_task for task in executor.tasks] == [
        TASK_ANCHORS,
        TASK_CENTRAL_ANCHOR_SELECTION,
    ]
    assert context.semantic_anchors["entities"] == ["bridge"]
    assert context.central_anchors == ["bridge", "flooded roads", "request support", "urgent"]
    params = executor.tasks[1].task_params
    assert params["referenceText"] == "Flooded roads near the bridge need urgent support."
    assert params["numCentralAnchors"] == 4


def test_generate_texts_reuses_same_central_anchors(test_config):
    test_config.set("experiment.n", 2)
    executor = TextRecordingExecutor(["first generated text.", "second generated text."])
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    central_anchors = ["bridge", "flooded roads", "request support", "urgent"]
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
    ]

    generated = builder._generate_texts(items, "reference text", central_anchors)

    assert len(generated) == 2
    synthetic_tasks = [task for task in executor.tasks if task.semantic_task == TASK_SYNTHETIC_TEXT]
    assert len(synthetic_tasks) == 2
    assert [task.task_params["centralAnchors"] for task in synthetic_tasks] == [central_anchors, central_anchors]
