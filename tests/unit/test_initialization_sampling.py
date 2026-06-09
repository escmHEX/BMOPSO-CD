from __future__ import annotations

import asyncio
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


class AsyncTextRecordingExecutor:
    def __init__(self, texts_by_prompt: dict[str, str], delays_by_prompt: dict[str, float] | None = None):
        self.tasks = []
        self.texts_by_prompt = dict(texts_by_prompt)
        self.delays_by_prompt = dict(delays_by_prompt or {})
        self.embedding_service = ConstantEmbeddingService()
        self.outdir = None
        self.active = 0
        self.max_active = 0

    async def execute_async(self, task):
        self.tasks.append(task)
        prompt = str(task.task_params["prompt"])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delays_by_prompt.get(prompt, 0.0))
            value = self.texts_by_prompt[prompt]
            if isinstance(value, Exception):
                raise value
            return value
        finally:
            self.active -= 1

    def execute(self, task):
        self.tasks.append(task)
        return self.texts_by_prompt[str(task.task_params["prompt"])]


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


def test_reference_context_extracts_semantic_anchors_without_central_selection(test_config):
    executor = RecordingExecutor()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))

    context = builder.build_reference_context("Flooded roads near the bridge need urgent support.")

    assert [task.semantic_task for task in executor.tasks] == [TASK_ANCHORS]
    assert context.semantic_anchors["entities"] == ["bridge"]


def test_select_central_anchors_uses_reference_context(test_config):
    executor = RecordingExecutor()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    context = builder.build_reference_context("Flooded roads near the bridge need urgent support.")
    central_anchors = builder.select_central_anchors(
        "Flooded roads near the bridge need urgent support.",
        context.semantic_anchors,
    )

    assert [task.semantic_task for task in executor.tasks] == [
        TASK_ANCHORS,
        TASK_CENTRAL_ANCHOR_SELECTION,
    ]
    assert central_anchors == ["bridge", "flooded roads", "request support", "urgent"]
    params = executor.tasks[1].task_params
    assert params["referenceText"] == "Flooded roads near the bridge need urgent support."
    assert params["numCentralAnchors"] == 4


def test_generate_texts_uses_base_prompt_without_central_anchors(test_config):
    test_config.set("experiment.n", 2)
    executor = TextRecordingExecutor(["first generated text.", "second generated text."])
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
    ]

    generated = builder._generate_texts(items, "reference text")

    assert len(generated) == 2
    synthetic_tasks = [task for task in executor.tasks if task.semantic_task == TASK_SYNTHETIC_TEXT]
    assert len(synthetic_tasks) == 2
    assert all("centralAnchors" not in task.task_params for task in synthetic_tasks)
    assert all("userPromptOverride" not in task.task_params for task in synthetic_tasks)
    assert all(solution.metadata["used_central_anchors"] is False for solution in generated)


def test_generate_texts_parallel_preserves_input_order(test_config):
    test_config.set("experiment.n", 2)
    test_config.set("parallelism.enabled", True)
    test_config.set("parallelism.initial_text_generation_max_concurrent", 2)
    executor = AsyncTextRecordingExecutor(
        {"prompt one": "first generated text.", "prompt two": "second generated text."},
        {"prompt one": 0.02, "prompt two": 0.0},
    )
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
    ]

    generated = builder._generate_texts(items, "reference text")

    assert [solution.generated_text for solution in generated] == ["first generated text.", "second generated text."]
    assert executor.max_active == 2


def test_generate_texts_parallel_excludes_failed_generation(test_config, tmp_path):
    test_config.set("experiment.n", 1)
    test_config.set("parallelism.enabled", True)
    test_config.set("parallelism.initial_text_generation_max_concurrent", 2)
    executor = AsyncTextRecordingExecutor(
        {"prompt one": RuntimeError("llm failed"), "prompt two": "second generated text."}
    )
    executor.outdir = tmp_path
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
    ]

    generated = builder._generate_texts(items, "reference text")

    assert [solution.generated_text for solution in generated] == ["second generated text."]
    assert "text_generation_failed" in (tmp_path / "initialization_rejections.jsonl").read_text(encoding="utf-8")
