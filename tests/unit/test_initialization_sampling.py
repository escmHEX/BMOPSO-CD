from __future__ import annotations

import json
from random import Random

import numpy as np

from binary_mopso_cd.initialization import InitialPopulationBuilder, greedy_max_min_indices
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


class DistinctEmbeddingService:
    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=float,
        )
        return vectors[: len(texts)]


class PromptRenderingExecutor:
    def __init__(self):
        self.tasks = []
        self.embedding_service = DistinctEmbeddingService()
        self.outdir = None

    def execute(self, task):
        self.tasks.append(task)
        components = task.task_params["components"]
        return " ".join(str(value) for value in components.values())


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
    def __init__(self, texts_by_prompt: dict[str, str], failures: set[str] | None = None):
        self.tasks = []
        self.texts_by_prompt = dict(texts_by_prompt)
        self.failures = set(failures or set())
        self.embedding_service = ConstantEmbeddingService()
        self.outdir = None
        self.active = 0
        self.max_active = 0

    async def execute_async(self, task):
        import asyncio

        self.tasks.append(task)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01 if task.task_params["prompt"].endswith("one") else 0.0)
            prompt = task.task_params["prompt"]
            if prompt in self.failures:
                raise RuntimeError(f"failed prompt: {prompt}")
            return self.texts_by_prompt[prompt]
        finally:
            self.active -= 1

    def execute(self, task):
        raise AssertionError("sync execute should not be used")


class RecordingProgress:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)


def test_prompt_reduction_logs_internal_diagnostic_steps(test_config):
    progress = RecordingProgress()
    builder = InitialPopulationBuilder(
        test_config,
        PassthroughRouter(),
        PromptRenderingExecutor(),
        Random(1),
        progress=progress,
    )
    candidates = [
        SemanticVector({"role": "resident", "topic": "quarantine", "action": "avoid phone"}),
        SemanticVector({"role": "worker", "topic": "anxiety", "action": "mute alerts"}),
        SemanticVector({"role": "neighbor", "topic": "routine loss", "action": "ask for help"}),
    ]

    reduced = builder._reduce_by_prompt_diversity(candidates, "crisis messages", 2)

    assert len(reduced) == 2
    assert any("rendering prompt candidates" in message for message in progress.messages)
    assert any("candidate prompts rendered" in message for message in progress.messages)
    assert any("embedding prompt candidates" in message for message in progress.messages)
    assert any("prompt embeddings ready" in message for message in progress.messages)
    assert any("selecting diverse prompt candidates" in message for message in progress.messages)
    assert any("diverse prompt candidates selected" in message for message in progress.messages)
    assert any("scoring prompt diversity" in message for message in progress.messages)
    assert any("prompt diversity scored" in message for message in progress.messages)


def test_greedy_prompt_reduction_does_not_require_numpy_matmul():
    assert greedy_max_min_indices(NoMatmulEmbeddings(), 2) == [0, 1]


class NoMatmulRow:
    def __init__(self, values: list[float]):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __matmul__(self, _other):
        raise AssertionError("prompt reduction should avoid ndarray matmul for small dot products")


class NoMatmulEmbeddings:
    shape = (3, 3)
    dtype = "diagnostic"

    def __getitem__(self, index):
        return [
            NoMatmulRow([1.0, 0.0, 0.0]),
            NoMatmulRow([0.0, 1.0, 0.0]),
            NoMatmulRow([0.0, 0.0, 1.0]),
        ][index]


class NoMatmulEmbeddingService:
    def encode(self, texts: list[str], text_type: str) -> NoMatmulEmbeddings:
        return NoMatmulEmbeddings()


def test_prompt_reduction_scoring_does_not_require_numpy_matmul(test_config):
    progress = RecordingProgress()
    executor = PromptRenderingExecutor()
    executor.embedding_service = NoMatmulEmbeddingService()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1), progress=progress)
    candidates = [
        SemanticVector({"role": "resident", "topic": "quarantine", "action": "avoid phone"}),
        SemanticVector({"role": "worker", "topic": "anxiety", "action": "mute alerts"}),
        SemanticVector({"role": "neighbor", "topic": "routine loss", "action": "ask for help"}),
    ]

    reduced = builder._reduce_by_prompt_diversity(candidates, "crisis messages", 2)

    assert len(reduced) == 2


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


def test_build_pool_writes_component_validation_diagnostics(test_config, tmp_path):
    class MixedPoolExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.outdir = tmp_path

        def execute(self, task):
            self.tasks.append(task)
            return [
                "valid role",
                "",
                "line\nbreak",
                "one two three four five six seven",
                "valid role",
                "existing role",
            ]

    executor = MixedPoolExecutor()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))

    pool = builder._build_pool(
        "role",
        6,
        "Flooded roads near the bridge need urgent support.",
        {"entities": ["bridge"]},
        central_anchor_count=1,
        domain="social media messages related to crises and emergencies",
        existing=["existing role"],
        task_name=TASK_POOL_EXPANSION,
    )

    assert pool == ["valid role"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "initialization_pool_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "role"
    assert row["task"] == TASK_POOL_EXPANSION
    assert row["requested_items"] == 6
    assert row["existing_items"] == ["existing role"]
    assert row["raw_count"] == 6
    assert row["valid_items"] == ["valid role"]
    assert row["pool_size_after"] == 2
    reasons = [item["reason"] for item in row["rejected_items"]]
    assert "empty" in reasons
    assert "contains_newline" in reasons
    assert "too_many_words" in reasons
    assert reasons.count("duplicate") == 2


def test_insufficient_pool_error_reports_component_sizes(test_config, tmp_path):
    class EmptyTopicExecutor(RecordingExecutor):
        def __init__(self):
            super().__init__()
            self.outdir = tmp_path

        def execute(self, task):
            self.tasks.append(task)
            if task.semantic_task == TASK_ANCHORS:
                return {
                    "entities": ["park"],
                    "topics": ["cleanup drive"],
                    "actions": ["organize"],
                    "constraints": ["9 AM"],
                }
            if task.task_params["component"] == "topic":
                return []
            return [f"{task.task_params['component']} item"]

    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), EmptyTopicExecutor(), Random(1))

    try:
        builder.build_with_context("We're organizing a cleanup drive in the park at 9 AM")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected insufficient pool error")

    assert "Initial semantic pools are insufficient" in message
    assert "pool_sizes={'role': 1, 'topic': 0, 'action': 1}" in message
    assert "initialization_pool_diagnostics.jsonl" in message
    rows = [
        json.loads(line)
        for line in (tmp_path / "initialization_pool_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["component"] == "topic" and row["valid_count"] == 0 for row in rows)


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
    assert all(task.operation_context == "initialization" for task in synthetic_tasks)
    assert all("centralAnchors" not in task.task_params for task in synthetic_tasks)
    assert all("userPromptOverride" not in task.task_params for task in synthetic_tasks)
    assert all(solution.metadata["used_central_anchors"] is False for solution in generated)


def test_generate_texts_parallel_preserves_order_and_concurrency_limit(test_config):
    test_config.set("experiment.n", 3)
    test_config.set("parallelism.initial_text_generation_max_concurrent", 2)
    executor = AsyncTextRecordingExecutor(
        {
            "prompt one": "first generated text.",
            "prompt two": "second generated text.",
            "prompt three": "third generated text.",
        }
    )
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
        (SemanticVector({"role": "role three", "topic": "topic three", "action": "action three"}), "prompt three", 0.1),
    ]

    generated = builder._generate_texts(items, "reference text")

    assert [solution.generated_text for solution in generated] == [
        "first generated text.",
        "second generated text.",
        "third generated text.",
    ]
    assert executor.max_active == 2


def test_generate_texts_parallel_logs_live_progress(test_config):
    test_config.set("experiment.n", 3)
    test_config.set("parallelism.initial_text_generation_max_concurrent", 2)
    executor = AsyncTextRecordingExecutor(
        {
            "prompt one": "first generated text.",
            "prompt two": "second generated text.",
            "prompt three": "third generated text.",
        }
    )
    progress = RecordingProgress()
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1), progress=progress)
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
        (SemanticVector({"role": "role three", "topic": "topic three", "action": "action three"}), "prompt three", 0.1),
    ]

    generated = builder._generate_texts(items, "reference text")

    assert len(generated) == 3
    assert any(
        "initial population | generating texts | candidates=3 | max_concurrent=2" in message
        for message in progress.messages
    )
    assert any(
        "initial population | text generation progress | completed=1/3" in message
        for message in progress.messages
    )
    assert any(
        "initial population | text generation progress | completed=3/3 (100%)" in message
        for message in progress.messages
    )
    assert any(
        "initial population | generated texts validated | accepted=3 | rejected=0" in message
        for message in progress.messages
    )


def test_generate_texts_parallel_records_generation_failures(test_config, tmp_path):
    test_config.set("experiment.n", 2)
    test_config.set("parallelism.initial_text_generation_max_concurrent", 2)
    executor = AsyncTextRecordingExecutor(
        {
            "prompt one": "first generated text.",
            "prompt three": "third generated text.",
        },
        failures={"prompt two"},
    )
    executor.outdir = tmp_path
    builder = InitialPopulationBuilder(test_config, PassthroughRouter(), executor, Random(1))
    items = [
        (SemanticVector({"role": "role one", "topic": "topic one", "action": "action one"}), "prompt one", 0.3),
        (SemanticVector({"role": "role two", "topic": "topic two", "action": "action two"}), "prompt two", 0.2),
        (SemanticVector({"role": "role three", "topic": "topic three", "action": "action three"}), "prompt three", 0.1),
    ]

    generated = builder._generate_texts(items, "reference text")

    assert len(generated) == 2
    rejection_log = (tmp_path / "initialization_rejections.jsonl").read_text(encoding="utf-8")
    assert "text_generation_failed" in rejection_log
    assert "failed prompt: prompt two" in rejection_log
