from __future__ import annotations

import asyncio
import json

from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.router import TASK_SYNTHETIC_TEXT, RouteTask, SemanticRouter


def test_executor_ollama_contract_is_single_shot_stream_false(test_config, tmp_path):
    router = SemanticRouter(test_config)
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path)
    task = RouteTask(
        "x",
        "test",
        TASK_SYNTHETIC_TEXT,
        {
            "prompt": (
                "Generate a short social media message related to crises and emergencies using the following semantic components: "
                "role = local official; topic = evacuation order; action = warn residents. "
                "The generated message must follow the role, address the topic, and satisfy the action."
            ),
            "reference_text": "Evacuation order",
        },
    )
    result = executor.execute(router.route(task))
    assert result
    call = json.loads((tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert call["message_count"] == 2
    assert call["message_roles"] == ["system", "user"]
    assert call["stream"] is False
    assert call["think"] is False
    assert call["options"]["temperature"] == 0.75
    assert call["options"]["top_p"] == 0.95


def test_executor_sync_llm_passes_routed_contract_to_client(test_config, tmp_path):
    class StubLLMClient:
        def __init__(self):
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return "generated text"

    router = SemanticRouter(test_config)
    test_config.set("router.phase_task_models.optimization.synthetic_text_generation", "gemma4:e4b")
    test_config.set("router.llm_params.synthetic_text_generation.thinking", "medium")
    llm_client = StubLLMClient()
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path, llm_client=llm_client)
    task = RouteTask(
        "x",
        "optimization",
        TASK_SYNTHETIC_TEXT,
        {
            "prompt": "Generate a short social media message related to crises and emergencies.",
            "reference_text": "Evacuation order",
        },
    )

    result = executor.execute(router.route(task))

    assert result == "generated text"
    assert llm_client.kwargs["semantic_task"] == TASK_SYNTHETIC_TEXT
    assert llm_client.kwargs["model"] == "gemma4:e4b"
    assert llm_client.kwargs["system_prompt"]
    assert llm_client.kwargs["user_prompt"]
    assert llm_client.kwargs["think"] == "medium"
    assert llm_client.kwargs["options"] == {"temperature": 0.75, "top_p": 0.95}
    assert llm_client.kwargs["response_format"] is None


def test_executor_async_llm_matches_sync_task_contract(test_config, tmp_path):
    class StubLLMClient:
        def __init__(self):
            self.kwargs = None

        async def chat_async(self, **kwargs):
            self.kwargs = kwargs
            return "async generated text"

    router = SemanticRouter(test_config)
    test_config.set("router.phase_task_models.optimization.synthetic_text_generation", "qwen3.5:2b")
    test_config.set("router.llm_params.synthetic_text_generation.thinking", True)
    llm_client = StubLLMClient()
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path, llm_client=llm_client)
    task = RouteTask(
        "x",
        "optimization",
        TASK_SYNTHETIC_TEXT,
        {
            "prompt": "Generate a short social media message related to crises and emergencies.",
            "reference_text": "Evacuation order",
        },
    )

    result = asyncio.run(executor.execute_async(router.route(task)))

    assert result == "async generated text"
    assert llm_client.kwargs["semantic_task"] == TASK_SYNTHETIC_TEXT
    assert llm_client.kwargs["model"] == "qwen3.5:2b"
    assert llm_client.kwargs["think"] is True
    assert llm_client.kwargs["options"] == {"temperature": 0.75, "top_p": 0.95}
    assert llm_client.kwargs["response_format"] is None
