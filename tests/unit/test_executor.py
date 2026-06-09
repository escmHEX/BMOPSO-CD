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
    assert call["stream"] is False
    assert call["options"]["temperature"] == 0.75


def test_executor_async_llm_path_matches_sync_parser(test_config, tmp_path):
    class StubClient:
        def chat(self, **_kwargs):
            return "sync generated text"

        async def chat_async(self, **_kwargs):
            return "async generated text"

    router = SemanticRouter(test_config)
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path, llm_client=StubClient())
    task = RouteTask(
        "x",
        "test",
        TASK_SYNTHETIC_TEXT,
        {"prompt": "Generate a short message.", "reference_text": "reference"},
    )

    result = asyncio.run(executor.execute_async(router.route(task)))

    assert result == "async generated text"
