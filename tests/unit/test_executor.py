from __future__ import annotations

from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.router import TASK_SYNTHETIC_TEXT, RouteTask, SemanticRouter


def test_executor_ollama_contract_is_single_shot_stream_false(test_config, tmp_path):
    router = SemanticRouter(test_config)
    executor = SemanticTaskExecutor(test_config, outdir=tmp_path)
    task = RouteTask("x", "test", TASK_SYNTHETIC_TEXT, {"prompt": "Write alert", "reference_text": "Alert"})
    result = executor.execute(router.route(task))
    assert result
    call = executor.llm_client.calls[-1]
    assert len(call["messages"]) == 2
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1]["role"] == "user"
    assert call["options"]["temperature"] == 0.75

