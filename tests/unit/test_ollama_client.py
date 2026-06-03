from __future__ import annotations

import json

from binary_mopso_cd.services.ollama_client import LLMCallLogger, OllamaChatClient, ollama_usage_metadata


def test_ollama_usage_metadata_converts_nanoseconds_and_token_counts():
    metadata = ollama_usage_metadata(
        {
            "total_duration": 2_500_000_000,
            "load_duration": 100_000_000,
            "prompt_eval_duration": 300_000_000,
            "eval_duration": 700_000_000,
            "prompt_eval_count": 12,
            "eval_count": 34,
        }
    )

    assert metadata["ollamaTotalDurationSeconds"] == 2.5
    assert metadata["ollamaLoadDurationSeconds"] == 0.1
    assert metadata["ollamaPromptEvalDurationSeconds"] == 0.3
    assert metadata["ollamaEvalDurationSeconds"] == 0.7
    assert metadata["promptEvalCount"] == 12
    assert metadata["evalCount"] == 34
    assert metadata["total_duration"] == 2_500_000_000


def test_ollama_chat_client_logs_usage_metadata_without_changing_content(tmp_path):
    class FakeClient:
        def chat(self, **_kwargs):
            return {
                "message": {"content": "generated text"},
                "total_duration": 1_000_000_000,
                "prompt_eval_count": 5,
                "eval_count": 7,
            }

    client = object.__new__(OllamaChatClient)
    client.host = "http://127.0.0.1:11434"
    client.timeout_seconds = 120
    client.think = False
    client.client = FakeClient()
    client.logger = LLMCallLogger(tmp_path / "llm_calls.jsonl")

    text = client.chat(
        task_id="task-1",
        semantic_task="synthetic_text_generation",
        model="llama3",
        system_prompt="system",
        user_prompt="user",
        options={"temperature": 0.75},
        response_format=None,
    )

    assert text == "generated text"
    call = json.loads((tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert call["ollamaTotalDurationSeconds"] == 1.0
    assert call["promptEvalCount"] == 5
    assert call["evalCount"] == 7
    assert call["message_count"] == 2
