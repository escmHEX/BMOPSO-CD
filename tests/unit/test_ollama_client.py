from __future__ import annotations

import asyncio
import json

from binary_mopso_cd.services.ollama_client import (
    LLMCallLogger,
    OllamaChatClient,
    ollama_usage_metadata,
    strip_content_thinking_tags,
)


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
    assert metadata["input_tokens"] == 12
    assert metadata["output_tokens"] == 34
    assert metadata["total_tokens"] == 46
    assert metadata["total_duration"] == 2_500_000_000


def test_strip_content_thinking_tags_removes_complete_or_open_sections():
    assert strip_content_thinking_tags("<think>hidden</think> final") == "final"
    assert strip_content_thinking_tags("prefix <think>hidden</think> suffix") == "prefix  suffix"
    assert strip_content_thinking_tags("<think>hidden without final") == ""


def test_ollama_chat_client_logs_usage_metadata_without_changing_content(tmp_path):
    captured = {}
    response_format = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    class StubClient:
        def chat(self, **kwargs):
            captured["chat"] = kwargs
            return {
                "message": {"content": "<think>hidden reasoning</think> generated text", "thinking": "server thinking"},
                "total_duration": 1_000_000_000,
                "prompt_eval_count": 5,
                "eval_count": 7,
            }

    client = object.__new__(OllamaChatClient)
    client.host = "http://127.0.0.1:11434"
    client.timeout_seconds = 120
    client.think = False
    client.model_profiles = {
        "default": {"response_thinking": {"strip_content_tags": False}},
        "lfm2.5:8b": {
            "response_thinking": {
                "strip_content_tags": True,
                "start_tag": "<think>",
                "end_tag": "</think>",
            }
        },
    }
    client.client = StubClient()
    client.logger = LLMCallLogger(tmp_path / "llm_calls.jsonl")

    text = client.chat(
        task_id="task-1",
        semantic_task="synthetic_text_generation",
        model="lfm2.5:8b",
        system_prompt="system",
        user_prompt="user",
        options={"temperature": 0.75, "top_p": 0.95},
        response_format=response_format,
        think=False,
    )

    assert text == "generated text"
    assert captured["chat"] == {
        "model": "lfm2.5:8b",
        "messages": [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        "options": {"temperature": 0.75, "top_p": 0.95},
        "stream": False,
        "think": False,
        "format": response_format,
    }
    call = json.loads((tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert call["think"] is False
    assert call["format"] == "structured"
    assert call["ollamaTotalDurationSeconds"] == 1.0
    assert call["promptEvalCount"] == 5
    assert call["evalCount"] == 7
    assert call["input_tokens"] == 5
    assert call["output_tokens"] == 7
    assert call["total_tokens"] == 12
    assert call["message_count"] == 2
    assert call["message_roles"] == ["system", "user"]
    assert call["system_prompt_chars"] == len("system")
    assert call["user_prompt_chars"] == len("user")
    assert call["thinking_chars"] == len("server thinking")
    assert call["raw_content_chars"] > call["content_chars"]
    assert call["content_thinking_stripped"] is True


def test_ollama_chat_client_uses_global_think_when_call_omits_think(tmp_path):
    captured = {}

    class StubClient:
        def chat(self, **kwargs):
            captured["chat"] = kwargs
            return {"message": {"content": "generated text"}}

    client = object.__new__(OllamaChatClient)
    client.host = "http://127.0.0.1:11434"
    client.timeout_seconds = 120
    client.think = "medium"
    client.model_profiles = {}
    client.client = StubClient()
    client.logger = LLMCallLogger(tmp_path / "llm_calls.jsonl")

    text = client.chat(
        task_id="task-1",
        semantic_task="synthetic_text_generation",
        model="qwen3.5:4b",
        system_prompt="system",
        user_prompt="user",
        options={"temperature": 0.75},
        response_format=None,
    )

    assert text == "generated text"
    assert captured["chat"]["think"] == "medium"


def test_ollama_chat_client_async_uses_async_client_and_logs(monkeypatch, tmp_path):
    captured = {}

    class StubAsyncClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.closed = False

        async def chat(self, **kwargs):
            captured["chat"] = kwargs
            return {
                "message": {"content": "async generated text"},
                "total_duration": 2_000_000_000,
                "prompt_eval_count": 3,
                "eval_count": 4,
            }

        def close(self):
            captured["closed"] = True

    import ollama

    monkeypatch.setattr(ollama, "AsyncClient", StubAsyncClient)
    client = object.__new__(OllamaChatClient)
    client.host = "http://127.0.0.1:11434"
    client.timeout_seconds = 120
    client.think = False
    client.model_profiles = {}
    client.logger = LLMCallLogger(tmp_path / "llm_calls.jsonl")

    text = asyncio.run(
        client.chat_async(
            task_id="task-async",
            semantic_task="synthetic_text_generation",
            model="llama3",
            system_prompt="system",
            user_prompt="user",
            options={"temperature": 0.75, "top_p": 0.95},
            response_format={"type": "object"},
            think=True,
        )
    )

    assert text == "async generated text"
    assert captured["init"] == {"host": "http://127.0.0.1:11434", "timeout": 120}
    assert captured["chat"]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    assert captured["chat"]["options"] == {"temperature": 0.75, "top_p": 0.95}
    assert captured["chat"]["stream"] is False
    assert captured["chat"]["think"] is True
    assert captured["chat"]["format"] == {"type": "object"}
    assert captured["closed"] is True
    call = json.loads((tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert call["think"] is True
    assert call["ollamaTotalDurationSeconds"] == 2.0
    assert call["message_count"] == 2
    assert call["message_roles"] == ["system", "user"]
