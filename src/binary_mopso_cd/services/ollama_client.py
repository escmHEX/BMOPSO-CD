from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

from binary_mopso_cd.async_utils import run_async


NANOSECONDS_PER_SECOND = 1_000_000_000


def response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def response_message_content(response: Any) -> str:
    message = response_value(response, "message", {})
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def seconds_from_nanoseconds(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number / NANOSECONDS_PER_SECOND


def ollama_usage_metadata(response: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for source_key, output_key in (
        ("total_duration", "ollamaTotalDurationSeconds"),
        ("load_duration", "ollamaLoadDurationSeconds"),
        ("prompt_eval_duration", "ollamaPromptEvalDurationSeconds"),
        ("eval_duration", "ollamaEvalDurationSeconds"),
    ):
        seconds = seconds_from_nanoseconds(response_value(response, source_key))
        if seconds is not None:
            metadata[output_key] = seconds
            metadata[source_key] = response_value(response, source_key)
    for source_key, output_key in (
        ("prompt_eval_count", "promptEvalCount"),
        ("eval_count", "evalCount"),
    ):
        value = response_value(response, source_key)
        if value is not None:
            metadata[output_key] = value
            metadata[source_key] = value
    return metadata


class LLMCallLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class OllamaChatClient:
    def __init__(
        self,
        host: str,
        timeout_seconds: int = 120,
        think: bool | str | None = False,
        logger: LLMCallLogger | None = None,
    ):
        from ollama import AsyncClient

        self.host = host
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.async_client = None
        self.async_client_factory = AsyncClient
        self.logger = logger or LLMCallLogger(None)

    def chat(
        self,
        *,
        task_id: str,
        semantic_task: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: dict[str, Any],
        response_format: Any,
    ) -> str:
        return run_async(
            self.chat_async(
                task_id=task_id,
                semantic_task=semantic_task,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                options=options,
                response_format=response_format,
            )
        )

    async def chat_async(
        self,
        *,
        task_id: str,
        semantic_task: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: dict[str, Any],
        response_format: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        client = getattr(self, "async_client", None)
        if client is not None:
            response = await client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=False,
                think=self.think,
                format=response_format,
            )
        else:
            async with self.async_client_factory(host=self.host, timeout=self.timeout_seconds) as scoped_client:
                response = await scoped_client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=False,
                    think=self.think,
                    format=response_format,
                )
        elapsed = time.perf_counter() - started
        content = response_message_content(response)
        self.logger.log(
            {
                "task_id": task_id,
                "semantic_task": semantic_task,
                "model": model,
                "options": options,
                "stream": False,
                "think": self.think,
                "format": "plain" if response_format is None else "structured",
                "message_count": len(messages),
                "elapsed_seconds": elapsed,
                "content_chars": len(str(content)),
                **ollama_usage_metadata(response),
            }
        )
        text = str(content).strip()
        if not text:
            raise RuntimeError(
                f"Ollama returned empty content for semantic task {semantic_task!r} with model {model!r}. "
                "Check model configuration and Ollama thinking settings."
            )
        return text
