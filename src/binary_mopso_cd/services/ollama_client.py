from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any


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
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
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
        from ollama import Client

        self.host = host
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.client = Client(host=host, timeout=timeout_seconds)
        self.logger = logger or LLMCallLogger(None)

    def _log_call(
        self,
        *,
        task_id: str,
        semantic_task: str,
        model: str,
        options: dict[str, Any],
        response_format: Any,
        message_count: int,
        elapsed: float,
        content: str,
        response: Any,
    ) -> None:
        self.logger.log(
            {
                "task_id": task_id,
                "semantic_task": semantic_task,
                "model": model,
                "options": options,
                "stream": False,
                "think": self.think,
                "format": "plain" if response_format is None else "structured",
                "message_count": message_count,
                "elapsed_seconds": elapsed,
                "content_chars": len(str(content)),
                **ollama_usage_metadata(response),
            }
        )

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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        response = self.client.chat(
            model=model,
            messages=messages,
            options=options,
            stream=False,
            think=self.think,
            format=response_format,
        )
        elapsed = time.perf_counter() - started
        content = response_message_content(response)
        self._log_call(
            task_id=task_id,
            semantic_task=semantic_task,
            model=model,
            options=options,
            response_format=response_format,
            message_count=len(messages),
            elapsed=elapsed,
            content=content,
            response=response,
        )
        text = str(content).strip()
        if not text:
            raise RuntimeError(
                f"Ollama returned empty content for semantic task {semantic_task!r} with model {model!r}. "
                "Check model configuration and Ollama thinking settings."
            )
        return text

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
        from ollama import AsyncClient

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        async_client = AsyncClient(host=self.host, timeout=self.timeout_seconds)
        try:
            started = time.perf_counter()
            response = await async_client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=False,
                think=self.think,
                format=response_format,
            )
            elapsed = time.perf_counter() - started
        finally:
            close_result = async_client.close()
            if inspect.isawaitable(close_result):
                await close_result
        content = response_message_content(response)
        self._log_call(
            task_id=task_id,
            semantic_task=semantic_task,
            model=model,
            options=options,
            response_format=response_format,
            message_count=len(messages),
            elapsed=elapsed,
            content=content,
            response=response,
        )
        text = str(content).strip()
        if not text:
            raise RuntimeError(
                f"Ollama returned empty content for semantic task {semantic_task!r} with model {model!r}. "
                "Check model configuration and Ollama thinking settings."
            )
        return text
