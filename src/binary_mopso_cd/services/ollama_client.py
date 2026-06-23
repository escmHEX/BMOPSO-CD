from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any


NANOSECONDS_PER_SECOND = 1_000_000_000
_UNSET = object()


def response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def response_message_content(response: Any) -> str:
    message = response_value(response, "message", {})
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def response_message_thinking(response: Any) -> str:
    message = response_value(response, "message", {})
    if isinstance(message, dict):
        return str(message.get("thinking") or "")
    return str(getattr(message, "thinking", "") or "")


def strip_content_thinking_tags(content: str, start_tag: str = "<think>", end_tag: str = "</think>") -> str:
    if not start_tag or start_tag not in content:
        return content
    pieces: list[str] = []
    position = 0
    while True:
        start = content.find(start_tag, position)
        if start < 0:
            pieces.append(content[position:])
            break
        pieces.append(content[position:start])
        end = content.find(end_tag, start + len(start_tag)) if end_tag else -1
        if end < 0:
            break
        position = end + len(end_tag)
    return "".join(pieces).strip()


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
    input_tokens = response_value(response, "prompt_eval_count")
    output_tokens = response_value(response, "eval_count")
    if input_tokens is not None:
        metadata["input_tokens"] = int(input_tokens)
    if output_tokens is not None:
        metadata["output_tokens"] = int(output_tokens)
    if input_tokens is not None and output_tokens is not None:
        metadata["total_tokens"] = int(input_tokens) + int(output_tokens)
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
        model_profiles: dict[str, dict[str, Any]] | None = None,
        logger: LLMCallLogger | None = None,
    ):
        from ollama import Client

        self.host = host
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.model_profiles = model_profiles or {}
        self.client = Client(host=host, timeout=timeout_seconds)
        self.logger = logger or LLMCallLogger(None)

    def _model_profile(self, model: str) -> dict[str, Any]:
        profile = dict(getattr(self, "model_profiles", {}).get("default", {}))
        model_profile = getattr(self, "model_profiles", {}).get(model, {})
        for key, value in model_profile.items():
            if isinstance(value, dict) and isinstance(profile.get(key), dict):
                profile[key] = {**profile[key], **value}
            else:
                profile[key] = value
        return profile

    def _normalize_content(self, model: str, content: str) -> str:
        response_thinking = self._model_profile(model).get("response_thinking", {})
        if not isinstance(response_thinking, dict):
            return content
        if not bool(response_thinking.get("strip_content_tags", False)):
            return content
        return strip_content_thinking_tags(
            content,
            str(response_thinking.get("start_tag", "<think>")),
            str(response_thinking.get("end_tag", "</think>")),
        )

    def _log_call(
        self,
        *,
        task_id: str,
        semantic_task: str,
        model: str,
        options: dict[str, Any],
        think: bool | str | None,
        response_format: Any,
        messages: list[dict[str, str]],
        elapsed: float,
        content: str,
        raw_content: str,
        response: Any,
    ) -> None:
        thinking = response_message_thinking(response)
        self.logger.log(
            {
                "task_id": task_id,
                "semantic_task": semantic_task,
                "model": model,
                "options": options,
                "stream": False,
                "think": think,
                "format": "plain" if response_format is None else "structured",
                "message_count": len(messages),
                "message_roles": [message["role"] for message in messages],
                "system_prompt_chars": len(messages[0]["content"]) if messages else 0,
                "user_prompt_chars": len(messages[1]["content"]) if len(messages) > 1 else 0,
                "elapsed_seconds": elapsed,
                "content_chars": len(str(content)),
                "raw_content_chars": len(str(raw_content)),
                "thinking_chars": len(thinking),
                "content_thinking_stripped": raw_content != content,
                **ollama_usage_metadata(response),
            }
        )

    def _resolve_think(self, think: Any) -> bool | str | None:
        if think is _UNSET:
            return self.think
        return think

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
        think: Any = _UNSET,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resolved_think = self._resolve_think(think)
        started = time.perf_counter()
        response = self.client.chat(
            model=model,
            messages=messages,
            options=options,
            stream=False,
            think=resolved_think,
            format=response_format,
        )
        elapsed = time.perf_counter() - started
        raw_content = response_message_content(response)
        content = self._normalize_content(model, raw_content)
        self._log_call(
            task_id=task_id,
            semantic_task=semantic_task,
            model=model,
            options=options,
            think=resolved_think,
            response_format=response_format,
            messages=messages,
            elapsed=elapsed,
            content=content,
            raw_content=raw_content,
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
        think: Any = _UNSET,
    ) -> str:
        from ollama import AsyncClient

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resolved_think = self._resolve_think(think)
        async_client = AsyncClient(host=self.host, timeout=self.timeout_seconds)
        try:
            started = time.perf_counter()
            response = await async_client.chat(
                model=model,
                messages=messages,
                options=options,
                stream=False,
                think=resolved_think,
                format=response_format,
            )
            elapsed = time.perf_counter() - started
        finally:
            close_result = async_client.close()
            if inspect.isawaitable(close_result):
                await close_result
        raw_content = response_message_content(response)
        content = self._normalize_content(model, raw_content)
        self._log_call(
            task_id=task_id,
            semantic_task=semantic_task,
            model=model,
            options=options,
            think=resolved_think,
            response_format=response_format,
            messages=messages,
            elapsed=elapsed,
            content=content,
            raw_content=raw_content,
            response=response,
        )
        text = str(content).strip()
        if not text:
            raise RuntimeError(
                f"Ollama returned empty content for semantic task {semantic_task!r} with model {model!r}. "
                "Check model configuration and Ollama thinking settings."
            )
        return text
