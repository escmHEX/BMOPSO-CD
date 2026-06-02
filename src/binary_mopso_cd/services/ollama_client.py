from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


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
        content = response["message"]["content"] if isinstance(response, dict) else response.message.content
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
            }
        )
        text = str(content).strip()
        if not text:
            raise RuntimeError(
                f"Ollama returned empty content for semantic task {semantic_task!r} with model {model!r}. "
                "Check model configuration and Ollama thinking settings."
            )
        return text
