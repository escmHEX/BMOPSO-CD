from __future__ import annotations

import json
import time
from hashlib import sha1
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
    def __init__(self, host: str, timeout_seconds: int = 120, logger: LLMCallLogger | None = None):
        from ollama import Client

        self.host = host
        self.timeout_seconds = timeout_seconds
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
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        response_format = None if semantic_task == "synthetic_text_generation" else "json"
        response = self.client.chat(
            model=model,
            messages=messages,
            options=options,
            stream=False,
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
                "message_count": len(messages),
                "elapsed_seconds": elapsed,
            }
        )
        return str(content).strip()


class FakeLLMClient:
    def __init__(self, logger: LLMCallLogger | None = None):
        self.logger = logger or LLMCallLogger(None)
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        *,
        task_id: str,
        semantic_task: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        options: dict[str, Any],
    ) -> str:
        started = time.perf_counter()
        self.calls.append(
            {
                "task_id": task_id,
                "semantic_task": semantic_task,
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": dict(options),
            }
        )
        if semantic_task == "semantic_anchor_extraction":
            content = '{"central_anchors":["evacuation","shelter","storm","road safety"]}'
        elif semantic_task in {"semantic_pool_generation", "semantic_pool_expansion"}:
            lowered = user_prompt.lower()
            if "component 'role'" in lowered or 'component "role"' in lowered:
                content = '["emergency coordinator","local official","shelter manager","public safety officer","field responder","community liaison","city spokesperson","rescue dispatcher"]'
            elif "component 'action'" in lowered or 'component "action"' in lowered:
                content = '["warn residents","give evacuation guidance","request calm","share safety steps","announce shelter status","discourage travel","provide updates","coordinate response"]'
            else:
                content = '["flooded roads","wildfire evacuation","storm damage","shelter capacity","boil water notice","high wind alert","power outage","debris cleanup"]'
        elif semantic_task == "semantic_component_influence_candidates":
            content = '["urgent public safety update","clear evacuation notice","resident safety guidance","storm response update","local emergency warning"]'
        elif semantic_task == "synthetic_text_generation":
            suffix = sha1(user_prompt.encode("utf-8")).hexdigest()[:8]
            content = (
                "Residents should follow official emergency guidance, avoid unnecessary travel, "
                f"and monitor update {suffix} until conditions improve."
            )
        else:
            content = "[]"
        elapsed = time.perf_counter() - started
        self.logger.log(
            {
                "task_id": task_id,
                "semantic_task": semantic_task,
                "model": model,
                "stream": False,
                "message_count": 2,
                "elapsed_seconds": elapsed,
                "fake": True,
            }
        )
        return content
