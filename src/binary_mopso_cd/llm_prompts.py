from __future__ import annotations

import json
import re
from typing import Any

from binary_mopso_cd.router import (
    TASK_ANCHORS,
    TASK_INFLUENCE,
    TASK_POOL_EXPANSION,
    TASK_POOL_GENERATION,
    TASK_SYNTHETIC_TEXT,
)


SYSTEM_JSON = (
    "You are a semantic task executor for emergency-message prompt optimization. "
    "Return only valid JSON. Do not include markdown, explanations, comments, or extra text."
)

SYSTEM_TEXT_GENERATION = (
    "You are a plain-text generator for natural-disaster scenario messages. "
    "Return plain text only: one final message, 1-4 sentences. "
    "Do not use quotes, hashtags, URLs, usernames, placeholders, tags, lists, or explanations."
)


def build_messages(semantic_task: str, params: dict[str, Any]) -> tuple[str, str]:
    if semantic_task == TASK_ANCHORS:
        return (
            SYSTEM_JSON,
            (
                "Extract concise central semantic anchors from this reference text. "
                "Return an object with key central_anchors containing 4 to 8 short strings.\n"
                f"Reference text: {params['reference_text']}"
            ),
        )
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION}:
        component = params["component"]
        quantity = int(params["quantity"])
        existing = params.get("existing", [])
        mode = "Expand" if semantic_task == TASK_POOL_EXPANSION else "Generate"
        return (
            SYSTEM_JSON,
            (
                f"{mode} a semantic pool for component {component!r}. "
                f"Return a JSON array with exactly {quantity} unique short strings. "
                "Respect these maximum word counts: role 6, topic 8, action 6. "
                f"Domain: {params.get('domain')}. "
                f"Reference text: {params.get('reference_text')}. "
                f"Anchors: {json.dumps(params.get('anchors', []), ensure_ascii=False)}. "
                f"Existing values to avoid: {json.dumps(existing, ensure_ascii=False)}."
            ),
        )
    if semantic_task == TASK_INFLUENCE:
        return (
            SYSTEM_JSON,
            (
                "Generate semantic component candidates that move the current component toward the target. "
                f"Return a JSON array with at most {int(params.get('max_candidates', 5))} short strings. "
                "Do not copy the current or target literally. "
                f"Component name: {params.get('component')}. "
                f"Current: {params.get('current')}. "
                f"Target: {params.get('target')}. "
                f"Reference text: {params.get('reference_text')}."
            ),
        )
    if semantic_task == TASK_SYNTHETIC_TEXT:
        return SYSTEM_TEXT_GENERATION, str(params["prompt"])
    raise ValueError(f"No prompt template for semantic task: {semantic_task}")


def parse_json_payload(raw: str) -> Any:
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1).strip())
    first_candidates = [pos for pos in [text.find("{"), text.find("[")] if pos >= 0]
    last_candidates = [pos for pos in [text.rfind("}"), text.rfind("]")] if pos >= 0]
    if first_candidates and last_candidates:
        start = min(first_candidates)
        end = max(last_candidates)
        return json.loads(text[start : end + 1])
    raise ValueError(f"LLM did not return valid JSON: {raw[:200]}")


def parse_task_result(semantic_task: str, raw: str) -> Any:
    if semantic_task == TASK_SYNTHETIC_TEXT:
        return raw.strip()
    payload = parse_json_payload(raw)
    if semantic_task == TASK_ANCHORS:
        anchors = payload.get("central_anchors") if isinstance(payload, dict) else payload
        if not isinstance(anchors, list):
            raise ValueError("Anchor extraction must return central_anchors list")
        return [str(item).strip() for item in anchors if str(item).strip()]
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION, TASK_INFLUENCE}:
        if isinstance(payload, dict):
            lists = [value for value in payload.values() if isinstance(value, list)]
            payload = lists[0] if lists else payload
        if not isinstance(payload, list):
            raise ValueError(f"{semantic_task} must return a JSON array")
        return [str(item).strip() for item in payload if str(item).strip()]
    return payload
