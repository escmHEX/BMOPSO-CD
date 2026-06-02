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
from binary_mopso_cd.utils import canonical_text


SYSTEM_JSON = (
    "You are a semantic task executor for emergency-message prompt optimization. "
    "Return only valid JSON. Do not include markdown, explanations, comments, or extra text."
)

SYSTEM_TEXT_GENERATION = """You are a plain-text generator for natural-disaster scenario messages.
You will receive one text-generation instruction from the user.
Follow the instruction and generate exactly one final text message.
Output rules:
- Return plain text only.
- Return the dataset content itself, not a prompt, explanation, title, label, list, code, or metadata.
- Do not describe the task.
- Do not add unsolicited safety advice.
- Do not use quotation marks, hashtags, URLs, usernames, placeholders, tags, or special markers.
- Limit the message to between 1 and 4 sentences.
- Return only the final message."""


JSON_ARRAY_OF_STRINGS_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}

ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": JSON_ARRAY_OF_STRINGS_SCHEMA,
        "topics": JSON_ARRAY_OF_STRINGS_SCHEMA,
        "actions": JSON_ARRAY_OF_STRINGS_SCHEMA,
        "constraints": JSON_ARRAY_OF_STRINGS_SCHEMA,
    },
    "required": ["entities", "topics", "actions", "constraints"],
    "additionalProperties": False,
}


def build_messages(semantic_task: str, params: dict[str, Any]) -> tuple[str, str]:
    if semantic_task == TASK_ANCHORS:
        return (
            SYSTEM_JSON,
            (
                "Extract concise central semantic anchors from this reference text. "
                "Return only valid JSON with exactly these keys: entities, topics, actions, constraints. "
                "Each value must be a list of non-empty short strings without exact duplicates after normalization.\n"
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
                f"Respect these maximum word counts: {json.dumps(params.get('max_words_by_component', {}), ensure_ascii=False)}. "
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
                f"Return only a JSON array with at most {int(params.get('max_candidates', 5))} short strings. "
                "Do not return an object, keys, labels, explanations, or nested structures. "
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


def response_format_for_task(semantic_task: str) -> Any:
    if semantic_task == TASK_SYNTHETIC_TEXT:
        return None
    if semantic_task == TASK_ANCHORS:
        return ANCHOR_SCHEMA
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION, TASK_INFLUENCE}:
        return JSON_ARRAY_OF_STRINGS_SCHEMA
    return "json"


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
        if not isinstance(payload, dict):
            raise ValueError("Anchor extraction must return a JSON object")
        expected = ["entities", "topics", "actions", "constraints"]
        missing = [key for key in expected if key not in payload]
        if missing:
            raise ValueError(f"Anchor extraction missing keys: {missing}")
        anchors: dict[str, list[str]] = {}
        for key in expected:
            value = payload[key]
            if not isinstance(value, list):
                raise ValueError(f"Anchor extraction key {key!r} must be a list")
            seen: set[str] = set()
            items: list[str] = []
            for item in value:
                text = str(item).strip()
                normalized = canonical_text(text)
                if text and normalized not in seen:
                    seen.add(normalized)
                    items.append(text)
            anchors[key] = items
        return anchors
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION, TASK_INFLUENCE}:
        if isinstance(payload, dict):
            lists = [value for value in payload.values() if isinstance(value, list)]
            payload = lists[0] if lists else payload
        if not isinstance(payload, list):
            raise ValueError(f"{semantic_task} must return a JSON array")
        return [str(item).strip() for item in payload if str(item).strip()]
    return payload
