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

SYSTEM_ANCHOR_EXTRACTION = """You are an information extraction module for a prompt optimization algorithm.
Task:
Extract short semantic anchors from the reference text.
Output format:
Return only valid JSON with exactly these keys:
{
"entities": ["..."],
"topics": ["..."],
"actions": ["..."],
"constraints": ["..."]
}
Rules:
- Use the reference text as the main source.
- Use the general domain only as context.
- Do not invent specific facts that are not supported by the reference text.
- Keep each item short.
- Use 1 to 5 items per list.
- Do not include explanations, markdown, numbering, or extra keys."""

SYSTEM_POOL_GENERATION = """You generate one pool of semantic components for a prompt optimization algorithm.
Task:
Generate only the requested component type: roles, topics, or actions.
Output format:
Return only valid JSON with exactly this structure:
{"items": ["...", "..."]}
Rules:
- The reference text is the main source, therefore every item must be associated with the reference text.
- Use the general domain only as context.
- Do not introduce new crisis concepts that are not supported by the reference text. Prefer reference-specific terms over generic emergency terms.
- Use central reference-specific anchors when needed to preserve meaning, but do not copy the full reference sentence.
- Keep each item short.
- Do not include explanations, markdown, numbering, or extra keys."""

SYSTEM_POOL_EXPANSION = """You expand one existing pool of semantic components.
Task:
Generate additional items for the requested component type.
Output format:
Return only valid JSON with exactly this structure:
{"items": ["...", "..."]}
Rules:
- The reference text is the main source, therefore every new item must be associated with the reference text.
- Use the general domain only as context.
- Do not introduce new crisis concepts that are not supported by the reference text. Prefer reference-specific terms over generic emergency terms.
- Use central reference-specific anchors when needed to preserve meaning, but do not copy the full reference sentence.
- Do not repeat existing items.
- Keep each item short.
- Do not include explanations, markdown, numbering, or extra keys."""

SYSTEM_TEXT_GENERATION = """You are a plain-text generator for social media messages related to crises and emergencies.
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

POOL_ITEMS_SCHEMA = {
    "type": "object",
    "properties": {
        "items": JSON_ARRAY_OF_STRINGS_SCHEMA,
    },
    "required": ["items"],
    "additionalProperties": False,
}

COMPONENT_TYPE_LABELS = {
    "role": "roles",
    "topic": "topics",
    "action": "actions",
}

COMPONENT_ADDITIONAL_INSTRUCTIONS = {
    "role": (
        "Return message-sender roles or stakeholder perspectives, not actions, topics, events, "
        "or full social media messages. Prefer short noun phrases grounded in the reference text."
    ),
    "topic": (
        "Return issue, event, or subject focuses, not message senders, actions, "
        "or full social media messages. Prefer short noun phrases grounded in the reference text."
    ),
    "action": (
        "Return communicative intents or discourse operations, not full social media messages. "
        "Prefer verb phrases such as 'inform users about...', 'announce...', 'warn about...', "
        "'direct users to...', 'request...', or 'report...'."
    ),
}


def component_type_label(component: str) -> str:
    normalized = str(component).strip().lower()
    return COMPONENT_TYPE_LABELS.get(normalized, normalized)


def component_additional_instruction(component: str) -> str:
    normalized = str(component).strip().lower()
    return COMPONENT_ADDITIONAL_INSTRUCTIONS.get(
        normalized,
        (
            "Return short semantic component values for the requested component type, not full social media messages. "
            "Prefer phrases grounded in the reference text."
        ),
    )


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _general_domain(params: dict[str, Any]) -> str:
    return str(params.get("domain") or params.get("general_domain") or "").strip()


def _pool_quantity(params: dict[str, Any], key: str) -> int:
    return int(params.get(key, params.get("quantity")))


def build_messages(semantic_task: str, params: dict[str, Any]) -> tuple[str, str]:
    if semantic_task == TASK_ANCHORS:
        return (
            SYSTEM_ANCHOR_EXTRACTION,
            (
                "Reference text:\n"
                f'"""{params["reference_text"]}"""\n\n'
                "General domain:\n"
                f"{_general_domain(params)}."
            ),
        )
    if semantic_task == TASK_POOL_GENERATION:
        component = str(params["component"])
        component_type = str(params.get("component_type") or component_type_label(component))
        additional_instruction = str(
            params.get("component_additional_instruction") or component_additional_instruction(component)
        )
        return (
            SYSTEM_POOL_GENERATION,
            (
                "Reference text:\n"
                f'"""{params.get("reference_text")}"""\n\n'
                "General domain:\n"
                f"{_general_domain(params)}.\n\n"
                "Component to generate:\n"
                f"{component_type}\n\n"
                "Required number of items:\n"
                f"{_pool_quantity(params, 'required_items')}\n\n"
                "Semantic anchors for support:\n"
                f"{_json_block(params.get('anchors', {}))}\n\n"
                "Additional instruction:\n"
                f"{additional_instruction}"
            ),
        )
    if semantic_task == TASK_POOL_EXPANSION:
        component = str(params["component"])
        component_type = str(params.get("component_type") or component_type_label(component))
        additional_instruction = str(
            params.get("component_additional_instruction") or component_additional_instruction(component)
        )
        return (
            SYSTEM_POOL_EXPANSION,
            (
                "Reference text:\n"
                f'"""{params.get("reference_text")}"""\n\n'
                "General domain:\n"
                f"{_general_domain(params)}.\n\n"
                "Component to expand:\n"
                f"{component_type}\n\n"
                "Required number of new items:\n"
                f"{_pool_quantity(params, 'required_new_items')}\n\n"
                "Existing items to avoid:\n"
                f"{_json_block(params.get('existing', []))}\n\n"
                "Semantic anchors for support:\n"
                f"{_json_block(params.get('anchors', {}))}\n\n"
                "Additional instruction:\n"
                f"{additional_instruction}"
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
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION}:
        return POOL_ITEMS_SCHEMA
    if semantic_task == TASK_INFLUENCE:
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
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION}:
        if isinstance(payload, dict):
            items = payload.get("items")
            if not isinstance(items, list):
                raise ValueError(f"{semantic_task} must return a JSON object with an items array")
            payload = items
        if not isinstance(payload, list):
            raise ValueError(f"{semantic_task} must return a JSON array")
        return [str(item).strip() for item in payload if str(item).strip()]
    if semantic_task == TASK_INFLUENCE:
        if isinstance(payload, dict):
            lists = [value for value in payload.values() if isinstance(value, list)]
            payload = lists[0] if lists else payload
        if not isinstance(payload, list):
            raise ValueError(f"{semantic_task} must return a JSON array")
        return [str(item).strip() for item in payload if str(item).strip()]
    return payload
