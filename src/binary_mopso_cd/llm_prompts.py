from __future__ import annotations

import json
import re
from typing import Any

from binary_mopso_cd.component_specs import component_spec
from binary_mopso_cd.router import (
    TASK_ANCHORS,
    TASK_CENTRAL_ANCHOR_SELECTION,
    TASK_INFLUENCE,
    TASK_POOL_EXPANSION,
    TASK_POOL_GENERATION,
    TASK_SYNTHETIC_TEXT,
)
from binary_mopso_cd.utils import canonical_text


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

SYSTEM_CENTRAL_ANCHOR_SELECTION = """You select central anchors for a synthetic text generation algorithm.

Task:
Select a compact set of central anchors from the reference text and the provided semantic anchors.

Output format:
Return only valid JSON with exactly this structure:
{"central_anchors": ["...", "..."]}

Rules:
- Select only anchors that preserve the main meaning of the reference text.
- Prefer specific phrases over generic words.
- Prefer phrases that are present in the reference text or directly supported by it.
- Avoid generic domain terms unless they are essential.
- Avoid redundant anchors.
- Do not invent entities, events, resources, or topics.
- Return between 3 and 5 anchors.
- Do not include explanations, markdown, numbering, or extra keys."""

SYSTEM_TEXT_GENERATION = """You are a plain-text generator for social media messages related to crises and emergencies.

You will receive one text-generation instruction from the user.
Follow the instruction and generate exactly one final text message.

Output rules:
- Return plain text only.
- Do not describe the task.
- Do not add unsolicited safety advice.
- Do not use quotation marks, hashtags, URLs, usernames, placeholders, tags, or special markers.
- Limit the message to between 1 and 4 sentences.
- Return only the final message."""

SEMANTIC_MOVE_SYSTEM_PROMPT = """You are a prompt-component editor for structured prompt optimization.
Task:
Generate candidate replacements for exactly one semantic prompt component.
Each candidate must move the current component semantically toward the target component.
General rules:
- Return only candidate components.
- Keep the same component type requested by the user.
- Each candidate must have 2-8 words.
- The other components are only there to give you context.
- Do not copy the current component exactly.
- Avoid copying the target component exactly, but semantic closeness is more important than lexical novelty.
- Candidates must be distinct from each other.
- Do not explain.
- Do not use quotes.
- Do not add titles, labels, comments, or extra text.
- Return exactly the number of candidates requested by the user.
- Use numbered lines with this format:
1) candidate
2) candidate
3) candidate
..."""


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

CENTRAL_ANCHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "central_anchors": JSON_ARRAY_OF_STRINGS_SCHEMA,
    },
    "required": ["central_anchors"],
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


def _first_param(params: dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = params.get(key)
        if value is not None:
            return value
    return default


def _format_other_components(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        return _json_block(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        return _json_block(value)
    text = str(value or "").strip()
    return text or "{}"


def _format_central_anchors(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def build_base_text_generation_user_prompt(prompt: str) -> str:
    return (
        "Prompt to follow:\n"
        f'"""{prompt}"""'
    )


def build_anchored_text_generation_user_prompt(prompt: str, central_anchors: list[str]) -> str:
    return (
        f"{build_base_text_generation_user_prompt(prompt)}\n\n"
        "Reference-specific anchors:\n"
        f"{_format_central_anchors(central_anchors)}\n\n"
        "Instruction:\n"
        "Follow the prompt as the main generation instruction. Use the reference-specific anchors only as "
        "semantic context to preserve important information when compatible with the prompt. Do not force "
        "all anchors into the message. Do not copy the full reference text."
    )


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
    if semantic_task == TASK_CENTRAL_ANCHOR_SELECTION:
        reference_text = str(_first_param(params, "referenceText", "reference_text")).strip()
        semantic_anchors = _first_param(params, "semanticAnchors", "semantic_anchors", "anchors", default={})
        central_anchor_count = int(_first_param(params, "numCentralAnchors", "num_central_anchors", default=4))
        return (
            SYSTEM_CENTRAL_ANCHOR_SELECTION,
            (
                "Reference text:\n"
                f'"""{reference_text}"""\n\n'
                "Semantic anchors:\n"
                f"{_json_block(semantic_anchors)}\n\n"
                "Number of central anchors to return:\n"
                f"{central_anchor_count}\n\n"
                "Coverage priority:\n"
                "Select anchors that cover distinct semantic roles when available:\n"
                "- main entity or stakeholder\n"
                "- main event or problem\n"
                "- main resource, topic, or information type\n"
                "- main action, channel, or information source\n\n"
                "Selection rules:\n"
                "- Prefer specific compound noun phrases over isolated generic words.\n"
                "- Convert action anchors into concise noun phrases when possible.\n"
                "- Do not select generic domain terms such as \"crisis\" or \"emergency\" when more specific anchors are available.\n"
                "- Avoid redundant anchors unless the repeated concept is needed to preserve a distinct semantic role.\n\n"
                "Select the central anchors that should be reused as semantic context during final text generation."
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
        component_name = str(_first_param(params, "componentName", "component_name", "component")).strip()
        spec = component_spec(component_name)
        num_candidates = int(_first_param(params, "numCandidates", "num_candidates", "max_candidates", default=5))
        component_definition = str(
            _first_param(params, "componentDefinition", "component_definition", default=spec.definition)
        ).strip()
        current_component = str(_first_param(params, "currentComponent", "current_component", "current")).strip()
        target_component = str(_first_param(params, "targetComponent", "target_component", "target")).strip()
        other_components = _format_other_components(
            _first_param(params, "otherComponents", "other_components", default={})
        )
        reference_text = str(_first_param(params, "referenceText", "reference_text")).strip()
        additional_instruction = str(
            _first_param(
                params,
                "componentAdditionalInstruction",
                "component_additional_instruction",
                default=spec.influence_instruction,
            )
        ).strip()
        return (
            SEMANTIC_MOVE_SYSTEM_PROMPT,
            (
                "Number of candidates: "
                f"{num_candidates}\n\n"
                "Component: "
                f"{component_name}\n"
                "Definition: "
                f"{component_definition}\n\n"
                "Current: "
                f"{current_component}\n"
                "Target: "
                f"{target_component}\n\n"
                "Other components:\n"
                f"{other_components}\n\n"
                "Reference text:\n"
                f"{reference_text}\n\n"
                "Additional important instruction:\n"
                f"{additional_instruction}"
            ),
        )
    if semantic_task == TASK_SYNTHETIC_TEXT:
        system_prompt = str(
            _first_param(params, "systemPromptOverride", "system_prompt_override", default=SYSTEM_TEXT_GENERATION)
        )
        user_prompt = _first_param(params, "userPromptOverride", "user_prompt_override", default=None)
        return (
            system_prompt,
            str(user_prompt) if user_prompt is not None else build_base_text_generation_user_prompt(str(params["prompt"])),
        )
    raise ValueError(f"No prompt template for semantic task: {semantic_task}")


def response_format_for_task(semantic_task: str) -> Any:
    if semantic_task == TASK_SYNTHETIC_TEXT:
        return None
    if semantic_task == TASK_ANCHORS:
        return ANCHOR_SCHEMA
    if semantic_task == TASK_CENTRAL_ANCHOR_SELECTION:
        return CENTRAL_ANCHOR_SCHEMA
    if semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION}:
        return POOL_ITEMS_SCHEMA
    if semantic_task == TASK_INFLUENCE:
        return None
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
    if semantic_task == TASK_INFLUENCE:
        return parse_influence_candidates(raw)
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
    if semantic_task == TASK_CENTRAL_ANCHOR_SELECTION:
        if not isinstance(payload, dict):
            raise ValueError("Central anchor selection must return a JSON object")
        value = payload.get("central_anchors")
        if not isinstance(value, list):
            raise ValueError("central_anchor_selection must return a central_anchors array")
        seen: set[str] = set()
        anchors: list[str] = []
        for item in value:
            text = str(item).strip()
            normalized = canonical_text(text)
            if text and normalized not in seen:
                seen.add(normalized)
                anchors.append(text)
        if len(anchors) < 3 or len(anchors) > 5:
            raise ValueError("central_anchor_selection must return between 3 and 5 central anchors")
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
    return payload


def parse_influence_candidates(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if text.startswith(("[", "{")) or "```" in text:
        try:
            payload = parse_json_payload(text)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            lists = [value for value in payload.values() if isinstance(value, list)]
            payload = lists[0] if lists else payload
        if isinstance(payload, list):
            return _clean_candidate_lines(str(item) for item in payload)
    return _clean_candidate_lines(text.splitlines())


def _clean_candidate_lines(lines: Any) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for line in lines:
        value = str(line).strip()
        if not value or value == "...":
            continue
        value = re.sub(r"^\s*\d+\s*[\).:-]\s*", "", value).strip()
        value = value.strip(" \t\r\n\"'`")
        normalized = canonical_text(value)
        if not normalized or normalized in seen or normalized == "candidate":
            continue
        seen.add(normalized)
        candidates.append(value)
    return candidates
