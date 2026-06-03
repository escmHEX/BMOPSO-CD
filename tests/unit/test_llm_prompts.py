from __future__ import annotations

import pytest

from binary_mopso_cd.llm_prompts import build_messages, parse_task_result, response_format_for_task
from binary_mopso_cd.router import TASK_ANCHORS, TASK_POOL_EXPANSION, TASK_POOL_GENERATION


DOMAIN = "social media messages related to crises and emergencies"


def test_anchor_extraction_messages_match_strategy():
    system, user = build_messages(
        TASK_ANCHORS,
        {
            "reference_text": "Flooded roads near the bridge need urgent support.",
            "domain": DOMAIN,
        },
    )

    assert system == """You are an information extraction module for a prompt optimization algorithm.
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
    assert user == (
        'Reference text:\n"""Flooded roads near the bridge need urgent support."""\n\n'
        "General domain:\n"
        f"{DOMAIN}."
    )


def test_pool_generation_messages_include_component_instruction():
    system, user = build_messages(
        TASK_POOL_GENERATION,
        {
            "component": "role",
            "component_type": "roles",
            "quantity": 2,
            "required_items": 2,
            "reference_text": "Flooded roads near the bridge need urgent support.",
            "domain": DOMAIN,
            "anchors": {"entities": ["bridge"], "topics": ["flooded roads"], "actions": ["request support"]},
            "component_additional_instruction": "Return message-sender roles.",
        },
    )

    assert system.startswith("You generate one pool of semantic components for a prompt optimization algorithm.")
    assert '{"items": ["...", "..."]}' in system
    assert "Component to generate:\nroles" in user
    assert "Required number of items:\n2" in user
    assert '"entities": [\n    "bridge"\n  ]' in user
    assert user.endswith("Additional instruction:\nReturn message-sender roles.")


def test_pool_expansion_messages_include_existing_items():
    _system, user = build_messages(
        TASK_POOL_EXPANSION,
        {
            "component": "action",
            "component_type": "actions",
            "quantity": 3,
            "required_new_items": 3,
            "reference_text": "Flooded roads near the bridge need urgent support.",
            "domain": DOMAIN,
            "anchors": {"actions": ["request support"]},
            "existing": ["request help", "warn neighbors"],
            "component_additional_instruction": "Return communicative intents.",
        },
    )

    assert "Component to expand:\nactions" in user
    assert "Required number of new items:\n3" in user
    assert "Existing items to avoid:\n[\n  \"request help\",\n  \"warn neighbors\"\n]" in user
    assert user.endswith("Additional instruction:\nReturn communicative intents.")


def test_pool_schema_and_parser_prefer_items_object_but_accept_legacy_array():
    schema = response_format_for_task(TASK_POOL_GENERATION)

    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert parse_task_result(TASK_POOL_GENERATION, '{"items": ["a", " b ", ""]}') == ["a", "b"]
    assert parse_task_result(TASK_POOL_EXPANSION, '["legacy a", "legacy b"]') == ["legacy a", "legacy b"]
    with pytest.raises(ValueError, match="items array"):
        parse_task_result(TASK_POOL_GENERATION, '{"values": ["a"]}')
