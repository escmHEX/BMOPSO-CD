from __future__ import annotations

import pytest

from binary_mopso_cd.llm_prompts import (
    SEMANTIC_MOVE_SYSTEM_PROMPT,
    build_anchored_text_generation_user_prompt,
    build_messages,
    parse_task_result,
    response_format_for_task,
)
from binary_mopso_cd.router import TASK_ANCHORS, TASK_INFLUENCE, TASK_POOL_EXPANSION, TASK_POOL_GENERATION
from binary_mopso_cd.router import TASK_CENTRAL_ANCHOR_SELECTION, TASK_SYNTHETIC_TEXT


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


def test_central_anchor_selection_messages_match_strategy():
    anchors = {
        "entities": ["small businesses", "local emergency center"],
        "topics": ["disaster financing options"],
        "actions": ["share verified updates"],
        "constraints": ["avoid rumors"],
    }

    system, user = build_messages(
        TASK_CENTRAL_ANCHOR_SELECTION,
        {
            "referenceText": "Small businesses can check the local emergency center for disaster financing options.",
            "semanticAnchors": anchors,
            "numCentralAnchors": 4,
        },
    )

    assert system == """You select central anchors for a synthetic text generation algorithm.

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
    assert user == (
        "Reference text:\n"
        '"""Small businesses can check the local emergency center for disaster financing options."""\n\n'
        "Semantic anchors:\n"
        "{\n"
        '  "entities": [\n'
        '    "small businesses",\n'
        '    "local emergency center"\n'
        "  ],\n"
        '  "topics": [\n'
        '    "disaster financing options"\n'
        "  ],\n"
        '  "actions": [\n'
        '    "share verified updates"\n'
        "  ],\n"
        '  "constraints": [\n'
        '    "avoid rumors"\n'
        "  ]\n"
        "}\n\n"
        "Number of central anchors to return:\n"
        "4\n\n"
        "Coverage priority:\n"
        "Select anchors that cover distinct semantic roles when available:\n"
        "- main entity or stakeholder\n"
        "- main event or problem\n"
        "- main resource, topic, or information type\n"
        "- main action, channel, or information source\n\n"
        "Selection rules:\n"
        "- Prefer specific compound noun phrases over isolated generic words.\n"
        "- Convert action anchors into concise noun phrases when possible.\n"
        '- Do not select generic domain terms such as "crisis" or "emergency" when more specific anchors are available.\n'
        "- Avoid redundant anchors unless the repeated concept is needed to preserve a distinct semantic role.\n\n"
        "Select the central anchors that should be reused as semantic context during final text generation."
    )


def test_central_anchor_schema_and_parser():
    schema = response_format_for_task(TASK_CENTRAL_ANCHOR_SELECTION)

    assert schema["type"] == "object"
    assert schema["required"] == ["central_anchors"]
    assert parse_task_result(
        TASK_CENTRAL_ANCHOR_SELECTION,
        '{"central_anchors": ["small businesses", " disaster financing options ", "local emergency center"]}',
    ) == ["small businesses", "disaster financing options", "local emergency center"]
    with pytest.raises(ValueError, match="between 3 and 5"):
        parse_task_result(TASK_CENTRAL_ANCHOR_SELECTION, '{"central_anchors": ["one", "two"]}')


def test_synthetic_text_messages_use_base_prompt_without_implicit_anchors():
    system, user = build_messages(
        TASK_SYNTHETIC_TEXT,
        {
            "prompt": "Generate a short social media message.",
            "centralAnchors": [
                "small businesses",
                "disaster financing options",
                "local emergency center",
                "verified updates",
            ],
        },
    )

    assert system == """You are a plain-text generator for social media messages related to crises and emergencies.

You will receive one text-generation instruction from the user.
Follow the instruction and generate exactly one final text message.

Output rules:
- Return plain text only.
- Do not describe the task.
- Do not add unsolicited safety advice.
- Do not use quotation marks, hashtags, URLs, usernames, placeholders, tags, or special markers.
- Limit the message to between 1 and 4 sentences.
- Return only the final message."""
    assert user == (
        "Prompt to follow:\n"
        '"""Generate a short social media message."""'
    )
    assert response_format_for_task(TASK_SYNTHETIC_TEXT) is None


def test_synthetic_text_messages_use_explicit_user_prompt_override():
    override = build_anchored_text_generation_user_prompt(
        "Generate a short social media message.",
        [
            "small businesses",
            "disaster financing options",
            "local emergency center",
            "verified updates",
        ],
    )

    system, user = build_messages(
        TASK_SYNTHETIC_TEXT,
        {
            "prompt": "Generate a short social media message.",
            "userPromptOverride": override,
            "systemPromptOverride": "custom system prompt",
        },
    )

    assert system == "custom system prompt"
    assert user == (
        "Prompt to follow:\n"
        '"""Generate a short social media message."""\n\n'
        "Reference-specific anchors:\n"
        '["small businesses", "disaster financing options", "local emergency center", "verified updates"]\n\n'
        "Instruction:\n"
        "Follow the prompt as the main generation instruction. Use the reference-specific anchors only as semantic "
        "context to preserve important information when compatible with the prompt. Do not force all anchors into the "
        "message. Do not copy the full reference text."
    )


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


def test_pool_schema_and_parser_require_items_object():
    schema = response_format_for_task(TASK_POOL_GENERATION)

    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert parse_task_result(TASK_POOL_GENERATION, '{"items": ["a", " b ", ""]}') == ["a", "b"]
    with pytest.raises(ValueError, match="items array"):
        parse_task_result(TASK_POOL_EXPANSION, '["old a", "old b"]')
    with pytest.raises(ValueError, match="items array"):
        parse_task_result(TASK_POOL_GENERATION, '{"values": ["a"]}')


def test_influence_messages_match_strategy_contract():
    system, user = build_messages(
        TASK_INFLUENCE,
        {
            "numCandidates": 3,
            "componentName": "action",
            "componentDefinition": (
                "Communicative intent or discourse operation that indicates how the message communicates information. "
                "It must be a verb phrase, not a resource, program, service, or support type."
            ),
            "currentComponent": "ask for aid",
            "targetComponent": "warn residents about flooding",
            "otherComponents": {"role": "local official", "topic": "flooded roads"},
            "referenceText": "Flooded roads near the bridge need urgent support.",
            "componentAdditionalInstruction": (
                "Keep each candidate as a communicative verb phrase. Preserve the target object when possible. If you "
                "change it, use only a close paraphrase supported by the reference text. Do not replace it with a narrower "
                "or broader resource, program, service, entity, or topic."
            ),
        },
    )

    assert system == SEMANTIC_MOVE_SYSTEM_PROMPT
    assert user == (
        "Number of candidates: 3\n\n"
        "Component: action\n"
        "Definition: Communicative intent or discourse operation that indicates how the message communicates information. "
        "It must be a verb phrase, not a resource, program, service, or support type.\n\n"
        "Current: ask for aid\n"
        "Target: warn residents about flooding\n\n"
        "Other components:\n"
        "{\n"
        '  "role": "local official",\n'
        '  "topic": "flooded roads"\n'
        "}\n\n"
        "Reference text:\n"
        "Flooded roads near the bridge need urgent support.\n\n"
        "Additional important instruction:\n"
        "Keep each candidate as a communicative verb phrase. Preserve the target object when possible. If you "
        "change it, use only a close paraphrase supported by the reference text. Do not replace it with a narrower "
        "or broader resource, program, service, entity, or topic."
    )


def test_influence_uses_plain_text_response_and_parses_numbered_lines():
    assert response_format_for_task(TASK_INFLUENCE) is None
    raw = "1) warn residents about flooding\n2. alert locals about flood risk\n3) notify bridge users"

    assert parse_task_result(TASK_INFLUENCE, raw) == [
        "warn residents about flooding",
        "alert locals about flood risk",
        "notify bridge users",
    ]


def test_influence_parser_requires_plain_numbered_lines():
    with pytest.raises(ValueError, match="plain numbered lines"):
        parse_task_result(TASK_INFLUENCE, '[" warn residents ", "alert locals"]')
