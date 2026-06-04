from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    name: str
    definition: str
    preferred_turbulence_pos: tuple[str, ...]
    influence_instruction: str


COMPONENT_SPECS: dict[str, ComponentSpec] = {
    "role": ComponentSpec(
        name="role",
        definition="Message-sender role or stakeholder perspective that indicates who is speaking or posting.",
        preferred_turbulence_pos=("NOUN", "PROPN"),
        influence_instruction=(
            "Keep each candidate as a message-sender role or stakeholder perspective. Do not turn the candidate into "
            "a topic, event, action, object, organization description, or full message. Do not introduce details that "
            "are absent from both the reference text and the target. Prefer short noun phrases close to the target "
            "and reference text."
        ),
    ),
    "topic": ComponentSpec(
        name="topic",
        definition="Issue, event, or subject focus that indicates what the message is about.",
        preferred_turbulence_pos=("NOUN", "PROPN", "ADJ"),
        influence_instruction=(
            "Keep each candidate as a short noun phrase. Preserve the central terms of the target as much as possible. "
            "Do not replace them with broader, narrower, or adjacent concepts unless those concepts appear in the "
            "reference text."
        ),
    ),
    "action": ComponentSpec(
        name="action",
        definition=(
            "Communicative intent or discourse operation that indicates how the message communicates information. "
            "It must be a verb phrase, not a resource, program, service, or support type."
        ),
        preferred_turbulence_pos=("VERB", "NOUN"),
        influence_instruction=(
            "Keep each candidate as a communicative verb phrase. Preserve the target object when possible. If you "
            "change it, use only a close paraphrase supported by the reference text. Do not replace it with a narrower "
            "or broader resource, program, service, entity, or topic."
        ),
    ),
}


def component_spec(component: str) -> ComponentSpec:
    normalized = str(component).strip().lower()
    if normalized in COMPONENT_SPECS:
        return COMPONENT_SPECS[normalized]
    return ComponentSpec(
        name=normalized,
        definition=f"Semantic prompt component of type {normalized}.",
        preferred_turbulence_pos=(),
        influence_instruction=(
            "Keep each candidate as the requested semantic component type. Do not turn it into a full message."
        ),
    )
