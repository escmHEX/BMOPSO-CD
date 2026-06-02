from __future__ import annotations

from binary_mopso_cd.utils import canonical_text


class DeterministicPromptRenderer:
    def render(self, components: dict[str, str], domain: str) -> str:
        normalized = {name: value.strip() for name, value in components.items() if value and value.strip()}
        role = normalized.get("role")
        topic = normalized.get("topic")
        action = normalized.get("action")
        if role and topic and action:
            return (
                f"From the perspective of {role}, write a {domain} message "
                f"about {topic} that should {action}."
            )
        if role and topic:
            return f"From the perspective of {role}, write a {domain} message about {topic}."
        if topic and action:
            return f"Write a {domain} message about {topic} that should {action}."
        if role and action:
            return f"From the perspective of {role}, write a {domain} message that should {action}."
        constraints = "; ".join(f"{name}: {value}" for name, value in sorted(normalized.items()))
        return f"Write a {domain} message that combines these semantic constraints: {constraints}."

    def normalize_component(self, text: str) -> str:
        return canonical_text(text)

