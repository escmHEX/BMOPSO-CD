from __future__ import annotations

import re

from binary_mopso_cd.utils import canonical_text


class DeterministicPromptRenderer:
    def render(self, components: dict[str, str], domain: str) -> str:
        normalized = {
            name: self._normalize_slot(value) for name, value in components.items() if value and self._normalize_slot(value)
        }
        names = set(normalized)
        if names == {"role"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic component: "
                f"role = {normalized['role']}. The generated message must follow the role."
            )
        if names == {"topic"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic component: "
                f"topic = {normalized['topic']}. The generated message must address the topic."
            )
        if names == {"action"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic component: "
                f"action = {normalized['action']}. The generated message must satisfy the action."
            )
        if names == {"role", "topic"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic components: "
                f"role = {normalized['role']}; topic = {normalized['topic']}. "
                "The generated message must follow the role and address the topic."
            )
        if names == {"role", "action"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic components: "
                f"role = {normalized['role']}; action = {normalized['action']}. "
                "The generated message must follow the role and satisfy the action."
            )
        if names == {"topic", "action"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic components: "
                f"topic = {normalized['topic']}; action = {normalized['action']}. "
                "The generated message must address the topic and satisfy the action."
            )
        if names == {"role", "topic", "action"}:
            return (
                "Generate a short natural-disaster scenario message using the following semantic components: "
                f"role = {normalized['role']}; topic = {normalized['topic']}; action = {normalized['action']}. "
                "The generated message must follow the role, address the topic, and satisfy the action."
            )
        if not normalized:
            raise ValueError("Deterministic prompt rendering requires at least one component")
        component_list = "; ".join(f"{name} = {value}" for name, value in normalized.items())
        return (
            "Generate a short natural-disaster scenario message using the following semantic components: "
            f"{component_list}. The generated message must satisfy all provided components."
        )

    def normalize_component(self, text: str) -> str:
        return canonical_text(text)

    def _normalize_slot(self, text: str) -> str:
        value = re.sub(r"\s+", " ", str(text).strip())
        return value.strip("\"'` ")
