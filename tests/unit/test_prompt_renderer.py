from __future__ import annotations

from binary_mopso_cd.services.prompt_renderer import DeterministicPromptRenderer


def test_prompt_renderer_uses_canonical_role_topic_action_template():
    renderer = DeterministicPromptRenderer()
    prompt = renderer.render(
        {
            "role": "affected resident",
            "topic": "urban floods and road closures",
            "action": "request help and report location",
        },
        "Natural-disaster and emergency scenario messages",
    )
    assert prompt == (
        "Generate a short natural-disaster scenario message using the following semantic components: "
        "role = affected resident; topic = urban floods and road closures; action = request help and report location. "
        "The generated message must follow the role, address the topic, and satisfy the action."
    )
