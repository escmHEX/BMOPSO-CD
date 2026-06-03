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
        "social media messages related to crises and emergencies",
    )
    assert prompt == (
        "Generate a short social media message related to crises and emergencies using the following semantic components: "
        "role = affected resident; topic = urban floods and road closures; action = request help and report location. "
        "The generated message must follow the role, address the topic, and satisfy the action."
    )


def test_prompt_renderer_uses_specific_single_component_templates():
    renderer = DeterministicPromptRenderer()
    domain = "social media messages related to crises and emergencies"

    assert renderer.render({"role": "affected resident"}, domain) == (
        "Generate a short social media message related to crises and emergencies using the following semantic component: "
        "role = affected resident. The generated message must follow the role."
    )
    assert renderer.render({"topic": "urban floods"}, domain) == (
        "Generate a short social media message related to crises and emergencies using the following semantic component: "
        "topic = urban floods. The generated message must address the topic."
    )
    assert renderer.render({"action": "request help"}, domain) == (
        "Generate a short social media message related to crises and emergencies using the following semantic component: "
        "action = request help. The generated message must satisfy the action."
    )


def test_prompt_renderer_uses_generalized_template_for_extra_components():
    renderer = DeterministicPromptRenderer()
    prompt = renderer.render(
        {
            "role": "affected resident",
            "topic": "urban floods",
            "action": "request help",
            "tone": "urgent",
        },
        "social media messages related to crises and emergencies",
    )
    assert prompt == (
        "Generate a short social media message related to crises and emergencies using the following semantic components: "
        "role = affected resident; topic = urban floods; action = request help; tone = urgent. "
        "The generated message must satisfy all provided components."
    )
