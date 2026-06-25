from __future__ import annotations

from pathlib import Path

import pytest

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.router import (
    ALG_DISTILBERT,
    ALG_LLM,
    ALG_WORDNET_PPDB,
    TASK_ANCHORS,
    TASK_CENTRAL_ANCHOR_SELECTION,
    TASK_INFLUENCE,
    TASK_POOL_EXPANSION,
    TASK_POOL_GENERATION,
    TASK_SYNTHETIC_TEXT,
    TASK_WORD_REPLACEMENT,
    RouteTask,
    SemanticRouter,
)


def test_router_anchor_heuristic_short_reference(test_config):
    router = SemanticRouter(test_config)
    task = RouteTask("1", "test", TASK_ANCHORS, {"reference_text": "Flood warning now"})
    execution = router.route(task)
    assert execution.alg_name == ALG_LLM
    assert execution.alg_params["temperature"] == 0.25
    assert execution.alg_params["top_p"] == 0.90


def test_router_routes_central_anchor_selection_with_default_llm_params(test_config):
    router = SemanticRouter(test_config)
    task = RouteTask(
        "1",
        "test",
        TASK_CENTRAL_ANCHOR_SELECTION,
        {"referenceText": "Flood warning now", "semanticAnchors": {}, "numCentralAnchors": 4},
    )
    execution = router.route(task)

    assert execution.alg_name == ALG_LLM
    assert execution.alg_params["temperature"] == 0.60
    assert execution.alg_params["top_p"] == 0.90


def test_default_config_routes_central_anchor_selection_to_gemma_high_thinking():
    config = RuntimeConfig.load(Path("configs/default.yaml"))
    config.validate()
    router = SemanticRouter(config)
    task = RouteTask(
        "1",
        "optimization",
        TASK_CENTRAL_ANCHOR_SELECTION,
        {"referenceText": "Flood warning now", "semanticAnchors": {}, "numCentralAnchors": 4},
    )

    execution = router.route(task)

    assert execution.alg_name == ALG_LLM
    assert execution.alg_params["model"] == "gemma4:e4b"
    assert execution.alg_params["thinking"] == "high"
    assert execution.alg_params["temperature"] == 0.60
    assert execution.alg_params["top_p"] == 0.90


def test_router_phase_task_model_override_takes_precedence(test_config):
    test_config.set("router.phase_task_models.initialization.synthetic_text_generation", "qwen3.5:2b")
    router = SemanticRouter(test_config)
    task = RouteTask("1", "initialization", TASK_SYNTHETIC_TEXT, {"prompt": "x", "reference_text": "y"})

    execution = router.route(task)

    assert execution.alg_name == ALG_LLM
    assert execution.alg_params["model"] == "qwen3.5:2b"
    assert execution.alg_params["temperature"] == 0.75
    assert execution.alg_params["top_p"] == 0.95


def test_router_optimization_synthetic_text_inherits_task_model_when_phase_model_is_null(test_config):
    test_config.set("router.task_models.synthetic_text_generation", "llama3.1:8b-custom")
    test_config.set("router.phase_task_models.optimization.synthetic_text_generation", None)
    router = SemanticRouter(test_config)
    task = RouteTask("1", "optimization", TASK_SYNTHETIC_TEXT, {"prompt": "x", "reference_text": "y"})

    execution = router.route(task)

    assert execution.alg_name == ALG_LLM
    assert execution.alg_params["model"] == "llama3.1:8b-custom"


def test_router_task_model_override_still_applies_when_phase_model_is_null(test_config):
    test_config.set("router.task_models.synthetic_text_generation", "qwen3.5:2b")
    test_config.set("router.phase_task_models.initialization.synthetic_text_generation", None)
    router = SemanticRouter(test_config)
    task = RouteTask("1", "initialization", TASK_SYNTHETIC_TEXT, {"prompt": "x", "reference_text": "y"})

    execution = router.route(task)

    assert execution.alg_name == ALG_LLM
    assert execution.alg_params["model"] == "qwen3.5:2b"


def test_router_llm_tasks_default_to_thinking_false(test_config):
    router = SemanticRouter(test_config)
    tasks = [
        RouteTask("anchors", "initialization", TASK_ANCHORS, {"reference_text": "Flood warning now"}),
        RouteTask(
            "central",
            "initialization",
            TASK_CENTRAL_ANCHOR_SELECTION,
            {"referenceText": "Flood warning now", "semanticAnchors": {}, "numCentralAnchors": 4},
        ),
        RouteTask(
            "pool",
            "initialization",
            TASK_POOL_GENERATION,
            {
                "component": "role",
                "reference_text": "Flood warning now",
                "central_anchor_count": 4,
            },
        ),
        RouteTask(
            "expand",
            "initialization",
            TASK_POOL_EXPANSION,
            {
                "component": "topic",
                "reference_text": "Flood warning now",
                "central_anchor_count": 4,
            },
        ),
        RouteTask(
            "influence",
            "optimization",
            TASK_INFLUENCE,
            {"iteration": 0, "iterations": 10, "component": "action"},
        ),
        RouteTask("synthetic", "optimization", TASK_SYNTHETIC_TEXT, {"prompt": "x", "reference_text": "y"}),
    ]

    for task in tasks:
        execution = router.route(task)
        assert execution.alg_name == ALG_LLM
        assert execution.alg_params["thinking"] is False


def test_router_task_thinking_override_only_affects_that_task(test_config):
    test_config.set("router.llm_params.synthetic_text_generation.thinking", True)
    router = SemanticRouter(test_config)

    synthetic = router.route(
        RouteTask("synthetic", "optimization", TASK_SYNTHETIC_TEXT, {"prompt": "x", "reference_text": "y"})
    )
    anchors = router.route(RouteTask("anchors", "initialization", TASK_ANCHORS, {"reference_text": "Flood warning now"}))

    assert synthetic.alg_params["thinking"] is True
    assert anchors.alg_params["thinking"] is False


def test_router_task_thinking_config_path_overrides_task_llm_params(test_config):
    test_config.set("router.llm_params.synthetic_text_generation.thinking", False)
    test_config.set("router.task_thinking.synthetic_text_generation", True)
    router = SemanticRouter(test_config)

    synthetic = router.route(
        RouteTask("synthetic", "optimization", TASK_SYNTHETIC_TEXT, {"prompt": "x", "reference_text": "y"})
    )
    anchors = router.route(RouteTask("anchors", "initialization", TASK_ANCHORS, {"reference_text": "Flood warning now"}))

    assert synthetic.alg_params["thinking"] is True
    assert anchors.alg_params["thinking"] is False


def test_validate_rejects_task_thinking_true_for_incompatible_model(test_config):
    test_config.set("router.task_models.synthetic_text_generation", "llama3.1:8b")
    test_config.set("router.task_thinking.synthetic_text_generation", True)

    with pytest.raises(ValueError, match="does not support thinking"):
        test_config.validate()


def test_validate_rejects_task_thinking_true_for_unknown_model(test_config):
    test_config.set("router.task_models.synthetic_text_generation", "custom-local-model")
    test_config.set("router.task_thinking.synthetic_text_generation", True)

    with pytest.raises(ValueError, match="not declared in ollama.model_capabilities"):
        test_config.validate()


@pytest.mark.parametrize(
    "model",
    [
        "gemma4:e4b",
        "qwen3.5:9b",
        "lfm2.5:8b",
    ],
)
@pytest.mark.parametrize("thinking", [True, "low", "medium", "high"])
def test_validate_allows_task_thinking_for_models_that_support_thinking(test_config, model, thinking):
    test_config.set("router.task_models.semantic_pool_generation", model)
    test_config.set("router.task_thinking.semantic_pool_generation", thinking)

    test_config.validate()


@pytest.mark.parametrize("thinking", [True, "low"])
def test_validate_allows_task_thinking_for_supported_model_without_task_validation(test_config, thinking):
    test_config.set("router.task_models.semantic_component_influence_candidates", "qwen3.5:4b")
    test_config.set("router.task_thinking.semantic_component_influence_candidates", thinking)

    test_config.validate()


def test_validate_exposes_validated_thinking_tasks_as_metadata(test_config):
    assert test_config.model_validated_thinking_tasks("qwen3.5:9b") == {"semantic_anchor_extraction"}

    test_config.set("router.task_models.semantic_pool_generation", "qwen3.5:9b")
    test_config.set("router.task_thinking.semantic_pool_generation", "low")
    test_config.validate()


def test_validate_rejects_unknown_task_thinking_value(test_config):
    test_config.set("router.task_models.synthetic_text_generation", "qwen3.5:4b")
    test_config.set("router.task_thinking.synthetic_text_generation", "maximum")

    with pytest.raises(ValueError, match="must be false, true, null, low, medium, or high"):
        test_config.validate()


def test_validate_allows_task_thinking_false_for_incompatible_model(test_config):
    test_config.set("router.task_models.synthetic_text_generation", "llama3.1:8b")
    test_config.set("router.task_thinking.synthetic_text_generation", False)

    test_config.validate()


def test_router_word_replacement_uses_distilbert_with_context(test_config):
    router = SemanticRouter(test_config)
    task = RouteTask(
        "1",
        "test",
        TASK_WORD_REPLACEMENT,
        {"text": "public safety alert", "tokens": ["public", "safety", "alert"], "target_index": 1, "max_variants": 5},
    )
    assert router.route(task).alg_name == ALG_DISTILBERT


def test_router_word_replacement_uses_default_kcand_when_missing(test_config):
    router = SemanticRouter(test_config)
    task = RouteTask(
        "1",
        "test",
        TASK_WORD_REPLACEMENT,
        {"text": "public safety alert", "tokens": ["public", "safety", "alert"], "target_index": 1},
    )
    execution = router.route(task)
    assert execution.alg_name == ALG_DISTILBERT
    assert execution.alg_params["max_variants"] == 6
    assert execution.alg_params["preliminary_top_k"] == 18


def test_router_word_replacement_accepts_strategy_context_fields(test_config):
    router = SemanticRouter(test_config)
    task = RouteTask(
        "1",
        "test",
        TASK_WORD_REPLACEMENT,
        {
            "component": "public safety alert",
            "componentType": "topic",
            "targetWord": "safety",
            "targetWordLeftTokens": 1,
            "targetWordRightTokens": 1,
            "target_index": 1,
            "maxVariants": 7,
        },
    )
    execution = router.route(task)

    assert execution.alg_name == ALG_DISTILBERT
    assert execution.alg_params["max_variants"] == 7
    assert execution.alg_params["preliminary_top_k"] == 21


def test_router_word_replacement_fallback_at_boundary(test_config):
    router = SemanticRouter(test_config)
    task = RouteTask(
        "1",
        "test",
        TASK_WORD_REPLACEMENT,
        {"text": "safety alert", "tokens": ["safety", "alert"], "target_index": 0, "max_variants": 5},
    )
    assert router.route(task).alg_name == ALG_WORDNET_PPDB
