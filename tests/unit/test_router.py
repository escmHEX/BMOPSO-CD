from __future__ import annotations

from binary_mopso_cd.router import (
    ALG_DISTILBERT,
    ALG_LLM,
    ALG_WORDNET_PPDB,
    TASK_ANCHORS,
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
