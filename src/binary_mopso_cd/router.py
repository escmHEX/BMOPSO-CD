from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.utils import progress_ratio, word_count


TASK_ANCHORS = "semantic_anchor_extraction"
TASK_CENTRAL_ANCHOR_SELECTION = "central_anchor_selection"
TASK_POOL_GENERATION = "semantic_pool_generation"
TASK_POOL_EXPANSION = "semantic_pool_expansion"
TASK_INFLUENCE = "semantic_component_influence_candidates"
TASK_SYNTHETIC_TEXT = "synthetic_text_generation"
TASK_WORD_REPLACEMENT = "word_replacement_candidates"
TASK_EMBEDDING = "embedding"
TASK_PROMPT_RENDERING = "prompt_rendering"

ALG_LLM = "LLM"
ALG_SBERT = "SBERT"
ALG_PROMPT_RENDERER = "DeterministicPromptGenerator"
ALG_DISTILBERT = "distilbert_fill_mask"
ALG_WORDNET_PPDB = "wordnet_ppdb"


@dataclass(slots=True)
class RouteTask:
    task_id: str
    operation_context: str
    semantic_task: str
    task_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionTask:
    task_id: str
    semantic_task: str
    alg_name: str
    task_params: dict[str, Any]
    alg_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteResponse:
    task_id: str
    operation_context: str
    semantic_task: str
    status: str
    result: Any


class SemanticRouter:
    def __init__(self, config: RuntimeConfig):
        self.config = config

    def route(self, task: RouteTask) -> ExecutionTask:
        semantic_task = task.semantic_task
        if semantic_task == TASK_EMBEDDING:
            return ExecutionTask(
                task.task_id,
                semantic_task,
                ALG_SBERT,
                task.task_params,
                {
                    "model": self.config.get("models.sbert.default"),
                    "batch_size": self.config.get("models.sbert.batch_size", 64),
                },
            )
        if semantic_task == TASK_PROMPT_RENDERING:
            return ExecutionTask(task.task_id, semantic_task, ALG_PROMPT_RENDERER, task.task_params, {})
        if semantic_task == TASK_WORD_REPLACEMENT:
            return self._route_word_replacement(task)
        if semantic_task in {
            TASK_ANCHORS,
            TASK_CENTRAL_ANCHOR_SELECTION,
            TASK_POOL_GENERATION,
            TASK_POOL_EXPANSION,
            TASK_INFLUENCE,
            TASK_SYNTHETIC_TEXT,
        }:
            return self._route_llm(task)
        raise ValueError(f"Unsupported semantic task: {semantic_task}")

    def _route_llm(self, task: RouteTask) -> ExecutionTask:
        params = self._llm_params(task)
        model = self.config.get(f"router.task_models.{task.semantic_task}") or self.config.get("ollama.default_model")
        params["model"] = model
        return ExecutionTask(task.task_id, task.semantic_task, ALG_LLM, task.task_params, params)

    def _llm_params(self, task: RouteTask) -> dict[str, Any]:
        enabled = bool(self.config.get(f"router.heuristics.{task.semantic_task}", True))
        if not enabled:
            return dict(self.config.get("router.llm_params.disabled_default"))
        if task.semantic_task == TASK_ANCHORS:
            reference = str(task.task_params.get("reference_text", ""))
            base = self.config.get("router.llm_params.semantic_anchor_extraction")
            bucket = "short" if word_count(reference) <= int(base["short_word_threshold"]) else "long"
            return dict(base[bucket])
        if task.semantic_task == TASK_CENTRAL_ANCHOR_SELECTION:
            return dict(
                self.config.get(
                    "router.llm_params.central_anchor_selection",
                    self.config.get("router.llm_params.disabled_default"),
                )
            )
        if task.semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION}:
            component = str(task.task_params.get("component", "topic")).lower()
            reference = str(task.task_params.get("reference_text", ""))
            anchor_count = int(task.task_params.get("central_anchor_count", 0))
            base = self.config.get("router.llm_params.semantic_pool_generation")
            low_evidence = (
                word_count(reference) <= int(base["low_evidence_word_threshold"])
                or anchor_count < int(base["low_evidence_anchor_threshold"])
            )
            bucket = "low_evidence" if low_evidence else "normal"
            component_params = base["components"].get(component)
            if component_params is None:
                component_params = self.config.get("router.llm_params.disabled_default")
                return dict(component_params)
            return dict(component_params[bucket])
        if task.semantic_task == TASK_INFLUENCE:
            iteration = int(task.task_params.get("iteration", 0))
            total = int(
                task.task_params.get(
                    "totalGenerations",
                    task.task_params.get("total_generations", task.task_params.get("iterations", 1)),
                )
            )
            rho = progress_ratio(iteration, total)
            base = self.config.get("router.llm_params.semantic_component_influence_candidates")
            temperature = float(base["temperature_start"]) - (
                float(base["temperature_start"]) - float(base["temperature_end"])
            ) * rho
            top_p = float(base["top_p_start"]) - (float(base["top_p_start"]) - float(base["top_p_end"])) * rho
            return {"temperature": temperature, "top_p": top_p}
        if task.semantic_task == TASK_SYNTHETIC_TEXT:
            return dict(self.config.get("router.llm_params.synthetic_text_generation"))
        return dict(self.config.get("router.llm_params.disabled_default"))

    def _route_word_replacement(self, task: RouteTask) -> ExecutionTask:
        if "targetSpan" not in task.task_params:
            raise ValueError("word_replacement_candidates requires targetSpan")
        max_variants = int(
            task.task_params.get(
                "maxVariants",
                task.task_params.get("max_variants", self.config.get("mopso.kcand", 7)),
            )
        )
        left_tokens = int(task.task_params.get("targetWordLeftTokens", 0))
        right_tokens = int(task.task_params.get("targetWordRightTokens", 0))
        has_context = left_tokens > 0 and right_tokens > 0
        if bool(self.config.get("router.heuristics.word_replacement_candidates", True)) and has_context:
            alg_params = {
                "preliminary_top_k": int(self.config.get("models.distilbert.top_k_multiplier", 3)) * max_variants,
                "max_variants": max_variants,
            }
            return ExecutionTask(task.task_id, task.semantic_task, ALG_DISTILBERT, task.task_params, alg_params)
        return ExecutionTask(
            task.task_id,
            task.semantic_task,
            ALG_WORDNET_PPDB,
            task.task_params,
            {"max_variants": max_variants, "use_ppdb": self.config.get("models.ppdb.enabled", True)},
        )
