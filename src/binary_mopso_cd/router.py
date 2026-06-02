from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.utils import word_count


TASK_ANCHORS = "semantic_anchor_extraction"
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
            return {"temperature": 0.60, "top_p": 0.90}
        if task.semantic_task == TASK_ANCHORS:
            reference = str(task.task_params.get("reference_text", ""))
            if word_count(reference) <= 6:
                return {"temperature": 0.25, "top_p": 0.90}
            return {"temperature": 0.20, "top_p": 0.85}
        if task.semantic_task in {TASK_POOL_GENERATION, TASK_POOL_EXPANSION}:
            component = str(task.task_params.get("component", "topic")).lower()
            reference = str(task.task_params.get("reference_text", ""))
            anchor_count = int(task.task_params.get("central_anchor_count", 0))
            low_evidence = word_count(reference) <= 6 or anchor_count < 4
            table = {
                "role": (0.68, 0.95) if low_evidence else (0.60, 0.90),
                "topic": (0.42, 0.90) if low_evidence else (0.40, 0.89),
                "action": (0.58, 0.94) if low_evidence else (0.50, 0.90),
            }
            temperature, top_p = table.get(component, (0.55, 0.90))
            return {"temperature": temperature, "top_p": top_p}
        if task.semantic_task == TASK_INFLUENCE:
            iteration = int(task.task_params.get("iteration", 0))
            total = int(task.task_params.get("iterations", 1))
            rho = 0.0 if total <= 1 else iteration / max(total - 1, 1)
            return {"temperature": 0.70 - 0.20 * rho, "top_p": 0.95 - 0.05 * rho}
        if task.semantic_task == TASK_SYNTHETIC_TEXT:
            return {"temperature": 0.75, "top_p": 0.95}
        return {"temperature": 0.60, "top_p": 0.90}

    def _route_word_replacement(self, task: RouteTask) -> ExecutionTask:
        tokens = list(task.task_params.get("tokens", []))
        index = int(task.task_params.get("target_index", -1))
        max_variants = int(task.task_params.get("max_variants", self.config.get("mopso.kcand", 5)))
        has_context = index > 0 and index < len(tokens) - 1
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

