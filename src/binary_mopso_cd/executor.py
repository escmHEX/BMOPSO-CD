from __future__ import annotations

from pathlib import Path
from typing import Any

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.llm_prompts import build_messages, parse_task_result, response_format_for_task
from binary_mopso_cd.router import (
    ALG_DISTILBERT,
    ALG_LLM,
    ALG_PROMPT_RENDERER,
    ALG_SBERT,
    ALG_WORDNET_PPDB,
    ExecutionTask,
)
from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService
from binary_mopso_cd.services.ollama_client import LLMCallLogger, OllamaChatClient
from binary_mopso_cd.services.ppdb import PPDBSQLiteIndex, resolve_config_path
from binary_mopso_cd.services.prompt_renderer import DeterministicPromptRenderer
from binary_mopso_cd.services.turbulence import DistilBertFillMaskProvider, TurbulenceService, WordNetPPDBProvider


class SemanticTaskExecutor:
    def __init__(
        self,
        config: RuntimeConfig,
        outdir: Path | None = None,
        embedding_service: EmbeddingService | None = None,
        llm_client: Any | None = None,
        prompt_renderer: DeterministicPromptRenderer | None = None,
        turbulence_provider: Any | None = None,
    ):
        self.config = config
        self.outdir = outdir
        self.prompt_renderer = prompt_renderer or DeterministicPromptRenderer()
        self.embedding_service = embedding_service or self._build_embedding_service(outdir)
        self.llm_client = llm_client or self._build_llm_client(outdir)
        self.turbulence_provider = turbulence_provider or self._build_turbulence_provider()
        if bool(config.get("runtime.eager_load_models", True)):
            self._eager_load_real_resources()

    def execute(self, task: ExecutionTask) -> Any:
        if task.alg_name == ALG_LLM:
            return self._execute_llm(task)
        if task.alg_name == ALG_SBERT:
            texts = list(task.task_params.get("texts", []))
            text_type = str(task.task_params.get("text_type", "component"))
            return self.embedding_service.encode(texts, text_type=text_type)
        if task.alg_name == ALG_PROMPT_RENDERER:
            return self.prompt_renderer.render(
                dict(task.task_params["components"]),
                str(task.task_params.get("domain", self.config.get("experiment.domain"))),
            )
        if task.alg_name == ALG_DISTILBERT:
            return self.turbulence_provider.distilbert_candidates(
                str(task.task_params.get("text", task.task_params.get("component", ""))),
                list(task.task_params["targetSpan"]),
                int(task.alg_params.get("preliminary_top_k", 15)),
                int(task.alg_params.get("max_variants", 5)),
            )
        if task.alg_name == ALG_WORDNET_PPDB:
            return self.turbulence_provider.wordnet_candidates(
                str(task.task_params.get("text", task.task_params.get("component", ""))),
                list(task.task_params["targetSpan"]),
                int(task.alg_params.get("max_variants", 5)),
                bool(task.alg_params.get("use_ppdb", True)),
                target_lemma=task.task_params.get("targetLemma") or task.task_params.get("target_lemma"),
                target_pos=task.task_params.get("targetPos") or task.task_params.get("target_pos"),
                target_word=task.task_params.get("targetWord"),
            )
        raise ValueError(f"Unsupported algorithm: {task.alg_name}")

    async def execute_async(self, task: ExecutionTask) -> Any:
        if task.alg_name == ALG_LLM:
            return await self._execute_llm_async(task)
        return self.execute(task)

    def _execute_llm(self, task: ExecutionTask) -> Any:
        system_prompt, user_prompt = build_messages(task.semantic_task, task.task_params)
        options = {
            "temperature": float(task.alg_params.get("temperature", 0.6)),
            "top_p": float(task.alg_params.get("top_p", 0.9)),
        }
        raw = self.llm_client.chat(
            task_id=task.task_id,
            semantic_task=task.semantic_task,
            model=str(task.alg_params["model"]),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            options=options,
            response_format=response_format_for_task(task.semantic_task),
        )
        return parse_task_result(task.semantic_task, raw)

    async def _execute_llm_async(self, task: ExecutionTask) -> Any:
        system_prompt, user_prompt = build_messages(task.semantic_task, task.task_params)
        options = {
            "temperature": float(task.alg_params.get("temperature", 0.6)),
            "top_p": float(task.alg_params.get("top_p", 0.9)),
        }
        raw = await self.llm_client.chat_async(
            task_id=task.task_id,
            semantic_task=task.semantic_task,
            model=str(task.alg_params["model"]),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            options=options,
            response_format=response_format_for_task(task.semantic_task),
        )
        return parse_task_result(task.semantic_task, raw)

    def save_caches(self) -> None:
        self.embedding_service.cache.save()

    def _build_embedding_service(self, outdir: Path | None) -> EmbeddingService:
        model_alias = str(self.config.get("models.sbert.default"))
        resolved = self.config.get(f"models.sbert.alternatives.{model_alias}", model_alias)
        cache_path = None
        if outdir is not None:
            cache_path = outdir / str(self.config.get("runtime.embedding_cache_file", "embedding_cache.json"))
        return EmbeddingService(
            model_name=model_alias,
            resolved_model_name=resolved,
            batch_size=int(self.config.get("models.sbert.batch_size", 64)),
            config_version=str(self.config.get("models.sbert.config_version", "sbert-v1")),
            cache=EmbeddingCache(cache_path),
        )

    def _build_llm_client(self, outdir: Path | None) -> Any:
        logger = LLMCallLogger(None if outdir is None else outdir / "llm_calls.jsonl")
        return OllamaChatClient(
            host=str(self.config.get("ollama.host")),
            timeout_seconds=int(self.config.get("ollama.timeout_seconds", 120)),
            think=self.config.get("ollama.think", False),
            logger=logger,
        )

    def _build_turbulence_provider(self) -> Any:
        distilbert = DistilBertFillMaskProvider(str(self.config.get("models.distilbert.model")))
        ppdb = None
        if bool(self.config.get("models.ppdb.enabled", True)):
            ppdb = PPDBSQLiteIndex(
                resolve_config_path(str(self.config.get("models.ppdb.index_path"))),
                resolve_config_path(self.config.get("models.ppdb.source_path")),
            )
        wordnet = WordNetPPDBProvider(ppdb, use_wordnet=bool(self.config.get("models.wordnet.enabled", True)))
        return TurbulenceService(distilbert, wordnet, str(self.config.get("models.spacy.model", "en_core_web_sm")))

    def _eager_load_real_resources(self) -> None:
        _ = self.embedding_service.model
        if hasattr(self.turbulence_provider, "distilbert"):
            _ = self.turbulence_provider.distilbert.pipeline
        if hasattr(self.turbulence_provider, "wordnet"):
            _ = self.turbulence_provider.wordnet.wordnet
        if hasattr(self.turbulence_provider, "nlp"):
            _ = self.turbulence_provider.nlp
