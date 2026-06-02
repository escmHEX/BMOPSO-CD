from __future__ import annotations

from pathlib import Path
from typing import Any

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.llm_prompts import build_messages, parse_task_result
from binary_mopso_cd.router import (
    ALG_DISTILBERT,
    ALG_LLM,
    ALG_PROMPT_RENDERER,
    ALG_SBERT,
    ALG_WORDNET_PPDB,
    ExecutionTask,
)
from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService, fake_embedding_service
from binary_mopso_cd.services.ollama_client import FakeLLMClient, LLMCallLogger, OllamaChatClient
from binary_mopso_cd.services.prompt_renderer import DeterministicPromptRenderer
from binary_mopso_cd.services.turbulence import DistilBertFillMaskProvider, FakeTurbulenceProvider, WordNetPPDBProvider


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
        backend = str(config.get("runtime.backend", "real"))
        self.embedding_service = embedding_service or self._build_embedding_service(backend, outdir)
        self.llm_client = llm_client or self._build_llm_client(backend, outdir)
        self.turbulence_provider = turbulence_provider or self._build_turbulence_provider(backend)
        if backend == "real" and bool(config.get("runtime.eager_load_models", True)):
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
                str(task.task_params["text"]),
                int(task.task_params["target_index"]),
                int(task.alg_params.get("preliminary_top_k", 15)),
                int(task.alg_params.get("max_variants", 5)),
            )
        if task.alg_name == ALG_WORDNET_PPDB:
            return self.turbulence_provider.wordnet_candidates(
                str(task.task_params["text"]),
                int(task.task_params["target_index"]),
                int(task.alg_params.get("max_variants", 5)),
                bool(task.alg_params.get("use_ppdb", True)),
            )
        raise ValueError(f"Unsupported algorithm: {task.alg_name}")

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
        )
        return parse_task_result(task.semantic_task, raw)

    def save_caches(self) -> None:
        self.embedding_service.cache.save()

    def _build_embedding_service(self, backend: str, outdir: Path | None) -> EmbeddingService:
        if backend == "fake":
            return fake_embedding_service(str(self.config.get("models.sbert.config_version", "test-v1")))
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

    def _build_llm_client(self, backend: str, outdir: Path | None) -> Any:
        logger = LLMCallLogger(None if outdir is None else outdir / "llm_calls.jsonl")
        if backend == "fake":
            return FakeLLMClient(logger)
        return OllamaChatClient(
            host=str(self.config.get("ollama.host")),
            timeout_seconds=int(self.config.get("ollama.timeout_seconds", 120)),
            logger=logger,
        )

    def _build_turbulence_provider(self, backend: str) -> Any:
        if backend == "fake":
            return FakeTurbulenceProvider()
        distilbert = DistilBertFillMaskProvider(str(self.config.get("models.distilbert.model")))
        ppdb_path = Path(str(self.config.get("models.ppdb.index_path")))
        wordnet = WordNetPPDBProvider(ppdb_path, use_wordnet=bool(self.config.get("models.wordnet.enabled", True)))

        class Provider:
            def __init__(self):
                self.distilbert = distilbert
                self.wordnet = wordnet

            def distilbert_candidates(self, text: str, target_index: int, preliminary_top_k: int, max_variants: int):
                return self.distilbert.candidates(text, target_index, preliminary_top_k, max_variants)

            def wordnet_candidates(self, text: str, target_index: int, max_variants: int, use_ppdb: bool = True):
                return self.wordnet.candidates(text, target_index, max_variants, use_ppdb)

        return Provider()

    def _eager_load_real_resources(self) -> None:
        _ = self.embedding_service.model
        if hasattr(self.turbulence_provider, "distilbert"):
            _ = self.turbulence_provider.distilbert.pipeline
        if hasattr(self.turbulence_provider, "wordnet"):
            _ = self.turbulence_provider.wordnet.wordnet
