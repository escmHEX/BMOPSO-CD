from __future__ import annotations

from pathlib import Path

import pytest

from binary_mopso_cd.config import RuntimeConfig


@pytest.fixture()
def test_config() -> RuntimeConfig:
    config = RuntimeConfig.load(Path("configs/test.yaml"))
    config.set("experiment.n", 2)
    config.set("experiment.iterations", 1)
    config.set("experiment.runs", 1)
    config.set("experiment.seed", 7)
    config.set("runtime.eager_load_models", False)
    config.set("ollama.timeout_seconds", 600)
    config.set("ollama.default_model", "llama3.1:8b")
    config.set("logging.console", False)
    for task in [
        "semantic_anchor_extraction",
        "central_anchor_selection",
        "semantic_pool_generation",
        "semantic_pool_expansion",
        "semantic_component_influence_candidates",
        "synthetic_text_generation",
    ]:
        config.set(f"router.task_models.{task}", "llama3.1:8b")
    config.set("selection.k", 2)
    config.validate()
    return config


@pytest.fixture(scope="session")
def real_embedding_service():
    from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService

    config = RuntimeConfig.load(Path("configs/test.yaml"))
    alias = str(config.get("models.sbert.default"))
    resolved = str(config.get(f"models.sbert.alternatives.{alias}", alias))
    return EmbeddingService(
        model_name=alias,
        resolved_model_name=resolved,
        batch_size=int(config.get("models.sbert.batch_size", 16)),
        config_version=str(config.get("models.sbert.config_version", "sbert-test-v1")),
        cache=EmbeddingCache(),
    )
