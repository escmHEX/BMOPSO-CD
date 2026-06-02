from __future__ import annotations

from binary_mopso_cd.services.embedding import EmbeddingCache, fake_embedding_service


def test_embedding_cache_key_includes_text_type_and_version():
    cache = EmbeddingCache()
    key_component = cache.key("component", "fake", "v1", "Same text")
    key_prompt = cache.key("prompt", "fake", "v1", "Same text")
    key_version = cache.key("component", "fake", "v2", "Same text")
    assert key_component != key_prompt
    assert key_component != key_version


def test_embedding_service_batches_missing_texts_once():
    service = fake_embedding_service()
    first = service.encode(["a repeated text", "a repeated text"], text_type="generated_text")
    cache_size = service.cache.size
    second = service.encode(["a repeated text"], text_type="generated_text")
    assert first.shape[0] == 2
    assert second.shape[0] == 1
    assert service.cache.size == cache_size

