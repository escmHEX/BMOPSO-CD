from __future__ import annotations

import math

import numpy as np
import pytest

from binary_mopso_cd.entities import SemanticVector, Solution
from binary_mopso_cd.monitor import (
    EVOLMD_MO_EMPTY_TEXT_PLACEHOLDER,
    ObservationalMonitor,
    calculate_entity_entropy,
    calculate_kmeans_inertia,
)


def test_disabled_monitor_has_no_overhead_dependency():
    result = ObservationalMonitor(enabled=False).observe(1, [])
    assert result.metrics == {"generation": 1}
    assert result.overhead_seconds == 0.0


def test_kmeans_inertia_matches_evolmd_mo_text_cleaning_and_parameters(monkeypatch):
    captured: dict[str, object] = {}

    class StubEmbeddingService:
        def encode(self, texts: list[str], text_type: str) -> np.ndarray:
            captured["texts"] = list(texts)
            captured["text_type"] = text_type
            return np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [1.0, 2.0],
                ],
                dtype=float,
            )

    class StubKMeans:
        def __init__(self, *, n_clusters: int, random_state: int, n_init: int):
            captured["n_clusters"] = n_clusters
            captured["random_state"] = random_state
            captured["n_init"] = n_init
            self.inertia_ = 12.0

        def fit(self, embeddings: np.ndarray) -> "StubKMeans":
            captured["fit_embeddings"] = embeddings.copy()
            return self

    import sklearn.cluster

    monkeypatch.setattr(sklearn.cluster, "KMeans", StubKMeans)

    value = calculate_kmeans_inertia(
        ["alpha", "   ", "beta", "gamma", "delta", "epsilon"],
        StubEmbeddingService(),
    )

    assert captured["texts"] == ["alpha", EVOLMD_MO_EMPTY_TEXT_PLACEHOLDER, "beta", "gamma", "delta", "epsilon"]
    assert captured["text_type"] == "monitor_generated_text"
    assert captured["n_clusters"] == 5
    assert captured["random_state"] == 0
    assert captured["n_init"] == 10
    assert value == pytest.approx(2.0)


def test_entity_entropy_matches_evolmd_mo_concept_lemmas_and_token_normalization():
    class Token:
        def __init__(self, pos_: str, lemma_: str):
            self.pos_ = pos_
            self.lemma_ = lemma_

    class StubNLP:
        def pipe(self, _texts: list[str]):
            return [
                [
                    Token("NOUN", "Bridge"),
                    Token("VERB", "Warn"),
                    Token("ADJ", "Urgent"),
                    Token("ADV", "ignored"),
                ],
                [
                    Token("NOUN", "Bridge"),
                    Token("VERB", "Evacuate"),
                ],
            ]

    value = calculate_entity_entropy(["first", "second"], StubNLP())

    counts = [2, 1, 1, 1]
    total_concepts = sum(counts)
    raw_entropy = -sum((count / total_concepts) * math.log2(count / total_concepts) for count in counts)
    assert value == pytest.approx(raw_entropy / math.log2(6))


def test_enabled_monitor_observes_generated_texts_only(monkeypatch):
    captured: dict[str, object] = {}

    def stub_kmeans(texts, embedding_service):
        captured["kmeans_texts"] = list(texts)
        captured["embedding_service"] = embedding_service
        return 1.5

    def stub_entropy(texts, nlp):
        captured["entropy_texts"] = list(texts)
        captured["nlp"] = nlp
        return 0.75

    monkeypatch.setattr("binary_mopso_cd.monitor.calculate_kmeans_inertia", stub_kmeans)
    monkeypatch.setattr("binary_mopso_cd.monitor.calculate_entity_entropy", stub_entropy)
    embedding_service = object()
    nlp = object()
    monitor = ObservationalMonitor(enabled=True, embedding_service=embedding_service)
    monitor._nlp = nlp

    result = monitor.observe(
        3,
        [
            Solution(SemanticVector({"role": "a"}), "prompt-a", "generated a"),
            Solution(SemanticVector({"role": "b"}), "prompt-b", "generated b"),
        ],
    )

    assert captured["kmeans_texts"] == ["generated a", "generated b"]
    assert captured["entropy_texts"] == ["generated a", "generated b"]
    assert captured["embedding_service"] is embedding_service
    assert captured["nlp"] is nlp
    assert result.metrics == {"generation": 3, "kmeans_inertia": 1.5, "entity_entropy": 0.75}
    assert result.overhead_seconds >= 0.0
