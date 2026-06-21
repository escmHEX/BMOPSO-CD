from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from binary_mopso_cd.entities import Solution


EVOLMD_MO_EMPTY_TEXT_PLACEHOLDER = "[texto vacío]"
EVOLMD_MO_KMEANS_CLUSTERS = 5
MONITOR_GENERATED_TEXT_TYPE = "monitor_generated_text"
POS_KEPT_FOR_ENTITY_ENTROPY = {"NOUN", "VERB", "ADJ"}


@dataclass(slots=True)
class MonitorResult:
    metrics: dict[str, Any]
    overhead_seconds: float


class ObservationalMonitor:
    def __init__(self, enabled: bool, spacy_model: str = "en_core_web_sm", embedding_service: Any | None = None):
        self.enabled = enabled
        self.spacy_model = spacy_model
        self.embedding_service = embedding_service
        self._nlp: Any | None = None

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def observe(self, generation: int, solutions: list[Solution]) -> MonitorResult:
        if not self.enabled:
            return MonitorResult({"generation": generation}, 0.0)
        started = time.perf_counter()
        generated_texts = [solution.generated_text for solution in solutions]
        inertia = calculate_kmeans_inertia(generated_texts, self.embedding_service)
        entropy = calculate_entity_entropy(generated_texts, self.nlp)
        overhead = time.perf_counter() - started
        return MonitorResult({"generation": generation, "kmeans_inertia": inertia, "entity_entropy": entropy}, overhead)


def calculate_kmeans_inertia(generated_texts: list[str], embedding_service: Any) -> float:
    texts = [text if text.strip() else EVOLMD_MO_EMPTY_TEXT_PLACEHOLDER for text in generated_texts]
    sample_count = len(texts)
    if sample_count == 0:
        return 0.0
    if embedding_service is None:
        raise ValueError("ObservationalMonitor requires an embedding_service when enabled")
    embeddings = embedding_service.encode(texts, text_type=MONITOR_GENERATED_TEXT_TYPE)
    clusters = min(EVOLMD_MO_KMEANS_CLUSTERS, sample_count)
    if clusters == 0:
        return 0.0
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=clusters, random_state=0, n_init=10)
    model.fit(np.asarray(embeddings, dtype=float))
    return float(model.inertia_ / sample_count)


def calculate_entity_entropy(generated_texts: list[str], nlp: Any) -> float:
    total_tokens_count = 0
    concepts: list[str] = []
    for doc in nlp.pipe(generated_texts):
        total_tokens_count += len(doc)
        for token in doc:
            if token.pos_ in POS_KEPT_FOR_ENTITY_ENTROPY:
                concepts.append(token.lemma_.lower())
    if not concepts:
        return 0.0
    if total_tokens_count <= 1:
        return 0.0
    counts = Counter(concepts)
    total = sum(counts.values())
    raw_entropy = -sum((count / total) * np.log2(count / total) for count in counts.values())
    return float(raw_entropy / np.log2(total_tokens_count))

