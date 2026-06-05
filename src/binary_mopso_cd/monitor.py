from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from binary_mopso_cd.entities import Solution


@dataclass(slots=True)
class MonitorResult:
    metrics: dict[str, Any]
    overhead_seconds: float


class ObservationalMonitor:
    def __init__(self, enabled: bool, spacy_model: str = "en_core_web_sm", kmeans_clusters: int = 3):
        self.enabled = enabled
        self.spacy_model = spacy_model
        self.kmeans_clusters = kmeans_clusters
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
        embeddings = np.asarray([solution.embedding for solution in solutions if solution.embedding is not None], dtype=float)
        inertia = kmeans_global_inertia(embeddings, self.kmeans_clusters)
        labels: list[str] = []
        for solution in solutions:
            for entity in self.nlp(solution.generated_text).ents:
                labels.append(entity.label_)
        entropy = entity_entropy(labels)
        overhead = time.perf_counter() - started
        return MonitorResult({"generation": generation, "kmeans_inertia": inertia, "entity_entropy": entropy}, overhead)


def entity_entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    return float(-sum((count / total) * np.log(count / total) for count in counts.values()))


def kmeans_global_inertia(embeddings: np.ndarray, kmeans_clusters: int) -> float:
    matrix = np.asarray(embeddings, dtype=float)
    if len(matrix) < 2:
        return 0.0
    from sklearn.cluster import KMeans

    clusters = min(int(kmeans_clusters), len(matrix))
    return float(KMeans(n_clusters=clusters, n_init="auto", random_state=0).fit(matrix).inertia_)
