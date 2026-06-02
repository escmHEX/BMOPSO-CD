from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from binary_mopso_cd.entities import Solution
from binary_mopso_cd.services.embedding import EmbeddingService
from binary_mopso_cd.utils import canonical_text


@dataclass
class ComponentMemoryIndex:
    components: list[str]
    embedding_service: EmbeddingService
    texts: dict[str, list[str]] = field(default_factory=dict)
    keys: dict[str, set[str]] = field(default_factory=dict)
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for component in self.components:
            self.texts.setdefault(component, [])
            self.keys.setdefault(component, {canonical_text(text) for text in self.texts[component]})
            self.embeddings.setdefault(component, np.empty((0, 0), dtype=float))

    @classmethod
    def from_snapshot(
        cls,
        components: list[str],
        embedding_service: EmbeddingService,
        snapshot: dict[str, list[str]] | None,
    ) -> "ComponentMemoryIndex":
        index = cls(components=components, embedding_service=embedding_service)
        if snapshot:
            index.add_components({component: list(values) for component, values in snapshot.items()})
        return index

    def add_solutions(self, solutions: list[Solution]) -> None:
        by_component: dict[str, list[str]] = {component: [] for component in self.components}
        for solution in solutions:
            for component in self.components:
                value = solution.vector.components.get(component)
                if value:
                    by_component[component].append(value)
        self.add_components(by_component)

    def add_components(self, values: dict[str, list[str]]) -> None:
        for component, candidates in values.items():
            if component not in self.texts:
                self.texts[component] = []
                self.keys[component] = set()
                self.embeddings[component] = np.empty((0, 0), dtype=float)
            new_texts: list[str] = []
            for candidate in candidates:
                key = canonical_text(candidate)
                if key and key not in self.keys[component]:
                    self.keys[component].add(key)
                    self.texts[component].append(candidate)
                    new_texts.append(candidate)
            if not new_texts:
                continue
            new_embeddings = self.embedding_service.encode(new_texts, text_type="component")
            current = self.embeddings[component]
            self.embeddings[component] = new_embeddings if current.size == 0 else np.vstack([current, new_embeddings])

    def max_similarity(self, component: str, candidate_embeddings: np.ndarray) -> np.ndarray:
        memory_embeddings = self.embeddings.get(component)
        if candidate_embeddings.size == 0:
            return np.zeros(candidate_embeddings.shape[0], dtype=float)
        if memory_embeddings is None or memory_embeddings.size == 0:
            return np.zeros(candidate_embeddings.shape[0], dtype=float)
        return np.max(candidate_embeddings @ memory_embeddings.T, axis=1)

    def to_snapshot(self) -> dict[str, list[str]]:
        return {component: list(values) for component, values in self.texts.items()}
