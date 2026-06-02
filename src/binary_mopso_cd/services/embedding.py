from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from binary_mopso_cd.utils import canonical_text, stable_digest


class EmbeddingCache:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._items: dict[str, dict[str, Any]] = {}
        if path and path.exists():
            self.load(path)

    def key(self, text_type: str, model_name: str, config_version: str, text: str) -> str:
        return stable_digest(
            {
                "text_type": text_type,
                "model_name": model_name,
                "config_version": config_version,
                "canonical_text": canonical_text(text),
            }
        )

    def get(self, key: str) -> list[float] | None:
        item = self._items.get(key)
        if item is None:
            return None
        return list(item["embedding"])

    def set(self, key: str, embedding: Iterable[float], metadata: dict[str, Any]) -> None:
        self._items[key] = {"embedding": [float(x) for x in embedding], "metadata": metadata}

    def load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._items = dict(payload.get("items", {}))

    def save(self, path: Path | None = None) -> None:
        target = path or self.path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False)

    def to_dict(self) -> dict[str, Any]:
        return {"items": self._items}

    def restore(self, payload: dict[str, Any]) -> None:
        self._items = dict(payload.get("items", {}))

    @property
    def size(self) -> int:
        return len(self._items)


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        resolved_model_name: str | None = None,
        batch_size: int = 64,
        config_version: str = "sbert-v1",
        cache: EmbeddingCache | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ):
        self.model_name = model_name
        self.resolved_model_name = resolved_model_name or model_name
        self.batch_size = int(batch_size)
        self.config_version = config_version
        self.cache = cache or EmbeddingCache()
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            if self._model_factory is not None:
                self._model = self._model_factory(self.resolved_model_name)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.resolved_model_name)
        return self._model

    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=float)
        keys = [self.cache.key(text_type, self.model_name, self.config_version, text) for text in texts]
        result: list[list[float] | None] = [self.cache.get(key) for key in keys]
        missing_indices = [idx for idx, value in enumerate(result) if value is None]
        if missing_indices:
            missing_texts = [texts[idx] for idx in missing_indices]
            embeddings = self.model.encode(
                missing_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for idx, embedding in zip(missing_indices, embeddings, strict=True):
                vector = [float(x) for x in embedding]
                metadata = {
                    "text_type": text_type,
                    "model_name": self.model_name,
                    "resolved_model_name": self.resolved_model_name,
                    "config_version": self.config_version,
                    "canonical_text": canonical_text(texts[idx]),
                }
                self.cache.set(keys[idx], vector, metadata)
                result[idx] = vector
        return np.asarray(result, dtype=float)

    def similarity(self, left: str, right: str, text_type: str = "component") -> float:
        embeddings = self.encode([left, right], text_type=text_type)
        if embeddings.shape[0] < 2:
            return 0.0
        return float(np.dot(embeddings[0], embeddings[1]))


class FakeEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = stable_digest({"text": canonical_text(text)})
            values = [int(digest[i : i + 8], 16) / 0xFFFFFFFF for i in range(0, 64, 8)]
            vector = np.asarray(values, dtype=float)
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=float)


def fake_embedding_service(config_version: str = "test-v1") -> EmbeddingService:
    return EmbeddingService(
        model_name="fake-sbert",
        resolved_model_name="fake-sbert",
        batch_size=16,
        config_version=config_version,
        cache=EmbeddingCache(),
        model_factory=lambda _: FakeEmbeddingModel(),
    )
