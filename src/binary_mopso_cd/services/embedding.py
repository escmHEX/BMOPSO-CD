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
        missing_by_key: dict[str, list[int]] = {}
        for idx, value in enumerate(result):
            if value is None:
                missing_by_key.setdefault(keys[idx], []).append(idx)
        if missing_by_key:
            first_missing_indices = [indices[0] for indices in missing_by_key.values()]
            missing_texts = [texts[idx] for idx in first_missing_indices]
            embeddings = self.model.encode(
                missing_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for idx, embedding in zip(first_missing_indices, embeddings, strict=True):
                vector = [float(x) for x in embedding]
                metadata = {
                    "text_type": text_type,
                    "model_name": self.model_name,
                    "resolved_model_name": self.resolved_model_name,
                    "config_version": self.config_version,
                    "canonical_text": canonical_text(texts[idx]),
                }
                self.cache.set(keys[idx], vector, metadata)
                for duplicate_idx in missing_by_key[keys[idx]]:
                    result[duplicate_idx] = vector
        return np.asarray(result, dtype=float)

    def similarity(self, left: str, right: str, text_type: str = "component") -> float:
        embeddings = self.encode([left, right], text_type=text_type)
        if embeddings.shape[0] < 2:
            return 0.0
        return float(np.dot(embeddings[0], embeddings[1]))
