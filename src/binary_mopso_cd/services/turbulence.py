from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Any

from binary_mopso_cd.utils import canonical_text, unique_preserve_order


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def tokenize_component(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


class DistilBertFillMaskProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipeline: Any | None = None

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline("fill-mask", model=self.model_name)
        return self._pipeline

    def candidates(self, text: str, target_index: int, preliminary_top_k: int, max_variants: int) -> list[str]:
        tokens = tokenize_component(text)
        if target_index < 0 or target_index >= len(tokens):
            return []
        target = tokens[target_index]
        masked_tokens = list(tokens)
        masked_tokens[target_index] = self.pipeline.tokenizer.mask_token
        masked_text = " ".join(masked_tokens)
        predictions = self.pipeline(masked_text, top_k=preliminary_top_k)
        replacements: list[str] = []
        for prediction in predictions:
            token = str(prediction.get("token_str", "")).strip()
            if not token or token.startswith("##"):
                continue
            if any(char in string.punctuation for char in token):
                continue
            if canonical_text(token) == canonical_text(target):
                continue
            replacements.append(token)
        return build_variants(tokens, target_index, unique_preserve_order(replacements), max_variants)


class PPDBIndex:
    def __init__(self, path: Path | None):
        self.path = path
        self.entries: dict[str, list[str]] = {}
        if path and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.entries = {canonical_text(key): list(value) for key, value in payload.get("entries", {}).items()}

    def lookup(self, word: str) -> list[str]:
        return list(self.entries.get(canonical_text(word), []))


class WordNetPPDBProvider:
    def __init__(self, ppdb_path: Path | None, use_wordnet: bool = True):
        self.ppdb = PPDBIndex(ppdb_path)
        self.use_wordnet = use_wordnet
        self._wordnet: Any | None = None

    @property
    def wordnet(self) -> Any:
        if self._wordnet is None:
            import nltk
            from nltk.corpus import wordnet

            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
            self._wordnet = wordnet
        return self._wordnet

    def candidates(self, text: str, target_index: int, max_variants: int, use_ppdb: bool = True) -> list[str]:
        tokens = tokenize_component(text)
        if target_index < 0 or target_index >= len(tokens):
            return []
        target = tokens[target_index]
        replacements: list[str] = []
        if self.use_wordnet:
            for synset in self.wordnet.synsets(target):
                for lemma in synset.lemmas():
                    replacements.append(lemma.name().replace("_", " "))
        if use_ppdb and len(unique_preserve_order(replacements)) < max_variants:
            replacements.extend(self.ppdb.lookup(target))
        filtered = []
        for replacement in replacements:
            value = replacement.strip()
            if not value or canonical_text(value) == canonical_text(target):
                continue
            if not TOKEN_RE.fullmatch(value.replace(" ", "")):
                continue
            filtered.append(value)
        return build_variants(tokens, target_index, unique_preserve_order(filtered), max_variants)


def build_variants(tokens: list[str], target_index: int, replacements: list[str], max_variants: int) -> list[str]:
    variants = []
    for replacement in replacements:
        next_tokens = list(tokens)
        next_tokens[target_index] = replacement
        variants.append(" ".join(next_tokens))
        if len(variants) >= max_variants:
            break
    return variants


class FakeTurbulenceProvider:
    def distilbert_candidates(self, text: str, target_index: int, preliminary_top_k: int, max_variants: int) -> list[str]:
        tokens = tokenize_component(text)
        if not tokens:
            return []
        idx = max(0, min(target_index, len(tokens) - 1))
        return build_variants(tokens, idx, ["urgent", "clear", "public", "safe"], max_variants)

    def wordnet_candidates(self, text: str, target_index: int, max_variants: int, use_ppdb: bool = True) -> list[str]:
        return self.distilbert_candidates(text, target_index, max_variants * 3, max_variants)

