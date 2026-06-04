from __future__ import annotations

import re
import string
from typing import Any

from binary_mopso_cd.services.ppdb import PPDBSQLiteIndex
from binary_mopso_cd.utils import canonical_text, unique_preserve_order


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
ALLOWED_TURBULENCE_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


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


class WordNetPPDBProvider:
    def __init__(self, ppdb: PPDBSQLiteIndex | None, use_wordnet: bool = True):
        self.ppdb = ppdb
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

    def candidates(
        self,
        text: str,
        target_index: int,
        max_variants: int,
        use_ppdb: bool = True,
        target_lemma: str | None = None,
        target_pos: str | None = None,
    ) -> list[str]:
        tokens = tokenize_component(text)
        if target_index < 0 or target_index >= len(tokens):
            return []
        target = tokens[target_index]
        replacements: list[str] = []
        if self.use_wordnet:
            wordnet_pos = spacy_pos_to_wordnet(target_pos)
            for synset in self.wordnet.synsets(target_lemma or target, pos=wordnet_pos):
                for lemma in synset.lemmas():
                    replacements.append(lemma.name().replace("_", " "))
        if use_ppdb and len(unique_preserve_order(replacements)) < max_variants:
            replacements.extend(self.ppdb.lookup(target) if self.ppdb else [])
            if target_lemma:
                replacements.extend(self.ppdb.lookup(target_lemma) if self.ppdb else [])
        filtered = []
        for replacement in replacements:
            value = replacement.strip()
            if not value or canonical_text(value) == canonical_text(target):
                continue
            if not TOKEN_RE.fullmatch(value):
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


def spacy_pos_to_wordnet(pos: str | None) -> str | None:
    mapping = {
        "ADJ": "a",
        "ADV": "r",
        "NOUN": "n",
        "PROPN": "n",
        "VERB": "v",
    }
    return mapping.get(str(pos or "").upper())


class TurbulenceService:
    def __init__(
        self,
        distilbert: DistilBertFillMaskProvider,
        wordnet: WordNetPPDBProvider,
        spacy_model: str,
    ):
        self.distilbert = distilbert
        self.wordnet = wordnet
        self.spacy_model = spacy_model
        self._nlp: Any | None = None

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def distilbert_candidates(self, text: str, target_index: int, preliminary_top_k: int, max_variants: int) -> list[str]:
        return self.distilbert.candidates(text, target_index, preliminary_top_k, max_variants)

    def wordnet_candidates(
        self,
        text: str,
        target_index: int,
        max_variants: int,
        use_ppdb: bool = True,
        target_lemma: str | None = None,
        target_pos: str | None = None,
    ) -> list[str]:
        tokens = tokenize_component(text)
        target = tokens[target_index] if 0 <= target_index < len(tokens) else ""
        lemma, pos = (target_lemma, target_pos) if target_lemma or target_pos else self._target_features(text, target)
        return self.wordnet.candidates(
            text,
            target_index,
            max_variants,
            use_ppdb=use_ppdb,
            target_lemma=lemma,
            target_pos=pos,
        )

    def modifiable_units(self, text: str, preferred_pos: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        units = self._all_modifiable_units(text)
        preferred = {pos.upper() for pos in preferred_pos}
        if preferred:
            preferred_units = [unit for unit in units if unit["pos"] in preferred]
            if preferred_units:
                return preferred_units
        return units

    def _all_modifiable_units(self, text: str) -> list[dict[str, Any]]:
        doc = self.nlp(text)
        tokens = tokenize_component(text)
        token_spans = [
            (idx, match.group(0), match.start(), match.end())
            for idx, match in enumerate(TOKEN_RE.finditer(text))
        ]
        used_indices: set[int] = set()
        units: list[dict[str, Any]] = []
        for token in doc:
            if not self._is_modifiable_token(token):
                continue
            target_index = self._target_index_for_token(token, token_spans, used_indices)
            if target_index is None:
                continue
            used_indices.add(target_index)
            pos = str(getattr(token, "pos_", "")).upper()
            start = int(getattr(token, "idx", token_spans[target_index][2]))
            token_text = str(getattr(token, "text", ""))
            units.append(
                {
                    "text": token_text,
                    "lemma": str(getattr(token, "lemma_", token_text)),
                    "pos": pos,
                    "span": [start, start + len(token_text)],
                    "leftTokens": len(units),
                    "rightTokens": 0,
                    "target_index": target_index,
                    "tokens": tokens,
                }
            )
        total_units = len(units)
        for idx, unit in enumerate(units):
            unit["leftTokens"] = idx
            unit["rightTokens"] = total_units - idx - 1
        return units

    def _is_modifiable_token(self, token: Any) -> bool:
        if bool(getattr(token, "is_punct", False)):
            return False
        if bool(getattr(token, "is_space", False)):
            return False
        if bool(getattr(token, "is_stop", False)):
            return False
        if bool(getattr(token, "like_num", False)):
            return False
        if str(getattr(token, "pos_", "")).upper() not in ALLOWED_TURBULENCE_POS:
            return False
        return bool(canonical_text(str(getattr(token, "text", ""))))

    def _target_index_for_token(
        self,
        token: Any,
        token_spans: list[tuple[int, str, int, int]],
        used_indices: set[int],
    ) -> int | None:
        token_text = canonical_text(str(getattr(token, "text", "")))
        token_start = int(getattr(token, "idx", -1))
        token_end = token_start + len(str(getattr(token, "text", "")))
        for idx, text, start, end in token_spans:
            if idx in used_indices:
                continue
            if canonical_text(text) != token_text:
                continue
            if token_start < 0 or (start < token_end and token_start < end):
                return idx
        for idx, text, _start, _end in token_spans:
            if idx not in used_indices and canonical_text(text) == token_text:
                return idx
        return None

    def _target_features(self, text: str, target: str) -> tuple[str | None, str | None]:
        if not target:
            return None, None
        doc = self.nlp(text)
        target_key = canonical_text(target)
        for token in doc:
            if canonical_text(token.text) == target_key:
                return token.lemma_, token.pos_
        return target, None
