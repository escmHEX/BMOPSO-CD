from __future__ import annotations

import re
import string
from typing import Any

from binary_mopso_cd.services.ppdb import PPDBSQLiteIndex
from binary_mopso_cd.utils import canonical_text, unique_preserve_order


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
ALLOWED_TURBULENCE_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
TargetSpan = list[int] | tuple[int, int]


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

    def candidates(self, text: str, target_span: TargetSpan, preliminary_top_k: int, max_variants: int) -> list[str]:
        span = normalize_target_span(text, target_span)
        if span is None:
            return []
        start, end = span
        target = text[start:end]
        masked_text = replace_span(text, span, self.pipeline.tokenizer.mask_token)
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
        return build_span_variants(text, span, unique_preserve_order(replacements), max_variants)


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
        target_span: TargetSpan,
        max_variants: int,
        use_ppdb: bool = True,
        target_lemma: str | None = None,
        target_pos: str | None = None,
        target_word: str | None = None,
    ) -> list[str]:
        span = normalize_target_span(text, target_span)
        if span is None:
            return []
        start, end = span
        target = target_word or text[start:end]
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
        return build_span_variants(text, span, unique_preserve_order(filtered), max_variants)


def normalize_target_span(text: str, target_span: TargetSpan) -> tuple[int, int] | None:
    if len(target_span) != 2:
        return None
    start = int(target_span[0])
    end = int(target_span[1])
    if start < 0 or end <= start or end > len(text):
        return None
    if not text[start:end].strip():
        return None
    return start, end


def replace_span(text: str, target_span: tuple[int, int], replacement: str) -> str:
    start, end = target_span
    return f"{text[:start]}{replacement}{text[end:]}"


def build_span_variants(text: str, target_span: tuple[int, int], replacements: list[str], max_variants: int) -> list[str]:
    variants = []
    for replacement in replacements:
        variants.append(replace_span(text, target_span, replacement))
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

    def distilbert_candidates(
        self,
        text: str,
        target_span: TargetSpan,
        preliminary_top_k: int,
        max_variants: int,
    ) -> list[str]:
        return self.distilbert.candidates(text, target_span, preliminary_top_k, max_variants)

    def wordnet_candidates(
        self,
        text: str,
        target_span: TargetSpan,
        max_variants: int,
        use_ppdb: bool = True,
        target_lemma: str | None = None,
        target_pos: str | None = None,
        target_word: str | None = None,
    ) -> list[str]:
        span = normalize_target_span(text, target_span)
        if span is None:
            return []
        start, end = span
        target = target_word or text[start:end]
        lemma, pos = (target_lemma, target_pos) if target_lemma or target_pos else self._target_features(text, target)
        return self.wordnet.candidates(
            text,
            span,
            max_variants,
            use_ppdb=use_ppdb,
            target_lemma=lemma,
            target_pos=pos,
            target_word=target,
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
        units: list[dict[str, Any]] = []
        modifiable_tokens = [token for token in doc if self._is_modifiable_token(token)]
        total_units = len(modifiable_tokens)
        for idx, token in enumerate(modifiable_tokens):
            if not self._is_modifiable_token(token):
                continue
            pos = str(getattr(token, "pos_", "")).upper()
            start = int(getattr(token, "idx", -1))
            token_text = str(getattr(token, "text", ""))
            if start < 0:
                continue
            units.append(
                {
                    "text": token_text,
                    "lemma": str(getattr(token, "lemma_", token_text)),
                    "pos": pos,
                    "span": [start, start + len(token_text)],
                    "leftTokens": idx,
                    "rightTokens": total_units - idx - 1,
                }
            )
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

    def _target_features(self, text: str, target: str) -> tuple[str | None, str | None]:
        if not target:
            return None, None
        doc = self.nlp(text)
        target_key = canonical_text(target)
        for token in doc:
            if canonical_text(token.text) == target_key:
                return token.lemma_, token.pos_
        return target, None
