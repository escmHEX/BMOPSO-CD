from __future__ import annotations

import math
from dataclasses import dataclass

from binary_mopso_cd.utils import canonical_text


REASON_EMPTY = "empty"
REASON_DUPLICATE = "duplicate_generated_text"
REASON_SAME_AS_REFERENCE = "same_as_reference"
REASON_SENTENCE_LIMIT = "sentence_limit"
REASON_REFUSAL_PHRASE = "refusal_phrase"
REASON_LOW_FIDELITY = "low_fidelity"

REFUSAL_PHRASES = (
    "not a feasible request",
    "cannot tailor them exactly to specific requests",
    "i cannot assist you with that request",
    "i can't assist you with that request",
    "i cannot assist with that request",
    "i cannot comply with that request",
    "i am unable to fulfill your request",
    "i cannot fulfill your request",
    "i cannot proceed with that request",
    "i cannot generate content for that request",
    "i am not allowed to generate that content",
    "i am not permitted to produce that content",
)


@dataclass(frozen=True, slots=True)
class GeneratedTextValidationResult:
    valid: bool
    reason: str | None = None


def contains_refusal_phrase(text: str) -> bool:
    normalized = canonical_text(text)
    return any(phrase in normalized for phrase in REFUSAL_PHRASES)


def generated_sentence_count(text: str) -> int:
    return max(1, text.count(".") + text.count("!") + text.count("?"))


def validate_generated_text(
    text: object,
    reference_text: str,
    accepted_text_keys: set[str] | None = None,
    f1: float | None = None,
    tau_gen_min: float = 0.15,
    max_sentences: int | None = None,
) -> GeneratedTextValidationResult:
    if not isinstance(text, str):
        return GeneratedTextValidationResult(False, REASON_EMPTY)

    normalized = canonical_text(text)
    if not normalized:
        return GeneratedTextValidationResult(False, REASON_EMPTY)
    if max_sentences is not None and generated_sentence_count(text) > max_sentences:
        return GeneratedTextValidationResult(False, REASON_SENTENCE_LIMIT)
    if normalized == canonical_text(reference_text):
        return GeneratedTextValidationResult(False, REASON_SAME_AS_REFERENCE)
    if accepted_text_keys is not None and normalized in accepted_text_keys:
        return GeneratedTextValidationResult(False, REASON_DUPLICATE)
    if f1 is not None:
        fidelity = float(f1)
        if not math.isfinite(fidelity) or fidelity < tau_gen_min:
            return GeneratedTextValidationResult(False, REASON_LOW_FIDELITY)
    if contains_refusal_phrase(text):
        return GeneratedTextValidationResult(False, REASON_REFUSAL_PHRASE)
    return GeneratedTextValidationResult(True)
