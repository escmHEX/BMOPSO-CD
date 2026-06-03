from __future__ import annotations

import json
from random import Random

import numpy as np
import pytest

from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.generated_text_validation import (
    REASON_DUPLICATE,
    REASON_EMPTY,
    REASON_LOW_FIDELITY,
    REASON_REFUSAL_PHRASE,
    REASON_SAME_AS_REFERENCE,
    REASON_SENTENCE_LIMIT,
    validate_generated_text,
)
from binary_mopso_cd.initialization import InitialPopulationBuilder
from binary_mopso_cd.mopso import BinaryMOPSOCDEngine
from binary_mopso_cd.utils import canonical_text


class StubEmbeddingService:
    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=float)

    def _vector(self, text: str) -> list[float]:
        normalized = canonical_text(text)
        if "bad generated" in normalized:
            return [-1.0, 0.0]
        return [1.0, 0.0]


class StubExecutor:
    def __init__(self, outdir, texts: list[str] | None = None):
        self.outdir = outdir
        self.embedding_service = StubEmbeddingService()
        self._texts = list(texts or [])

    def execute(self, _task):
        if not self._texts:
            raise AssertionError("No execution result configured")
        return self._texts.pop(0)


class PassthroughRouter:
    def route(self, task):
        return task


def test_generated_text_validator_reports_expected_reasons():
    assert validate_generated_text("", "reference").reason == REASON_EMPTY
    assert validate_generated_text("reference", "reference").reason == REASON_SAME_AS_REFERENCE
    assert (
        validate_generated_text("duplicate", "reference", accepted_text_keys={canonical_text("duplicate")}).reason
        == REASON_DUPLICATE
    )
    assert validate_generated_text("One. Two.", "reference", max_sentences=1).reason == REASON_SENTENCE_LIMIT
    assert validate_generated_text("plausible text", "reference", f1=0.01).reason == REASON_LOW_FIDELITY
    assert (
        validate_generated_text("I cannot assist you with that request", "reference", f1=1.0).reason
        == REASON_REFUSAL_PHRASE
    )
    assert validate_generated_text("plausible text", "reference", f1=0.2).valid


def test_initialization_rejects_low_fidelity_generated_text(test_config, tmp_path):
    test_config.set("experiment.n", 1)
    builder = InitialPopulationBuilder(
        test_config,
        PassthroughRouter(),
        StubExecutor(tmp_path, ["bad generated text.", "valid generated text."]),
        Random(1),
    )
    items = [
        (SemanticVector({"role": "role", "topic": "topic", "action": "action"}), "prompt bad", 0.2),
        (SemanticVector({"role": "role", "topic": "topic", "action": "warn"}), "prompt good", 0.1),
    ]

    generated = builder._generate_texts(items, "reference")

    assert len(generated) == 1
    assert generated[0].generated_text == "valid generated text."
    assert generated[0].objectives is not None
    assert generated[0].objectives.f1 == pytest.approx(1.0)
    rows = [
        json.loads(line)
        for line in (tmp_path / "initialization_rejections.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["phase"] == "initialization"
    assert rows[0]["reason"] == REASON_LOW_FIDELITY


def test_mopso_rejects_invalid_generated_text_and_preserves_velocity(test_config, tmp_path):
    test_config.set("mopso.p_tur_max", 1.0)
    test_config.set("mopso.p_tur_min", 1.0)
    test_config.set("mopso.dmax", 1)
    rng = Random(2)
    particle = Solution(
        SemanticVector({"role": "role", "topic": "topic", "action": "action"}),
        "old prompt",
        "old generated",
        Objectives(0.5, 0.5),
        velocity={"role": 2.0, "topic": 2.0, "action": 2.0},
        initial_components={"role": "role", "topic": "topic", "action": "action"},
        changed=False,
    )
    engine = BinaryMOPSOCDEngine(
        test_config,
        PassthroughRouter(),
        StubExecutor(tmp_path),
        rng,
        tmp_path,
        "reference",
    )
    engine._candidate_for_mode = lambda component, *_args: f"{component} changed"
    engine._render_prompt = lambda _vector: "new prompt"
    engine._generate_text = lambda _prompt: "not a feasible request"

    updated = engine._update_particle(particle, particle.clone(), particle.clone(), generation=1)

    assert not updated.changed
    assert updated.vector.components == particle.vector.components
    assert updated.prompt == particle.prompt
    assert updated.generated_text == particle.generated_text
    assert updated.velocity["role"] == pytest.approx(1.8)
    rows = [
        json.loads(line)
        for line in (tmp_path / "optimization_rejections.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["phase"] == "optimization"
    assert rows[0]["reason"] == REASON_REFUSAL_PHRASE
