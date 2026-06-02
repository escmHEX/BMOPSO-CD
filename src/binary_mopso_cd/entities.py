from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from binary_mopso_cd.utils import canonical_text


@dataclass(slots=True)
class SemanticVector:
    components: dict[str, str]

    def copy(self) -> "SemanticVector":
        return SemanticVector(dict(self.components))

    def signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((name, canonical_text(value)) for name, value in self.components.items()))


@dataclass(slots=True)
class Objectives:
    f1: float
    f2: float

    def values(self) -> tuple[float, float]:
        return self.f1, self.f2


@dataclass(slots=True)
class Solution:
    vector: SemanticVector
    prompt: str
    generated_text: str
    objectives: Objectives | None = None
    solution_id: str = field(default_factory=lambda: uuid4().hex)
    velocity: dict[str, float] = field(default_factory=dict)
    last_guided_move: dict[str, str] = field(default_factory=dict)
    initial_components: dict[str, str] = field(default_factory=dict)
    changed: bool = True
    generation: int = 0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self, keep_id: bool = False) -> "Solution":
        return Solution(
            vector=self.vector.copy(),
            prompt=self.prompt,
            generated_text=self.generated_text,
            objectives=None if self.objectives is None else Objectives(self.objectives.f1, self.objectives.f2),
            solution_id=self.solution_id if keep_id else uuid4().hex,
            velocity=dict(self.velocity),
            last_guided_move=dict(self.last_guided_move),
            initial_components=dict(self.initial_components),
            changed=self.changed,
            generation=self.generation,
            embedding=None if self.embedding is None else list(self.embedding),
            metadata=dict(self.metadata),
        )


def solution_to_dict(solution: Solution) -> dict[str, Any]:
    return {
        "solution_id": solution.solution_id,
        "components": dict(solution.vector.components),
        "prompt": solution.prompt,
        "generated_text": solution.generated_text,
        "objectives": None
        if solution.objectives is None
        else {"f1": solution.objectives.f1, "f2": solution.objectives.f2},
        "velocity": dict(solution.velocity),
        "last_guided_move": dict(solution.last_guided_move),
        "initial_components": dict(solution.initial_components),
        "changed": solution.changed,
        "generation": solution.generation,
        "embedding": solution.embedding,
        "metadata": dict(solution.metadata),
    }


def solution_from_dict(payload: dict[str, Any]) -> Solution:
    objectives_payload = payload.get("objectives")
    return Solution(
        solution_id=payload["solution_id"],
        vector=SemanticVector(dict(payload["components"])),
        prompt=payload.get("prompt", ""),
        generated_text=payload.get("generated_text", ""),
        objectives=None
        if objectives_payload is None
        else Objectives(float(objectives_payload["f1"]), float(objectives_payload["f2"])),
        velocity=dict(payload.get("velocity", {})),
        last_guided_move=dict(payload.get("last_guided_move", {})),
        initial_components=dict(payload.get("initial_components", {})),
        changed=bool(payload.get("changed", False)),
        generation=int(payload.get("generation", 0)),
        embedding=payload.get("embedding"),
        metadata=dict(payload.get("metadata", {})),
    )

