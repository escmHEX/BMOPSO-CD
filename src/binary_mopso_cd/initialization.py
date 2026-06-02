from __future__ import annotations

import itertools
import math
from random import Random
from typing import Any
from uuid import uuid4

import numpy as np

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.entities import SemanticVector, Solution
from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.objectives import evaluate_solutions
from binary_mopso_cd.router import (
    TASK_ANCHORS,
    TASK_POOL_EXPANSION,
    TASK_POOL_GENERATION,
    TASK_PROMPT_RENDERING,
    TASK_SYNTHETIC_TEXT,
    RouteTask,
    SemanticRouter,
)
from binary_mopso_cd.utils import canonical_text, unique_preserve_order, word_count


POOL_MAX_WORDS = {"role": 6, "topic": 8, "action": 6}


class InitialPopulationBuilder:
    def __init__(self, config: RuntimeConfig, router: SemanticRouter, executor: SemanticTaskExecutor, rng: Random):
        self.config = config
        self.router = router
        self.executor = executor
        self.rng = rng

    def build(self, reference_text: str) -> list[Solution]:
        n = self.config.n
        domain = str(self.config.get("experiment.domain"))
        anchors = self._extract_anchors(reference_text)
        pool_sizes = choose_pool_sizes(n, self.config)
        pools: dict[str, list[str]] = {}
        for component, quantity in pool_sizes.items():
            pools[component] = self._build_pool(component, quantity, reference_text, anchors, domain)
        product = math.prod(len(values) for values in pools.values())
        min_product = int(self.config.get("initialization.min_product_multiplier", 3)) * n
        if product < min_product:
            self._expand_one_pool(pools, min_product, reference_text, anchors, domain)
        product = math.prod(len(values) for values in pools.values())
        if product < min_product:
            raise RuntimeError(f"Initial semantic pools are insufficient: product={product}, required={min_product}")
        candidates = self._candidate_vectors(pools)
        reduced = self._reduce_by_prompt_diversity(candidates, domain, 2 * n)
        generated = self._generate_texts(reduced, reference_text)
        evaluate_solutions(generated, reference_text, self.executor.embedding_service)
        generated.sort(
            key=lambda solution: (
                solution.objectives.f1 if solution.objectives else -2.0,
                solution.metadata.get("prompt_diversity_score", 0.0),
            ),
            reverse=True,
        )
        selected = generated[:n]
        evaluate_solutions(selected, reference_text, self.executor.embedding_service)
        for solution in selected:
            solution.initial_components = dict(solution.vector.components)
            solution.velocity = {component: 0.0 for component in self.config.components}
            solution.last_guided_move = {}
            solution.changed = False
        return selected

    def _extract_anchors(self, reference_text: str) -> list[str]:
        route = RouteTask(uuid4().hex, "initialization", TASK_ANCHORS, {"reference_text": reference_text})
        return list(self.executor.execute(self.router.route(route)))

    def _build_pool(
        self,
        component: str,
        quantity: int,
        reference_text: str,
        anchors: list[str],
        domain: str,
        existing: list[str] | None = None,
        task_name: str = TASK_POOL_GENERATION,
    ) -> list[str]:
        route = RouteTask(
            uuid4().hex,
            "initialization",
            task_name,
            {
                "component": component,
                "quantity": quantity,
                "reference_text": reference_text,
                "anchors": anchors,
                "central_anchor_count": len(anchors),
                "domain": domain,
                "existing": existing or [],
            },
        )
        raw = self.executor.execute(self.router.route(route))
        return validate_pool(component, raw)

    def _expand_one_pool(
        self,
        pools: dict[str, list[str]],
        required_product: int,
        reference_text: str,
        anchors: list[str],
        domain: str,
    ) -> None:
        for component in ["role", "action", "topic"]:
            if component not in pools:
                continue
            other_sizes = [len(values) for name, values in pools.items() if name != component]
            other_product = math.prod(other_sizes) if other_sizes else 1
            needed_total = math.ceil(required_product / max(other_product, 1))
            extra = max(1, needed_total - len(pools[component]))
            additions = self._build_pool(
                component,
                extra,
                reference_text,
                anchors,
                domain,
                existing=pools[component],
                task_name=TASK_POOL_EXPANSION,
            )
            pools[component] = validate_pool(component, pools[component] + additions)
            if math.prod(len(values) for values in pools.values()) >= required_product:
                return

    def _candidate_vectors(self, pools: dict[str, list[str]]) -> list[SemanticVector]:
        components = self.config.components
        values = [pools[component] for component in components]
        all_vectors = [SemanticVector(dict(zip(components, combo, strict=True))) for combo in itertools.product(*values)]
        limit = int(self.config.get("initialization.candidate_multiplier", 4)) * self.config.n
        if len(all_vectors) <= limit:
            return all_vectors
        return self.rng.sample(all_vectors, limit)

    def _reduce_by_prompt_diversity(
        self,
        candidates: list[SemanticVector],
        domain: str,
        target_count: int,
    ) -> list[tuple[SemanticVector, str, float]]:
        prompts = [self._render_prompt(vector, domain) for vector in candidates]
        embeddings = self.executor.embedding_service.encode(prompts, text_type="prompt")
        selected_indices = greedy_max_min_indices(embeddings, min(target_count, len(candidates)))
        reduced: list[tuple[SemanticVector, str, float]] = []
        for idx in selected_indices:
            score = 0.0
            if len(selected_indices) > 1:
                others = [j for j in selected_indices if j != idx]
                score = float(np.min(1.0 - (embeddings[idx] @ embeddings[others].T)))
            reduced.append((candidates[idx], prompts[idx], score))
        return reduced

    def _render_prompt(self, vector: SemanticVector, domain: str) -> str:
        route = RouteTask(
            uuid4().hex,
            "initialization",
            TASK_PROMPT_RENDERING,
            {"components": vector.components, "domain": domain},
        )
        return str(self.executor.execute(self.router.route(route)))

    def _generate_texts(self, items: list[tuple[SemanticVector, str, float]], reference_text: str) -> list[Solution]:
        accepted: list[Solution] = []
        seen: set[str] = {canonical_text(reference_text)}
        for vector, prompt, diversity_score in items:
            route = RouteTask(
                uuid4().hex,
                "initialization",
                TASK_SYNTHETIC_TEXT,
                {"prompt": prompt, "reference_text": reference_text},
            )
            text = str(self.executor.execute(self.router.route(route))).strip()
            key = canonical_text(text)
            if not key or key in seen:
                continue
            sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
            if sentence_count > int(self.config.get("initialization.generated_sentences_max", 4)):
                continue
            seen.add(key)
            accepted.append(
                Solution(
                    vector=vector.copy(),
                    prompt=prompt,
                    generated_text=text,
                    metadata={"prompt_diversity_score": diversity_score},
                )
            )
        if len(accepted) < self.config.n:
            raise RuntimeError(f"Generated only {len(accepted)} valid initial texts; required {self.config.n}")
        return accepted


def choose_pool_sizes(n: int, config: RuntimeConfig) -> dict[str, int]:
    alpha_role = float(config.get("initialization.alpha_role", 1.4))
    alpha_action = float(config.get("initialization.alpha_action", 1.2))
    alpha_topic = float(config.get("initialization.alpha_topic", 1.0))
    c = max(2, round((4 * n / (alpha_role * alpha_action * alpha_topic)) ** (1 / 3)))
    sizes = {
        "role": math.ceil(alpha_role * c),
        "action": math.ceil(alpha_action * c),
        "topic": math.ceil(alpha_topic * c),
    }
    order = ["role", "action", "topic"]
    idx = 0
    while math.prod(sizes.values()) < 4 * n:
        sizes[order[idx % len(order)]] += 1
        idx += 1
    return {component: sizes.get(component, c) for component in config.components}


def validate_pool(component: str, values: list[str]) -> list[str]:
    max_words = POOL_MAX_WORDS.get(component, 8)
    valid = []
    for value in values:
        text = str(value).strip()
        if not text or "\n" in text:
            continue
        if word_count(text) > max_words:
            continue
        valid.append(text)
    return unique_preserve_order(valid)


def greedy_max_min_indices(embeddings: np.ndarray, count: int) -> list[int]:
    if count <= 0 or embeddings.shape[0] == 0:
        return []
    selected = [0]
    remaining = set(range(1, embeddings.shape[0]))
    while remaining and len(selected) < count:
        best_idx = max(
            remaining,
            key=lambda idx: min(1.0 - float(embeddings[idx] @ embeddings[sel]) for sel in selected),
        )
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

