from __future__ import annotations

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
from binary_mopso_cd.settings import ComponentSettings, InitializationSettings
from binary_mopso_cd.utils import canonical_text, unique_preserve_order, word_count


class InitialPopulationBuilder:
    def __init__(self, config: RuntimeConfig, router: SemanticRouter, executor: SemanticTaskExecutor, rng: Random):
        self.config = config
        self.router = router
        self.executor = executor
        self.rng = rng
        self.components = ComponentSettings.from_config(config)
        self.settings = InitializationSettings.from_config(config)

    def build(self, reference_text: str) -> list[Solution]:
        n = self.config.n
        domain = str(self.config.get("experiment.domain"))
        anchors = self._extract_anchors(reference_text)
        central_anchor_count = count_anchors(anchors)
        pool_sizes = choose_pool_sizes(n, self.config)
        pools: dict[str, list[str]] = {}
        for component, quantity in pool_sizes.items():
            pools[component] = self._build_pool(
                component,
                quantity,
                reference_text,
                anchors,
                central_anchor_count,
                domain,
            )
        product = math.prod(len(values) for values in pools.values())
        min_product = self.settings.min_product_multiplier * n
        if product < min_product:
            self._expand_one_pool(pools, min_product, reference_text, anchors, central_anchor_count, domain)
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
            solution.velocity = {component: 0.0 for component in self.components.order}
            solution.last_guided_move = {}
            solution.changed = False
        return selected

    def _extract_anchors(self, reference_text: str) -> dict[str, list[str]]:
        route = RouteTask(uuid4().hex, "initialization", TASK_ANCHORS, {"reference_text": reference_text})
        return dict(self.executor.execute(self.router.route(route)))

    def _build_pool(
        self,
        component: str,
        quantity: int,
        reference_text: str,
        anchors: dict[str, list[str]],
        central_anchor_count: int,
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
                "central_anchor_count": central_anchor_count,
                "domain": domain,
                "existing": existing or [],
                "max_words_by_component": dict(self.components.max_words),
            },
        )
        raw = self.executor.execute(self.router.route(route))
        return validate_pool(component, raw, self.config)

    def _expand_one_pool(
        self,
        pools: dict[str, list[str]],
        required_product: int,
        reference_text: str,
        anchors: dict[str, list[str]],
        central_anchor_count: int,
        domain: str,
    ) -> None:
        for component in self.components.expansion_order:
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
                central_anchor_count,
                domain,
                existing=pools[component],
                task_name=TASK_POOL_EXPANSION,
            )
            pools[component] = validate_pool(component, pools[component] + additions, self.config)
            if math.prod(len(values) for values in pools.values()) >= required_product:
                return

    def _candidate_vectors(self, pools: dict[str, list[str]]) -> list[SemanticVector]:
        components = self.components.order
        values = [pools[component] for component in components]
        limit = self.settings.candidate_multiplier * self.config.n
        total = math.prod(len(items) for items in values)
        if total <= limit:
            return [
                SemanticVector(dict(zip(components, combo, strict=True)))
                for combo in materialize_product(values)
            ]
        return self._stratified_lazy_vectors(components, values, total, limit)

    def _stratified_lazy_vectors(
        self,
        components: list[str],
        values: list[list[str]],
        total: int,
        limit: int,
    ) -> list[SemanticVector]:
        vectors: list[SemanticVector] = []
        seen_indices: set[int] = set()
        for stratum in range(limit):
            start = math.floor(stratum * total / limit)
            end = max(start, math.floor((stratum + 1) * total / limit) - 1)
            index = self.rng.randint(start, end)
            if index in seen_indices:
                index = start
                while index <= end and index in seen_indices:
                    index += 1
            if index >= total or index in seen_indices:
                continue
            seen_indices.add(index)
            vectors.append(vector_from_product_index(components, values, index))
        fill_index = 0
        while len(vectors) < limit and fill_index < total:
            if fill_index not in seen_indices:
                seen_indices.add(fill_index)
                vectors.append(vector_from_product_index(components, values, fill_index))
            fill_index += 1
        return vectors

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
        rejections: list[dict[str, Any]] = []
        seen: set[str] = {canonical_text(reference_text)}
        for index, (vector, prompt, diversity_score) in enumerate(items):
            route = RouteTask(
                uuid4().hex,
                "initialization",
                TASK_SYNTHETIC_TEXT,
                {"prompt": prompt, "reference_text": reference_text},
            )
            text = str(self.executor.execute(self.router.route(route))).strip()
            key = canonical_text(text)
            if not key or key in seen:
                rejections.append({"index": index, "reason": "empty_or_duplicate", "text": text})
                continue
            sentence_count = max(1, text.count(".") + text.count("!") + text.count("?"))
            if sentence_count > self.settings.generated_sentences_max:
                rejections.append({"index": index, "reason": "sentence_limit", "text": text})
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
            self._write_rejections(rejections)
            raise RuntimeError(
                f"Generated only {len(accepted)} valid initial texts after {len(items)} specified generation calls; "
                f"required {self.config.n}. See initialization_rejections.jsonl."
            )
        if rejections:
            self._write_rejections(rejections)
        return accepted

    def _write_rejections(self, rows: list[dict[str, Any]]) -> None:
        if not rows or self.executor.outdir is None:
            return
        import json

        path = self.executor.outdir / "initialization_rejections.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_pool_sizes(n: int, config: RuntimeConfig) -> dict[str, int]:
    components = ComponentSettings.from_config(config)
    alphas = dict(components.alpha)
    alpha_product = math.prod(alphas.values())
    c = max(2, round((4 * n / alpha_product) ** (1 / max(len(components.order), 1))))
    sizes = {component: math.ceil(alpha * c) for component, alpha in alphas.items()}
    order = list(components.expansion_order)
    idx = 0
    while math.prod(sizes.values()) < 4 * n:
        sizes[order[idx % len(order)]] += 1
        idx += 1
    return {component: sizes.get(component, c) for component in components.order}


def validate_pool(component: str, values: list[str], config: RuntimeConfig) -> list[str]:
    max_words = ComponentSettings.from_config(config).max_words[component]
    valid = []
    for value in values:
        text = str(value).strip()
        if not text or "\n" in text:
            continue
        if word_count(text) > max_words:
            continue
        valid.append(text)
    return unique_preserve_order(valid)


def count_anchors(anchors: dict[str, list[str]]) -> int:
    return sum(len(values) for values in anchors.values())


def materialize_product(values: list[list[str]]) -> list[tuple[str, ...]]:
    if not values:
        return []
    result: list[tuple[str, ...]] = [()]
    for pool in values:
        result = [prefix + (item,) for prefix in result for item in pool]
    return result


def vector_from_product_index(components: list[str], values: list[list[str]], index: int) -> SemanticVector:
    selected: list[str] = []
    remainder = index
    for pool in reversed(values):
        selected.append(pool[remainder % len(pool)])
        remainder //= len(pool)
    selected.reverse()
    return SemanticVector(dict(zip(components, selected, strict=True)))


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
