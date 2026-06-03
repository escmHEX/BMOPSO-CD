from __future__ import annotations

import math
from random import Random
from typing import Any
from uuid import uuid4

import numpy as np

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.generated_text_validation import validate_generated_text
from binary_mopso_cd.llm_prompts import component_additional_instruction, component_type_label
from binary_mopso_cd.objectives import evaluate_solutions, semantic_fidelity_scores
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
        self.tau_gen_min = float(config.get("generated_text_validation.tau_gen_min", 0.05))

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
        is_expansion = task_name == TASK_POOL_EXPANSION
        route = RouteTask(
            uuid4().hex,
            "initialization",
            task_name,
            {
                "component": component,
                "component_type": component_type_label(component),
                "quantity": quantity,
                "required_items": quantity,
                "required_new_items": quantity if is_expansion else None,
                "reference_text": reference_text,
                "anchors": anchors,
                "central_anchor_count": central_anchor_count,
                "domain": domain,
                "existing": existing or [],
                "component_additional_instruction": component_additional_instruction(component),
                "max_words_by_component": dict(self.components.max_words),
                "reference_word_count": word_count(reference_text),
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
        return self._balanced_stratified_vectors(components, values, limit)

    def _balanced_stratified_vectors(
        self,
        components: list[str],
        values: list[list[str]],
        limit: int,
    ) -> list[SemanticVector]:
        combinations = materialize_product(values)
        self.rng.shuffle(combinations)
        counts = [{value: 0 for value in pool} for pool in values]
        vectors: list[SemanticVector] = []
        while combinations and len(vectors) < limit:
            best_index = min(
                range(len(combinations)),
                key=lambda idx: sum(counts[pos][value] for pos, value in enumerate(combinations[idx])),
            )
            combo = combinations.pop(best_index)
            for pos, value in enumerate(combo):
                counts[pos][value] += 1
            vectors.append(SemanticVector(dict(zip(components, combo, strict=True))))
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
        candidates: list[tuple[int, SemanticVector, str, float, str]] = []
        accepted: list[Solution] = []
        rejections: list[dict[str, Any]] = []
        for index, (vector, prompt, diversity_score) in enumerate(items):
            route = RouteTask(
                uuid4().hex,
                "initialization",
                TASK_SYNTHETIC_TEXT,
                {"prompt": prompt, "reference_text": reference_text},
            )
            text = str(self.executor.execute(self.router.route(route))).strip()
            validation = validate_generated_text(
                text,
                reference_text,
                max_sentences=self.settings.generated_sentences_max,
            )
            if not validation.valid:
                rejections.append(self._rejection_row(validation.reason, text, index=index))
                continue
            candidates.append((index, vector, prompt, diversity_score, text))

        if candidates:
            f1_values = semantic_fidelity_scores(
                [item[4] for item in candidates],
                reference_text,
                self.executor.embedding_service,
            )
        else:
            f1_values = np.zeros(0, dtype=float)

        accepted_keys: set[str] = set()
        for (index, vector, prompt, diversity_score, text), f1_value in zip(candidates, f1_values, strict=True):
            f1 = float(f1_value)
            validation = validate_generated_text(
                text,
                reference_text,
                accepted_text_keys=accepted_keys,
                f1=f1,
                tau_gen_min=self.tau_gen_min,
                max_sentences=self.settings.generated_sentences_max,
            )
            if not validation.valid:
                rejections.append(self._rejection_row(validation.reason, text, index=index, f1=f1))
                continue
            accepted_keys.add(canonical_text(text))
            accepted.append(
                Solution(
                    vector=vector.copy(),
                    prompt=prompt,
                    generated_text=text,
                    objectives=Objectives(f1, 0.0),
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

    def _rejection_row(self, reason: str | None, text: str, **metadata: Any) -> dict[str, Any]:
        row = {
            "phase": "initialization",
            "reason": reason,
            "f1": metadata.pop("f1", None),
            "text": text,
        }
        row.update(metadata)
        return row

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
