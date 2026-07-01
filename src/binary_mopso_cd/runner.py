from __future__ import annotations

from pathlib import Path
import json
from random import Random

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.entities import solution_from_dict
from binary_mopso_cd.executor import SemanticTaskExecutor
from binary_mopso_cd.initialization import InitialPopulationBuilder
from binary_mopso_cd.mopso import BinaryMOPSOCDEngine, non_dominated
from binary_mopso_cd.outputs import RuntimeTimer, create_run_dir, write_config, write_cost_metrics, write_json, write_solutions
from binary_mopso_cd.progress import ProgressLogger
from binary_mopso_cd.router import SemanticRouter
from binary_mopso_cd.selection import mmr_select, rank_solutions
from binary_mopso_cd.utils import rng_from_text


class ExperimentRunner:
    def __init__(self, config: RuntimeConfig, reference_text: str):
        self.config = config
        self.reference_text = reference_text

    def run_all(self) -> list[Path]:
        outdirs = []
        for run_index in range(1, self.config.runs + 1):
            run_config = RuntimeConfig(self.config.as_dict())
            run_config.set("experiment.seed", self.config.seed + run_index - 1)
            outdirs.append(self._run_one(run_config, run_index if self.config.runs > 1 else None))
        return outdirs

    def _run_one(self, config: RuntimeConfig, run_index: int | None) -> Path:
        timer = RuntimeTimer()
        outdir = create_run_dir(Path(str(config.get("runtime.outdir_base", "exec"))), run_index)
        progress = ProgressLogger(config, outdir, run_index or 1, config.runs)
        try:
            progress.stage(1, 6, "Preparando directorio de salida")
            progress.start(config.n, config.iterations, outdir)
            (outdir / "reference.txt").write_text(self.reference_text, encoding="utf-8")
            write_config(outdir / "config_effective.yaml", config.as_dict())
            progress.stage(2, 6, "Cargando recursos semanticos")
            router = SemanticRouter(config)
            executor = SemanticTaskExecutor(config, outdir=outdir)
            resume_path = config.get("runtime.resume_from")
            checkpoint_payload = self._load_checkpoint(Path(resume_path)) if resume_path else None
            if checkpoint_payload:
                cache_name = str(checkpoint_payload.get("embedding_cache_file", config.get("runtime.embedding_cache_file")))
                cache_path = Path(resume_path).parent.parent / cache_name
                if cache_path.exists():
                    executor.embedding_service.cache.load(cache_path)
            rng = rng_from_text(config.seed, checkpoint_payload.get("rng_state") if checkpoint_payload else None)
            if checkpoint_payload:
                progress.stage(3, 6, "Cargando poblacion desde checkpoint")
                initial_population = [solution_from_dict(item) for item in checkpoint_payload["population"]]
                pbest_state = [solution_from_dict(item) for item in checkpoint_payload["pbest"]]
                archive_state = [solution_from_dict(item) for item in checkpoint_payload["archive"]]
                archive_stats = self._archive_stats_from_checkpoint(checkpoint_payload)
                component_memory = dict(checkpoint_payload.get("component_memory", {}))
                semantic_anchors = {
                    str(key): [str(item).strip() for item in values if str(item).strip()]
                    for key, values in dict(checkpoint_payload.get("semantic_anchors", {})).items()
                    if isinstance(values, list)
                }
                start_generation = int(checkpoint_payload["generation"])
                should_select_central_anchors = self._should_select_central_anchors(config, start_generation)
                central_anchors: list[str] = [
                    str(item).strip()
                    for item in checkpoint_payload.get("central_anchors", [])
                    if str(item).strip()
                ] if should_select_central_anchors else []
                if not central_anchors and should_select_central_anchors:
                    context_builder = InitialPopulationBuilder(config, router, executor, rng, progress=progress)
                    if not semantic_anchors:
                        reference_context = context_builder.build_reference_context(self.reference_text)
                        semantic_anchors = reference_context.semantic_anchors
                    central_anchors = context_builder.select_central_anchors(self.reference_text, semantic_anchors)
                write_json(
                    outdir / "reference_context.json",
                    {
                        "semantic_anchors": semantic_anchors,
                        "central_anchors": central_anchors,
                    },
                )
                write_solutions(outdir / "data_initial_population.json", initial_population)
                write_solutions(outdir / "data_inicial_evaluada.json", initial_population)
            else:
                pbest_state = None
                archive_state = None
                archive_stats = None
                component_memory = None
                start_generation = 0
                population_input_path = self._optional_path(config.get("initialization.population_input_path"))
                reference_context_input_path = self._optional_path(config.get("initialization.reference_context_input_path"))
                if population_input_path:
                    progress.stage(3, 6, "Cargando poblacion inicial externa")
                    initial_population = self._load_initial_population(population_input_path, config)
                    semantic_anchors, central_anchors = self._load_or_build_reference_context(
                        config,
                        reference_context_input_path,
                        router,
                        executor,
                        rng,
                        progress,
                        start_generation,
                    )
                else:
                    progress.stage(3, 6, "Construyendo poblacion inicial")
                    initial_builder = InitialPopulationBuilder(config, router, executor, rng, progress=progress)
                    initial_result = initial_builder.build_with_context(self.reference_text)
                    initial_population = initial_result.population
                    semantic_anchors = initial_result.semantic_anchors
                    central_anchors = (
                        initial_builder.select_central_anchors(self.reference_text, semantic_anchors)
                        if self._should_select_central_anchors(config, start_generation)
                        else []
                    )
                write_json(
                    outdir / "reference_context.json",
                    {
                        "semantic_anchors": semantic_anchors,
                        "central_anchors": central_anchors,
                    },
                )
                write_solutions(outdir / "data_initial_population.json", initial_population)
                write_solutions(outdir / "data_inicial_evaluada.json", initial_population)
            engine = BinaryMOPSOCDEngine(
                config,
                router,
                executor,
                rng,
                outdir,
                self.reference_text,
                central_anchors,
                progress,
                semantic_anchors,
            )
            progress.stage(4, 6, "Ejecutando MOPSO-CD")
            population, archive = engine.run(
                initial_population,
                start_generation=start_generation,
                pbest_state=pbest_state,
                archive_state=archive_state,
                archive_stats=archive_stats,
                component_memory=component_memory,
            )
            progress.stage(5, 6, "Escribiendo frente y seleccion final")
            pareto = non_dominated(archive.solutions)
            write_solutions(outdir / "population_evaluated.json", population)
            write_solutions(outdir / "pareto_front.json", pareto)
            if bool(config.get("selection.enabled", True)):
                ranked = rank_solutions(
                    pareto,
                    tau_min=float(config.get("selection.tau_min", 0.20)),
                    tau_max=float(config.get("selection.tau_max", 0.94)),
                    epsilon=float(config.get("selection.epsilon", 0.0001)),
                )
                selected = mmr_select(
                    ranked,
                    executor.embedding_service,
                    k=int(config.get("selection.k", 5)),
                    lambda_mmr=float(config.get("selection.lambda_mmr", 0.35)),
                )
                write_json(
                    outdir / "pareto_ranked.json",
                    [
                        {
                            "solution": item.solution.solution_id,
                            "topsis_score": item.topsis_score,
                            "selected": item.selected,
                        }
                        for item in ranked
                    ],
                )
                write_solutions(outdir / "final_selection_hybrid.json", [item.solution for item in selected])
            executor.save_caches()
            runtime_payload = timer.write(outdir / "runtime.txt", {"run_index": run_index or 1})
            write_cost_metrics(
                outdir / "cost_metrics.json",
                llm_log_path=outdir / "llm_calls.jsonl",
                wall_clock_seconds=float(runtime_payload["runtime_seconds"]),
            )
            progress.stage(6, 6, "Finalizando corrida")
            progress.finish(outdir, archive.update_count, archive.prune_count)
            return outdir
        except Exception:
            progress.exception("run %s/%s failed", run_index or 1, config.runs)
            raise
        finally:
            progress.close()

    def _load_checkpoint(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _optional_path(self, value: object) -> Path | None:
        text = str(value or "").strip()
        return Path(text) if text else None

    def _load_json_file(self, path: Path, label: str) -> object:
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_initial_population(self, path: Path, config: RuntimeConfig) -> list:
        payload = self._load_json_file(path, "Initial population input")
        if not isinstance(payload, list):
            raise ValueError(f"Initial population input must be a JSON array: {path}")
        if len(payload) != config.n:
            raise ValueError(f"Initial population input must contain {config.n} solutions; got {len(payload)}.")
        population = [solution_from_dict(item) for item in payload]
        expected_components = set(config.components)
        for index, solution in enumerate(population, start=1):
            components = set(solution.vector.components)
            if components != expected_components:
                raise ValueError(
                    "Initial population input solution "
                    f"{index} components must match semantic_components.order: {sorted(expected_components)}"
                )
            if solution.objectives is None:
                raise ValueError(f"Initial population input solution {index} must include objectives.")
        return population

    def _load_reference_context(self, path: Path) -> tuple[dict[str, list[str]], list[str]]:
        payload = self._load_json_file(path, "Reference context input")
        if not isinstance(payload, dict):
            raise ValueError(f"Reference context input must be a JSON object: {path}")
        raw_semantic_anchors = payload.get("semantic_anchors", {})
        if not isinstance(raw_semantic_anchors, dict):
            raise ValueError("Reference context input semantic_anchors must be an object.")
        semantic_anchors = {
            str(key): [str(item).strip() for item in values if str(item).strip()]
            for key, values in raw_semantic_anchors.items()
            if isinstance(values, list)
        }
        raw_central_anchors = payload.get("central_anchors", [])
        if not isinstance(raw_central_anchors, list):
            raise ValueError("Reference context input central_anchors must be an array.")
        central_anchors = [str(item).strip() for item in raw_central_anchors if str(item).strip()]
        return semantic_anchors, central_anchors

    def _load_or_build_reference_context(
        self,
        config: RuntimeConfig,
        reference_context_input_path: Path | None,
        router: SemanticRouter,
        executor: SemanticTaskExecutor,
        rng: Random,
        progress: ProgressLogger,
        start_generation: int,
    ) -> tuple[dict[str, list[str]], list[str]]:
        if reference_context_input_path:
            return self._load_reference_context(reference_context_input_path)
        if not self._should_select_central_anchors(config, start_generation):
            return {}, []
        context_builder = InitialPopulationBuilder(config, router, executor, rng, progress=progress)
        reference_context = context_builder.build_reference_context(self.reference_text)
        central_anchors = context_builder.select_central_anchors(
            self.reference_text,
            reference_context.semantic_anchors,
        )
        return reference_context.semantic_anchors, central_anchors

    def _archive_stats_from_checkpoint(self, checkpoint_payload: dict) -> dict[str, int]:
        archive_stats = checkpoint_payload.get("archive_stats")
        if not isinstance(archive_stats, dict):
            raise ValueError("Checkpoint is missing archive_stats; cannot resume archive metrics accurately")
        return {
            "update_count": int(archive_stats["update_count"]),
            "prune_count": int(archive_stats["prune_count"]),
        }

    def _should_select_central_anchors(self, config: RuntimeConfig, start_generation: int) -> bool:
        active_components = set(config.components).difference(config.frozen_components)
        return (
            config.get("mopso.p_anchor_enabled", False) is True
            and start_generation < config.iterations
            and bool(active_components)
        )
