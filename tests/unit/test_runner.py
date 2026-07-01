from __future__ import annotations

import json

import pytest

from binary_mopso_cd import runner as runner_module
from binary_mopso_cd.entities import Objectives, SemanticVector, Solution, solution_to_dict
from binary_mopso_cd.initialization import InitialPopulationResult
from binary_mopso_cd.runner import ExperimentRunner


class RecordingExecutor:
    embedding_service = object()

    def save_caches(self) -> None:
        return None


class RecordingInitialBuilder:
    select_calls = 0

    def __init__(self, *_args, **_kwargs):
        return None

    def build_with_context(self, _reference_text: str) -> InitialPopulationResult:
        return InitialPopulationResult(
            population=[
                Solution(
                    SemanticVector({"role": "resident", "topic": "bridge flooding", "action": "warn neighbors"}),
                    "prompt",
                    "generated text",
                    Objectives(0.5, 0.5),
                    metadata={"used_central_anchors": False, "anchor_inclusion_probability": None},
                )
            ],
            semantic_anchors={"entities": ["bridge"], "topics": ["flooding"]},
        )

    def select_central_anchors(self, _reference_text: str, _semantic_anchors: dict[str, list[str]]) -> list[str]:
        type(self).select_calls += 1
        return ["bridge", "flooding", "warning"]


class FailingInitialBuilder:
    def __init__(self, *_args, **_kwargs):
        return None

    def build_with_context(self, _reference_text: str) -> InitialPopulationResult:
        raise AssertionError("external initial population should skip build_with_context")

    def build_reference_context(self, _reference_text: str):
        raise AssertionError("reference context input should skip build_reference_context")

    def select_central_anchors(self, _reference_text: str, _semantic_anchors: dict[str, list[str]]) -> list[str]:
        raise AssertionError("reference context input should skip select_central_anchors")


class EmptyArchive:
    solutions: list[Solution] = []
    update_count = 0
    prune_count = 0


class InitialArchive:
    def __init__(self, solutions: list[Solution]):
        self.solutions = list(solutions)
        self.update_count = 1
        self.prune_count = 0


class RecordingEngine:
    central_anchors_received: list[str] | None = None
    initial_population_received: list[str] | None = None

    def __init__(
        self,
        _config,
        _router,
        _executor,
        _rng,
        _outdir,
        _reference_text,
        central_anchors,
        *_args,
    ):
        type(self).central_anchors_received = list(central_anchors)

    def run(self, initial_population, **_kwargs):
        type(self).initial_population_received = [solution.solution_id for solution in initial_population]
        return list(initial_population), EmptyArchive()


class RecordingInitialArchiveEngine(RecordingEngine):
    def run(self, initial_population, **_kwargs):
        population = list(initial_population)
        return population, InitialArchive(population)


def external_population_payload(config, count: int | None = None, components: dict[str, str] | None = None):
    count = config.n if count is None else count
    component_values = components or {name: f"{name} value" for name in config.components}
    return [
        solution_to_dict(
            Solution(
                SemanticVector(dict(component_values)),
                f"prompt {index}",
                f"generated text {index}",
                Objectives(0.5, 0.2),
                solution_id=f"external-{index}",
                metadata={"used_central_anchors": False, "anchor_inclusion_probability": None},
            )
        )
        for index in range(count)
    ]


def run_with_recording_services(test_config, tmp_path, selection_enabled: bool = False):
    test_config.set("experiment.iterations", 1)
    test_config.set("selection.enabled", selection_enabled)
    test_config.set("runtime.outdir_base", str(tmp_path / "exec"))
    RecordingInitialBuilder.select_calls = 0
    RecordingEngine.central_anchors_received = None
    RecordingEngine.initial_population_received = None
    RecordingInitialArchiveEngine.central_anchors_received = None
    RecordingInitialArchiveEngine.initial_population_received = None
    return ExperimentRunner(test_config, "Flooding near the bridge.").run_all()[0]


def test_runner_skips_central_anchor_selection_when_anchor_prompting_is_disabled(
    test_config,
    tmp_path,
    monkeypatch,
):
    test_config.set("mopso.p_anchor_enabled", False)
    monkeypatch.setattr(runner_module, "SemanticTaskExecutor", lambda *_args, **_kwargs: RecordingExecutor())
    monkeypatch.setattr(runner_module, "InitialPopulationBuilder", RecordingInitialBuilder)
    monkeypatch.setattr(runner_module, "BinaryMOPSOCDEngine", RecordingEngine)

    outdir = run_with_recording_services(test_config, tmp_path)

    payload = json.loads((outdir / "reference_context.json").read_text(encoding="utf-8"))
    assert RecordingInitialBuilder.select_calls == 0
    assert RecordingEngine.central_anchors_received == []
    assert payload["central_anchors"] == []


def test_runner_selects_central_anchors_once_when_anchor_prompting_is_enabled(
    test_config,
    tmp_path,
    monkeypatch,
):
    test_config.set("mopso.p_anchor_enabled", True)
    monkeypatch.setattr(runner_module, "SemanticTaskExecutor", lambda *_args, **_kwargs: RecordingExecutor())
    monkeypatch.setattr(runner_module, "InitialPopulationBuilder", RecordingInitialBuilder)
    monkeypatch.setattr(runner_module, "BinaryMOPSOCDEngine", RecordingEngine)

    outdir = run_with_recording_services(test_config, tmp_path)

    payload = json.loads((outdir / "reference_context.json").read_text(encoding="utf-8"))
    assert RecordingInitialBuilder.select_calls == 1
    assert RecordingEngine.central_anchors_received == ["bridge", "flooding", "warning"]
    assert payload["central_anchors"] == ["bridge", "flooding", "warning"]


def test_runner_skips_central_anchor_selection_when_all_components_are_frozen(
    test_config,
    tmp_path,
    monkeypatch,
):
    test_config.set("experiment.frozen_components", ["role", "topic", "action"])
    test_config.set("mopso.p_anchor_enabled", True)
    monkeypatch.setattr(runner_module, "SemanticTaskExecutor", lambda *_args, **_kwargs: RecordingExecutor())
    monkeypatch.setattr(runner_module, "InitialPopulationBuilder", RecordingInitialBuilder)
    monkeypatch.setattr(runner_module, "BinaryMOPSOCDEngine", RecordingInitialArchiveEngine)

    outdir = run_with_recording_services(test_config, tmp_path, selection_enabled=True)

    payload = json.loads((outdir / "reference_context.json").read_text(encoding="utf-8"))
    initial = json.loads((outdir / "data_initial_population.json").read_text(encoding="utf-8"))
    population = json.loads((outdir / "population_evaluated.json").read_text(encoding="utf-8"))
    pareto = json.loads((outdir / "pareto_front.json").read_text(encoding="utf-8"))
    ranked = json.loads((outdir / "pareto_ranked.json").read_text(encoding="utf-8"))
    selected = json.loads((outdir / "final_selection_hybrid.json").read_text(encoding="utf-8"))
    assert RecordingInitialBuilder.select_calls == 0
    assert RecordingInitialArchiveEngine.central_anchors_received == []
    assert payload["central_anchors"] == []
    assert population == initial
    assert pareto == initial
    assert ranked[0]["solution"] == initial[0]["solution_id"]
    assert selected == initial


def test_runner_uses_external_initial_population_and_reference_context(test_config, tmp_path, monkeypatch):
    test_config.set("mopso.p_anchor_enabled", True)
    external_population = [
        Solution(
            SemanticVector({"role": "resident", "topic": "bridge flooding", "action": "warn neighbors"}),
            "prompt a",
            "generated text a",
            Objectives(0.7, 0.1),
            solution_id="external-a",
            metadata={"used_central_anchors": False, "anchor_inclusion_probability": None},
        ),
        Solution(
            SemanticVector({"role": "volunteer", "topic": "road closure", "action": "share shelter info"}),
            "prompt b",
            "generated text b",
            Objectives(0.6, 0.2),
            solution_id="external-b",
            metadata={"used_central_anchors": False, "anchor_inclusion_probability": None},
        ),
    ]
    population_payload = [solution_to_dict(solution) for solution in external_population]
    population_path = tmp_path / "shared" / "data_initial_population.json"
    context_path = tmp_path / "shared" / "reference_context.json"
    population_path.parent.mkdir(parents=True)
    population_path.write_text(json.dumps(population_payload), encoding="utf-8")
    context_path.write_text(
        json.dumps(
            {
                "semantic_anchors": {"entities": ["bridge"], "topics": ["flooding"]},
                "central_anchors": ["bridge", "flooding", "warning"],
            }
        ),
        encoding="utf-8",
    )
    test_config.set("initialization.population_input_path", str(population_path))
    test_config.set("initialization.reference_context_input_path", str(context_path))
    monkeypatch.setattr(runner_module, "SemanticTaskExecutor", lambda *_args, **_kwargs: RecordingExecutor())
    monkeypatch.setattr(runner_module, "InitialPopulationBuilder", FailingInitialBuilder)
    monkeypatch.setattr(runner_module, "BinaryMOPSOCDEngine", RecordingEngine)

    outdir = run_with_recording_services(test_config, tmp_path)

    written_initial = json.loads((outdir / "data_initial_population.json").read_text(encoding="utf-8"))
    written_context = json.loads((outdir / "reference_context.json").read_text(encoding="utf-8"))
    assert RecordingEngine.initial_population_received == ["external-a", "external-b"]
    assert RecordingEngine.central_anchors_received == ["bridge", "flooding", "warning"]
    assert written_initial == population_payload
    assert written_context["semantic_anchors"] == {"entities": ["bridge"], "topics": ["flooding"]}
    assert written_context["central_anchors"] == ["bridge", "flooding", "warning"]


def test_runner_rejects_missing_external_initial_population(test_config, tmp_path):
    runner = ExperimentRunner(test_config, "reference")

    with pytest.raises(FileNotFoundError, match="Initial population input not found"):
        runner._load_initial_population(tmp_path / "missing.json", test_config)


def test_runner_rejects_external_initial_population_that_is_not_a_list(test_config, tmp_path):
    path = tmp_path / "data_initial_population.json"
    path.write_text(json.dumps({"solutions": []}), encoding="utf-8")
    runner = ExperimentRunner(test_config, "reference")

    with pytest.raises(ValueError, match="must be a JSON array"):
        runner._load_initial_population(path, test_config)


def test_runner_rejects_external_initial_population_with_wrong_length(test_config, tmp_path):
    path = tmp_path / "data_initial_population.json"
    path.write_text(json.dumps(external_population_payload(test_config, count=1)), encoding="utf-8")
    runner = ExperimentRunner(test_config, "reference")

    with pytest.raises(ValueError, match=f"must contain {test_config.n} solutions"):
        runner._load_initial_population(path, test_config)


def test_runner_rejects_external_initial_population_with_incompatible_components(test_config, tmp_path):
    path = tmp_path / "data_initial_population.json"
    path.write_text(
        json.dumps(external_population_payload(test_config, components={"role": "resident"})),
        encoding="utf-8",
    )
    runner = ExperimentRunner(test_config, "reference")

    with pytest.raises(ValueError, match="components must match semantic_components.order"):
        runner._load_initial_population(path, test_config)


def test_runner_rejects_legacy_checkpoint_without_archive_stats(test_config):
    runner = ExperimentRunner(test_config, "reference")

    with pytest.raises(ValueError, match="archive_stats"):
        runner._archive_stats_from_checkpoint({"generation": 1})
