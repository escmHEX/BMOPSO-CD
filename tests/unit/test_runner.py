from __future__ import annotations

import json

import pytest

from binary_mopso_cd import runner as runner_module
from binary_mopso_cd.entities import Objectives, SemanticVector, Solution
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
        return list(initial_population), EmptyArchive()


class RecordingInitialArchiveEngine(RecordingEngine):
    def run(self, initial_population, **_kwargs):
        population = list(initial_population)
        return population, InitialArchive(population)


def run_with_recording_services(test_config, tmp_path, selection_enabled: bool = False):
    test_config.set("experiment.iterations", 1)
    test_config.set("selection.enabled", selection_enabled)
    test_config.set("runtime.outdir_base", str(tmp_path / "exec"))
    RecordingInitialBuilder.select_calls = 0
    RecordingEngine.central_anchors_received = None
    RecordingInitialArchiveEngine.central_anchors_received = None
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


def test_runner_rejects_legacy_checkpoint_without_archive_stats(test_config):
    runner = ExperimentRunner(test_config, "reference")

    with pytest.raises(ValueError, match="archive_stats"):
        runner._archive_stats_from_checkpoint({"generation": 1})
