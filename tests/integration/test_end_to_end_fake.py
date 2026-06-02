from __future__ import annotations

from binary_mopso_cd.runner import ExperimentRunner


def test_reduced_end_to_end_fake_run(test_config, tmp_path):
    test_config.set("runtime.outdir_base", str(tmp_path))
    outdirs = ExperimentRunner(
        test_config,
        "Evacuation orders remain in effect for Zone A until further notice.",
    ).run_all()
    outdir = outdirs[0]
    assert (outdir / "reference.txt").exists()
    assert (outdir / "data_initial_population.json").exists()
    assert (outdir / "pareto_front.json").exists()
    assert (outdir / "final_selection_hybrid.json").exists()
    assert (outdir / "checkpoints").exists()

