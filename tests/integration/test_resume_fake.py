from __future__ import annotations

import json

from binary_mopso_cd.runner import ExperimentRunner


def test_resume_from_checkpoint_fake_run(test_config, tmp_path):
    test_config.set("runtime.outdir_base", str(tmp_path / "first"))
    test_config.set("experiment.iterations", 1)
    first = ExperimentRunner(test_config, "Evacuation orders remain in effect.").run_all()[0]
    checkpoint = first / "checkpoints" / "generation_0001.json"
    assert checkpoint.exists()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert "embedding_cache" in payload
    assert payload["embedding_cache"]["items"]

    test_config.set("runtime.outdir_base", str(tmp_path / "resumed"))
    test_config.set("runtime.resume_from", str(checkpoint))
    test_config.set("experiment.iterations", 2)
    resumed = ExperimentRunner(test_config, "Evacuation orders remain in effect.").run_all()[0]
    assert (resumed / "checkpoints" / "generation_0002.json").exists()
