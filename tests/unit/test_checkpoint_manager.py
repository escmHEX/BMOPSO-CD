from __future__ import annotations

import json

from binary_mopso_cd.checkpoint import CheckpointManager


def test_disabled_checkpoint_manager_creates_no_directory(tmp_path):
    manager = CheckpointManager(False, tmp_path, "checkpoints", 1)
    manager.submit(1, {"generation": 1})
    manager.close()
    assert not (tmp_path / "checkpoints").exists()


def test_deferred_checkpoint_manager_writes_atomic_snapshot(tmp_path):
    manager = CheckpointManager(True, tmp_path, "checkpoints", 1)
    manager.submit(1, {"generation": 1, "embedding_cache_file": "embedding_cache.json"})
    manager.close()
    path = tmp_path / "checkpoints" / "generation_0001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["generation"] == 1
    assert "embedding_cache" not in payload
