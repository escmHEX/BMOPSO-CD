from __future__ import annotations

import pytest

from binary_mopso_cd.config import RuntimeConfig


@pytest.fixture()
def test_config() -> RuntimeConfig:
    config = RuntimeConfig.load()
    config.set("runtime.backend", "fake")
    config.set("experiment.n", 4)
    config.set("experiment.iterations", 2)
    config.set("experiment.runs", 1)
    config.set("experiment.seed", 7)
    config.set("models.sbert.default", "fake-sbert")
    config.set("models.sbert.config_version", "test-v1")
    config.set("selection.k", 2)
    config.validate()
    return config

