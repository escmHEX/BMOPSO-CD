from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from binary_mopso_cd.external_monitoring import (
    ExternalMonitoringComparator,
    MonitoringInputSpec,
    load_monitoring_specs,
)


class DeterministicEmbeddingService:
    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        vectors = []
        for idx, text in enumerate(texts):
            base = float(sum(ord(char) for char in text) % 17)
            vectors.append([base, float(idx + 1), 1.0])
        return np.asarray(vectors, dtype=float)


class StaticEntityLabeler:
    def __init__(self, labels_by_text: dict[str, list[str]]):
        self.labels_by_text = labels_by_text

    def labels_for_texts(self, texts: list[str]) -> list[str]:
        labels: list[str] = []
        for text in texts:
            labels.extend(self.labels_by_text.get(text, []))
        return labels


def comparator(labels_by_text: dict[str, list[str]] | None = None) -> ExternalMonitoringComparator:
    return ExternalMonitoringComparator(
        embedding_service=DeterministicEmbeddingService(),
        entity_labeler=StaticEntityLabeler(labels_by_text or {}),
        kmeans_clusters=2,
    )


def write_spec(path, items) -> None:
    path.write_text(json.dumps(items), encoding="utf-8")


def test_text_rows_are_recomputed_on_common_scale(tmp_path):
    source = tmp_path / "evolmd.csv"
    pd.DataFrame(
        [
            {"generation": 1, "generated_text": "evacuation update"},
            {"generation": 1, "generated_text": "shelter notice"},
            {"generation": 1, "generated_text": "road closure"},
            {"generation": 2, "generated_text": "evacuation update"},
            {"generation": 2, "generated_text": "relief center"},
        ]
    ).to_csv(source, index=False)
    labels = {
        "evacuation update": ["EVENT"],
        "shelter notice": ["FAC"],
        "road closure": ["GPE"],
        "relief center": ["ORG"],
    }

    output = comparator(labels).compare(
        [MonitoringInputSpec(method="evolmd", path=source, source="reconstructed", run="r1")],
        tmp_path / "comparison",
    )

    rows = pd.read_csv(output.metrics_csv)
    assert list(rows["metric_source"].unique()) == ["reconstructed"]
    assert rows["global_inertia_norm"].between(0, 1).all()
    assert rows["entity_entropy_norm"].between(0, 1).all()
    assert output.global_inertia_svg.exists()
    assert output.entity_entropy_svg.exists()


def test_internal_metrics_are_preserved_when_texts_allow_canonical_recalculation(tmp_path):
    source = tmp_path / "evolmd_mo.csv"
    pd.DataFrame(
        [
            {
                "generation": 1,
                "generated_text": "shelter notice",
                "global_inertia_raw": 99.0,
                "entity_entropy_raw": 8.0,
            },
            {
                "generation": 1,
                "generated_text": "road closure",
                "global_inertia_raw": 99.0,
                "entity_entropy_raw": 8.0,
            },
        ]
    ).to_csv(source, index=False)
    labels = {"shelter notice": ["FAC"], "road closure": ["GPE"]}

    output = comparator(labels).compare(
        [MonitoringInputSpec(method="evolmd-mo", path=source, source="canonical", run="r1")],
        tmp_path / "comparison",
    )

    row = pd.read_csv(output.metrics_csv).iloc[0]
    assert row["metric_source"] == "canonical"
    assert row["reported_global_inertia_raw"] == 99.0
    assert row["reported_entity_entropy_raw"] == 8.0
    assert row["global_inertia_raw"] != 99.0


def test_reported_directory_uses_effective_population_size(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {"generation": 1, "kmeans_inertia": 2.0, "entity_entropy": 0.5},
            {"generation": 2, "kmeans_inertia": 1.0, "entity_entropy": 0.25},
        ]
    ).to_csv(run_dir / "monitor_metrics.csv", index=False)
    (run_dir / "config_effective.yaml").write_text(
        yaml.safe_dump({"experiment": {"n": 4}}),
        encoding="utf-8",
    )

    output = comparator().compare(
        [MonitoringInputSpec(method="mopso-cd", path=run_dir, source="canonical")],
        tmp_path / "comparison",
    )

    rows = pd.read_csv(output.metrics_csv)
    assert rows["metric_source"].tolist() == ["reported", "reported"]
    assert rows["n_texts"].tolist() == [4, 4]
    assert rows["global_inertia_norm"].tolist() == [0.125, 0.0625]


def test_input_without_texts_or_metrics_fails_clearly(tmp_path):
    source = tmp_path / "evolmd.csv"
    pd.DataFrame([{"generation": 1, "other": "value"}]).to_csv(source, index=False)

    with pytest.raises(ValueError, match="generated texts or reported monitoring metrics"):
        comparator().compare(
            [MonitoringInputSpec(method="evolmd", path=source, source="reconstructed")],
            tmp_path / "comparison",
        )


def test_load_monitoring_specs_reads_expected_contract(tmp_path):
    spec_path = tmp_path / "spec.json"
    write_spec(
        spec_path,
        [{"method": "mopso-cd", "path": "exec/run_a", "source": "canonical", "run": "1"}],
    )

    specs = load_monitoring_specs(spec_path)

    assert specs == [
        MonitoringInputSpec(method="mopso-cd", path=Path("exec/run_a"), source="canonical", run="1")
    ]
    assert specs[0].path.as_posix() == "exec/run_a"
