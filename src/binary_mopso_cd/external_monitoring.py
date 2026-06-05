from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from binary_mopso_cd.config import RuntimeConfig
from binary_mopso_cd.monitor import entity_entropy, kmeans_global_inertia
from binary_mopso_cd.services.embedding import EmbeddingCache, EmbeddingService
from binary_mopso_cd.utils import safe_float


INERTIA_COLUMNS = ("global_inertia_raw", "global_inertia", "kmeans_inertia")
ENTROPY_COLUMNS = ("entity_entropy_raw", "entity_entropy")


@dataclass(frozen=True, slots=True)
class MonitoringInputSpec:
    method: str
    path: Path
    source: str = "canonical"
    run: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonOutput:
    metrics_csv: Path
    global_inertia_svg: Path
    entity_entropy_svg: Path


class EmbeddingEncoder(Protocol):
    def encode(self, texts: list[str], text_type: str) -> np.ndarray:
        ...


class EntityLabeler(Protocol):
    def labels_for_texts(self, texts: list[str]) -> list[str]:
        ...


class SpacyEntityLabeler:
    def __init__(self, spacy_model: str):
        self.spacy_model = spacy_model
        self._nlp: Any | None = None

    @property
    def nlp(self) -> Any:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def labels_for_texts(self, texts: list[str]) -> list[str]:
        labels: list[str] = []
        for doc in self.nlp.pipe(texts):
            labels.extend(entity.label_ for entity in doc.ents)
        return labels


class ExternalMonitoringComparator:
    def __init__(
        self,
        embedding_service: EmbeddingEncoder,
        entity_labeler: EntityLabeler,
        kmeans_clusters: int = 3,
        generation_column: str = "generation",
        text_column: str = "generated_text",
        run_column: str = "run",
    ):
        if kmeans_clusters <= 0:
            raise ValueError("kmeans_clusters must be positive")
        self.embedding_service = embedding_service
        self.entity_labeler = entity_labeler
        self.kmeans_clusters = int(kmeans_clusters)
        self.generation_column = generation_column
        self.text_column = text_column
        self.run_column = run_column

    def compare(self, specs: list[MonitoringInputSpec], outdir: Path) -> ComparisonOutput:
        if not specs:
            raise ValueError("At least one comparison input is required")
        outdir.mkdir(parents=True, exist_ok=True)
        rows, label_universe_size = self._collect_rows(specs)
        self._normalize_rows(rows, label_universe_size)
        ordered_rows = sorted(rows, key=lambda item: (str(item["method"]), str(item["run"]), int(item["generation"])))
        metrics_csv = outdir / "comparison_monitor_metrics.csv"
        pd.DataFrame(ordered_rows).to_csv(metrics_csv, index=False)
        global_inertia_svg = outdir / "comparison_global_inertia_by_generation.svg"
        entity_entropy_svg = outdir / "comparison_entity_entropy_by_generation.svg"
        write_line_chart_svg(
            ordered_rows,
            global_inertia_svg,
            value_key="global_inertia_norm",
            title="Global Inertia By Generation",
            y_label="Global Inertia (normalized)",
        )
        write_line_chart_svg(
            ordered_rows,
            entity_entropy_svg,
            value_key="entity_entropy_norm",
            title="Entity Entropy By Generation",
            y_label="Entity Entropy (normalized)",
        )
        return ComparisonOutput(metrics_csv, global_inertia_svg, entity_entropy_svg)

    def _collect_rows(self, specs: list[MonitoringInputSpec]) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        all_labels: list[str] = []
        for spec in specs:
            table, default_n_texts = read_monitoring_table(spec.path)
            table = self._prepare_table(spec, table)
            if self.text_column in table.columns:
                text_rows, text_labels = self._rows_from_texts(spec, table)
                if text_rows:
                    rows.extend(text_rows)
                    all_labels.extend(text_labels)
                    continue
            reported_rows = self._rows_from_reported_metrics(spec, table, default_n_texts)
            if not reported_rows:
                raise ValueError(
                    f"Input for method {spec.method!r} at {spec.path} must contain generated texts "
                    "or reported monitoring metrics"
                )
            rows.extend(reported_rows)
        return rows, len(set(all_labels))

    def _prepare_table(self, spec: MonitoringInputSpec, table: pd.DataFrame) -> pd.DataFrame:
        result = table.copy()
        if self.generation_column not in result.columns:
            raise ValueError(f"Input for method {spec.method!r} is missing column {self.generation_column!r}")
        result[self.generation_column] = pd.to_numeric(result[self.generation_column], errors="raise").astype(int)
        result["method"] = spec.method
        if self.run_column in result.columns:
            result["run"] = result[self.run_column].astype(str)
        elif spec.run is not None:
            result["run"] = str(spec.run)
        else:
            result["run"] = spec.path.stem
        return result

    def _rows_from_texts(self, spec: MonitoringInputSpec, table: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
        rows: list[dict[str, Any]] = []
        all_labels: list[str] = []
        grouped = table.groupby(["method", "run", self.generation_column], sort=True)
        for (method, run, generation), group in grouped:
            texts = [
                str(value).strip()
                for value in group[self.text_column].tolist()
                if not pd.isna(value) and str(value).strip()
            ]
            if not texts:
                continue
            embeddings = self.embedding_service.encode(texts, text_type="generated_text")
            labels = self.entity_labeler.labels_for_texts(texts)
            all_labels.extend(labels)
            row = {
                "method": str(method),
                "run": str(run),
                "generation": int(generation),
                "n_texts": len(texts),
                "global_inertia_raw": kmeans_global_inertia(embeddings, self.kmeans_clusters),
                "entity_entropy_raw": entity_entropy(labels),
                "metric_source": spec.source,
            }
            self._attach_reported_audit_columns(row, group)
            rows.append(row)
        return rows, all_labels

    def _rows_from_reported_metrics(
        self,
        spec: MonitoringInputSpec,
        table: pd.DataFrame,
        default_n_texts: int | None,
    ) -> list[dict[str, Any]]:
        inertia_column = first_existing_column(table, INERTIA_COLUMNS)
        entropy_column = first_existing_column(table, ENTROPY_COLUMNS)
        if inertia_column is None and entropy_column is None:
            return []
        grouped = table.groupby(["method", "run", self.generation_column], sort=True)
        rows: list[dict[str, Any]] = []
        for (method, run, generation), group in grouped:
            n_texts = reported_n_texts(group, default_n_texts)
            if n_texts is None:
                raise ValueError(
                    f"Reported monitoring metrics for method {spec.method!r} need an n_texts column "
                    "or a config_effective.yaml with experiment.n"
                )
            inertia = first_metric_value(group, inertia_column) if inertia_column else 0.0
            entropy = first_metric_value(group, entropy_column) if entropy_column else 0.0
            rows.append(
                {
                    "method": str(method),
                    "run": str(run),
                    "generation": int(generation),
                    "n_texts": n_texts,
                    "global_inertia_raw": inertia,
                    "entity_entropy_raw": entropy,
                    "reported_global_inertia_raw": inertia,
                    "reported_entity_entropy_raw": entropy,
                    "metric_source": "reported",
                }
            )
        return rows

    def _attach_reported_audit_columns(self, row: dict[str, Any], group: pd.DataFrame) -> None:
        inertia_column = first_existing_column(group, INERTIA_COLUMNS)
        entropy_column = first_existing_column(group, ENTROPY_COLUMNS)
        if inertia_column is not None:
            row["reported_global_inertia_raw"] = first_metric_value(group, inertia_column)
        if entropy_column is not None:
            row["reported_entity_entropy_raw"] = first_metric_value(group, entropy_column)

    def _normalize_rows(self, rows: list[dict[str, Any]], label_universe_size: int) -> None:
        for row in rows:
            n_texts = int(row.get("n_texts", 0))
            inertia_denom = 4.0 * n_texts
            row["global_inertia_norm"] = clip01(
                safe_float(row.get("global_inertia_raw")) / inertia_denom if inertia_denom > 0 else 0.0
            )
            raw_entropy = safe_float(row.get("entity_entropy_raw"))
            row["entity_entropy_norm"] = (
                clip01(raw_entropy / math.log(label_universe_size)) if label_universe_size > 1 else 0.0
            )


def build_comparison_embedding_service(config: RuntimeConfig, outdir: Path) -> EmbeddingService:
    model_alias = str(config.get("models.sbert.default"))
    resolved = str(config.get(f"models.sbert.alternatives.{model_alias}", model_alias))
    return EmbeddingService(
        model_name=model_alias,
        resolved_model_name=resolved,
        batch_size=int(config.get("models.sbert.batch_size", 64)),
        config_version=str(config.get("models.sbert.config_version", "sbert-v1")),
        cache=EmbeddingCache(outdir / "comparison_embedding_cache.json"),
    )


def load_monitoring_specs(path: Path) -> list[MonitoringInputSpec]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Comparison input spec must be a JSON array")
    specs: list[MonitoringInputSpec] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Comparison input spec item {index} must be an object")
        method = str(item.get("method", "")).strip()
        source_path = str(item.get("path", "")).strip()
        if not method or not source_path:
            raise ValueError(f"Comparison input spec item {index} needs method and path")
        specs.append(
            MonitoringInputSpec(
                method=method,
                path=Path(source_path),
                source=str(item.get("source", "canonical")).strip() or "canonical",
                run=None if item.get("run") is None else str(item["run"]),
            )
        )
    return specs


def read_monitoring_table(path: Path) -> tuple[pd.DataFrame, int | None]:
    if path.is_dir():
        return read_monitoring_directory(path)
    if not path.exists():
        raise FileNotFoundError(f"Monitoring input not found: {path}")
    return read_monitoring_file(path), None


def read_monitoring_directory(path: Path) -> tuple[pd.DataFrame, int | None]:
    monitor_metrics = path / "monitor_metrics.csv"
    if monitor_metrics.exists():
        table = pd.read_csv(monitor_metrics)
        return table, read_experiment_population_size(path / "config_effective.yaml")
    for candidate in [
        path / "comparison_monitoring_input.csv",
        path / "generated_texts_by_generation.csv",
        path / "generated_texts.csv",
    ]:
        if candidate.exists():
            return read_monitoring_file(candidate), None
    archive_history = path / "archive_history.jsonl"
    if archive_history.exists():
        return read_monitoring_file(archive_history), None
    raise FileNotFoundError(
        f"No monitoring input found in {path}. Expected monitor_metrics.csv or a generated-text table."
    )


def read_monitoring_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.extend(flatten_monitoring_record(json.loads(line)))
        return pd.DataFrame(rows)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            rows: list[dict[str, Any]] = []
            for item in payload:
                rows.extend(flatten_monitoring_record(item))
            return pd.DataFrame(rows)
        if isinstance(payload, dict):
            for key in ("rows", "items", "data"):
                if isinstance(payload.get(key), list):
                    rows = []
                    for item in payload[key]:
                        rows.extend(flatten_monitoring_record(item))
                    return pd.DataFrame(rows)
            return pd.DataFrame(flatten_monitoring_record(payload))
    raise ValueError(f"Unsupported monitoring input format: {path}")


def flatten_monitoring_record(record: Any) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []
    if isinstance(record.get("archive"), list):
        rows: list[dict[str, Any]] = []
        generation = record.get("generation")
        for item in record["archive"]:
            if isinstance(item, dict):
                row = dict(item)
                row["generation"] = generation
                rows.append(row)
        return rows
    return [dict(record)]


def read_experiment_population_size(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    try:
        return int(payload["experiment"]["n"])
    except (KeyError, TypeError, ValueError):
        return None


def first_existing_column(table: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in table.columns:
            return column
    return None


def first_metric_value(group: pd.DataFrame, column: str) -> float:
    for value in group[column].tolist():
        if not pd.isna(value):
            return safe_float(value)
    return 0.0


def reported_n_texts(group: pd.DataFrame, default_n_texts: int | None) -> int | None:
    if "n_texts" in group.columns:
        value = first_metric_value(group, "n_texts")
        if value > 0:
            return int(value)
    return default_n_texts


def clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def write_line_chart_svg(
    rows: list[dict[str, Any]],
    path: Path,
    value_key: str,
    title: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    averaged = average_rows_by_method_generation(rows, value_key)
    svg = build_line_chart_svg(averaged, value_key, title, y_label)
    path.write_text(svg, encoding="utf-8")


def average_rows_by_method_generation(rows: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    table = pd.DataFrame(rows)
    grouped = table.groupby(["method", "generation"], as_index=False)[value_key].mean()
    return grouped.sort_values(["method", "generation"]).to_dict("records")


def build_line_chart_svg(rows: list[dict[str, Any]], value_key: str, title: str, y_label: str) -> str:
    width = 900
    height = 520
    margin_left = 80
    margin_right = 180
    margin_top = 52
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    methods = sorted({str(row["method"]) for row in rows})
    generations = [int(row["generation"]) for row in rows]
    x_min = min(generations) if generations else 0
    x_max = max(generations) if generations else 1
    if x_min == x_max:
        x_max = x_min + 1
    palette = ["#2f6f9f", "#c64e4e", "#4f8a3b", "#8a5fbf", "#a66a2f", "#008c82"]

    def x_pos(generation: int) -> float:
        return margin_left + ((generation - x_min) / (x_max - x_min)) * plot_width

    def y_pos(value: float) -> float:
        return margin_top + (1.0 - clip01(safe_float(value))) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-family="Arial" font-size="20" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1"/>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        parts.append(
            f'<line x1="{margin_left - 4}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#e3e3e3" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12" fill="#444">{tick:.2f}</text>'
        )
    x_ticks = sorted(set(generations))
    if len(x_ticks) > 6:
        step = max(1, math.ceil(len(x_ticks) / 6))
        x_ticks = x_ticks[::step]
    for generation in x_ticks:
        x = x_pos(generation)
        parts.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" x2="{x:.2f}" y2="{margin_top + plot_height + 5}" stroke="#333" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#444">{generation}</text>'
        )
    parts.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">Generation</text>'
    )
    parts.append(
        f'<text transform="translate(22 {margin_top + plot_height / 2:.2f}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="13" fill="#333">{html.escape(y_label)}</text>'
    )
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for row in rows:
        by_method[str(row["method"])].append(row)
    for idx, method in enumerate(methods):
        color = palette[idx % len(palette)]
        points = [
            f'{x_pos(int(row["generation"])):.2f},{y_pos(safe_float(row[value_key])):.2f}'
            for row in sorted(by_method[method], key=lambda item: int(item["generation"]))
        ]
        if points:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{" ".join(points)}"/>'
            )
            for point in points:
                x, y = point.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
        legend_y = margin_top + 22 * idx
        legend_x = margin_left + plot_width + 28
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#333">{html.escape(method)}</text>'
        )
    if not rows:
        parts.append(
            f'<text x="{width / 2:.2f}" y="{height / 2:.2f}" text-anchor="middle" font-family="Arial" font-size="14" fill="#777">No data</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
