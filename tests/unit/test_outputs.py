from __future__ import annotations

import json

from binary_mopso_cd.outputs import RuntimeTimer, summarize_llm_costs, write_cost_metrics


def test_summarize_llm_costs_aggregates_calls_tokens_and_task_model_breakdown(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    rows = [
        {
            "semantic_task": "synthetic_text_generation",
            "model": "llama3.1:8b",
            "input_tokens": 5,
            "output_tokens": 7,
        },
        {
            "semantic_task": "synthetic_text_generation",
            "model": "llama3.1:8b",
            "promptEvalCount": 3,
            "evalCount": 4,
        },
        {
            "semantic_task": "semantic_anchor_extraction",
            "model": "qwen3.5:2b",
            "prompt_eval_count": 2,
            "eval_count": 1,
        },
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_llm_costs(log_path, wall_clock_seconds=12.5)

    assert summary["wall_clock_seconds"] == 12.5
    assert summary["llm_calls_total"] == 3
    assert summary["input_tokens_total"] == 10
    assert summary["output_tokens_total"] == 12
    assert summary["tokens_total"] == 22
    assert summary["calls_by_task_model"] == {
        "synthetic_text_generation::llama3.1:8b": 2,
        "semantic_anchor_extraction::qwen3.5:2b": 1,
    }
    assert summary["input_tokens_by_task_model"]["synthetic_text_generation::llama3.1:8b"] == 8
    assert summary["output_tokens_by_task_model"]["synthetic_text_generation::llama3.1:8b"] == 11


def test_write_cost_metrics_writes_json_summary(tmp_path):
    log_path = tmp_path / "llm_calls.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "semantic_task": "synthetic_text_generation",
                "model": "llama3.1:8b",
                "input_tokens": 5,
                "output_tokens": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = write_cost_metrics(tmp_path / "cost_metrics.json", llm_log_path=log_path, wall_clock_seconds=3.25)

    persisted = json.loads((tmp_path / "cost_metrics.json").read_text(encoding="utf-8"))
    assert persisted == summary
    assert persisted["wall_clock_seconds"] == 3.25
    assert persisted["llm_calls_total"] == 1
    assert persisted["tokens_total"] == 12


def test_runtime_timer_writes_total_sec_alias(tmp_path):
    timer = RuntimeTimer()

    payload = timer.write(tmp_path / "runtime.txt", {"run_index": 1})

    content = (tmp_path / "runtime.txt").read_text(encoding="utf-8")
    assert "runtime_seconds:" in content
    assert "total_sec:" in content
    assert payload["runtime_seconds"] == payload["total_sec"]
    assert payload["run_index"] == 1
