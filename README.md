# Binary MOPSO-CD

Python 3.13 implementation of the updated semantic Binary MOPSO-CD proposal.
It uses Ollama for LLM calls, SBERT embeddings for both objectives, a semantic
router/executor split, a discrete MOPSO-CD optimizer, and the final
Entropy-TOPSIS-MMR selection module.

## Setup

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev,models]"
.\.venv\Scripts\python -m spacy download en_core_web_sm
ollama pull llama3.1:8b
ollama pull qwen3.5:2b
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull phi4-mini
ollama pull ministral-3:3b
ollama pull gemma4:e4b
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull lfm2.5:8b
```

The setup commands are normally run once. Python dependencies stay inside
`.venv`, Ollama models stay in the local Ollama store, and Hugging Face models
stay in the user cache.

Pull only the Ollama models you plan to use. `phi4-mini` requires Ollama 0.5.13
or newer, and `ministral-3:3b` currently requires Ollama 0.13.1 pre-release.
Local contract checks for this implementation were run with Ollama 0.30.8.

PPDB is also local. By default, the project builds a SQLite index once under the
active Python environment:

```text
{venv}/var/binary_mopso_cd/ppdb_index.sqlite
```

The default PPDB source is:

```text
data/ppdb/ppdb-2.0-s-all
```

Use `--set models.ppdb.source_path=...` or `--set models.ppdb.index_path=...`
if your paths differ. The raw PPDB file and the generated SQLite index are not
tracked by Git.

## Run

```powershell
.\.venv\Scripts\python -m binary_mopso_cd `
  --reference-text "Evacuation orders remain in effect for Zone A until further notice."
```

Use `--config` to load an additional YAML file on top of `configs/default.yaml`:

```powershell
.\.venv\Scripts\python -m binary_mopso_cd `
  --reference-text "Evacuation orders remain in effect for Zone A until further notice." `
  --config configs/default.yaml
```

Every configurable value from the effective YAML can be overridden with repeated
`--set path.to.value=value` arguments. Values are parsed as YAML, so booleans,
numbers and lists keep their real types:

```powershell
.\.venv\Scripts\python -m binary_mopso_cd `
  --reference-text "Evacuation orders remain in effect for Zone A until further notice." `
  --set experiment.n=100 `
  --set experiment.iterations=100 `
  --set experiment.runs=3 `
  --set runtime.outdir_base=exec `
  --set mopso.archive_multiplier=0.5 `
  --set parallelism.enabled=true
```

More examples:

```powershell
--set ollama.default_model=llama3.1:8b
--set router.task_models.synthetic_text_generation=qwen3.5:2b
--set router.task_models.semantic_pool_generation=qwen3:4b-instruct-2507-q4_K_M
--set router.task_models.synthetic_text_generation=phi4-mini
--set router.task_models.synthetic_text_generation=gemma4:e4b
--set router.task_models.semantic_anchor_extraction=qwen3.5:4b
--set router.task_models.central_anchor_selection=qwen3.5:9b
--set router.task_models.semantic_pool_generation=lfm2.5:8b
--set router.phase_task_models.initialization.synthetic_text_generation=qwen3.5:2b
--set router.phase_task_models.optimization.synthetic_text_generation=ministral-3:3b
--set router.llm_params.synthetic_text_generation.thinking=true
--set router.llm_params.synthetic_text_generation.thinking=medium
--set router.llm_params.semantic_anchor_extraction.short.thinking=false
--set router.heuristics.semantic_pool_generation=false
--set models.sbert.default=all-MiniLM-L6-v2
--set models.ppdb.source_path=data/ppdb/ppdb-2.0-s-all
--set models.ppdb.index_path=.venv/var/binary_mopso_cd/ppdb_index.sqlite
--set experiment.frozen_components='["role","topic"]'
--set selection.enabled=false
--set checkpoint.enabled=true
--set checkpoint.interval=1
--set runtime.resume_from=exec/<run>/checkpoints/generation_0001.json
```

Task-level LLM model overrides apply to every phase. Phase-task overrides apply
only to the matching `RouteTask.operation_context` and inherit the task-level
model when set to `null`.

LLM thinking is disabled by default in every router LLM parameter bucket. A
task-level `router.llm_params.<task>.thinking` value is passed as Ollama's
per-call `think` field; if the task-level key is absent, the executor falls back
to `ollama.think`. Accepted values are `false`, `true`, `null`, `low`,
`medium`, and `high`; enabled thinking is rejected during config validation for
models declared without thinking support in `ollama.model_capabilities`.

The Ollama client uses the standard Chat request for every model:
`messages=[system,user]`, `options.temperature`, `options.top_p`, `format` for
structured JSON tasks, `stream=false`, and an explicit per-call `think`. Model
differences that affect response handling belong in `ollama.model_profiles`.
`lfm2.5:8b` is profiled to strip in-band `<think>...</think>` content from
responses because local verification showed that tag can appear in
`message.content` even when `think=false`.

Speculative decoding is intentionally not supported in this version. The config
contains a blocked flag so accidental activation fails during validation.
Checkpoints are disabled by default. When enabled, snapshots are written by a
single deferred worker and reference the persistent embedding cache file instead
of embedding the cache in every checkpoint.

## Outputs

Each run writes an EVOLMD-MO-style folder under `exec/<timestamp>/` containing:

- `reference.txt`
- `config_effective.yaml`
- `data_initial_population.json`
- `data_inicial_evaluada.json`
- `population_evaluated.json`
- `pareto_front.json`
- `pareto_ranked.json`
- `final_selection_hybrid.json`
- `evolucion_metricas.csv`
- `runtime.txt`
- `runtime.log`
- `llm_calls.jsonl`
- `cost_metrics.json`
- `monitor_metrics.csv` by default; set `monitor.enabled=false` to disable it
- `archive_history.jsonl`
- `checkpoints/generation_*.json` when checkpointing is enabled

These generated artifacts are ignored by Git.

`runtime.txt` includes both `runtime_seconds` and the EVOLMD-MO-compatible
`total_sec` alias. `cost_metrics.json` summarizes wall-clock seconds, LLM calls,
input tokens and output tokens from `llm_calls.jsonl`.

`evolucion_metricas.csv` includes archive counters for each generation.
`archive_update_count` is cumulative and only increases when the ordered
normalized signatures in the external archive change. `archive_prune_count` is
cumulative and increases once per update that performs pruning, regardless of
how many solutions are removed.

`config_effective.yaml` stores the final resolved configuration after `default.yaml`,
`--config` and all `--set` overrides have been applied.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```

The tests use real services for executor, embeddings and reduced integration
runs. Unit tests that cover pure optimizer logic avoid unnecessary model loading
but do not replace the runtime backend. Coverage includes router schedules,
executor/Ollama contract, embedding cache, objective calculation, dominance,
crowding distance, archive pruning, frozen components, final selection,
checkpoint behavior, lazy sampling and reduced end-to-end runs.
