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
```

The setup commands are normally run once. Python dependencies stay inside
`.venv`, Ollama models stay in the local Ollama store, and Hugging Face models
stay in the user cache.

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
--set router.phase_task_models.initialization.synthetic_text_generation=qwen3.5:2b
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
