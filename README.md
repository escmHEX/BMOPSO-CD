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

Use `--ppdb-source` or `--ppdb-index` if your paths differ. The raw PPDB file and
the generated SQLite index are not tracked by Git.

## Run

```powershell
.\.venv\Scripts\python -m binary_mopso_cd --reference-text "Evacuation orders remain in effect for Zone A until further notice." --n 100 --iterations 100
```

Useful flags:

```text
--runs 3
--model llama3.1:8b
--bert-model all-MiniLM-L6-v2
--outdir-base exec
--freeze-components role,topic
--enable-monitor
--disable-selection
--router-heuristic semantic_pool_generation=false
--task-model synthetic_text_generation=qwen3.5:2b
--ppdb-source data/ppdb/ppdb-2.0-s-all
--ppdb-index .venv/var/binary_mopso_cd/ppdb_index.sqlite
--enable-checkpoint
--checkpoint-every 1
--resume-from exec/<run>/checkpoints/generation_0001.json
```

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
- `archive_history.jsonl`
- `checkpoints/generation_*.json` when checkpointing is enabled

These generated artifacts are ignored by Git.

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
