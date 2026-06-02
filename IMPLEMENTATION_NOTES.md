# Implementation Notes

This repository implements the updated HTML strategy set as the normative
source. The older anteproyecto and baseline repositories are used only for
comparability decisions such as output folder shape, fixed-iteration stopping,
Ollama usage, and external observational metrics.

## Decisions

- Semantic individuals are component vectors with default components `role`,
  `topic`, and `action`, but the code stores them as mappings for extension.
- The semantic router returns execution tasks only; all model calls and resource
  ownership live in the executor and services.
- Ollama calls are single-shot `system` plus `user` requests with `stream=false`.
  No conversational history is retained between calls.
- SBERT embeddings are batched and cached with keys that include text type,
  model alias, config version, and canonical text.
- Heavy resources are service-owned and loaded once per executor process.
- Checkpoints are disabled by default. If enabled, a single deferred writer saves
  a minimal optimizer state and references the persistent embedding cache file.
- `f2` is vectorized through an NxN cosine matrix and never loops through
  repeated embedding calls.
- The monitor is observational. It writes metrics and overhead but is not read
  by the optimizer.
- Speculative decoding is blocked because no exact Ollama option was selected
  for this implementation.

## Baseline Compatibility

Default experimental values match the current proposal and baseline protocol:

- `N=100`
- `iterations=100`
- `runs=3`
- final selection `K=5`

The CLI preserves baseline-style names where useful but uses `--reference-text`
and `--iterations` as requested.

## Fidelity Traceability

| Normative source | Strategy rule | Implementation | Config | Test |
| --- | --- | --- | --- | --- |
| `estrategia_modulo_enrutamiento_semantico_v2_top_p_synthetic.html` | Router selects algorithms and LLM parameters; influence schedule uses zero-based iteration in `[0, G-1]`. | `src/binary_mopso_cd/router.py` | `router.heuristics`, `router.llm_params`, `router.task_models` | `tests/unit/test_router.py`, `tests/unit/test_runtime_schedules.py` |
| `estrategia_modulo_ejecucion_tareas_semanticas_v1.html` | Executor dispatches `LLM`, `SBERT`, `DeterministicPromptGenerator`, `distilbert_fill_mask`, `wordnet_ppdb`; no router-side execution. | `src/binary_mopso_cd/executor.py`, `src/binary_mopso_cd/services/*` | `models.*`, `ollama.*` | `tests/unit/test_executor.py` |
| `estrategia_generacion_texto_gi_v3_executor_clase_estatica.html` | Synthetic generation sends exactly the rendered prompt as user prompt, plain text only, `stream=false`. | `src/binary_mopso_cd/llm_prompts.py`, `src/binary_mopso_cd/services/ollama_client.py` | `router.llm_params.synthetic_text_generation`, `ollama.stream` | `tests/unit/test_executor.py` |
| `estrategia_composicion_deterministica_prompt_v6_generalizada_corregida.html` | One deterministic canonical template per component subset, generalized template for extra components, no LLM composer. | `src/binary_mopso_cd/services/prompt_renderer.py` | `semantic_components.order` | `tests/integration/test_reduced_runs.py` |
| `estrategia_inicializacion_poblacion_hibrida_v12_Router.html` | Extract anchors, build/expand pools, never exceed `4N` sampled combinations, reduce to `2N` by prompt diversity, generate texts and select top `N`. | `src/binary_mopso_cd/initialization.py` | `initialization.*`, `semantic_components.rules`, `semantic_components.expansion_order` | `tests/unit/test_initialization_sampling.py`, `tests/integration/test_reduced_runs.py` |
| `estrategia_mopso_cd_semantica_v16_pbest_archive_lote.html` | Discrete MOPSO-CD with active/frozen components, schedules in `[0,1]`, `Dmax`, `Kcand`, no retries, evaluate modified generation in batch. | `src/binary_mopso_cd/mopso.py`, `src/binary_mopso_cd/component_memory.py` | `mopso.*`, `experiment.frozen_components` | `tests/unit/test_runtime_schedules.py`, `tests/unit/test_frozen_components.py`, `tests/integration/test_reduced_runs.py` |
| `mopso_cd_lider_poda_archivo_v4.html` | Global archive `Amax=2N`, non-dominated update, exact duplicate first occurrence, pruning crowding distance, leader tournament `q=3`. | `src/binary_mopso_cd/mopso.py` | `mopso.archive_multiplier`, `mopso.leader_tournament_size` | `tests/unit/test_mopso_core.py` |
| `pbest_Ux_suma_ponderada_dominios_teoricos.html` | `pbest` update by dominance; if incomparable, choose higher `U(x)=(f1+1+f2)/4`; ties keep previous. | `src/binary_mopso_cd/mopso.py` | `mopso.utility_weights` | `tests/unit/test_mopso_core.py` |
| `estrategia_turbulencia_distilbert_fillmask_v8_formulas_corregidas.html` | DistilBERT receives target word/span and returns up to `Kcand` one-word replacement variants with preliminary top-k `3Kcand`. | `src/binary_mopso_cd/services/turbulence.py`, `src/binary_mopso_cd/router.py` | `models.distilbert.top_k_multiplier`, `mopso.kcand` | `tests/unit/test_router.py` |
| `estrategia_turbulencia_wordnet_ppdb_v11_formulas_corregidas.html` | Boundary replacement uses WordNet by lemma/POS with PPDB fallback, returning variants only; semantic filtering stays in MOPSO. | `src/binary_mopso_cd/services/turbulence.py`, `src/binary_mopso_cd/mopso.py` | `models.wordnet.enabled`, `models.ppdb.index_path`, `models.ppdb.enabled` | `tests/unit/test_router.py` |
| `estrategia_filtro_similitud_mmr_topsis_entropy_v5_red_no_negativa.html` | Final selection applies similarity filter, Entropy Method, TOPSIS and MMR with non-negative redundancy. | `src/binary_mopso_cd/selection.py` | `selection.*` | `tests/unit/test_selection.py` |
| `arquitecturaEstrategia.png` | Reference data to initial population, routing/executor, optimizer, archive, final selection and generated dataset outputs. | `src/binary_mopso_cd/runner.py`, `src/binary_mopso_cd/outputs.py` | `runtime.outdir_base`, `selection.enabled` | `tests/integration/test_reduced_runs.py` |
| `diagrama_optimizador_mopso_cd_corregido.png` | Per generation: update particles, evaluate changed solutions or reuse cache, update `pbest`, update archive, then stop/continue. | `src/binary_mopso_cd/mopso.py` | `mopso.*`, `checkpoint.*` | `tests/integration/test_reduced_runs.py` |
| `defecto.png` | Defaults `G=100`, `K=3`, `N=100`. | `configs/default.yaml` | `experiment.iterations`, `experiment.runs`, `experiment.n` | `tests/unit/test_runtime_schedules.py` |
