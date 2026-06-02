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

