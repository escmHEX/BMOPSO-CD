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
- Progress logging is observational. It writes one low-cost line per generation
  using already evaluated objectives and never feeds optimizer decisions.
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

## Mathematical Documentation Format

Markdown tables are kept here as a searchable text fallback, but they are not the
best surface for formula rendering or color grouping. For rendered LaTeX with
real color differentiation by strategy, use the companion HTML document:
[IMPLEMENTATION_NOTES_FORMULAS.html](IMPLEMENTATION_NOTES_FORMULAS.html).

The group identifier is the authoritative marker in both documents. The HTML
document uses the same group identifiers with colored section borders, cards and
hyperparameter rows.

| Group | Color label | Strategy family |
| --- | --- | --- |
| `MODEL` | blue | Semantic individual, objective model and baseline defaults |
| `ROUTER` | violet | Semantic routing and LLM sampling policy |
| `EXEC` | gray | Semantic task executor, Ollama and embeddings |
| `PROMPT` | teal | Deterministic prompt composition |
| `INIT` | green | Hybrid initial population generation |
| `OBJ` | cyan | Objective functions and embedding cache |
| `MOPSO` | orange | Discrete binarized MOPSO-CD update |
| `ARCHIVE` | amber | External archive, dominance, crowding distance and leader selection |
| `PBEST` | yellow | Personal best update by dominance and utility |
| `TURB` | red | DistilBERT and WordNet/PPDB turbulence |
| `SELECT` | purple | Similarity filter, Entropy Method, TOPSIS and MMR |
| `MONITOR` | slate | Observational metrics outside optimizer decisions |
| `CHECKPOINT` | brown | Deferred checkpointing and resume support |

## Mathematical Conventions

- \(N\) is the population size, configured by `experiment.n` and CLI `--n`.
- \(G\) is the number of optimizer iterations, configured by
  `experiment.iterations` and CLI `--iterations`.
- \(K_{runs}\) is the number of independent executions, configured by
  `experiment.runs`.
- \(K_{sel}\) is the final number of selected solutions, configured by
  `selection.k`.
- \(K_{cand}\) is the maximum number of candidate variants per modification
  operator, configured by `mopso.kcand`.
- \(D\) is the number of semantic components. By default
  \(\mathcal{D}=\{role,topic,action\}\) and \(D=3\).
- Generation schedules use zero-based progress. For generation
  \(g\in\{1,\ldots,G\}\), the schedule index is \(t=g-1\), so
  \(t\in\{0,\ldots,G-1\}\).
- Both objective functions are maximized. Semantic fidelity \(f_1\) uses cosine
  similarity and semantic diversity \(f_2\) uses average cosine distance.
- Unless otherwise stated, \(E(s)\) is the SBERT embedding of text \(s\). The
  implementation normalizes embeddings in `EmbeddingService`, so dot products
  between encoded vectors are cosine similarities.
- Fidelity statuses document the formula currently executed by the code after
  contrast against the normative HTML strategies.

## Contrast Findings

| Area | Corrected formula | Strategy formula | Review status |
| --- | --- | --- | --- |
| `MOPSO` semantic distance in velocity | \(\Delta(a,b)=\frac{1-\operatorname{SimCos}(E(a),E(b))}{2}\) | \(\Delta(a,b)=\frac{1-\operatorname{SimCos}(E(a),E(b))}{2}\) | OK |
| `SELECT` Entropy Method column shift | \(\delta_j=\max(0,-\min_i a_{ij})+\varepsilon\), \(b_{ij}=a_{ij}+\delta_j\) | \(\delta_j=\max(0,-\min_i a_{ij})+\varepsilon\), \(b_{ij}=a_{ij}+\delta_j\) | OK |

## Funciones matemáticas implementadas

| Group | Function or rule | LaTeX formula used by the implementation | Implementation and config | Fidelity |
| --- | --- | --- | --- | --- |
| `MODEL` | Semantic individual | \(x_i=(x_{i,role},x_{i,topic},x_{i,action})\), generalized as \(x_i=\{(c_j,v_{i,j})\}_{j=1}^{D}\). | `entities.SemanticVector`; `experiment.components`, `semantic_components.order`. | OK |
| `MODEL` | Default stopping protocol | \(N=100,\quad G=100,\quad K_{runs}=3\). | `configs/default.yaml`; `defecto.png`. | OK |
| `PROMPT` | Deterministic prompt rendering | \(P_i=R_S(x_i)\), where \(S\subseteq\mathcal{D}\) is the available component subset. | `PromptRenderer.render`; `semantic_components.order`. No LLM composer is used. | OK |
| `PROMPT` | Prompt render cost | \(C_{render}^{Template}=0\), \(\Delta C_{render}=NG-0=NG\). | Only normalization and string concatenation are used. | OK |
| `ROUTER` | Router contract | \(\mathcal{R}(\tau,p)=(a,p',\theta)\), where \(\tau\) is the semantic task, \(a\) the selected algorithm and \(\theta\) the runtime config. | `SemanticRouter.route`; `router.heuristics`, `router.task_models`, `router.llm_params`. | OK |
| `ROUTER` | Progress ratio | \(\rho(t,G)=0\) if \(G\le1\); otherwise \(\rho(t,G)=\frac{t}{G-1}\), with \(t\in[0,G-1]\). | `utils.progress_ratio`; router passes zero-based iteration. | OK |
| `ROUTER` | Influence candidate sampling | \(T(t)=T_{start}-(T_{start}-T_{end})\rho(t,G)\); \(p_{top}(t)=p_{start}-(p_{start}-p_{end})\rho(t,G)\). Defaults: \(T(t)=0.70-0.20\rho\), \(p_{top}(t)=0.95-0.05\rho\). | `router.llm_params.semantic_component_influence_candidates`. | OK |
| `ROUTER` | Anchor extraction policy | If \(n_y\le6\), use \(T=0.25,p_{top}=0.90\); otherwise \(T=0.20,p_{top}=0.85\). | `router.llm_params.semantic_anchor_extraction.short_word_threshold`. | OK |
| `ROUTER` | Pool evidence policy | Low evidence if \(n_y\le6\) or \(a_c<4\). Select \((T,p_{top})\) by component and evidence state. | `router.llm_params.semantic_pool_generation`. | OK |
| `EXEC` | LLM execution | \(y=LLM(S_{\tau},U_{\tau}(p');\theta)\). | `SemanticTaskExecutor`; `ollama.stream=false`; no conversational history is retained. | OK |
| `EXEC` | Embedding execution | \(e=E(s)\in\mathbb{R}^{d_z}\). | `EmbeddingService.encode`; SBERT model from `models.sbert.*`. | OK |
| `EXEC` | Synthetic text generation | \(G_i=LLM(S_{gen},P_i;\theta_{gen})\), with \(T_{gen}=0.75\) and \(p_{gen}=0.95\). | `llm_prompts.py`, `TASK_SYNTHETIC_TEXT`; `router.llm_params.synthetic_text_generation`. | OK |
| `EXEC` | Generated text feasibility | Accept generated text only if it is non-empty, not the reference, not a refusal phrase, and \(f_1\ge\tau_{gen}^{min}\). Initialization also enforces duplicate and sentence-limit filters. | `generated_text_validation.validate_generated_text`; `generated_text_validation.tau_gen_min=0.05`. | OK |
| `INIT` | Pool base size | \(c=\max\left(2,\operatorname{round}\left(\left(\frac{4N}{\prod_{d\in\mathcal{D}}\alpha_d}\right)^{1/D}\right)\right)\). | `choose_pool_sizes`; `semantic_components.rules.*.alpha`. | OK |
| `INIT` | Per-component target size | \(q_d=\lceil\alpha_d c\rceil\). Increase \(q_d\) in expansion order until \(\prod_d q_d\ge4N\). | `choose_pool_sizes`; `semantic_components.expansion_order`. | OK |
| `INIT` | Minimum pool product | \(\prod_{d\in\mathcal{D}}\lvert P_d\rvert\ge3N\). | `initialization.min_product_multiplier=3`; one pool expansion is attempted before failing. | OK |
| `INIT` | Candidate cap | \(M_{cand}=\min\left(\prod_d\lvert P_d\rvert,4N\right)\). | `_candidate_vectors`; lazy stratified sampling when product exceeds `4N`. | OK |
| `INIT` | Prompt distance | \(d_P(P_i,P_j)=1-\operatorname{SimCos}(E(P_i),E(P_j))\). | `_reduce_by_prompt_diversity`; `text_type=prompt`. | OK |
| `INIT` | Greedy max-min reduction | \(P^*=\arg\max_{P_i\in C\setminus S}\min_{P_j\in S}d_P(P_i,P_j)\). | `greedy_max_min_indices`; reduces to \(\min(2N,M_{cand})\). | OK |
| `INIT` | Initial final ranking | Select top \(N\) from generated valid texts by descending \((f_1,\ score_{div,prompt})\). | `InitialPopulationBuilder.build`; fidelity is computed on generated texts, not prompts. | OK |
| `OBJ` | Cosine similarity | \(\operatorname{SimCos}(u,v)=\frac{u^\top v}{\lVert u\rVert_2\lVert v\rVert_2}\). | `utils.cosine_matrix`, normalized SBERT embeddings. | OK |
| `OBJ` | Semantic fidelity | \(f_1(x_i)=\operatorname{SimCos}(E(G_i),E(y_{ref}))\). | `evaluate_solutions`; `text_type=generated_text` and `reference_text`. | OK |
| `OBJ` | Semantic diversity | \(f_2(x_i)=\frac{1}{N-1}\sum_{j\ne i}\left(1-\operatorname{SimCos}(E(G_i),E(G_j))\right)\). If \(N\le1\), \(f_2(x_i)=0\). | `semantic_diversity_scores`; one NxN matrix. | OK |
| `OBJ` | Vectorized distance matrix | \(D_{ij}=1-S_{ij}\), where \(S=ZZ^\top\) for normalized embeddings \(Z\), and \(D_{ii}=0\). | `semantic_diversity_scores`; no repeated text-by-text embedding calls. | OK |
| `OBJ` | MOPSO evaluation set | \(S_t=\operatorname{unique}(P_{t+1}\cup PB_t\cup A_t)\), using `signature(x)` from normalized semantic components and keeping the first occurrence. | `evaluate_unique_solutions_by_signature`; same identity as `ExternalArchive.update`. | OK |
| `OBJ` | Embedding cache key | \(key=\operatorname{SHA256}(\{type,model,version,canonical(text)\})\). | `EmbeddingCacheKey.to_digest`; `models.sbert.config_version`. | OK |
| `MOPSO` | Active components | \(\mathcal{D}_{active}=\mathcal{D}\setminus\mathcal{D}_{frozen}\), \(F=\lvert\mathcal{D}_{frozen}\rvert\), \(0\le F\le D-1\). | `MOPSOOptimizer.active_components`; `experiment.frozen_components`. | OK |
| `MOPSO` | Semantic distance in velocity | \(\Delta(a,b)=\frac{1-\operatorname{SimCos}(E(a),E(b))}{2}\). | `semantic_velocity_delta`; used by `mopso._update_particle` for `delta_p` and `delta_l`. | OK |
| `MOPSO` | Inertia schedule | \(\omega(t)=\omega_{max}-(\omega_{max}-\omega_{min})\rho(t,G)\). Defaults: \(\omega_{max}=0.9,\omega_{min}=0.4\). | `mopso._update_particle`; `mopso.omega_max`, `mopso.omega_min`. | OK |
| `MOPSO` | Turbulence schedule | \(p_{tur}(t)=p_{tur}^{max}-(p_{tur}^{max}-p_{tur}^{min})\rho(t,G)\). Defaults: \(0.07\to0.02\). | `mopso._update_particle`; `mopso.p_tur_max`, `mopso.p_tur_min`. | OK |
| `MOPSO` | Velocity terms | \(s_{in}=\omega(t)v_{i,d}^{t}\); \(s_{cog}=c_1r_1\Delta(x_{i,d}^{t},pbest_{i,d}^{t})\); \(s_{soc}=c_2r_2\Delta(x_{i,d}^{t},L_{i,d}^{t})\); \(v_{raw}=s_{in}+s_{cog}+s_{soc}\). | `mopso._update_particle`; `c1`, `c2`. | OK |
| `MOPSO` | Inertia selection weight | \(a_{in}=\omega(t)\lvert v_{i,d}^{t}\rvert\). | `s_in_weight`; used only for move-source weighting. | OK |
| `MOPSO` | Velocity clamp | \(\hat v_{i,d}^{t+1}=\min(V_{max},\max(-V_{max},v_{raw}))\). | `mopso._update_particle`; `mopso.vmax`. | OK |
| `MOPSO` | V-shaped transfer | \(q_{pso,i,d}^{t}=T_\alpha(\hat v)=\lvert\tanh(\alpha\hat v)\rvert\). | `math.tanh`; `mopso.alpha=1`. | OK |
| `MOPSO` | Effective candidate probability | \(q_{eff}=1-(1-q_{pso})(1-p_{tur})\). | `mopso._update_particle`; turbulence checked first, guided move second. | OK |
| `MOPSO` | Per-generation change cap | \(\lvert M_i^t\rvert\le D_{max}\le\lvert\mathcal{D}_{active}\rvert\). | Candidate list is truncated by weighted sampling; frozen components are skipped. | OK |
| `MOPSO` | Guided candidate validation | Accept candidate \(c\) only if \(\operatorname{SimCos}(E(c),E(target))>\operatorname{SimCos}(E(current),E(target))\) and \(\max_{m\in Mem_d}\operatorname{SimCos}(E(c),E(m))<\tau_{dup}\). | `_select_guided_candidate`; batched memory index. | OK |
| `MOPSO` | Turbulence candidate validation | Accept candidate \(c\) only if \(\tau_{tur}^{min}\le\operatorname{SimCos}(E(c),E(current))\le\tau_{tur}^{max}\) and \(\max_{m\in Mem_d}\operatorname{SimCos}(E(c),E(m))<\tau_{dup}\). | `_select_turbulence_candidate`; `tau_tur_min`, `tau_tur_max`, `tau_dup`. | OK |
| `ARCHIVE` | Pareto dominance | \(x\succ y\iff f_1(x)\ge f_1(y)\land f_2(x)\ge f_2(y)\land(f_1(x)>f_1(y)\lor f_2(x)>f_2(y))\). | `dominates`. | OK |
| `ARCHIVE` | External archive update | \(A^t=ND(A^{t-1}\cup P^t)\), then prune if \(\lvert A^t\rvert>A_{max}\). | `ExternalArchive.update`. | OK |
| `ARCHIVE` | Archive maximum size | \(A_{max}=2N\). | `archive_multiplier=2`; `ExternalArchive(max_size=2N)`. | OK |
| `ARCHIVE` | Crowding distance | \(CD(x_r)\mathrel{+}= \frac{f_j(x_{r+1})-f_j(x_{r-1})}{f_j^{max}-f_j^{min}}\) for each objective \(j\); extremes receive \(\infty\). | `crowding_distance`; skips objectives with equal min and max. | OK |
| `ARCHIVE` | Pruning by CD | While \(\lvert A\rvert>A_{max}\), remove one solution with minimum \(CD\), breaking ties randomly with run seed. | `ExternalArchive._prune`. | OK |
| `ARCHIVE` | Leader tournament | \(q_{eff}^{leader}=\min(q,\lvert A\rvert)\), \(q=3\); winner has maximum crowding distance. | `ExternalArchive.select_leader`; `leader_tournament_size=3`. | OK |
| `PBEST` | pbest dominance rules | If \(x_{actual}\succ pbest_i\), update; if \(pbest_i\succ x_{actual}\), keep; otherwise compare \(U(x)\). | `PBestUpdater.update`. | OK |
| `PBEST` | Utility normalization | \(\tilde f_1(x)=\frac{f_1(x)+1}{2}\), \(\tilde f_2(x)=\frac{f_2(x)}{2}\). | `utility`. | OK |
| `PBEST` | Utility tie-break | \(U(x)=w_1\tilde f_1(x)+w_2\tilde f_2(x)\). With \(w_1=w_2=0.5\), \(U(x)=\frac{f_1(x)+1+f_2(x)}{4}\). Ties keep previous pbest. | `mopso.utility`; `mopso.utility_weights`. | OK |
| `TURB` | DistilBERT preliminary top-k | \(M_{tur}=3K_{cand}\). | `models.distilbert.top_k_multiplier=3`; router sends `preliminaryTopK`. | OK |
| `TURB` | DistilBERT route condition | Use DistilBERT if \(L_w\ge1\land R_w\ge1\); otherwise use WordNet first with PPDB fallback. | `router._route_word_replacement`. | OK |
| `TURB` | Variant count | \(\lvert C_K\rvert\le K_{cand}\). | `TurbulenceService`; `mopso.kcand=7`. | OK |
| `SELECT` | Similarity filter | \(\mathcal{C}=\{x_i\in\mathcal{F}:\tau_{min}\le F_1(x_i)\le\tau_{max}\}\). | `rank_solutions`; `selection.tau_min`, `selection.tau_max`. | OK |
| `SELECT` | Decision matrix | \(A=[a_{ij}]\in\mathbb{R}^{n_f\times2}\), \(a_{i1}=F_1(x_i)\), \(a_{i2}=F_2(x_i)\). | `rank_solutions`. | OK |
| `SELECT` | Entropy column shift | \(\delta_j=\max(0,-\min_i a_{ij})+\varepsilon\), \(b_{ij}=a_{ij}+\delta_j\). | `entropy_weights`. | OK |
| `SELECT` | Entropy weights | \(p_{ij}=b_{ij}/\sum_i b_{ij}\); \(e_j=-\frac{1}{\ln n_f}\sum_i p_{ij}\ln p_{ij}\); \(d_j=1-e_j\); \(w_j=d_j/\sum_k d_k\). | `entropy_weights`; uniform weights if \(n_f\le1\) or total divergence is zero. | OK |
| `SELECT` | TOPSIS normalization and score | \(r_{ij}=a_{ij}/\sqrt{\sum_i a_{ij}^2}\); \(v_{ij}=w_jr_{ij}\); \(A_j^+=\max_i v_{ij}\); \(A_j^-=\min_i v_{ij}\); \(C_i=\frac{D_i^-}{D_i^++D_i^-}\). | `topsis_scores`; zero denominators are set to 1. | OK |
| `SELECT` | Non-negative redundancy | \(\operatorname{Sim}_+(u,v)=\max(0,\operatorname{SimCos}(u,v))\). | `mmr_select`; embeddings for generated texts. | OK |
| `SELECT` | MMR score | \(\operatorname{MMR}(x)=\lambda C_x-(1-\lambda)\max_{y\in S}\operatorname{Sim}_+(E(G_x),E(G_y))\). | `mmr_select`; `selection.lambda_mmr=0.35`. | OK |
| `SELECT` | Final selection count | \(K_{eff}=\min(K_{sel},n_f)\). | `selection.k`; loop stops at \(K_{sel}\) or no remaining candidates. | OK |
| `MONITOR` | KMeans inertia | \(I=\sum_i\lVert z_i-\mu_{cluster(i)}\rVert_2^2\). | `ObservationalMonitor.observe`; not read by optimizer. | OK |
| `MONITOR` | Entity entropy | \(H_{ent}=-\sum_{\ell}p_\ell\ln p_\ell\), where \(p_\ell\) is the empirical frequency of entity label \(\ell\). | `entity_entropy`; spaCy labels only; not read by optimizer. | OK |
| `RUNTIME` | HV logging normalization | \(x=\operatorname{clip}((f_1+1)/2,0,1)\), \(y=\operatorname{clip}(f_2/2,0,1)\). | `metrics.normalized_objective_point`; used only for progress metrics. | OK |
| `RUNTIME` | Hypervolume progress metric | \(HV=\sum_k(x_k-x_{k-1})y_k\), over collapsed non-dominated points sorted by \(x\), with reference \((0,0)\). | `metrics.calculate_hypervolume`; not read by optimizer. | OK |
| `RUNTIME` | Spread progress metric | \(Spread=\frac{\sum_i\lvert d_i-\bar d\rvert}{m\bar d}\), where \(d_i\) are consecutive distances in the normalized non-dominated front. | `metrics.calculate_spread`; not read by optimizer. | OK |
| `CHECKPOINT` | Checkpoint trigger | Save only if checkpoints are enabled and \(g\bmod interval=0\). | `CheckpointManager`; disabled by default. | OK |

## Hiperparámetros implementados

| Group | Hyperparameter | Symbol | Default | Config or CLI | Use | Fidelity |
| --- | --- | --- | --- | --- | --- | --- |
| `MODEL` | Population size | \(N\) | 100 | `experiment.n`, `--n` | Number of particles/individuals. | OK |
| `MODEL` | Optimizer iterations | \(G\) | 100 | `experiment.iterations`, `--iterations` | Fixed stopping criterion. | OK |
| `MODEL` | Independent runs | \(K_{runs}\) | 3 | `experiment.runs`, `--runs` | Repeated independent executions for comparability. | OK |
| `MODEL` | Random seed | \(seed\) | 42 | `experiment.seed` | Reproducible sampling, archive tie breaks and tournaments. | Configurable implementation detail |
| `MODEL` | Domain | \(D_{gen}\) | crisis and emergency-related social media messages | `experiment.domain` | Domain used in prompts and initial pools. | OK |
| `MODEL` | Semantic components | \(\mathcal{D}\) | `role`, `topic`, `action` | `experiment.components`, `semantic_components.order` | Component dimensions of each individual. | OK |
| `MODEL` | Frozen components | \(\mathcal{D}_{frozen}\) | empty | `experiment.frozen_components`, `--freeze-components` | Components skipped by MOPSO update, still present in prompt. | OK |
| `PROMPT` | Component order | \(S\) order | `role`, `topic`, `action` | `semantic_components.order` | Stable deterministic prompt rendering. | OK |
| `PROMPT` | Render LLM calls | \(C_{render}^{Template}\) | 0 | Not configurable | Eliminates LLM prompt composition. | OK |
| `ROUTER` | Router heuristics | \(h_\tau\) | all enabled | `router.heuristics.*`, `--router-heuristic` | Enables/disables each specified routing heuristic. | OK |
| `ROUTER` | LLM model per task | \(LLM_\tau\) | `llama3.1:8b` | `router.task_models.*`, `--task-model` | Selects model for LLM semantic tasks. | OK |
| `ROUTER` | Alternative LLM | \(LLM_{alt}\) | `qwen3.5:2b` | `ollama.alternative_model` | Available configurable model option. | OK |
| `ROUTER` | Anchor short threshold | \(n_y^{short}\) | 6 words | `router.llm_params.semantic_anchor_extraction.short_word_threshold` | Chooses short vs long anchor extraction sampling. | OK |
| `ROUTER` | Anchor short sampling | \((T,p_{top})\) | `(0.25, 0.90)` | `router.llm_params.semantic_anchor_extraction.short` | LLM anchor extraction for short references. | OK |
| `ROUTER` | Anchor long sampling | \((T,p_{top})\) | `(0.20, 0.85)` | `router.llm_params.semantic_anchor_extraction.long` | LLM anchor extraction for longer references. | OK |
| `ROUTER` | Pool low-evidence word threshold | \(n_y^{low}\) | 6 words | `router.llm_params.semantic_pool_generation.low_evidence_word_threshold` | Part of low-evidence pool sampling rule. | OK |
| `ROUTER` | Pool low-evidence anchor threshold | \(a_c^{low}\) | 4 anchors | `router.llm_params.semantic_pool_generation.low_evidence_anchor_threshold` | Part of low-evidence pool sampling rule. | OK |
| `ROUTER` | Role pool sampling | \((T,p_{top})\) | low `(0.68,0.95)`, normal `(0.60,0.90)` | `router.llm_params.semantic_pool_generation.components.role` | Pool generation and expansion for `role`. | OK |
| `ROUTER` | Topic pool sampling | \((T,p_{top})\) | low `(0.42,0.90)`, normal `(0.40,0.89)` | `router.llm_params.semantic_pool_generation.components.topic` | Pool generation and expansion for `topic`. | OK |
| `ROUTER` | Action pool sampling | \((T,p_{top})\) | low `(0.58,0.94)`, normal `(0.50,0.90)` | `router.llm_params.semantic_pool_generation.components.action` | Pool generation and expansion for `action`. | OK |
| `ROUTER` | Influence temperature range | \(T_{start},T_{end}\) | `0.70`, `0.50` | `router.llm_params.semantic_component_influence_candidates.temperature_start/end` | Dynamic sampling for guided semantic candidates. | OK |
| `ROUTER` | Influence top-p range | \(p_{start},p_{end}\) | `0.95`, `0.90` | `router.llm_params.semantic_component_influence_candidates.top_p_start/end` | Dynamic top-p for guided semantic candidates. | OK |
| `ROUTER` | Synthetic generation sampling | \(T_{gen},p_{gen}\) | `0.75`, `0.95` | `router.llm_params.synthetic_text_generation` | Final text generation from rendered prompt. | OK |
| `EXEC` | Ollama host | \(host\) | `http://127.0.0.1:11434` | `ollama.host` | Ollama endpoint. | Runtime setting |
| `EXEC` | Ollama timeout | \(timeout\) | 120 seconds | `ollama.timeout_seconds` | Request timeout. | Runtime setting |
| `EXEC` | Ollama stream | \(stream\) | `false` | `ollama.stream` | Enforced single response, no streaming. | OK |
| `EXEC` | Ollama thinking | \(think\) | `false` | `ollama.think` | Disables extra thinking payload when supported. | OK |
| `EXEC` | Speculative decoding | \(specdec\) | `false` | `ollama.speculative_decoding_enabled` | Blocked and documented as unsupported for this version. | OK |
| `EXEC` | SBERT default model | \(E\) | `all-MiniLM-L6-v2` | `models.sbert.default`, `--bert-model` | Embeddings for objectives, prompts and validation. | OK |
| `EXEC` | SBERT alternative | \(E_{alt}\) | `gte-small` | `models.sbert.alternatives.gte-small` | Configurable embedding model. | OK |
| `EXEC` | Embedding batch size | \(B_E\) | 64 | `models.sbert.batch_size` | Batching for SBERT encode calls. | OK |
| `EXEC` | Embedding config version | \(v_E\) | `sbert-v1` | `models.sbert.config_version` | Included in embedding cache key. | OK |
| `EXEC` | DistilBERT model | \(MLM\) | `distilbert-base-uncased` | `models.distilbert.model` | Fill-mask turbulence provider. | OK |
| `EXEC` | DistilBERT top-k multiplier | \(m_{tur}\) | 3 | `models.distilbert.top_k_multiplier` | Computes \(M_{tur}=3K_{cand}\). | OK |
| `EXEC` | spaCy model | \(NLP\) | `en_core_web_sm` | `models.spacy.model` | Entity entropy monitor only. | OK |
| `EXEC` | WordNet enabled | \(WN\) | `true` | `models.wordnet.enabled` | WordNet replacement source. | OK |
| `EXEC` | PPDB enabled/path | \(PPDB\) | `true`, source `data/ppdb/ppdb-2.0-s-all`, index `{venv}/var/binary_mopso_cd/ppdb_index.sqlite` | `models.ppdb.enabled`, `models.ppdb.source_path`, `models.ppdb.index_path` | SQLite PPDB fallback replacement source, built once inside the active Python environment. | OK |
| `INIT` | Component alpha: role | \(\alpha_{role}\) | 1.4 | `semantic_components.rules.role.alpha` | Pool size weighting for role. | OK |
| `INIT` | Component alpha: topic | \(\alpha_{topic}\) | 1.0 | `semantic_components.rules.topic.alpha` | Pool size weighting for topic. | OK |
| `INIT` | Component alpha: action | \(\alpha_{action}\) | 1.2 | `semantic_components.rules.action.alpha` | Pool size weighting for action. | OK |
| `INIT` | Max words: role | \(W_{role}\) | 6 | `semantic_components.rules.role.max_words` | Pool and candidate validation. | OK |
| `INIT` | Max words: topic | \(W_{topic}\) | 8 | `semantic_components.rules.topic.max_words` | Pool and candidate validation. | OK |
| `INIT` | Max words: action | \(W_{action}\) | 6 | `semantic_components.rules.action.max_words` | Pool and candidate validation. | OK |
| `INIT` | Expansion order | \(O_{exp}\) | `role`, `action`, `topic` | `semantic_components.expansion_order` | Pool expansion priority. | OK |
| `INIT` | Candidate multiplier | \(M_{cand}/N\) | 4 | `initialization.candidate_multiplier` | Absolute cap of `4N` semantic combinations. | OK |
| `INIT` | Minimum product multiplier | \(M_{min}/N\) | 3 | `initialization.min_product_multiplier` | Requires pool product at least `3N`. | OK |
| `INIT` | Prompt reduction multiplier | \(M_{red}/N\) | 2 | `initialization.prompt_reduction_multiplier` | Reduces candidate prompts to `2N` before generation. | OK |
| `INIT` | Generated sentence minimum | \(S_{min}\) | 1 | `initialization.generated_sentences_min` | Non-empty text implies at least one sentence in implementation; value is not separately checked. | OK with implicit enforcement |
| `INIT` | Generated sentence maximum | \(S_{max}\) | 4 | `initialization.generated_sentences_max` | Rejects generated initial texts above sentence limit. | OK |
| `EXEC` | Generated text fidelity threshold | \(\tau_{gen}^{min}\) | 0.05 | `generated_text_validation.tau_gen_min` | Rejects low-fidelity generated texts before accepting initialization or optimization changes. | OK |
| `MOPSO` | Archive multiplier | \(A_{max}/N\) | 2 | `mopso.archive_multiplier` | Sets \(A_{max}=2N\). | OK |
| `MOPSO` | Leader tournament size | \(q\) | 3 | `mopso.leader_tournament_size` | Tournament by crowding distance. | OK |
| `MOPSO` | Max modified components | \(D_{max}\) | 1 | `mopso.dmax` | Caps changed active components per particle and generation. | OK |
| `MOPSO` | Candidate variants | \(K_{cand}\) | 7 | `mopso.kcand` | Max variants from guided/turbulence operators. | OK |
| `MOPSO` | Inertia endpoints | \(\omega_{max},\omega_{min}\) | `0.9`, `0.4` | `mopso.omega_max`, `mopso.omega_min` | Linear decreasing inertia. | OK |
| `MOPSO` | Cognitive/social constants | \(c_1,c_2\) | `1.5`, `1.5` | `mopso.c1`, `mopso.c2` | Balance pbest and leader influence. | OK |
| `MOPSO` | Velocity clamp | \(V_{max}\) | 4.0 | `mopso.vmax` | Bounds semantic velocity. | OK |
| `MOPSO` | V-shaped scale | \(\alpha\) | 1 | `mopso.alpha` | Controls \(\lvert\tanh(\alpha v)\rvert\) saturation. | OK |
| `MOPSO` | Turbulence endpoints | \(p_{tur}^{max},p_{tur}^{min}\) | `0.07`, `0.02` | `mopso.p_tur_max`, `mopso.p_tur_min` | Linear decreasing turbulence probability. | OK |
| `MOPSO` | Retry count | \(K_{retry}\) | 0 | `mopso.k_retry` | No extra LLM retries; invalid candidates fail clearly. | OK |
| `MOPSO` | Duplicate threshold | \(\tau_{dup}\) | 0.92 | `mopso.tau_dup` | Rejects near-duplicate component candidates. | OK |
| `MOPSO` | Turbulence min fidelity | \(\tau_{tur}^{min}\) | 0.65 | `mopso.tau_tur_min` | Lower semantic similarity bound for turbulence. | OK |
| `MOPSO` | Turbulence max fidelity | \(\tau_{tur}^{max}\) | 0.90 | `mopso.tau_tur_max` | Upper semantic similarity bound for turbulence. | OK |
| `PBEST` | Utility weight f1 | \(w_1\) | 0.5 | `mopso.utility_weights.f1` | pbest tie-break relevance of semantic fidelity. | OK |
| `PBEST` | Utility weight f2 | \(w_2\) | 0.5 | `mopso.utility_weights.f2` | pbest tie-break relevance of semantic diversity. | OK |
| `SELECT` | Final selection enabled | \(sel\) | `true` | `selection.enabled`, `--disable-selection` | Enables final post-processing selection. | OK |
| `SELECT` | Final selection count | \(K_{sel}\) | 5 | `selection.k` | Max number of final selected solutions. | OK |
| `SELECT` | Similarity lower bound | \(\tau_{min}\) | 0.20 | `selection.tau_min` | Minimum \(F_1\) for final selection. | OK |
| `SELECT` | Similarity upper bound | \(\tau_{max}\) | 0.94 | `selection.tau_max` | Maximum \(F_1\) for final selection. | OK |
| `SELECT` | MMR lambda | \(\lambda_{MMR}\) | 0.35 | `selection.lambda_mmr` | Tradeoff between TOPSIS relevance and redundancy penalty. | OK |
| `SELECT` | Entropy epsilon | \(\varepsilon\) | 0.0001 | `selection.epsilon` | Positive displacement for Entropy Method column shift. | OK |
| `MONITOR` | Monitor enabled | \(mon\) | `false` | `monitor.enabled`, `--enable-monitor` | Observational metrics only; no optimizer feedback. | OK |
| `MONITOR` | KMeans clusters | \(k_{km}\) | 3 | `monitor.kmeans_clusters` | Cluster count for external inertia metric. | OK |
| `CHECKPOINT` | Checkpoint enabled | \(ckpt\) | `false` | `checkpoint.enabled` | Deferred checkpoints disabled by default. | OK |
| `CHECKPOINT` | Checkpoint interval | \(I_{ckpt}\) | 1 | `checkpoint.interval`, `--checkpoint-every` | Save cadence when checkpoints are enabled. | OK |
| `CHECKPOINT` | Checkpoint directory | \(dir_{ckpt}\) | `checkpoints` | `checkpoint.directory` | Subdirectory under run output. | OK |
| `RUNTIME` | Output directory base | \(outdir\) | `exec` | `runtime.outdir_base`, `--outdir-base` | EVOLMD-MO style execution folders. | OK |
| `RUNTIME` | Resume path | \(resume\) | null | `runtime.resume_from`, `--resume-from` | Resume from checkpoint snapshot. | OK |
| `RUNTIME` | Embedding cache file | \(cache_E\) | `embedding_cache.json` | `runtime.embedding_cache_file` | Persistent embedding cache path in outputs. | OK |
| `RUNTIME` | Eager model loading | \(load_{eager}\) | `true` | `runtime.eager_load_models` | Loads heavy services once at executor startup. | OK |
| `RUNTIME` | Progress logging | \(log\) | `true`, `runtime.log` | `logging.enabled`, `logging.console`, `logging.file`, `logging.level` | Logs run start/end, generation progress, modified count, HV, spread and errors without extra model calls. | OK |

## Fidelity Traceability

| Normative source | Strategy rule | Implementation | Config | Test |
| --- | --- | --- | --- | --- |
| `estrategia_modulo_enrutamiento_semantico_v2_top_p_synthetic.html` | Router selects algorithms and LLM parameters; influence schedule uses zero-based iteration in `[0, G-1]`. | `src/binary_mopso_cd/router.py` | `router.heuristics`, `router.llm_params`, `router.task_models` | `tests/unit/test_router.py`, `tests/unit/test_runtime_schedules.py` |
| `estrategia_modulo_ejecucion_tareas_semanticas_v1.html` | Executor dispatches `LLM`, `SBERT`, `DeterministicPromptGenerator`, `distilbert_fill_mask`, `wordnet_ppdb`; no router-side execution. | `src/binary_mopso_cd/executor.py`, `src/binary_mopso_cd/services/*` | `models.*`, `ollama.*` | `tests/unit/test_executor.py`, `tests/unit/test_ppdb_sqlite.py` |
| `estrategia_generacion_texto_gi_v3_executor_clase_estatica.html` | Synthetic generation sends exactly the rendered prompt as user prompt, plain text only, `stream=false`. | `src/binary_mopso_cd/llm_prompts.py`, `src/binary_mopso_cd/services/ollama_client.py` | `router.llm_params.synthetic_text_generation`, `ollama.stream` | `tests/unit/test_executor.py` |
| `estrategia_composicion_deterministica_prompt_v6_generalizada_corregida.html` | One deterministic canonical template per component subset, generalized template for extra components, no LLM composer. | `src/binary_mopso_cd/services/prompt_renderer.py` | `semantic_components.order` | `tests/integration/test_reduced_runs.py` |
| `estrategia_inicializacion_poblacion_hibrida_v12_Router.html` | Extract anchors, build/expand pools, never exceed `4N` sampled combinations, reduce to `2N` by prompt diversity, generate texts and select top `N`. | `src/binary_mopso_cd/initialization.py` | `initialization.*`, `semantic_components.rules`, `semantic_components.expansion_order` | `tests/unit/test_initialization_sampling.py`, `tests/integration/test_reduced_runs.py` |
| `estrategia_mopso_cd_semantica_v16_pbest_archive_lote.html` | Discrete MOPSO-CD with active/frozen components, schedules in `[0,1]`, `Dmax`, `Kcand`, no retries, evaluate modified generation in batch. | `src/binary_mopso_cd/mopso.py`, `src/binary_mopso_cd/component_memory.py` | `mopso.*`, `experiment.frozen_components` | `tests/unit/test_runtime_schedules.py`, `tests/unit/test_frozen_components.py`, `tests/integration/test_reduced_runs.py` |
| `mopso_cd_lider_poda_archivo_v4.html` | Global archive `Amax=2N`, non-dominated update, exact duplicate first occurrence, pruning crowding distance, leader tournament `q=3`. | `src/binary_mopso_cd/mopso.py` | `mopso.archive_multiplier`, `mopso.leader_tournament_size` | `tests/unit/test_mopso_core.py` |
| `pbest_Ux_suma_ponderada_dominios_teoricos.html` | `pbest` update by dominance; if incomparable, choose higher `U(x)=(f1+1+f2)/4`; ties keep previous. | `src/binary_mopso_cd/mopso.py` | `mopso.utility_weights` | `tests/unit/test_mopso_core.py` |
| `estrategia_turbulencia_distilbert_fillmask_v8_formulas_corregidas.html` | DistilBERT receives target word/span and returns up to `Kcand` one-word replacement variants with preliminary top-k `3Kcand`. | `src/binary_mopso_cd/services/turbulence.py`, `src/binary_mopso_cd/router.py` | `models.distilbert.top_k_multiplier`, `mopso.kcand` | `tests/unit/test_router.py` |
| `estrategia_turbulencia_wordnet_ppdb_v11_formulas_corregidas.html` | Boundary replacement uses WordNet by lemma/POS with PPDB fallback, returning word variants only; semantic filtering stays in MOPSO. | `src/binary_mopso_cd/services/turbulence.py`, `src/binary_mopso_cd/services/ppdb.py`, `src/binary_mopso_cd/mopso.py` | `models.wordnet.enabled`, `models.ppdb.source_path`, `models.ppdb.index_path`, `models.ppdb.enabled` | `tests/unit/test_router.py`, `tests/unit/test_ppdb_sqlite.py` |
| `estrategia_filtro_similitud_mmr_topsis_entropy_v5_red_no_negativa.html` | Final selection applies similarity filter, Entropy Method, TOPSIS and MMR with non-negative redundancy. | `src/binary_mopso_cd/selection.py` | `selection.*` | `tests/unit/test_selection.py` |
| `modelamiento1.png` | Semantic individuals use `role`, `topic`, `action`; objectives maximize semantic fidelity and semantic diversity with SBERT cosine metrics. | `src/binary_mopso_cd/entities.py`, `src/binary_mopso_cd/objectives.py` | `experiment.components`, `models.sbert.*` | `tests/unit/test_objectives.py` |
| `cf.png` | Fixed iteration stopping criterion with \(G=100\). | `src/binary_mopso_cd/mopso.py`, `src/binary_mopso_cd/runner.py` | `experiment.iterations` | `tests/unit/test_runtime_schedules.py` |
| `arquitecturaEstrategia.png` | Reference data to initial population, routing/executor, optimizer, archive, final selection and generated dataset outputs. | `src/binary_mopso_cd/runner.py`, `src/binary_mopso_cd/outputs.py` | `runtime.outdir_base`, `selection.enabled` | `tests/integration/test_reduced_runs.py` |
| `diagrama_optimizador_mopso_cd_corregido.png` | Per generation: update particles, evaluate changed solutions or reuse cache, update `pbest`, update archive, log observational progress, then stop/continue. | `src/binary_mopso_cd/mopso.py`, `src/binary_mopso_cd/progress.py` | `mopso.*`, `checkpoint.*`, `logging.*` | `tests/integration/test_reduced_runs.py`, `tests/unit/test_progress_metrics.py` |
| `defecto.png` | Defaults `G=100`, `K_runs=3`, `N=100`. | `configs/default.yaml` | `experiment.iterations`, `experiment.runs`, `experiment.n` | `tests/unit/test_runtime_schedules.py` |
