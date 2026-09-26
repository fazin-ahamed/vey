# Vey 2.0.0-rc1 release notes

**Vey 2 introduces CRUX: a semantic decision architecture that grounds natural
language into typed evidence and executes decisions deterministically.**

## What's in the RC

- **`vey.decide()` public API**: `question` + `candidates` (+ optional `state`,
  `explain`) → `answer`, `probabilities`, `decision_mode`, `trust`.
- **Lane router**: `structured` (deterministic, no model) when candidates expose
  numeric/enum facts; `crux` (frozen ModernBERT-base-NLI predicate grounder +
  trained antisymmetric ordinal comparator, ~150M, lazy-loaded) for qualitative
  language grounding. The 150M backbone runs only when needed.
- **CRUX executor**: instruction → typed program (`FILTER concept`,
  `FILTER key>=n`, `MAX axis`, `MIN axis`); identity-free grounding; ε-lexicographic
  composition (Semantic Resolution); content-deterministic tie-break. The decision
  is a pure function of the unordered candidate consequence set, order- and
  rename-invariant by construction (CRUX-P).
- **`explain=True`**: machine-evidence certificate (program, per-candidate typed
  values, survivors). No generated reasoning.
- **Trust/abstention**: no eligible candidate → `answer=None`, `mode="abstain"`;
  ties reported via `trust.state`.
- **Capability-matrix tests**: 11 hermetic tests (no model download): compiler,
  router dispatch, lexicographic routing, abstention, permutation invariance,
  rename invariance, exact-tie determinism, Semantic Resolution, certificate
  shape. `tests/test_crux_runtime.py`.
- **Examples**: `examples/route_selection.py` (model routing, structured lane),
  `examples/crux_choice.py`, `examples/semantic_constraints.py` (CRUX lane).
- **Docs**: `docs/ARCHITECTURE.md`, `docs/CRUX.md`, `docs/BENCHMARKS.md`.

## Benchmark results (immutable; see docs/BENCHMARKS.md)

- Planner-assisted Snake (seeds 711–714, run once): **CRUX 136 food / 1.000
  agreement / 0 deaths** vs Laya 421M: 81 / 0.910 / 1.
- Raw-consequence Snake (seeds 611–614, run once, NO planner verdicts): **CRUX 155
  food / 0.971 agreement / 0 deaths** vs Laya: 17 / 0.404 / 0.
- Permutation at the raw-consequence measurement was **0.965**; the later
  permutation-exact change achieved **1.000** order+rename invariance on 1,500
  generic held-out decisions. Snake was not rerun.
- **Disclosed tradeoff:** removing the candidate-name signal cost ~4 points of
  generic composition accuracy (0.951 → 0.914), still above the ≥0.90 gate, in
  exchange for exact invariance.

## Known limitations (RC)

- The CRUX crux lane loads the trained ordinal-comparator weights
  (`comparator.safetensors`, fine-tuned, not frozen-zero-shot) from the Hugging
  Face Hub (`fazinahamed/vey`, pinned revision). Override with
  `VEY_CRUX_COMPARATOR` (a local file) or `VEY_CRUX_COMPARATOR_HF`
  (`repo_id[@revision]`). The lane **fails closed** only if that fetch fails;
  the structured lane needs no artifact and is model-free.
- This RC ships the `structured` and `crux` lanes plus the Vey 2 router. The
  Vey 1 lanes (`semantic`, `retrieval`, `structured`/FactMemory, `act`, `trust`)
  are unchanged under `src/vey`; folding them into the Vey 2 router is follow-up
  work.
- Snake is a closed benchmark family. No new Snake runs exist or are planned;
  capability selection uses generic decision suites.

## Regression matrix (this RC)

- `pytest tests/` → **53 passed** (1.1s), including the 11 new Vey 2
  capability-matrix tests. No legacy regression.

## Upgrade from Vey 1

- Vey 1 is unchanged. Vey 2 adds `src/vey/crux`, `src/vey/runtime.py`, and a
  top-level `vey.decide`; the Vey 1 trust-policy decision remains at
  `vey.trust.decide`. Both import side-by-side from the same `vey` package.

## Next (separate tracks, not in this RC)

- **CRUX-S**: relation distillation 150M → 71M DeBERTa → ~30M student.
- **CRUX Runtime**: apply the spine to model routing, then browser and tools.
