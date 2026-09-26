# Benchmarks

Every number below was measured on the hardware stated with it. No number is
carried across incompatible hardware or incomparable workloads. Raw result
artifacts live under `benchmarks/results/`.

## Definitions used throughout

- **End-to-end semantic decision**: tokenize the state, run the 24M encoder on
  the query, and score against **cached** candidate embeddings. Caching the
  candidate side is the deployed configuration: candidate representations are
  built once at install time.
- **Scorer-only**: dot products against an already-encoded query vector. A
  separate characterization, not the headline, and not a parity claim.
- **Memory**: reported three ways because they answer different questions:
  model weight payload (comparable across systems), peak process RSS (dominated
  by the language runtime), and RSS delta after model load.

## Closed-set intent (Banking77 / CLINC150 / MASSIVE en-US)

Measured on a deterministically stratified export so the training split covers
every class. Frozen 24M encoder; a trained pooled head for the fixed label set.

| task | pooled head accuracy |
|---|---|
| Banking77 | 0.884 |
| CLINC150 | 0.832 |
| MASSIVE | 0.780 |

## Open-candidate matching (same rows, full candidate set)

The frozen matcher, scored by cosine over cached candidate embeddings.

| task | accuracy | chance |
|---|---|---|
| Banking77 (77 candidates) | 0.658 | ~0.24 |
| CLINC150 (151) | 0.675 | ~0.29 |
| MASSIVE (60) | 0.471 | ~0.24 |

The matcher is more accurate than the pooled head on unseen *classes* (it is not
trained on any of them) and less accurate on trained ones, hence two lanes.

## CPU latency (Intel i5-12400F, 6 torch threads, batch=1)

| path | p50 | p95 | p99 |
|---|---:|---:|---:|
| semantic decision, end-to-end | 2.6 – 2.9 ms | 3.2 – 3.9 ms | 3.4 – 4.2 ms |
| semantic decision, scorer-only | 0.030 to 0.070 ms | n/a | n/a |
| long-context decision (after BM25 sentence selection) | n/a | 9 to 11 ms | n/a |

Memory: 24.1M parameters, **92.0 MB FP32** weight payload, ~825 MB peak process
RSS.

## Candidate structural invariance

Permuting, inserting, or removing candidates leaves every existing candidate's
score **exactly** unchanged (drift 0.0) and the winner unchanged (agreement
1.000). This is why no set-architecture is required.

## Tool selection (ToolACE, Apache-2.0)

| split | semantic matcher accuracy | chance |
|---|---|---|
| known tools | 0.669 | 0.239 |
| novel tools (held-out identities) | 0.828 | 0.291 |

The candidate-compiler path: unsupported-argument rate is exactly **0.000** (no
emitted argument lacks an evidence span), schema validity and type validity are
**1.000**, and end-to-end exact tool calls are ~0.09–0.11 (the system prefers to
ask over guessing). See [LIMITATIONS.md](LIMITATIONS.md).

## Reproducing

Benchmarks retrieve their upstream datasets at run time; no dataset is bundled.
See `benchmarks/reproduce/` for the scripts and their expected environment.

## Vey 2 / CRUX (immutable, preregistered, run-once)

Preregistered zero-shot Snake control benchmarks. Numbers are **immutable**:
the values measured at the time of each run, not retro-edited. Thresholds were
frozen before scoring; each seed block was consumed exactly once.

### Planner-assisted variant (options carry planner verdicts), seeds 711–714

| system | params | food | deaths | planner agreement | permutation |
|---|---:|---:|---:|---:|---:|
| **CRUX** | ~150M | **136** | **0** | **1.000** | **1.000** |
| Laya | 421M | 81 | 1 | 0.910 | 0.880 |

### Raw-consequence variant (NO planner verdicts; model must derive utility), seeds 611–614

| system | params | food | deaths | agreement | permutation (at measurement) |
|---|---:|---:|---:|---:|---:|
| **CRUX** | ~150M | **155** | **0** | **0.971** | 0.965 |
| Laya | 421M | 17 | 0 | 0.404 | 0.433 |

The raw-consequence gap (155 vs 17 food) is the load-bearing result: CRUX
reconstructs the decision from grounded consequences rather than reading verdict
labels, at a ~150M frozen backbone versus a 421M decision model.

### Permutation invariance

The raw-consequence run measured **0.965** permutation agreement at the time.
A later change removed the candidate-order/name dependence **architecturally**
(identity-free grounding + content-deterministic tie-break + value
quantization), reaching **1.000 order- and rename-invariance on 1,500 generic
held-out decisions**. Snake was **not** rerun after this change; the historical
0.965 number above is left unchanged. Composition accuracy on those held-out
decisions moved 0.951 → **0.914**, a disclosed tradeoff: removing the
illegitimate candidate-name signal costs ~4 points of composition accuracy in
exchange for exact invariance, still above the ≥0.90 preregistered gate.

Not claimed: a universal "Vey > Laya". These are specific preregistered control
benchmarks. Snake is a closed benchmark family (all seed blocks consumed);
future capability selection uses generic decision suites, not Snake.

## CRUX runtime characterization (CPU, i5-12400F, 6 torch threads)

Measured on the frozen `v2.0.0-rc1` comparator with no changes to the model.
Warm numbers are the median of repeated calls after warmup on one loaded
`Runtime`, so model load is excluded. The pairwise ordinal comparison encodes
`K*(K-1)` sequences per decision, so latency grows roughly with `K²`.

| | |
|---|---|
| model load | 6.1 s (first call only) |
| artifact on disk | 598 MB (`comparator.safetensors`, fp32) |
| RSS before / after load | 375 MB / 1,289 MB (delta 914 MB) |
| peak process RSS | 1,980 MB |
| warm decision, K=4, p50 / p95 / p99 | 262 ms / 434 ms / 435 ms |

### Candidate scaling (one ordinal axis)

| candidates K | pairwise comparisons | warm p50 | warm p95 |
|---:|---:|---:|---:|
| 2 | 2 | 59 ms | 88 ms |
| 4 | 12 | 260 ms | 433 ms |
| 8 | 56 | 1,228 ms | 1,454 ms |
| 16 | 240 | 5,494 ms | 6,133 ms |
| 32 | 992 | 23,514 ms | 26,202 ms |

The growth is close to quadratic: K=32 costs about 19× K=8 while doing 18× the
comparisons. For a large candidate set (a router over many models, a tool
library) the cost is dominated by the pairwise stage, which is the point at
which a cheap preselection step should narrow the set before CRUX ranks it.

### Predicate scaling (K=8)

Categorical filters are linear in the number of predicates, and cheaper than an
ordinal stage because each predicate is one entailment pass over the candidates
rather than a pairwise comparison.

| predicates | warm p50 |
|---:|---:|
| 1 | 464 ms |
| 2 | 1,022 ms |
| 4 | 1,323 ms |
| one ordinal stage (for comparison) | 1,263 ms |

### Other factors (K=8 unless noted)

| factor | warm p50 |
|---|---:|
| candidate text 16 / 32 / 64 / 128 tokens | 1,174 / 1,277 / 1,322 / 1,583 ms |
| CPU threads 1 / 2 / 4 / 6 | 5,810 / 3,121 / 1,737 / 1,386 ms |
| batched grounding vs one sequence at a time (K=16) | 5,424 ms vs 7,427 ms |
| structured lane dispatch (no model) | 0.12 ms |

The structured lane is four orders of magnitude cheaper than the CRUX lane on
the same host, which is why the router uses it whenever the candidates expose
numeric or enum facts.
