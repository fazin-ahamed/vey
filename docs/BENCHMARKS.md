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
