---
license: apache-2.0
library_name: vey
language:
  - en
tags:
  - decision-making
  - embeddings
  - retrieval
  - tool-use
  - cpu
  - non-generative
---

# Vey

A lightweight machine-native decision runtime for fast, typed decisions without
autoregressive generation.

**Vey is a runtime, not a single neural model.** It composes narrow,
independently measured mechanisms: an exact structured lane, a frozen 24M
semantic encoder with two heads, BM25 and semantic rank fusion, long-context
lexical pre-selection, a typed tool/action compiler, and calibrated trust with
abstention. It emits no output tokens.

## Encoder dependency

Vey does not ship its own generative weights. The default English semantic
encoder is a separate, upstream Apache-2.0 work:

| | |
|---|---|
| Repository | `mixedbread-ai/mxbai-embed-xsmall-v1` |
| Parameters | 24.1M |
| License | Apache-2.0 |
| Usage | frozen; Vey trains only heads and selectors |

No upstream weights are duplicated in this repository. They are fetched at run
time. Copyright and attribution remain with the Mixedbread AI authors.

## Lanes

| lane | mechanism | role |
|---|---|---|
| structured | exact scoring over typed candidates, online reliability, exact history | structured decisions |
| semantic (fixed) | frozen encoder + trained pooled head | known label sets |
| semantic (open) | frozen encoder + cosine over cached candidate embeddings | novel candidate sets |
| retrieval | BM25 + semantic, fused by reciprocal rank fusion | documents |
| long context | BM25 sentence selection, then the encoder | long states |
| act | ToolCard selection, typed extraction, exact schema compiler | tool calls |
| trust | calibrated risk + deterministic policy | abstention, escalation |

## Measured

Intel i5-12400F, CPU, batch=1.

| | |
|---|---|
| Semantic decision, end-to-end | 2.6 to 2.9 ms p50 |
| Long-context decision, after lexical pre-selection | 9 to 11 ms p95 |
| Model weight payload | 92 MB FP32 |
| Closed-set intent (Banking77 / CLINC150 / MASSIVE) | 0.884 / 0.832 / 0.780 |

## Intended use

Closed-set routing, matching against a candidate set that may change between
calls, long-state decisions under a latency budget, and typed tool-call
construction where a missing required value should become `ASK_FOR_INFO` rather
than a guess.

## Not intended use

Vey is not a language model and does not replace one. It does not produce
open-ended text, perform general reasoning, or execute external tools. It is not
a drop-in replacement for an LLM on knowledge or multi-step inference tasks.

## Limitations

- **OOD detection at a tight false-abstain budget is weak.** Risk ranking is
  useful (AUROC 0.72 to 0.81), but OOD recall at a 10% false-UNKNOWN budget is
  only about 0.065 to 0.12.
- **Tool calls are conservative.** Free-text slots resolve only on an explicit
  cue, so end-to-end exact tool calls are about 0.09 to 0.11. Vey prefers to ask.
- **The open matcher returns similarity scores, not calibrated probabilities.**
  Do not treat its cosine values as posteriors.
- **English only in v1.**
- **Closed-set accuracy is moderate** by design, in exchange for a 24M CPU path.

## Source

- GitHub: <https://github.com/fazin-ahamed/vey>
- Documentation: <https://github.com/fazin-ahamed/vey/tree/main/docs>
- Issue tracker: <https://github.com/fazin-ahamed/vey/issues>
- Demo Space: <https://huggingface.co/spaces/fazinahamed/vey-demo>
- Security policy: <https://github.com/fazin-ahamed/vey/blob/main/SECURITY.md>

## License

Vey code is Apache-2.0. The default encoder is a separate Apache-2.0 work by the
Mixedbread AI authors.
