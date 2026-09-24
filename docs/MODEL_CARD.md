# Model card: Vey v1 (English)

## Purpose

Vey produces decisions (which candidate, which value, whether to act, whether
to ask) rather than text. It is not a language model and does not replace one.

## Architecture

Independent lanes, not one model. [ARCHITECTURE.md](ARCHITECTURE.md) has the
full picture.

| lane | mechanism | role |
|---|---|---|
| structured | exact scoring over typed candidates, online EMA reliability, exact history | structured decisions |
| semantic (fixed) | frozen 24M encoder + trained pooled head | known label sets |
| semantic (open) | frozen 24M encoder + cosine over cached candidate embeddings | novel candidate sets |
| retrieval | BM25 + semantic, fused by reciprocal rank fusion | documents |
| long context | BM25 sentence selection → ~128 tokens | long states |
| act | ToolCard selection → typed extraction → exact schema compiler | tool calls |
| trust | calibrated risk + deterministic policy | abstention / escalation |

## Backbone and provenance

- **Encoder:** `mixedbread-ai/mxbai-embed-xsmall-v1`
- **Parameters:** 24.1M · **License:** Apache-2.0 · **Support:** English
- **Usage:** frozen. Vey trains heads and selectors, never the backbone.
- The encoder is fetched from the Hugging Face hub at run time; no weights are
  committed here. The upstream encoder is a separate work. See [NOTICE](../NOTICE).

## Benchmark methodology

- Closed-set intent (Banking77 / CLINC150 / MASSIVE en-US) is evaluated on a
  **deterministically stratified** export, so the training split covers every
  class. An earlier label-sorted, capped export is not used for any release
  number.
- Strict Laya parity uses **identical rows, candidate descriptions, and labels**,
  on the same host, CPU, batch=1, with upstream Laya pinned to a known revision.
- Latency is reported as a distribution (p50/p95/p99), never a single number.
  Vey's "semantic decision" is **end-to-end**: tokenize, encode the query, and
  score against cached candidate embeddings. Memory is reported as model weight
  payload, peak process RSS, and RSS delta. Three distinct quantities.

## Intended uses

- Closed-set intent or routing with a known label set.
- Matching against a candidate set that may change between calls (tools, labels,
  actions) without retraining.
- Long-document or long-state decisions under a strict latency budget.
- Typed tool-call construction where a missing required value should produce
  `ASK_FOR_INFO` rather than a guess.
- Cheap abstention / escalation decisions on top of a decision the system has
  already made.

## Non-intended uses

- General reasoning, open-ended generation, or free-text answer production.
- A replacement for a language model on tasks that require world knowledge or
  multi-step inference.
- Unsupervised tool execution: Vey compiles a validated *decision object*; it
  does not invoke external tools.
- Safety-critical deployment without a human or deterministic fallback. The
  exact lanes are auditable; the semantic lanes are not.

## Known limitations (measured)

- **OOD detection at low false-abstain is weak.** Risk ranking improved
  (AUROC 0.72–0.81 on intent/tool trust tasks), but OOD recall at a 10%
  false-UNKNOWN budget is only ~0.065–0.12. Vey cannot reliably catch
  out-of-distribution input at a tight abstention budget.
- **Tool calls are conservative.** Free-text slots resolve only on an explicit
  label cue, so many calls end as `ASK_FOR_INFO`. End-to-end exact tool calls
  are ~0.09–0.11.
- **The open matcher returns similarity scores, not probabilities.** The pooled
  head and trust layer produce probabilities; the matcher does not.
- **English only in v1.**
- **Closed-set accuracy is moderate.** The pooled head reaches 0.884 / 0.832 /
  0.780 on Banking77 / CLINC150 / MASSIVE. That is the price of a 24M CPU path.

## Datasets

Vey does not redistribute any dataset. Closed-set intent benchmarks are fetched
from their upstream sources at reproduction time under their own licenses
(Banking77 CC-BY-4.0, CLINC150 CC-BY-3.0, MASSIVE CC-BY-4.0). Tool-selection
numbers use ToolACE (Apache-2.0). See [BENCHMARKS.md](BENCHMARKS.md).

## Ethical and safety notes

- Decisions carry a calibrated risk estimate, and the action policy is
  deterministic and can abstain. Set thresholds to match the cost of a wrong
  action in your deployment.
- Vey refuses rather than invents when a required value is missing. Downstream
  code should handle `ASK_FOR_INFO`.
