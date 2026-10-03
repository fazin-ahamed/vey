# Jev / TypeSafe AI: current primary-source dossier

Retrieved 2026-10-03; refresh completed at **2026-10-03T17:37:41Z**. Audience: Vey maintainers deciding what a fair, authorized comparison must cover. This dossier records public documentation and source inspection, not an API experiment.

## Evidence boundary

`MEASURED_SOURCE_INSPECTION` means that a public page or repository was inspected and contains the described contract or implementation documentation. It does **not** mean the hosted service was exercised. All quality, calibration, speed, cost-saving, type-safety and scaling claims below remain **HYPOTHESIS / vendor-reported, independently unreproduced**. No authenticated request, model inference, account action, paid call, package installation, weight download, test, build or formatter was performed. Neither Jev's superiority nor Vey's superiority is confirmed.

The companion [JEV.json](JEV.json) records explicit nulls for unsupported quantities. Website/docs content has no inspected immutable revision or supplied SHA-256; a retrieval date is not a model hash. SDK and ecosystem commits are pinned separately from model weights.

## Version and API identity

The [Models page](https://docs.typesafe.ai/models.md) currently documents **Jev 1.13**, with accepted versioned ID `jev-1.13.0`. Both `jev-latest` and `jev-preview` resolve to `jev-1.13.0`; no distinct preview is currently documented. These are moving aliases. The response's `model` field reports the serving versioned ID. Pin `jev-1.13.0` in any future permitted study and record the returned model rather than assuming the alias never changes.

The [HTTP API](https://docs.typesafe.ai/api.md) is `POST https://api.typesafe.ai/v1/systemone`, requiring bearer authentication and JSON. Its request consists of `model`, `state` (string/object/array) and `questions` (a map of caller-selected IDs to typed questions). IDs are response keys, not model input. All question types allow string/object/array `instructions`. The response contains `model`, keyed `answers`, and `usage.input_tokens` / `usage.output_tokens`. Documented errors include 401, 422, 429 and 529; SDKs provide default backoff. The authenticated `GET /v1/models` currently lists aliases, while versioned IDs may be accepted without appearing there. Neither endpoint was called.

The website footer's `Version 0.01` is a site label, **not** the Jev model version. A public release date for `jev-1.13.0`, weight hash, parameter count, architecture dimensions and checkpoint are unknown in the inspected sources. The launch article is visibly dated September 15, 2026; its page metadata says September 28, 2026. These are page dates, not proof of the current checkpoint's release date.

## Typed questions and answers

| Primitive | Question contract | Answer contract | Comparison implication |
| --- | --- | --- | --- |
| [Choice](https://docs.typesafe.ai/primitives/choice.md) | `type: choice`; `criteria` maps option names to string/object/array/null descriptions. Maximum **255 options** per question. Names and descriptions reach the model. | `choice` is the highest-probability option; `probabilities` maps every supplied option to a probability summing to 1; `confidence` is derived from that distribution. | Direct bounded-set selection and ranking are documented. A forced winner does not establish that any candidate is suitable; include a none/abstain option when required. |
| [Score](https://docs.typesafe.ai/primitives/score.md) | `type: score`; ordered descriptive `criteria` array, recommended at least 2 levels, API accepts at most **10**. Level indices start at 0. | `score = sum(i * p_i)`, ranging from 0 to `n-1`; `probabilities` keyed by level index strings; `legend` maps indices to descriptions; `confidence`. | Expected ordinal level, not a general real-number extractor or calibrated interpolation of exact numeric values. Different distributions can share a score. |
| [Noul](https://docs.typesafe.ai/primitives/noul.md) | `type: noul`; yes/no instruction; optional `criteria.true` and `criteria.false`. | `type` plus `noul`, a scalar probability of yes in [0,1]. **No separate confidence field.** | Truth probability is not a Boolean until caller code applies a threshold. |

Question primitives may be mixed in a single request. Docs say the state is ingested once and questions are evaluated independently in parallel; the maximum question count per request is not explicitly established by the inspected references beyond the token budgets. Multi-question fan-out is not the same as a multi-state asynchronous batch API. No dedicated offline job/batch endpoint was identified in the documentation index.

### Confidence is not another learned correctness probability

The [Confidence reference](https://docs.typesafe.ai/confidence.md) gives exact distribution-summary formulas:

- Choice, `n > 1`: `confidence = (p_max - 1/n) / (1 - 1/n)`.
- Score: `confidence = max(0, 1 - sum(p_i * abs(i-m)) / MAD_unif)`, where `m` is the most likely level and `MAD_unif = sum(abs(i-(n-1)/2))/n`.
- Noul supplies no confidence. Caller code can optionally compute `abs(2*p-1)`.

These are observable documentation facts, not empirical calibration evidence. A 0.9 confidence value is **not** a separately established 90% chance of correctness. TypeSafe claims calibrated probabilities through Reinforcement Learning for Calibrated Decisions (RLCD), but the inspected pages do not establish held-out ECE, Brier score, log loss, coverage/risk guarantees or per-domain calibration bounds. A Vey comparison must keep the distribution, its derived confidence, and empirically measured correctness separate. Score numerical calibration is explicitly listed as weak by the vendor.

## Limits, languages and deployment

From [Models](https://docs.typesafe.ai/models.md):

- Text only, including structured textual JSON. No image/audio/video input; preprocess other modalities externally.
- **64k tokens total per request**, covering state plus all questions. **32k tokens for state plus the longest individual question**. These are documented token budgets, not independently tested boundaries or quality guarantees.
- Published rate limits: **100,000 input tokens/second and 80 requests/second**. The vendor explicitly says limits change dynamically without notice; higher limits require custom/enterprise arrangements. These are service quotas, **not measured sustained model throughput**.
- English is the primary training language and currently best-performing. Other languages including CJK scripts are handled, but not equally well. No supported-language enumeration or per-language accuracy table was established.
- Same weights serve every account; vendor says no customer fine-tuning or LoRA. Domain customization is through state, instructions, criteria and downstream code.
- The documented deployment is TypeSafe-hosted API/console. No public Jev weights, local runtime, ONNX export, quantization options, GPU memory requirement, CPU latency or on-prem/self-hosting package was identified. These remain unknown; public SDKs are HTTP clients, not inference engines.
- [Models/legal](https://docs.typesafe.ai/legal.md) say customer requests/responses are not used for training; enterprise ZDR is offered. This is a vendor policy claim, not a verified retention audit or a blanket ZDR guarantee for all accounts.

## SDKs and ecosystem

| Surface | Inspected evidence and pin | What is not established |
| --- | --- | --- |
| Python | Official [`typesafe-sdk`](https://docs.typesafe.ai/sdk/python.md), sync/async clients, typed answers, Pydantic response models and optional HTTP/2. [Changelog](https://docs.typesafe.ai/sdk/python/changelog.md): **0.7.2, 2026-09-26**, adding HTTP/2 extra; 0.7.0 moved serialization to Pydantic. Tag [v0.7.2](https://github.com/typesafe-ai/typesafe-sdk-python/tree/f078f1e208a0d885154dc758344ae4fce77ac168) resolves to `f078f1e208a0d885154dc758344ae4fce77ac168`. [Package metadata](https://github.com/typesafe-ai/typesafe-sdk-python/blob/f078f1e208a0d885154dc758344ae4fce77ac168/pyproject.toml) requires Python >=3.10. MIT SDK license. | No installed/runtime validation; this is the inspected documented release, not a registry-wide latest-version audit. |
| JS/TypeScript | Official [`@typesafe-ai/sdk`](https://docs.typesafe.ai/sdk/javascript.md), typed inferred answers, ESM/CommonJS/declarations. [Changelog](https://docs.typesafe.ai/sdk/javascript/changelog.md): **0.6.0, 2026-09-15**, ordered Score criteria. Tag [v0.6.0](https://github.com/typesafe-ai/typesafe-sdk-js/tree/66880ccded6cb642dc1809620c2b108c33730214) resolves to `66880ccded6cb642dc1809620c2b108c33730214`; package requires Node >=20 and declares MIT. | No runtime exercise or browser-runtime support certification. |
| HTTP | Documented authenticated `/v1/systemone` and `/v1/models`, callable from other languages. | No unauthenticated or authenticated endpoint execution. |
| Coding agents | Official [agent skill](https://docs.typesafe.ai/agent-skill.md) for Claude Code/Codex/other agents teaches integration. [Coding-agent docs](https://docs.typesafe.ai/introduction/coding-agents.md) explicitly say Jev is not a chat, code-completion or coding-agent model replacement. | A skill/plugin is not an MCP server or evidence that Jev itself edits code. |
| LangChain | Partner [`langchain-typesafe`](https://github.com/langchain-ai/langchain/blob/115dbbd158c95044c2a2281e326dc8e6a44a3c23/libs/partners/typesafe/README.md) contains `TypeSafeClassifier` as a Runnable with `invoke`/`ainvoke`, request-scoped state/questions, message conversion, and experimental model-router/risk-blocking middleware. [Metadata](https://github.com/langchain-ai/langchain/blob/115dbbd158c95044c2a2281e326dc8e6a44a3c23/libs/partners/typesafe/pyproject.toml): **0.0.1a2**, MIT, Python >=3.10,<4. Pin `115dbbd158c95044c2a2281e326dc8e6a44a3c23` (README last-modifying commit, 2026-09-20). | Experimental APIs may change. No inference, installed package, or middleware behavior test. |
| LangGraph | LangChain Runnable/agent composition is documented, but no separate dedicated Jev LangGraph integration was established in the inspected TypeSafe index and linked integration README. | Native LangGraph package, node/checkpoint support and end-to-end operation remain null/unmeasured. Do not turn absence in this inspection into proof of incompatibility. |
| MCP | Community [`MarkChu-git/typesafe-mcp`](https://github.com/MarkChu-git/typesafe-mcp/blob/10737f071d4b299c3db30243093d2e739c2d03ad/README.md), pin `10737f071d4b299c3db30243093d2e739c2d03ad` (2026-09-22), documents Bun stdio tools wrapping hosted Jev: models/check/classify/score/mixed ask. Its act/review/abstain gate is wrapper code, not a Jev API field. | Not vendor-maintained. README lists Streamable HTTP transport as unimplemented. No official MCP server was identified in the inspected index/organization page; community runtime and published package version were not verified. |
| Workflow adapters | Vendor links [`system-one-adapter-python`](https://github.com/typesafe-ai/system-one-adapter-python) to constrain LLM comparisons to the same primitive interface. Public [`WorkflowEvals`](https://github.com/typesafe-ai/WorkflowEvals/tree/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb), commit `0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb` (2026-09-29), documents default `jev-1.13.0` and reference-agreement metrics; code Apache-2.0, dataset licenses separate. | Public evaluation code is not independently reproduced model evidence. Dataset revision must be pinned; its default is latest `main`. |

## Published metrics: vendor-reported, not reproduced

| Claim | Primary source | Limitations |
| --- | --- | --- |
| 193.6x faster, 444.6x cheaper for System One workflows | [Homepage](https://typesafe.ai/), [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Vendor says these gains are on the higher end of real-world gains. Vendor workflow builders may introduce bias. This is not a same-host Vey comparison or a universal speed ratio. |
| 70-500 ms end-to-end latency | [Launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Generally measured from vendor laptops on the US West Coast, near the service. No disclosed percentile SLA, comparable workload distribution or our independent measurement. |
| Demo: TypeSafe 0.114 s / $0.000081; LLM 8.566 s / $0.013880 | [Homepage](https://typesafe.ai/) | One recorded short/dense demo; the launch article admits this input favors Jev. Separate workload from the headline workflow aggregates. |
| Parallel questions: 13-question batch 0.27 s / $0.000497 versus 13 separate calls 2.71 s / $0.006090; 10.0x faster, 12.2x cheaper | [Cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions.md) | Explicitly **`jev-1.12`**, five repeats, 53,777-character GDPR article. Separate-call time is a **sum assuming sequential calls**, not concurrent baseline. Mostly identical answers, but two Nouls show sampling noise and different means; no universal determinism guarantee. Do not attribute this result to 1.13.0. |
| CLERC reranking: top-1 5% to 18%, top-10 38% to 62% | [Cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md) | 40 queries, 30-passage BM25 shortlist from 3,565 passages; finite example, not broad semantic accuracy. Model version was not established from the inspected excerpt. |
| High-K examples: 218 line IDs in one Choice; Wikiracing | [Line-search cookbook](https://docs.typesafe.ai/cookbooks/semantic_find.md), [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Direct Choice maximum is 255. Above 255 the Wikiracing demo uses two stages (independent scoring, then explicit choice). No held-out accuracy/calibration curve vs K, permutation sweep or large-K complexity measurement established. |
| Workflow intelligence / accuracy frontier | [Evaluation site](https://evals.typesafe.ai/) and [WorkflowEvals README](https://github.com/typesafe-ai/WorkflowEvals/blob/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb/README.md) | Equal weight across four workflows. Reference labels average GPT-6 Astra and Claude Fable 5.1 at high thinking; comparison models use provider-default reasoning. Metrics are agreement with assumed reference/workflow, **not independently verified ground-truth accuracy**. |
| Zero hallucinations / zero type errors / calibrated confidence | [Homepage](https://typesafe.ai/), [launch article](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [Confidence](https://docs.typesafe.ai/confidence.md) | Vendor's zero figure is a schema-construction argument, explicitly not empirical. Schema validity does not imply semantic correctness, adversarial safety or calibrated correctness; the vendor's own limitations and contract allow wrong answers. |

## Pricing and licenses

[Models](https://docs.typesafe.ai/models.md) and the launch article publish **$42 per billion input tokens = $0.042 per million input tokens**, with output tokens free. This is a token tariff, not a measured per-decision cost or a local compute/energy comparison. A multi-question request amortizes state tokens, but still includes questions in its aggregate token budget. We did not inspect an invoice, checkout or authenticated balance. Minimum spend, promotions/free credits, negotiated enterprise pricing and caching discounts remain unknown. The public [Master Customer Agreement](https://typesafe.ai/legal/mca), last updated September 23, 2026, describes purchased/promotional credits, taxes excluded and generally 12-month purchased-credit expiry unless an Order says otherwise.

Jev is a **proprietary hosted service** under the customer agreement. MIT licenses on SDKs do not license Jev weights or the hosted service; no open-weight license was established. The agreement grants limited service/API integration rights and reserves TypeSafe's technology rights.

**Authorization prerequisite for any later Vey comparison:** MCA §2.3(b) prohibits use of Services or Output for distillation, training an imitator, or developing/facilitating a similar or competing product/service. An API key or ordinary paid access alone does not establish authorization for Vey competitor research. A separate written agreement/permission must resolve applicable restrictions before a direct competitive evaluation. This dossier neither accepts terms nor makes calls. MCA §9.3 acknowledges inaccurate/erroneous outputs and assigns independent evaluation to the customer.

## Known limitations and comparison status

The [Jev 1.13 jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md), **last reviewed 2026-10-02**, lists literal reading, unreliable counting/numeric precision/date comparisons, weak numerical Score calibration, multi-hop/negation indirection, irrelevant long-state degradation, adversarial steering, contradictory criteria, **Choice option-order sensitivity with first-option preference**, and no effective text generation. Its recommendation to reorder options is a diagnostic, not a permutation-invariance guarantee.

The introduction's claim that adding independent questions avoids question-induced context rot must not be expanded into immunity to irrelevant **state** content; the jaggedness page explicitly reports the latter failure.

Direct evaluation status: **NOT RUN**. Unknowns include parameter count/weight bytes, runtime hardware and memory, ONNX/quantization/local deployment, percentile latency/sustained throughput, accuracy on Vey tasks, causal criterion swaps, high-K degradation, candidate-order invariance, calibration metrics, selective-risk/coverage guarantees and energy per verified decision. All comparisons stay hypotheses until a separately authorized, pinned, held-out evaluation measures the full end-to-end behavior.
