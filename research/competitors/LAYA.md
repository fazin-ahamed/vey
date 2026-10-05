# Laya: current primary-source dossier

Current metadata was retrieved 2026-10-05, from 07:11:30 to 07:23:43 UTC; see the [refresh section](#metadata-refresh-retrieved-2026-10-05) for source, release and model pins. The 2026-10-03 source snapshot and old-pinned local runtime evidence below are retained. Audience: Vey researchers choosing a fair competitor configuration. The companion `LAYA.json` records source hashes and machine-readable facts.

## 2026-10-03 source snapshot and evidence boundary

This is source and metadata inspection, not an inference benchmark. API existence and configuration are **MEASURED source inspection**. Every quality, latency, throughput, calibration and numerical-parity result below remains **HYPOTHESIS / vendor- or externally reported, not independently reproduced by Vey**. No model weights were downloaded, no model/API calls were made, and no tests or builds were run.

At the original inspection, the Python package was **0.3.25**, not 0.3.20: [GitHub release](https://github.com/NandhaKishorM/laya/releases/tag/v0.3.25) published 2026-10-03T15:58:35Z, [PyPI metadata](https://pypi.org/pypi/laya/json) retrieved that day, and the pinned package manifest agreed. Release commit: `8a976468b57c1b53541363dc86523263c9b68d34`. That snapshot pins source main at `859b8ee595cc04f84dd2af476d6d1d90ec1fea46`, committed 2026-10-03T17:31:47Z. Its API/configuration descriptions and original source references are historical unless explicitly updated in the refresh section. Published benchmark environments are not silently relabeled with a newer runtime.

Primary source entry points: [README][readme], [BENCHMARKS][bench], [package manifest][package], [routing guide][routing], [question/answer semantics][questions], [calibration implementation][calibrate], [HTTP API][http].

## Strongest appropriate checkpoints, not a weak baseline

| Role | Exact standalone HF repository and revision | Encoder / total parameters | Shipped total / option budgets |
|---|---|---|---|
| English | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya/tree/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851), `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | `answerdotai/ModernBERT-large`; 421,293,830 | `max_len=512`, `head_max_len=192` |
| Multilingual | [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual/tree/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67), `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67` | `jhu-clsp/mmBERT-base`; 321,908,998 | 1,024 / 256; opt-in total limit 8,192 |
| Typed workflows | [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/1a793eb568e6718f15941d08f85432581df534e3), `1a793eb568e6718f15941d08f85432581df534e3` | `answerdotai/ModernBERT-large`; 421,293,830 | 1,024 / 256 |

Parameter totals are HF metadata, not a locally counted resident model. All three HF cards declare Apache-2.0; the Python repository [license][license] is Apache-2.0. Commercial use follows those license terms, not an independently audited legal opinion.

**Important default:** [router source][router-src] loads all three from the **bundle** `convaiinnovations/laya`, with subfolders `None`, `multilingual`, and `typed-decisions`. Thus a default bundled run should pin bundle revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and record its selected subfolder. `Router(standalone_repos=True)` uses the three standalone repositories above. [Reviewed pins][pins] are opt-in, including `LAYA_REVISION=reviewed`; ordinary loads otherwise follow HF's default revision. Do not apply a standalone multilingual SHA to the bundle repository.

The recommended `Router` uses English for English, multilingual for recognized non-English text, and a configurable default for ambiguous text. Explicit `model` wins, followed by `task`, optional exact question-ID workflow matching, `lang`, `lang_guess`, then heuristic script/language detection. **Typed-decisions is not selected automatically by default**: use `model="typed-decisions"` or `task="typed_decisions"`. `auto_task_detection=True` matches known workflow question-ID sets exactly, not arbitrary question meaning. A comparison restricted to base English on typed workflows would omit the competitor's strongest published specialist.

The router is lazy; it keeps at most two loaded checkpoints by default. Preloading avoids first-use download/load latency and raises the resident limit to fit the chosen set. Mixed batches group compatible schemas and checkpoints, then restore input order. Memory use and fit on Vey's 12GB target are unmeasured.

## Decision interface and context

One state can be text or JSON, with multiple typed questions and no generated text:

- `choice`: ordered labels/descriptions, returned argmax and option probabilities.
- `noul`: binary head, returned P(true), not a boolean. The caller chooses the decision threshold.
- `score`: ordered levels, returned expected level index plus distribution/legend, not necessarily the argmax level.
- A separate `action.act_probability` scores whether to act; it is not answer confidence.

`decide` / `decide_batch` project JSON schema or Pydantic fields into these primitives. They do not implement arbitrary schema generation: enums, booleans and bounded integers are supported; free strings, arrays and nested structures are outside this fixed-option interface.

The 8,192-token statement applies specifically to multilingual with `max_len=8192`, preferably explicit `model="multilingual"` so English documents do not route to the 512-default English model. This is a total sequence budget, not 8,192 state tokens plus unlimited questions. [Committed long-context results][long] use runtime 0.3.18, Apple MPS, one four-option department question, and only 20 padded examples per row. At about 1,000/2,000/3,000/4,000 padding tokens the wide-window accuracies are .80/.85/.85/.90; at 5,000/6,000/7,000 they are .55/.85/.40. About 4,000 tokens take 1.707s median, versus .016s on short inputs. This demonstrates a small padded workload, not broad long-context reliability.

## Published quality and correction history

[Benchmark source][bench] separates runs rather than assigning every headline to one model:

| Reported result | Checkpoint / population | Qualification |
|---|---|---|
| Typed-decisions accuracy **.766**, soft accuracy .471, table Brier .061, ECE .213, score MAE .242 | Fine-tuned typed specialist, 400 cases / 2,000 decisions, four workflows | **No committed result file backs this specialist row or workflow scores.** Fine-tuned on the benchmark's training split; independent reproduction and full row-level artifact are absent. |
| Typed workflow scores .730/.764/.804/.766 | Agent trace / customer service / invoice / security | Same uncommitted specialist run. |
| Base typed accuracy .3620 / .3515 | English / multilingual, committed T4 run | Not the strongest specialist result; majority baseline .461 and random .318. CPU English run is .3615. |
| MASSIVE51 macro accuracy **.4008**, macro ECE .3911, 48/51 languages above 3× random | Multilingual, 5,100 cases, 20 labels, refreshed runtime 0.3.20 | Supersedes old .3661 / .3869 / 45-of-51 values. Vendor reports agreement across three machines, not a Vey reproduction. |
| MASSIVE51 macro accuracy .2269, ECE .5709 | English, current clamped confidence in same refreshed sweep | English remains better on English-only cases (.820 vs multilingual .710); using English for all languages is an avoidable weak baseline. |
| AG News .953, DAIR Emotion .600, Banking77 .492 | Typed specialist, Applications run, 400/task, CPU, runtime 0.2.1 | Different population from T4 English results .947/.573. AG News was in the training mix; Emotion/Banking77 were held out. |
| Spam/phishing .993; guardrails up to .762; toxicity .530 | Best suitable checkpoint varies per task | Spam/phishing were in training; toxicity and jailbreak sources held out. Scores do not imply universal task competence. |

The [refreshed sweep][refresh] retains superseded columns with provenance. The English `choice:11+` raw shipped temperature .1005828 is clamped to .5 by current runtime. [Clamp rerun][clamped] reports macro ECE .7331 raw to .5709 served, with unchanged accuracy .2269. Positive temperature scaling preserves argmax. Old raw calibration values are not today's served values. Model-card language claims cover 100+ languages; the named MASSIVE sweep tests 51, and XNLI tests 15. Coverage claims are not equal quality in every language.

## High K, ordering, and available remedies

The recommendation is roughly **20 choice options or fewer at default budgets**, not a hard 20-option API limit. HTTP source allows 100 choice options and 64 questions, with default configurable total token cap 8,192. Options share the head budget; rendering trims descriptions, potentially collapsing distinct labels. `usage.options` reports total/distinct spans and tokens per option; state truncation is reported separately.

The renderer uses a four-token per-option floor when over budget. At very high K it can exceed the nominal head cap: the README's English K=77 example renders a 319-token head, leaving 192 state tokens in a total 512, not the naive 320. Actual state room is `max_len - actual_head_len - 1`.

Do not treat default Banking77 .425/.425/.492 as an immutable architectural quality ceiling. Current runtime supports per-request `head_max_len=512, max_len=1024`, coarse-to-fine routing, and embedding shortlist (`predict_shortlist`, `k=20`, using the already-loaded encoder or a supplied embedder). Shortlist recall is a separate error source and probabilities become conditional on retained labels. [LangChain guide][langchain] reports synthetic 48- and 72-option recovery from 1/48 and 1/72 to 43/48 and 63/72 after widening; its 24-option result worsens from 24/24 to 20/24, so wider is not uniformly better. These are reported label tests, not a reproduced Banking77 improvement.

The [T4 artifact][t4] reports answer flip rates under option permutations, N=200/suite: English .150/.040/.000 and multilingual .230/.090/.015 on MASSIVE English / Emotion / XNLI English. Option order is positional; order invariance is not guaranteed. No equivalent typed-specialist permutation evidence was located for the .766 headline.

## Calibration procedure for a fair deployment

RLCD / proper-scoring-rule training does **not** establish deployment calibration. Multilingual ships all-one temperatures and an empty option map. English and the typed specialist contain fitted entries, including the same legacy `choice:11+` .10058 entry clamped to .5. Refitting is task/checkpoint/language/option-count/precision dependent; a contributed routing task was under-confident even where other tasks were over-confident.

1. Pin checkpoint, runtime, question renderings, budgets, backend and dtype before collecting labeled calibration data. Keep fitting/threshold selection separate from final evaluation; do not train on the evaluation labels.
2. Use `records_from_labeled(agent, pairs)` for raw logits and targets. Current `fit_temperature_map` / `Agent.fit_temperatures` fits type and option-count buckets by NLL on log-temperature, clamping to [.5, 5]. Type floor is 10 records, bucket floor **2,000**. Smaller buckets fall back to type-level fits.
3. `compute_ece=True` reserves 20% per eligible bucket and reports `n_eval` and excluded buckets. Small buckets remain fit-only and cannot earn held-out calibration credit. Default `compute_ece=False` fits all supplied records and emits no evaluation report.
4. Validate max-probability `answer_confidence`, Brier/NLL/ECE, selective accuracy, coverage and AURC by bucket on separate data. Entropy `confidence` is not the same statistic. Current 0.3.25 optionally applies histogram binning to `answer_confidence` without replacing the full option distribution; binning needs 200 records/bucket by default, and its separate calibration payload persists with temperatures. Per-language temperature overrides bypass the global binning map in the inspected decoder.
5. Choose empirical per-bucket abstention cuts with `fit_abstention_thresholds` (default 100 records/bucket, target error .10, one-error conservative margin), then validate on untouched data. This is **not a formal correctness/coverage guarantee**. Unknown map buckets use the map's `default`, otherwise threshold 0.0 (gate nothing), so supply and validate a default policy. `unevaluated` means confidence was unusable, not an unknown bucket. Standard `predict` keeps answers and marks abstentions; `decide` projects low-confidence fields to `None`.
6. Persist with `save_calibration`, restore with `load_calibration`, and recalibrate after backend/dtype/question changes. If histogram binning is enabled, select thresholds on that recalibrated confidence scale: the standalone threshold fitter shown in source consumes temperature-scaled raw logits, not the binning map.

The historical T4 mean ECE improvements .4656→.0812 and .3135→.1059 are means across suite-level held-out halves, not a current universal deployment number. That notebook fits buckets with at least 25 examples, unlike the current package's 2,000-example floor. Neither value belongs to the .766 specialist row by implication.

## Performance: retain checkpoint, hardware and timed path

| Reported path | English | Multilingual | Caveat |
|---|---:|---:|---|
| Tesla T4, warm SDK, 1 question, p50 / p95 | 39.5 / 44.8ms | 32.8 / 38.8ms | T4 torch 2.11+cu128; synchronized 20 calls after 3 warmups, `system_one`, including serialization/tokenization/forward/decoding. |
| T4, 10 questions, p50 | 158.6ms | 72.3ms | Many questions over one state, not a server concurrency capacity result. |
| T4, 50 questions, p50 / p95 | 771.3 / 806.7ms | 337.4 / 693.7ms | No typed-specialist timing in this T4 artifact. |
| RTX 4070 Ti SUPER, warm `predict`, 1 question / 72 tokens, stock→TileLang | 17.7→4.6ms | 14.1→2.8ms | Different GPU and short input; optional fused kernels/CUDA graphs, not stock T4. |
| EPYC 9R14, 4 cores, fp32, runtime 0.3.20, 1 question p50 | 580ms | 193ms | Same reported CPU run gives typed specialist 584ms. |

The README's **32.8ms belongs to multilingual**, while **.766 belongs to typed-decisions**. Combining them into one strongest-quality/fastest-latency operating point is unsupported. The T4 artifact's own throughput range is 103–332 questions/s across suite workloads; it is not derived by inverting the single-call p50.

A newer [external NVIDIA capacity study][capacity] exists, pinned to older Laya `6a5819129eb220570792e417e49723d697efd76f`: English procurement notices, three typed questions/request, 1,000 notices. TensorRT FP16 capacity at p99≤130ms is reported as 42/146/175 decisions/s on RTX PRO 5000 / RTX PRO 6000 / H100 NVL; H100 reaches 105 decisions/s at p99≤50ms. TensorRT is external ONNX Runtime serving, **not a native Laya backend**. One run/configuration and constrained workload; reported independent of Laya's author does not mean independently reproduced by Vey. No same-host Vey advantage, joules/decision, 12GB residency or workload-normalized superiority has been measured.

## Deployment and ecosystem

- **Python:** 3.10+, PyTorch/HF/safetensors; Agent, Router, typed/schema decisions, batching, long-window scan, CLI, calibration and hooks. CPU/CUDA/MPS/XPU selection and fallback exist; actual fallback devices need recording.
- **HTTP:** `laya-serve`, FastAPI, `POST /v1/systemone`, `/v1/systemone/batch`, health, bearer auth, request/token limits, preload/resident controls, current idle-unload option. SDK/framework schema decisions use these typed endpoints; there is no separate `/v1/decide` route in the inspected server. The explicit batch endpoint shares compatible forwards; the stock server does not implement cross-request dynamic queue batching (the external capacity study used its own serving harness). Jev-compatible transport is not quality parity; strict wire projection is opt-in.
- **MCP:** stdio tools for route/predict/presets/shortlist, batches and schema decisions. 0.3.25 `LAYA_BASE_URL` shares an HTTP server instead of resident weights in each MCP process; remote shortlist/hooks are unsupported, and remote heterogeneous prediction batches issue per-item HTTP requests rather than shared forwards.
- **LangChain / LangGraph:** actual Python integration modules and optional dependencies; `LayaRouter`, `LayaGuardrail`, `LayaTriage`, `LayaEvaluator`, `LayaDecision`, local or HTTP, `batch` / `abatch`. LlamaIndex and CrewAI wrappers also exist. API availability is source-inspected, not exercised here.
- **TypeScript:** separate `laya-client` HTTP ESM/CJS source and `laya-ts` local split-ONNX inference for Node CPU/CUDA and browser WebGPU/WASM. Local runtime is ESM-only, requires exported encoder/head/tokenizer/config artifacts, and does not yet port Python length sorting. Both package manifests say .1.0. Public npm `latest` lookups for **both names returned 404** during retrieval, despite README installation/release guidance; source availability is confirmed, npm publication is not.
- **ONNX / quantization:** ONNXAgent exposes batch/scan/schema operations; export supports CPU dynamic INT8 with per-tensor default. Documentation reports eager agreement only ~67% English / ~83% multilingual per-tensor, versus ~32% / ~40% per-channel; calibration-sensitive deployment cannot assume parity. Claimed CPU acceleration is not measured here. CUDA lacks the relevant INT8 MatMul kernel and may fall back per node.
- **Optimized native paths:** eager/compile/TileLang/ONNX selection, compile warmup/cache controls, and external AOTInductor packaging. AOTI's reported 2.27–3.60ms forwards on seven multilingual rows exclude tokenization and compilation, so they are not interchangeable with end-to-end SDK timings. CPU TileLang lowering is an explicit scalar FP32 kernel specialization, not a full-model CPU speedup claim.

## Local routing and artifact custody

MEASURED: the isolated checkout at `859b8ee595cc04f84dd2af476d6d1d90ec1fea46`
imports as 0.3.25 without replacing the installed training package. Actual
`Router.route` calls select English for `lang="en"`, typed-decisions for
`task="typed_decisions"` and multilingual for `lang="hi"`. No model loads,
Torch imports or model forwards occur.

All fifteen required checkpoint/config/tokenizer files match the pinned public
bundle revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` by upstream Git-blob
SHA-1 or LFS SHA-256. Safetensors headers are readable. English and typed model
files contain 842,609,210 and 842,609,220 bytes; multilingual contains
643,835,514 bytes. Stored scalar counts include buffers and are not live
parameter counts or resident-memory measurements.

The first inventory implementation wrongly assumed repository-local blob
storage and raw SHA-256 cache filenames. Current Hub shared Xet storage violates
both assumptions. The failed source is retained; corrected verification uses
upstream content identities, not Xet filenames.
[The custody manifest](laya_runtime_custody_manifest.json) pins the exercised
code, upstream identities, complete receipt and retained failure.

The subsequent preregistered CPU smoke loads all three pinned routes and
completes identical mixed Choice/Noul/Score requests with valid typed payloads.
All run eager FP32 without AMP. Actual loaded parameter counts are 421,293,827
for English and typed-decisions, and 321,908,995 for multilingual.
Tokenizers are copied into isolated smoke directories; all three config hashes
remain unchanged after the vendor compatibility helper.

The first smoke used an unsupported `1e-5` probability-sum tolerance.
Pinned source rounds each served probability to four decimals; the corrected
bound is `K*0.00005 + 1e-6`, registered before corrected execution. The original
failure is retained. [The CPU result manifest](laya_cpu_compatibility_result_manifest.json)
pins both protocols, scripts, failure receipt and complete successful outputs.
This verifies CPU integration only. GPU compatibility, quality, calibration,
latency, memory residency and neutral comparisons remain unmeasured.
No critical comparison cell is green.

## Remaining evidence gaps and fair comparison configuration

Use the **typed specialist for typed workflows**, routed English/multilingual for language-aware generic decisions, and the supported wider-head/shortlist path for high K. Calibrate on disjoint labels with the actual deployed backend. Pin versions and report each operating point separately.

Missing: committed raw rows for .766, independently reproduced quality/performance, same-host Vey comparison, high-K accuracy after a controlled strong remedy, typed-specialist order robustness, long-context diversity, deployment risk guarantees, actual 12GB memory, energy and cost-per-verified-decision. Forced-choice negation, boolean/`noul` label bias, held-out toxicity and ordinal quality remain documented task-specific limitations, not proof that the whole competitor is weak. Open-source download has no per-decision software fee; self-hosted hardware/energy/operations costs and any paid hosting price are unknown.

[readme]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/README.md
[bench]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/BENCHMARKS.md
[package]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/pyproject.toml
[license]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/LICENSE
[routing]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/docs/routing.md
[router-src]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/laya/router.py
[pins]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/laya/revisions.py
[questions]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/docs/questions-and-answers.md
[calibrate]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/laya/calibrate.py
[http]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/docs/http-api.md
[langchain]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/docs/langchain.md
[t4]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/research/results/t4_colab_benchmark.json
[refresh]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/research/results/cpu_51_language_sweep_refreshed.json
[clamped]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/research/results/cpu_51_language_sweep_clamped.json
[long]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/research/results/long_context_multilingual.json
[capacity]: https://github.com/NandhaKishorM/laya/blob/859b8ee595cc04f84dd2af476d6d1d90ec1fea46/research/results/nvidia_capacity_20260925.json

## Metadata refresh retrieved 2026-10-05

This refresh inspects public primary sources and metadata only. It downloads no package or model payload, loads no model, executes no inference or benchmark, and runs no tests/builds/linters/formatters. Vendor quality and performance reports remain **HYPOTHESIS / not independently reproduced by Vey**. Current source inspection earns no new local-runtime or neutral-comparison status.

### Released package and later source head

[GitHub latest release](https://github.com/NandhaKishorM/laya/releases/tag/v0.3.27) and [PyPI](https://pypi.org/pypi/laya/0.3.27/json), retrieved during this refresh, report **0.3.27**. The release was published 2026-10-04T18:18:11Z and resolves to commit `b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd`. PyPI advertises wheel SHA-256 `7d3c30b1afc8e8ea5ab0285b02de2a46ee71563ba1590e1278a4a637a988fbfe` and sdist SHA-256 `8a1ad028f2b02289fad0cb69a6058c7eb0cbf42506945726693e822b2f7f601b`; these are metadata identities, not locally downloaded package hashes.

The [observed main head](https://github.com/NandhaKishorM/laya/commit/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2) is `8a6e1328cce2460a0e5aa348ad465bb1b5821cd2`, committed 2026-10-04T18:49:24Z, with tree `684b182420daa34bcda50f95fad7603625c035f2`. The [old-to-current comparison](https://github.com/NandhaKishorM/laya/compare/859b8ee595cc04f84dd2af476d6d1d90ec1fea46...8a6e1328cce2460a0e5aa348ad465bb1b5821cd2) contains 40 commits. Its package manifest still says 0.3.27, but the [release-to-head comparison](https://github.com/NandhaKishorM/laya/compare/b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd...8a6e1328cce2460a0e5aa348ad465bb1b5821cd2) includes later changes: the shared `laya.train` loop, rejection of unpublished/path-like HTTP model IDs, and the remaining fine-tune temperature-bound fixes. Do not attribute those later changes to the published 0.3.27 wheel.

### Current recommended model pins, unchanged weights

| Role | Latest observed standalone repository revision | Shipped total / option budget |
|---|---|---|
| English | [`convaiinnovations/laya`, `7b928d828b7b0e022f929d9bd2e44165aa270148`](https://huggingface.co/convaiinnovations/laya/tree/7b928d828b7b0e022f929d9bd2e44165aa270148) | 512 / 192 |
| Multilingual | [`convaiinnovations/laya-multilingual`, `1720e3e3357cfe1e281542e223f8273b0890ca34`](https://huggingface.co/convaiinnovations/laya-multilingual/tree/1720e3e3357cfe1e281542e223f8273b0890ca34) | 1,024 / 256; opt-in total limit 8,192 |
| Typed workflows | [`convaiinnovations/laya-typed-decisions`, `e929ae5cf69bc34259cd2f95c9e91145b818b1f0`](https://huggingface.co/convaiinnovations/laya-typed-decisions/tree/e929ae5cf69bc34259cd2f95c9e91145b818b1f0) | 1,024 / 256 |

The commits are dated 2026-10-03T18:00:07Z, 18:00:08Z and 18:00:10Z respectively. Every commit adds only a 160-byte root `config.json` for Hub download tracking. Comparing the old and current pinned HF trees finds all previous entries unchanged, including cards, `rl_agent_config.json`, tokenizers, bundled subfolders and advertised LFS weight identities. The root config addition does not establish a new trained checkpoint or a quality improvement. This is metadata comparison, not a fresh weight-byte hash, current-revision model load, numerical-equivalence test or AutoModel compatibility claim.

English still uses ModernBERT-large; multilingual still uses mmBERT-base; typed-decisions remains the ModernBERT-large specialist. Parameter metadata, budgets, temperatures and cards are unchanged. The English card still describes runtime 0.3.20. The strongest task-appropriate configuration remains routed English/multilingual for generic language-aware decisions and the explicit typed specialist for its matching workflows.

The [current router](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/router.py) still defaults to the bundle `convaiinnovations/laya`. For a latest-observed bundle snapshot, pin **`7b928d828b7b0e022f929d9bd2e44165aa270148`** and record subfolder `None`, `multilingual` or `typed-decisions`. For `standalone_repos=True`, pin the three distinct revisions above through the per-model revision map. The [vendor's opt-in reviewed-pin table](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/revisions.py) still names the old `55cf4c4…` / `e4e9ddf…` / `1a793eb…` snapshots. `LAYA_REVISION=reviewed` therefore does not select the latest metadata revisions; ordinary unpinned loads still follow the Hub default.

Language routing, explicit typed selection, the two-checkpoint default resident cap, wider-head/high-K controls and the roughly 20-option default-budget recommendation are unchanged. Multilingual 8,192 remains an opt-in total sequence budget, with no new broad long-context quality evidence.

### Calibration and temperature corrections

The following corrections supersede the relevant 0.3.25 caveats above:

1. Released 0.3.27 [`fit_abstention_thresholds`](https://github.com/NandhaKishorM/laya/blob/b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd/laya/calibrate.py) accepts `binning_map`. Pass the installed map when the deployed gate reads histogram-binned `answer_confidence`; fitting a cut on unbinned max-probability and applying it to binned confidence is no longer the prescribed procedure. The [decoder](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/laya/agent.py) still bypasses global binning for a per-language temperature override. Fit those language-specific effective temperatures without applying the global map, then validate cuts on held-out served confidences.
2. Released 0.3.27 [confidence gating](https://github.com/NandhaKishorM/laya/blob/b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd/laya/confidence.py) clears a stale `low_confidence` flag when re-evaluation clears the threshold or the confidence is unusable. The old result dict no longer remains permanently abstained solely because an earlier gate flagged it.
3. Released 0.3.27 [coverage metrics](https://github.com/NandhaKishorM/laya/blob/b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd/laya/evals.py) use whole confidence-tie groups and record `config.coverage_metric_definition=2`. `selective_accuracy@50` can cover more than 50% when the cut intersects a tie. AURC weights each level by its answer count. Regenerate old definition-1 coverage reports before comparing them; cross-definition baseline comparisons are refused. This is a metric-definition correction, not evidence of better model accuracy or calibration.
4. The later head's [Apple-silicon fine-tune script](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/notebooks/laya_finetune_typed_decisions_mps.py) and [single-device fine-tune script](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/research/scripts/finetune_single_device.py) now clamp fitted temperatures with shared `TEMP_MIN=.5` / `TEMP_MAX=5`, rather than [.1, 10]. This removes a fit-versus-served-scale mismatch in those training entry points. It does not change the already-shipped checkpoint temperatures.

The legacy English and typed `choice:11+` raw temperature remains `.10058280825614929`, and served temperature remains clamped to `.5`. Multilingual still ships all-one temperatures and no option map. Type/bucket fit floors 10/2,000, the 20% eligible-bucket ECE holdout, binning floor 200 and abstention floor 100 are unchanged. These procedures remain empirical; no deployment-risk guarantee or fresh held-out calibration result is established.

### Vendor reports and runtime surfaces

The [current BENCHMARKS](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/BENCHMARKS.md) change only adds the paired M1 Pro MPS fp32/fp16 study. The typed specialist `.766` headline, missing committed specialist rows, multilingual MASSIVE51 `.4008` and existing T4 operating points are unchanged. The `.766` specialist accuracy and multilingual 32.8ms T4 p50 still belong to different checkpoints/workloads and cannot be combined.

The new [English](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/benchmarks/results/mps_autocast_english_m1pro.json) and [multilingual](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/benchmarks/results/mps_autocast_multilingual_m1pro.json) artifacts report runtime **0.3.26**, Apple M1 Pro, macOS 26.1, torch 2.14.0 and transformers 5.17.0 on a non-idle machine. The [harness](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/benchmarks/bench_mps_autocast.py) times warm `Agent.system_one` requests and compares the returned outputs from the two modes. It alternates order on the same loaded model, with 72 pairs per cell, and excludes initial load. No HF revision is recorded in the result or load call, so these measurements are not attributed to the latest HF pins or runtime 0.3.27.

Upstream reports fp16 slower on every multilingual row and all English rows except the long eight-question case. It suggests `LAYA_MPS_AMP_MIN_ROWS=1000000` to keep fp32 on that M1 Pro; the runtime default remains five rows. Reported comparisons change 0/1,248 English decisions and 1/1,248 multilingual decisions. The score comparison uses argmax level rather than equality of the returned expected score, and the `noul` comparison uses the .5 boundary. This is workload-specific upstream precision evidence, not a universal optimal configuration, broad quality parity or a Vey measurement.

[Current README](https://github.com/NandhaKishorM/laya/blob/8a6e1328cce2460a0e5aa348ad465bb1b5821cd2/README.md) also documents released batch per-call hooks, long-window scans outside hook dispatch, the `USE_TF=0` default guard and a Java ONNX inference package. These are source-inspected capabilities only; Vey has not built or exercised the new package or its advisory parity lane. The HTTP unpublished/path-like model-ID refusal is a later-head change, not part of the released-wheel attribution above.

### Raw-source custody and unchanged local evidence

The refresh manifest is `/home/fazinahamed/Documents/vey-data/decisionmix/competitors/refresh-2026-10-04/laya/refresh_manifest.json`, SHA-256 `11528aab8def77bd51772c3d7cbca95ab0b80a00df23298130f1a71b0338bbfa`. Its complete raw retrieval index is `retrieval_index.json` in the same directory, SHA-256 `794215dd803928b24f9b1da70005f0ba75106dba7fb3a4770ef030010d8a9ce7`. The folder date names the refresh series; actual retrieval occurred 2026-10-05. The index retains 56 raw HTTP/source bodies, URLs, hashes and capture timing. Four entries retain a bounded capture interval instead of an exact request-start timestamp.

| Raw source body at the current pinned source revision | SHA-256 |
|---|---|
| README | `0ea9fc1c99c358dbf5ef2971fd55d848eed0addefac27850bcca027bfd8fa374` |
| BENCHMARKS | `90b723a9a2398b2559712911a065339b4d0ee4245b0052138ff8ebe80f6e3937` |
| Calibration implementation | `1e27411f4275264755a3fc0d7daf95643fa8dd463143b941f0fbeedad178d316` |

Those hashes cover raw fetched bytes, not reader-rendered markdown. Model LFS SHA-256 values in the manifest are advertised upstream identities compared across metadata snapshots, not newly downloaded or rehashed weights.

The existing [runtime custody manifest](laya_runtime_custody_manifest.json) and [CPU compatibility manifest](laya_cpu_compatibility_result_manifest.json) remain unchanged and apply only to source `859b8ee595cc04f84dd2af476d6d1d90ec1fea46`, runtime 0.3.25 and bundle `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. They are not current-head or current-HF-revision verification. Quality, calibration, GPU compatibility, latency, residency, energy and neutral comparisons remain unmeasured; no critical comparison cell becomes green.

## Published 0.3.27 CPU compatibility

MEASURED under the separately committed [published-release protocol](laya_published_0_3_27_protocol.json):
the exact complete release source at `b09832bdd3819e375fe8b0d26dbd7a8f75c4f0bd`
loads all three pinned current-reference routes on CPU FP32. Each completes the
same synthetic mixed Choice/Noul/Score request, with finite bounded values,
correct question/probability keys and the registered four-decimal sum bound.
This is nine typed answers on one synthetic state, not a quality benchmark.
Live parameter counts are 421,293,827 for English and typed, and 321,908,995
for multilingual; these are loaded parameters, not the stored-scalar metadata totals.

The published wheel's registered `backend='eager'` constructor first failed with
`ModuleNotFoundError: No module named 'laya.backends'`, before any completed route.
The wheel omits five backend files present in the exact release tree. All 35
shipped package members match the release; all 40 complete-source members have
verified SHA-256 and Git blob identities. The successful run imports that complete
unmodified release source, without a missing-module shim, installation or later-head
substitution. It does not establish that every wheel API fails or that wheel packaging
has been repaired.

The multilingual isolated tokenizer copy is rewritten from a list of extra tokens
to the runtime-compatible mapping. Both before/after hashes are retained; the pinned
standalone source bytes remain intact. Its authoritative `resolved_path` overrides
the historical bundled display path in the prospective custody record. English and
typed tokenizer configurations are unchanged. The fixture's Choice has K=3; raw
`choice:11+` temperature metadata and the served floor do not earn high-K evidence.

The [post-execution result manifest](laya_published_0_3_27_result_manifest.json)
pins the successful receipt, original wheel failure, immutable prospective inputs,
release-source custody and route summaries. Historical 0.3.25 evidence above remains
unchanged. GPU compatibility, neutral quality, calibration, latency, residency and
Pareto comparison remain unmeasured; all 21 critical cells stay non-green.

## Pinned raw-bundle neutral development measurements

MEASURED on guarded MASSIVE dev records, using bundle
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` and its own `rl_common` runtime:

| Endpoint | English | Multilingual | Rows per route |
|---|---:|---:|---:|
| K=60 intent Choice accuracy | 0.116739 | 0.178021 | 63,852 |
| Choice ECE, predicted-label confidence | 0.709775 | 0.268751 | 63,852 |
| Grammar normalized ordinal MAE | 0.713256 | 0.478175 | 62,594 |

Independent replay receipts and limitations are recorded in
[`docs/BENCHMARKS.md`](../../docs/BENCHMARKS.md) and
[`COMPETITOR_MATRIX.json`](COMPETITOR_MATRIX.json). All 51 Choice locales have
1,252 rows, so macro and micro accuracy coincide. The original ECE report was
wrong and is explicitly superseded; model predictions are unchanged.
Historical multilingual grammar was exploratory, not prospectively registered.

This is not a current 0.3.27 served-runtime calibration comparison. The study
applied the raw English `choice:11+` temperature, rather than the current served
floor of `.5`. Rounded saved probabilities cannot recover the original logits
for an exact served-temperature replay. No timing, residency, fresh transfer
or Pareto claim follows; all 21 critical cells remain non-green.
