# Laya: current primary-source dossier

Retrieved 2026-10-03, beginning at 17:35:16 UTC. Audience: Vey researchers choosing a fair competitor configuration. The companion `LAYA.json` records source hashes and machine-readable facts.

## Evidence boundary and current version

This is source and metadata inspection, not an inference benchmark. API existence and configuration are **MEASURED source inspection**. Every quality, latency, throughput, calibration and numerical-parity result below remains **HYPOTHESIS / vendor- or externally reported, not independently reproduced by Vey**. No model weights were downloaded, no model/API calls were made, and no tests or builds were run.

The current Python package is **0.3.25**, not 0.3.20: [GitHub latest release](https://github.com/NandhaKishorM/laya/releases/tag/v0.3.25) published 2026-10-03T15:58:35Z, [PyPI metadata](https://pypi.org/pypi/laya/json), and the pinned package manifest agree. Release commit: `8a976468b57c1b53541363dc86523263c9b68d34`. This inspection pins source main at `859b8ee595cc04f84dd2af476d6d1d90ec1fea46`, committed 2026-10-03T17:31:47Z. The model cards still describe runtime 0.3.20, so GitHub/package sources take precedence for current API behavior. Published benchmark environments are older and are not silently relabeled 0.3.25.

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
