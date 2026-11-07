# System One server

`vey-serve` answers typed questions about a state over the wire format that
TypeSafe's Jev, Cloudflare's Clef and Laya's server share (`POST /v1/systemone`),
so a client written for one of them works by changing its base URL. The base
profile runs on ONNX Runtime with no PyTorch, and serves a console at `/`.
The opt-in d1 lane below still uses PyTorch.

```bash
pip install -e .          # no PyTorch
vey-serve --port 8000     # 127.0.0.1 by default; console at http://127.0.0.1:8000/
```

```bash
curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "vey-latest",
  "state": "I was charged twice for my subscription. Please refund the extra charge.",
  "questions": {
    "billing": {"type": "noul", "instructions": "Is this message about billing?"},
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "Charges and invoices", "technical": "Bugs and outages"}},
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]}}}'
```

The models download from the Hugging Face Hub on first use as ONNX files, pinned
to a revision: the 24M `mixedbread-ai/mxbai-embed-xsmall-v1` encoder (Apache-2.0)
and, on the first yes/no question, `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`
(MIT).

## Compatibility

Checked against TypeSafe's published OpenAPI document: `tests/systemone/test_conformance.py`
validates this server's responses, its error bodies and `/v1/models` against those
schemas, and answers the three request examples in that document. It is opt-in
(`VEY_NETWORK_TESTS=1`, `pip install jsonschema`) because it fetches the spec.

| | Supported |
|---|---|
| `noul`, `choice`, `score` | yes, with the published request and response fields |
| `state` as string, object or array; instructions and criteria as string, object or array | yes |
| Limits: 1 to 255 choices, 1 to 10 score levels, at least one question | yes, enforced with the published 422 shape |
| `400 {"detail": ...}`, `422 {"detail": [{loc, msg, type}]}`, bearer auth, `GET /v1/models` | yes |
| `assets` (images) as `data_base64` | with `--d1` (Tier 2, below), locally; otherwise through escalation |
| `box` questions, multipart image parts | only through escalation (below); otherwise a 400 that says so |
| `TypeSafe-Danger-Zone-*` headers | forwarded to the escalation target, ignored locally |
| d1's `multiple` choices, `point`, `read`, `estimate` answer types | no |

Differences a client can see:

- `model` in the response is `vey-1` (`vey-latest` is an alias). Any model name in a
  request is accepted and answered by Vey, so a client sending `jev-latest` works.
- `usage.output_tokens` is always 0: nothing is generated.
- Add `"explain": true` to a request, or the header `X-Vey-Explain: 1`, for a `trace`
  with the lane, confidence, temperature and time of every question. Without it the
  response has exactly the published fields. `explain` is never forwarded.
- `confidence` on a `noul` answer is not in the published schema, so none is returned.
  The trace carries it.

## How a question is answered

Each question runs on the cheapest lane that can answer it.

| Lane | Handles | How |
|---|---|---|
| `exact` | `choice`, `noul`, `score` | The state is an object with a field named like the question (case and punctuation blind, up to 3 levels deep) whose value is a valid answer: a listed choice, a boolean or `yes`/`no`, an integer level. Probability is 1 or 0. No model runs. `--no-exact` disables it |
| `matcher` | `choice`, `score` | The encoder embeds the state once and each option; cosine becomes a probability through a temperature that depends on the option count |
| `matcher+nli` | `choice` of 2 to 20 options | When the matcher's confidence is under `rerank_below` (0.6), the NLI model re-orders the matcher's top 5. The matcher's top-versus-rest probability mass is kept; the NLI model only orders within the top |
| `nli` | `noul` | A zero-shot NLI model checks the state against a statement. A yes/no question is first rewritten into a statement (see below). P(yes) is a sigmoid of the entailment margin |
| `d1` | With `--d1`: images, `score`, and a `choice` the encoder is under 80% sure of | Liquid AI's d1-omni-600M, int8, local (Tier 2, below) |
| `escalated` | anything below `--floor`, `box` questions, and images without `--d1` | The request is sent to another System One server |

The light lanes cut long states down with BM25 sentence selection against the
question and options. A state that fits is embedded once and shared by every
question in the request. d1 instead uses its own context and right truncation.
A choice or score with only one legal answer is forced, with probability 1, even
with `--no-exact`; no model needs to read it.

**Phrase yes/no questions as statements, or give `criteria.true`.** The NLI model was
trained on statements. On 200 SST-2 reviews, "This movie review is positive." scored
0.795 and "Is this movie review positive?" scored 0.485, chance. The engine rewrites
common yes/no questions (`Is this message about billing?` becomes `This message is
about billing.`) with a small deterministic rule set, and passes anything it cannot
parse through unchanged. `to_statement` in `systemone/nli.py` shows exactly what it does.

### Determinism

A text's embedding does not depend on what else is in the request. This is deliberate
and measured: ONNX Runtime's dynamic int8 quantization scales activations over the
whole batch, so the same text embedded inside a batch differed from the text alone by
cosine 0.984 at worst (NLI margins by up to 0.57 logits). The fp32 export is exactly
unchanged. Each text, and each NLI pair, is therefore encoded alone (`batch_size=1`),
which keeps the matcher's guarantee that permuting or removing options cannot move
another option's score. The cost is about 1.5 ms per new text; each is cached.

## Measured

Host for every number below: 2 vCPU Intel Xeon at 2.6 GHz with AVX-512 VNNI, no GPU,
2 threads per model. Latency is hardware specific and is not comparable to numbers
measured elsewhere.

### Memory

| | Resident |
|---|---:|
| 24M encoder, PyTorch (`vey[torch]`) | 465 MB |
| 24M encoder, ONNX fp32 | 204 MB |
| 24M encoder, ONNX int8 (default) | 126 MB |
| `vey-serve` after start, no NLI | 126 MB |
| `vey-serve` after the NLI model loads and 376 requests | 455 MB (cgroup peak 441 MB) |

The server ran under a hard 1 GiB `memory.max` with no swap and no OOM events. The NLI
model adds about 330 MB whatever ONNX Runtime options were tried (arena off, prepacking
off, graph optimization levels); it loads on the first yes/no question, or at start with
`--preload`, and `--no-nli` keeps the server at 126 MB.

### Latency (`scripts/systemone_bench.py`, client round trip, 60 requests each)

| Request | p50 | p95 |
|---|---:|---:|
| exact lane | 0.8 ms | 1.0 ms |
| choice, 4 options | 1.2 ms | 1.8 ms |
| score, 3 levels | 1.1 ms | 1.5 ms |
| noul (NLI) | 14.8 ms | 18.5 ms |
| 3 questions (noul + choice + score) | 16.6 ms | 23.5 ms |

Four concurrent clients sent three-question requests at 78 requests per second with 0
errors. The first NLI request took 1,065 ms because the model loads then. Choice and
score timings are with the option embeddings cached, which a server does after the
first request that uses them.

### Does the ONNX encoder match PyTorch? (`scripts/onnx_parity.py`, 500 rows each)

Open-candidate matcher accuracy, label names as the candidates:

| Backend | Banking77 | CLINC150 | Cosine to fp32 | p50 per query |
|---|---:|---:|---:|---:|
| PyTorch fp32 | 0.608 | 0.704 | 1.0000 | 10.8 ms |
| ONNX fp32 | 0.608 | 0.704 | 1.0000 | 2.6 ms |
| ONNX int8 (default) | 0.602 | 0.706 | 0.992 | 1.9 ms |

int8 is within 0.6 points of fp32, and ONNX fp32 reproduces PyTorch exactly. These
accuracies are not comparable to the README's Laya table, which used different candidate
descriptions.

### How accurate are the answers? (`scripts/systemone_eval.py`, 300 held-out rows per task)

Temperatures were fitted on 400 train rows of each task and tested on disjoint test rows.
Options are label names or a few words, so your own criteria will behave differently.

| Task (options) | Matcher alone | Cascade | Rows reranked |
|---|---:|---:|---:|
| SST-2 sentiment (2) | 0.673 | 0.773 | 28% |
| AG News topic (4) | 0.770 | 0.867 | 29% |
| Emotion (6) | 0.473 | 0.680 | 76% |
| Banking77 intent (77) | 0.627 | 0.627 | 0% (over the 20-option cap) |
| CLINC150 intent (150) | 0.710 | 0.710 | 0% (over the 20-option cap) |
| SST-5 score, 5 levels | exact 0.180, within one level 0.703 | | |
| Yes/no, sentiment and topic questions | accuracy 0.947 | | |

- **The cap at 20 options is measured, not assumed.** An earlier run with the NLI rerank on
  at 150 options lowered CLINC150 from 0.707 to 0.680 and moved Banking77 by +1.3 points, so
  the rerank is off above 20 options. Between 7 and 20 options is unmeasured.
- **`score` is at chance on exact level** (0.180 against 0.200). It is useful only as a
  rough ordering, and the within-one-level figure is the honest one. Do not rely on it for
  fine ratings; set `--floor` and escalate.
- **Tone and urgency are weak.** The console's tone example, an angry and urgent message,
  gets "calm" at 54% and urgency "no" at 58% from this stack. Those answers sit under a 60%
  floor and are flagged rather than shipped, which is what the floor is for.

### Calibration

Probabilities come from one temperature per question type and option count
(`choice:2`, `choice:3-5`, `choice:6-10`, `choice:11+`, `score`, `noul`, `choice_nli`).
`scripts/systemone_eval.py` fits them and ships a fitted value only if it beat the untuned
default on held-out expected calibration error, which left `choice:3-5` (0.0424),
`choice:6-10` (0.0621) and `choice_nli` (3.04) fitted and the rest at their defaults. These
defaults fit public datasets. Fit your own, then pass the file with `--calibration`:

```bash
pip install datasets
python scripts/systemone_eval.py --fit-rows 400 --eval-rows 300 --write calibration.json
vey-serve --calibration calibration.json
```

`vey.systemone.fit_temperature`, `fit_noul_temperature` and `ece` fit and check your own
labelled rows directly.

## Tier 2: d1-omni-600M, local

`--d1` adds Liquid AI's open [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M)
(a 381M bidirectional LFM2.5-Encoder-350M trunk and decision head, a 94M SigLIP2 vision
tower) as a local lane. It improves the measured text score and small-choice tasks
below. Local vision is **experimental**, not an accuracy-validated image service.
This lane uses PyTorch, not ONNX Runtime or ggml, and does not meet the 1 GiB target.

**License: read this first.** The weights are under the
[LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-omni-600M/blob/main/LICENSE),
not Vey's. Commercial use is licensed only while you and your affiliates have annual
revenue under $10M; redistribution must carry the license text and a notice of changes; and
an int8 conversion is a Derivative Work. So Vey never ships or hosts these weights.
`vey-d1-prepare` downloads them from the Hub at a pinned revision and converts them on
your machine, writing the license and a modification notice beside the result. It runs the
model's own code from that pinned revision, the same trust as `trust_remote_code`. This is
not legal advice.

```bash
pip install -e ".[d1]"        # PyTorch, transformers>=5.15, torchvision, Pillow
vey-d1-prepare                # about 2.3 GB download, converted to ~480 MB int8, once
vey-serve --d1 --no-nli --d1-noul
```

How questions route when `--d1` is on:

| Question | Route | Why |
|---|---|---|
| exact field match | `exact` | unchanged, no model |
| `score` | d1 | exact level 0.220 to 0.487, within one level 0.693 to 0.960 |
| `choice` | the encoder first; d1 when its confidence is under `--d1-choice-below` (0.8) and there are at most 50 options | see the sweep below |
| `noul` | the NLI model; d1 with `--d1-noul` | d1 was 0.953 against 0.939 at 15 times the time |
| any question in a request with images | d1, all in one call so the images are encoded once | nothing lighter can read them |

Images arrive as `assets` with `data_base64` (png, jpg, jpeg; up to 4 per request, 8 MiB
decoded bytes each, within the server's overall `--max-body` limit). A placeholder
matched as a whole token inside a string becomes "the attached image", a state that is only a
placeholder becomes empty so d1 reads the images alone, and each image is shrunk to a
longest side of 512 px (`--d1-max-image-side`), one 256-token tile. If d1 fails on a text
request the answer falls back to the lane Vey used before; on an image request nothing can,
so it is a 503. Image choices above 50 options need escalation (or return 400),
rather than silently falling back to a text-only matcher.

Measured (`scripts/systemone_eval.py --d1-only --d1-choice-below 0.6`, 150 held-out rows per task, the same rows in
every column, 2 vCPU):

| Task (options) | Light lanes | Routed at 0.6 | d1 alone | Rows sent to d1 | ms light / routed |
|---|---:|---:|---:|---:|---:|
| SST-2 sentiment (2) | 0.800 | 0.813 | 0.933 | 27% | 24 / 57 |
| AG News topic (4) | 0.787 | 0.813 | 0.847 | 36% | 42 / 104 |
| Emotion (6) | 0.673 | 0.813 | 0.853 | 77% | 68 / 181 |
| Banking77 intent (77) | 0.633 | 0.633 | not run | 0% (over 50 options) | 3 / 0 |
| SST-5 score (5) | 0.220 exact | 0.487 exact | | all | 3 / 225 |
| Yes/no (148 rows) | NLI 0.939 | d1 0.953 | | | 14 / 220 |

The "Routed" column above used a confidence threshold of 0.6. Sweeping it, on the same rows:

| `--d1-choice-below` | SST-2 | AG News | Emotion | Share sent to d1 (SST-2 / AG / Emotion) | ms (SST-2 / AG / Emotion) |
|---:|---:|---:|---:|---|---|
| 0 (encoder only) | 0.720 | 0.687 | 0.500 | 0 / 0 / 0 | 3 / 5 / 3 |
| 0.6 | 0.813 | 0.813 | 0.813 | 27% / 36% / 77% | 56 / 95 / 176 |
| **0.8 (default)** | 0.927 | 0.827 | 0.847 | 78% / 69% / 93% | 154 / 179 / 231 |
| 0.95 | 0.933 | 0.847 | 0.853 | 100% / 93% / 97% | 200 / 253 / 223 |
| 1.0 (always d1) | 0.933 | 0.847 | 0.853 | 100% / 100% / 100% | 194 / 305 / 231 |

The encoder is often confidently wrong on these label sets, so at 0.6 many mistakes never
reach d1. At 0.8 most rows do, and the encoder mainly saves time on easy intents and on
choices with more than 50 options. Above 0.95 there is almost nothing left to gain.
The zero-threshold control is the encoder alone, not the NLI-reranked light cascade.
The threshold was chosen using this sweep, so this is tuning evidence, not an
untouched final estimate. Yes/no in the first table explicitly tests `--d1-noul`;
default routing still uses NLI. d1's higher accuracy does not uniformly improve ECE:
SST-5 ECE was 0.051 to 0.094, and yes/no 0.022 to 0.040.

### Memory and latency recheck (2026-11-07)

Lean profile: `vey-serve --d1 --no-nli --d1-noul --threads 2`, with the same 2 vCPU
host above, Python 3.11.6, torch 2.15.0+cpu, transformers 5.22.1. No concurrent
training or browser. `scripts/systemone_bench.py --images --skip-throughput --requests 9`
checks each response's typed probabilities and weighted score as well as timing it.
These are a new measurement with that script's longer support-ticket fixture, not
the earlier short-ticket timings. Three warmups per workload, then nine round trips:

| Workload | p50 | RSS after workload |
|---|---:|---:|
| exact field | 0.9 ms | 851.6 MiB |
| choice, 4 options (cached matcher) | 1.4 ms | 854.7 MiB |
| score, 3 levels (d1) | 239.4 ms | 870.4 MiB |
| noul (d1) | 220.4 ms | 863.6 MiB |
| 3 mixed questions | 368.4 ms | 872.6 MiB |
| 512px receipt, 2 questions | 1,670.9 ms | 986.0 MiB |
| 2000px receipt, 2 questions | 1,679.4 ms | 1009.0 MiB |

Startup RSS: 799.6 MiB. Peak process RSS (`VmHWM`): 1283.4 MiB. Under a hard
1.25 GiB cgroup, `memory.peak` reached the limit, 1,342,177,280 bytes, with 88
`memory.events.max` events but zero `oom` and `oom_kill`. **There is no demonstrated
headroom.** RSS and cgroup charge are different metrics. The historical wire field
`resident_mb` holds MiB on Linux; the CLI and console now label it accordingly.

Under a fresh hard **1 GiB** cap, with swap disabled, the same script completed
the text workloads (902.1 MiB RSS after mixed questions) but the server was killed
on the first image warmup: `memory.peak=1,073,741,824`, `oom=1`, `oom_kill=1`.
This does not satisfy either a 1 GiB or a decimal 1 GB whole-server budget. The
1.25 GiB result applies only to this lean, serial, bounded workload, not to
`--d1` with the additional NLI model or to arbitrary long inputs and concurrency.
Audio is not loaded. Conversion and training need more RAM than inference.

Quantization: large weights are per-channel int8 and activations are quantized per token and
multiplied with `torch._int_mm` when supported by the Torch build; otherwise the
runtime falls back to weight-only int8, which has a different speed and memory
envelope. The 16 trunk `out_proj` layers stay weight-only. Int8 is not bitwise
equivalent to fp32.

### Vision control and shared-state head

`python scripts/systemone_eval.py --d1-vision-control` checks solid red, green and
blue 512px images, comparing quantized inference, fp32 vision with the int8 trunk,
and a full-fp32 control. **All three got only 2/3 right**: red was classified as
blue, including 91.7% confidence in the quantized path. Green and blue were correct.
The failure is not specific to vision quantization and survives the full-fp32
control. This tiny control is not a broad vision benchmark, but it rules out a
claim of reliable image accuracy. A confidence floor does not catch confident errors.

The shared-state prototype encodes the state once and caches each trunk layer's
attention keys/values and convolution boundary. A newly adapted head achieved
1.86x speedup at six questions, but fails score agreement and unseen-task accuracy.
It is **not wired into the server**. See the protocol, controls, results and commands
in [SHARED_STATE_HEAD.md](research/SHARED_STATE_HEAD.md).

## Escalation

Nothing leaves the machine unless you set `--escalate-url`. With it set:

- a question answered below `--floor` goes to the target, and its answer replaces the local one;
- a request with a `box` question, or with `assets` when `--d1` is off, goes to the target
  whole, and a multipart request is proxied byte for byte (up to `--max-proxy-body`);
- the model is replaced with `--escalate-model`, `explain` is never forwarded, and the
  `TypeSafe-Danger-Zone-*` headers are;
- the target's key comes from the `VEY_ESCALATE_API_KEY` environment variable.

If the target is unreachable, a low-confidence question keeps its local answer and the
trace records `escalation_error`; a request that only the target could read returns 502.
The target can be Jev, Clef, a larger d1 behind SGLang, or another Vey. Whatever you send
there leaves your control, so check the target's terms first.

## Console

`GET /` serves one self-contained page: paste a state (text or JSON), edit the questions,
run, and read each answer with its probability bars, the lane that produced it, and its
time. The "ship at or above" slider shows which answers would be escalated at a given
confidence. The header shows resident memory and which lanes are loaded, and the page
lists recent runs and the request and response JSON. It makes requests only to its own
origin and renders all text with `textContent`.

## Running it safely

- It binds to 127.0.0.1. To serve other machines, put a reverse proxy with TLS in front
  of it. It is a small threaded server, not a hardened one.
- Set `VEY_API_KEY` (preferred over `--api-key`, which shows in the process list) to require
  `Authorization: Bearer ...` on `/v1`. `/healthz` and the console page stay open; the page
  asks for the key.
- Bodies over 1 MiB (`--max-body`) are refused with 413.
- The embedding cache is a bounded LRU (`--cache-size`, default 50,000 texts).

## Limits

- English. Both models are English-focused.
- CPU only, and tuned for small requests. The encoder reads 96 tokens and the NLI model 256.
- Yes/no is only as good as the NLI model: 0.947 on sentiment and topic questions, weaker on
  properties such as urgency.
- Not supported: `multiple` choices, `point`, `read` and `estimate` answers, `box` questions,
  audio. Images are read locally only with `--d1`; otherwise they need escalation.

## Reproducing

```bash
python scripts/onnx_parity.py --rows 500 --torch      # encoder parity, memory, latency
python scripts/systemone_eval.py --write cal.json     # accuracy, calibration, routing
vey-serve --port 8000 & python scripts/systemone_bench.py http://127.0.0.1:8000
VEY_NETWORK_TESTS=1 pytest tests/systemone            # includes the OpenAPI conformance tests
```

For the T2 memory test, prepare the weights outside the cap, then start only the
server inside a dedicated cgroup v2 scope. On a Linux host with systemd and sudo:

```bash
sudo systemd-run --scope --unit=vey-t2-test \
  -p MemoryMax=1073741824 -p MemorySwapMax=0 \
  runuser -u "$USER" -- env VEY_CACHE="${VEY_CACHE:-$HOME/.cache/vey}" \
  "$(command -v vey-serve)" --port 8000 --d1 --no-nli --d1-noul --threads 2
```

In another terminal, outside that scope:

```bash
python scripts/systemone_bench.py http://127.0.0.1:8000 --requests 9 --images --skip-throughput
```

For 1.25 GiB use `MemoryMax=1342177280`. Record the server's `VmRSS` and `VmHWM`
from `/proc/<server-pid>/status`, and `memory.max`, `memory.swap.max`,
`memory.peak` and `memory.events` from its cgroup before removing the group.
The scope path is given by `systemctl show vey-t2-test.scope -p ControlGroup`.
Our orb runs used manually created, persistent cgroups with the same limits so
the event files remained readable after the OOM; the benchmark client, browser
and model conversion were outside those cgroups.
