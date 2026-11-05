# System One server

`vey-serve` answers typed questions about a state over the wire format that
TypeSafe's Jev, Cloudflare's Clef and Laya's server share (`POST /v1/systemone`),
so a client written for one of them works by changing its base URL. It runs on
ONNX Runtime with no PyTorch, and serves a console at `/`.

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
| `box` questions, `assets` (images), multipart requests | only through escalation (below); otherwise a 400 that says so |
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
| `escalated` | anything below `--floor`, and everything with images or `box` | The request is sent to another System One server |

A state longer than a model can read is cut down with BM25 sentence selection against
the question and options (the same long-context lane Vey uses elsewhere), not
truncated from the right. A state that fits is embedded once and shared by every
question in the request.

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

## Escalation

Nothing leaves the machine unless you set `--escalate-url`. With it set:

- a question answered below `--floor` goes to the target, and its answer replaces the local one;
- a request with `assets` or a `box` question goes to the target whole, and a multipart
  request is proxied byte for byte (up to `--max-proxy-body`);
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
- Not supported: `multiple` choices, `point`, `read` and `estimate` answers, a vision lane.
  Images work only through escalation.

## Reproducing

```bash
python scripts/onnx_parity.py --rows 500 --torch      # encoder parity, memory, latency
python scripts/systemone_eval.py --write cal.json     # accuracy, calibration, routing
vey-serve --port 8000 & python scripts/systemone_bench.py http://127.0.0.1:8000
VEY_NETWORK_TESTS=1 pytest tests/systemone            # includes the OpenAPI conformance tests
```
