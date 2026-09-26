<a href="https://github.com/fazin-ahamed/vey/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/fazin-ahamed/vey/ci.yml?label=CI" alt="CI"></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python versions">
<img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License">
<a href="https://huggingface.co/fazinahamed/vey"><img src="https://img.shields.io/badge/Hugging%20Face-runtime-ff9d00" alt="Hugging Face"></a>
<a href="https://github.com/fazin-ahamed/vey/releases"><img src="https://img.shields.io/github/v/release/fazin-ahamed/vey?label=release" alt="Release"></a>

# Vey

**A lightweight machine-native decision runtime for fast, typed decisions without
autoregressive generation.**

Vey answers *which candidate, which value, should I act, should I ask* with
narrow mechanisms you can measure one at a time. It emits no output tokens.

## Vey 2: semantic decisions (CRUX)

Vey 2 adds `vey.decide`, a semantic decision runtime that grounds natural
language into typed evidence and executes the decision deterministically. It
routes each decision to the cheapest lane that can answer it: a deterministic
`structured` lane when candidates expose numeric/enum facts (no model), or the
`crux` lane (a frozen NLI predicate grounder + a trained antisymmetric ordinal
comparator) when qualitative language grounding is required.

```python
import vey

result = vey.decide(
    question="minimize latency, then cost",
    candidates={
        "a": "latency 30 ms; cost 5",
        "b": "latency 30 ms; cost 2",
        "c": "latency 90 ms; cost 1",
    },
    explain=True,
)
result.answer          # "b"
result.decision_mode   # "structured" (resolved on facts; no model loaded)
result.certificate.to_dict()  # versioned machine-evidence, not generated text
```

The decision is a pure function of the unordered *set* of candidate
consequence-texts, order- and rename-invariant by construction. Every decision
returns a versioned machine-evidence `Certificate` (the typed decision program,
per-candidate grounded values, and survivors), never generated reasoning. The
crux lane's comparator artifact is configured via `VEY_CRUX_COMPARATOR` /
`VEY_CRUX_COMPARATOR_HF`; without it the crux lane fails closed while the
structured lane keeps running. See [docs/CRUX.md](docs/CRUX.md). The Vey 1
trust-policy decision is unchanged and remains at `vey.trust.decide`.

## What it combines

- **Exact structured decisions**: typed candidate sets with hard eligibility
  masks, online per-candidate reliability, exact recent-history predicates.
- **A 24M semantic encoder**: `mixedbread-ai/mxbai-embed-xsmall-v1`, frozen.
  Pooled head for fixed labels; cosine matcher for open/novel candidates.
- **BM25 + semantic rank fusion**: reciprocal rank fusion over two independent
  retrieval lanes.
- **Long-context lexical pre-selection**: reduce 4096 tokens to ~128 useful
  tokens before the encoder runs.
- **Typed tool/action compilation**: the schema supplies the structure and Vey
  only fills the values, then validates them exactly. It never writes tool-call
  JSON.
- **Calibrated trust and abstention**: an explicit `UNKNOWN` decision, ranked
  risk, and a deterministic action policy.

## Measured (Intel i5-12400F, CPU, batch=1)

| | |
|---|---|
| Semantic decision, end-to-end | **2.6 – 2.9 ms** p50 |
| Long-context decision, after lexical pre-selection | **9 – 11 ms** p95 |
| Semantic model | 24.1M parameters, ~92 MB FP32 weights |
| Peak process RSS | ~825 MB (includes the PyTorch runtime) |

## Strict-parity comparison

Identical rows, candidate descriptions, and labels; same CPU; batch=1. Upstream
Laya 0.3.20 pinned at revision `55cf4c4e`.

| task (options) | Vey acc | Laya acc | Vey p50 | Laya p50 |
|---|---:|---:|---:|---:|
| Banking77 (77) | **0.658** | 0.431 | **2.863 ms** | 478.969 ms |
| CLINC150 (151) | **0.675** | 0.593 | **2.760 ms** | 1014.777 ms |
| MASSIVE (60) | **0.471** | 0.403 | **2.598 ms** | 378.022 ms |

Vey wins accuracy on all three, runs **145–368× faster end-to-end**, and ships
**17.5× less model weight** (92 MB vs 1607 MB FP32). The trade is real: Laya
returns calibrated probabilities and Vey does not, and Vey's own pooled head
(0.884 / 0.832 / 0.780) beats the matcher on trained labels. Neither system is
better everywhere. See [docs/COMPARISON.md](docs/COMPARISON.md) for the hardware
and methodology caveats.

## Install

```bash
pip install -e .
```

The default encoder is fetched from the Hugging Face hub on first use; no weights
are committed to this repository.

## Use

```python
from vey import ToolCard, ArgSpec, extract_slot, compile_call

card = ToolCard(
    tool_id="book_flight",
    name="book_flight",
    description="Book a flight",
    args=(
        ArgSpec("destination", "string", True, "arrival city"),
        ArgSpec("date", "date", True, "travel date"),
        ArgSpec("passengers", "integer", False, "traveler count"),
    ),
)

state = "Book a flight; date 2026-09-25; passengers 2"
decisions = {a.name: extract_slot(a, state) for a in card.args}
call = compile_call(card, decisions)

call.outcome   # "ASK_FOR_INFO": "destination" has no explicit cue in the state
call.missing   # ('destination',)
call.args      # {'date': '2026-09-25', 'passengers': 2}
```

That result is the point. Typed and labeled values are extracted and validated
exactly; a required value with no explicit cue is left unresolved rather than
guessed. This is why the unsupported-argument rate is zero, and why Vey answers
`ASK_FOR_INFO` more often than a generative caller would emit a call. See
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the coverage cost of that choice.

## Demo

```bash
python -m examples.snake.snake
```

The Snake demo is a feature-assisted demonstration of Vey's structured decision
loop. **It is not evidence that the routing head perceives raw game frames.**

## Documentation

- [Architecture](docs/ARCHITECTURE.md): the lanes and why they are separate
- [CRUX](docs/CRUX.md): the Vey 2 semantic decision API and executor
- [Development](docs/DEVELOPMENT.md): how to extend the package
- [Model card](docs/MODEL_CARD.md): intended uses, non-intended uses, data
- [Decision API](docs/DECISION_API.md): the public interfaces
- [Benchmarks](docs/BENCHMARKS.md): how every number was measured
- [Comparison](docs/COMPARISON.md): Laya / Jev, two tracks, never one ranking
- [Limitations](docs/LIMITATIONS.md): what Vey does not do, and what was
  measured and rejected

## Community

- [Issues](https://github.com/fazin-ahamed/vey/issues)
- [Discussions](https://github.com/fazin-ahamed/vey/discussions)
- [Contributing](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Support](SUPPORT.md)
- Hugging Face: [runtime card](https://huggingface.co/fazinahamed/vey) |
  [demo Space](https://huggingface.co/spaces/fazinahamed/vey-demo)

## Contributing

Contributions are welcome. Bug fixes, documentation, benchmark reproductions,
runtime optimizations, new exact mechanisms, and well-tested integrations are
especially useful.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Benchmark
claims require reproducible evidence, and new mechanisms are compared against a
simpler control first.

## License

Apache-2.0. The default semantic encoder is a separate Apache-2.0 work by the
Mixedbread AI authors; see [NOTICE](NOTICE).
