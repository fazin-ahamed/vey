# Architecture

Vey is a set of narrow lanes, each independently measured and replaceable.
Structure, semantics, and trust are separate problems, and one model does not
have to solve all three.

```
                      user / state
                           │
              ┌────────────┴────────────┐
              │                         │
        structured input            natural language
              │                         │
        ┌─────▼─────┐           ┌───────▼────────┐
        │ exact lane │           │ semantic  24M   │
        │ R0.5 +     │           │ frozen encoder  │
        │ FactMemory  │           └───────┬────────┘
        └─────┬─────┘                   │
              │              ┌───────────┴───────────┐
              │              │                       │
              │        fixed labels            open candidates
              │        (pooled head)           (cosine matcher)
              │              │                       │
              │              └───────────┬───────────┘
              │                          │
              │                  ┌───────▼────────┐
              │                  │  BM25 lexical  │
              │                  │  lane          │
              │                  └───────┬────────┘
              │                          │
              │                  ┌───────▼────────┐
              │                  │  RRF fusion    │
              │                  └───────┬────────┘
              └──────────┬───────────────┘
                         │
                    ┌────▼─────┐
                    │ decision │
                    └────┬─────┘
                         │
         ┌───────────────┼────────────────┐
         │               │                │
   ┌─────▼──────┐  ┌─────▼──────┐  ┌──────▼───────┐
   │ ToolCard   │  │  typed     │  │ trust        │
   │ selection  │→ │  extraction│→ │ observation  │
   │ (24M)      │  │  + exact   │  │ + risk       │
   └────────────┘  │  compiler  │  └──────┬───────┘
                   └─────┬──────┘         │
                         │           ┌────▼─────┐
                    ACT  │           │ policy   │
                         │           │(determ.) │
                    ASK_FOR_INFO    └──────────┘
```

## The lanes

### Exact structured lane

Scores a typed candidate set with hard eligibility masks applied *before*
scoring, online per-candidate reliability (EMA/LCB), and exact recent-history
predicates. It is stateful by design: reliability and history are live, not
decorative. It consumes structured input, not natural language.

### Semantic lane

A frozen 24.1M-parameter sentence encoder
(`mixedbread-ai/mxbai-embed-xsmall-v1`, Apache-2.0) with two heads:

- **PooledClassifier**: a trained softmax head, for a *fixed* label set. More
  accurate than the matcher on trained labels (0.884 / 0.832 / 0.780 on
  Banking77 / CLINC150 / MASSIVE).
- **CandidateMatcher**: cosine over *cached* candidate embeddings, for
  open/novel candidate sets. Because embeddings are cached by text and the query
  is encoded once, a candidate's score is independent of the set it appears in:
  permuting, inserting, or removing candidates changes nothing. Measured score
  drift is exactly 0.0 and winner agreement is 1.000.

That invariance is why Vey has no set-transformer, DeepSets, or candidate
self-attention layer: independent scoring is sufficient.

### Retrieval lane

BM25 and the semantic lane are fused with one fixed Reciprocal Rank Fusion rule
(k=60). The two lanes fail differently out of domain, and fusion beat either lane
alone on every FreshStack technical domain.

### Long-context lane

BM25 sentence selection reduces a long state to ~128 useful tokens *before* the
encoder runs. This is the decision path, not a preprocessing convenience: on a
K-way answer-choice task the selected 128 tokens score 105–147% of the full
context, because mean-pooling a long state dilutes the signal.

### Action compiler

A tool is a typed `ToolCard`. Each slot resolves to exactly one of
`VALUE(span) | MISSING | AMBIGUOUS`. The compiler emits `ACT` only when every
required slot is resolved, every type validates, every enum constraint holds,
and no undeclared argument appears; otherwise `ASK_FOR_INFO` or `REJECT`. It
never invents a value, which is why the unsupported-argument rate is exactly 0.

### Trust lane

The trust layer estimates *risk*; it never emits an action. Exact schema state
decides first, and a calibrated risk estimate gates only the uncertain remainder.
A lane disagreement is a request to run the other cheap lane (`RERUN_CHEAP_LANE`),
never an `ACT`. The policy is deterministic and readable.

## Why the lanes are separate

Each boundary exists because a measurement forced it:

- The exact lane exists because structured decisions should not depend on a
  language model.
- Two semantic heads exist because pooled wins on trained labels and the matcher
  is the only one that generalizes to candidates it never saw.
- BM25 stays a lane because it wins some out-of-domain cases the semantic lane
  loses.
- The action compiler exists because generating structured output is the wrong
  mechanism for exactness.

See [LIMITATIONS.md](LIMITATIONS.md) for what was measured and deliberately not
retained.
