# Development guide

How the public package is organized, and how to extend it without turning the
repository into a research directory.

## Package layout

```
src/vey/
  __init__.py        public surface: ToolCard, ArgSpec, compile_call, decide, ...
  structured/        exact decisions over typed state
    fact_memory.py   exact recent-history predicates
    success_head.py  P(success | features, online reliability)
    trigger_bank.py  per-candidate experience memory
  semantic/          frozen 24M encoder
    encoder.py       load_encoder, Embedder, PooledClassifier
    candidates.py    CandidateMatcher (open candidates)
  retrieval/         BM25, semantic scoring, reciprocal-rank fusion
  act/               typed action compiler
    tool_card.py     ToolCard, ArgSpec
    extract.py       exact slot extraction
    compile.py       compile_call, is_schema_valid
  trust/             calibrated risk
    observation.py   the per-decision observation record
    controls.py      risk controls and calibration
    policy.py        deterministic action policy
  evaluation/        calibration and ranking metrics
```

## Why lanes are separate

Each lane solves one problem and can be measured or replaced on its own:

- `structured` handles typed inputs and does not need a language model.
- `semantic` has two heads because a pooled head is more accurate on a fixed
  label set, while a cosine matcher is the only one that handles a candidate set
  it was never trained on.
- `retrieval` keeps BM25 because it wins cases the semantic lane loses.
- `act` is deterministic because generating structured output is the wrong tool
  for exactness.
- `trust` estimates risk and never emits an action.

The public API in `vey/__init__.py` is stable. New mechanisms enter through a
lane, not through special cases in a central dispatcher.

## Adding a semantic lane

1. Add a module under `src/vey/semantic/` that exposes an object with the same
   shape as the existing heads: a `score`-like method returning one value per
   candidate.
2. Do not change `CandidateMatcher` to special-case your lane. If a new lane
   needs a different scoring rule, give it its own class.
3. Measure it against `CandidateMatcher` on the same rows. If it does not win,
   do not add it; [docs/LIMITATIONS.md](LIMITATIONS.md) records several
   plausible mechanisms that were measured and dropped.

## Adding a retrieval backend

1. Add a scoring function that returns one score per document, in the same order
   as the document list.
2. If it should participate in fusion, return a score vector compatible with
   `vey.retrieval.rrf`. Keep the fusion rule fixed. Do not add learned fusion
   weights.
3. Report NDCG and recall on the same evaluation set as the existing lanes.

## Adding an action field type

1. Add the type name to `ARG_TYPES` in `src/vey/act/tool_card.py`.
2. Implement its exact coercion in `coerce()` in `src/vey/act/compile.py`. An
   unknown type must be a construction error, never a best-effort guess.
3. Add the matching exact extractor in `src/vey/act/extract.py`.
4. Add tests: a valid value coerces, an invalid value does not, and an unresolved
   required field yields `ASK_FOR_INFO`.

Nested `object` types compile recursively. Do not add a path that generates
JSON or nested structure text; the schema supplies the shape and the extractor
only fills values.

## Adding trust signals

1. Add the observable feature to `Observation` in `src/vey/trust/observation.py`
   and include it in the `FEATURES` tuple. It must be computable at inference
   time from the decision alone.
2. Never add a gold-derived quantity. Outcomes (`outcome_correct`,
   `outcome_should_abstain`) exist only as labels attached after prediction.
3. Add a simple control first. A new signal earns its place only if it improves
   risk or coverage at matched operating point over the existing controls.
4. Keep `policy.py` deterministic. New risk estimates gate the uncertain part;
   they do not become decision labels.

## Writing tests

- Assert an observable contract, not an implementation detail. A test that
  breaks when you rename an internal helper is testing the wrong thing.
- Test boundaries and error paths: invalid values, missing required fields,
  ambiguous fields, empty candidate sets.
- A test earns its place when a plausible bug would make it fail.

## Benchmark discipline

- Every published number is reproducible from a committed script on a named
  host. Report CPU, thread count, batch size, and dtype with any timing.
- Measure end-to-end paths by default. If you report a cached or scorer-only
  path, label it as such and keep it out of the headline.
- Report memory as model weight payload, peak process RSS, and RSS delta. They
  answer different questions and must not be interchanged.
- When changing a claim, show the before and after on the same hardware.
- New mechanisms need a comparison against the simpler control that already
  exists.
