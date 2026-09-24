# Decision API

Every public interface Vey exposes, and the contract it keeps.

## Action compiler (`vey.act`)

```python
from vey import ToolCard, ArgSpec, FieldDecision, compile_call, is_schema_valid
```

### `ToolCard`

A typed, canonical description of a callable tool.

```python
ToolCard(
    tool_id: str,
    name: str,
    description: str,
    args: tuple[ArgSpec, ...] = (),
    family: str = "",
)
```

`ArgSpec(name, type, required, description, enum, fields, item_type)` supports
`string`, `enum`, `boolean`, `integer`, `number`, `date`, `datetime`, `email`,
`phone`, `url`, `id`, `object` (nested, compiled recursively), and `array`.
An unknown type is a construction error, never a best-effort guess.

`card.card_text()` is the single canonical string the semantic lane embeds, so
no caller can leak a second view of the same tool.

### `FieldDecision`

Every slot resolves to exactly one of:

- `VALUE(value, span=...)`: a resolved, type-validated value with an evidence
  span in the state,
- `MISSING`: no supported evidence,
- `AMBIGUOUS(candidates)`: more than one incompatible candidate.

There is no fourth state, and no path from `MISSING` to a value.

### `compile_call(card, decisions, extra_args=()) -> CompiledCall`

Emits `ACT` only when **every** gate passes: the tool exists, every required
slot is `VALUE`, every type validates, every enum constraint holds, and no
undeclared argument is present. Otherwise it returns `ASK_FOR_INFO`
(a required slot is missing or ambiguous) or `REJECT` (an undeclared argument).

`CompiledCall` carries `outcome`, `tool_id`, `args`, `missing`, `ambiguous`,
`evidence_refs`, and `reasons`. It is a decision object; Vey does not execute it.

## Trust and abstention (`vey.trust`)

```python
from vey.trust import Observation, from_scores, PolicyConfig, decide
```

`from_scores(semantic_scores, bm25_scores, ...)` builds a per-decision
observation from two score vectors. The calibrated risk estimate only ever
gates the uncertain part; exact schema state is decided first.

`decide(observation, risk, PolicyConfig())` returns one of:

- `ACT`: low estimated risk, schema complete,
- `ASK_FOR_INFO`: a required slot is missing or ambiguous (deterministic),
- `RERUN_CHEAP_LANE`: the two lanes disagree with a small margin; a *request* to
  re-score with the other lane, never an `ACT`,
- `ESCALATE`: high risk and escalation is available,
- `ABSTAIN`: otherwise.

## Semantic matching (`vey.semantic`)

```python
from vey.semantic import CandidateMatcher, Embedder, load_encoder

tok, enc = load_encoder(device="cpu")
matcher = CandidateMatcher(Embedder(tok, enc, "cpu"))
matcher.prepare(candidates)              # encode once at install time
choice, score = matcher.select(state, candidates)
margin = CandidateMatcher.margin(matcher.score(state, candidates))
```

A candidate's score is independent of the candidate set: permuting, inserting,
or removing candidates cannot change an existing candidate's score. This is
structural, not learned.

## Retrieval (`vey.retrieval`)

```python
from vey.retrieval import BM25, rrf

bm25 = BM25(documents)
lexical = bm25.score(query)
fused = rrf(semantic_scores, lexical)   # fixed k=60
```

## Evaluation (`vey.evaluation`)

Calibration and selective-risk primitives: `expected_calibration_error`,
`risk_coverage_curve`, `risk_at_coverage`, `coverage_at_risk`, `ndcg_at_k`,
`recall_at_k`, `mrr_at_k`.

## Guarantees

- The compiler never emits a value it has no evidence span for.
- The policy never returns `ACT` directly from a lane disagreement.
- Candidate/tool scores do not depend on the order or size of the candidate set.
