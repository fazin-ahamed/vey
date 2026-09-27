# CRUX: the Vey 2 decision API

```python
import vey

result = vey.decide(
    question="Choose the permitted option with the most headroom, then the lowest latency.",
    candidates={
        "a": "permitted; ample headroom; latency 40 ms",
        "b": "not permitted; ample headroom; latency 20 ms",
        "c": "permitted; little headroom; latency 15 ms",
    },
    explain=False,   # set True for the machine-evidence certificate
)

result.answer          # "a"
result.probabilities   # {"a": ..., "b": 0.0, "c": ...}
result.decision_mode   # "structured" | "field" | "crux" | "abstain"
result.trust           # {"state": "confident"|"tie"|"abstain", "eligible": n, "primary_margin": ...}
```

## The decision contract

- `question`: a natural-language priority instruction. The compiler reads an
  ordered policy: filters (`permitted`, `supports tools`, `meets the quality
  floor 0.7`, `that are <X>`) and a lexicographic list of preferences
  (`most/greatest/highest <axis>`, `least/lowest/minimize <axis>`, `then <axis>`).
- `candidates`: `{id: description}`. Descriptions are consequence text; the
  candidate id/name is never read by the grounder.
- `state`: optional shared context string.

## Program grammar (compiled, auditable)

```
FILTER <concept>          keep candidates satisfying a categorical predicate
FILTER <key> >= <number>  keep candidates whose numeric field meets a floor
MAX <axis>                prefer more of a graded axis   (lexicographic tier)
MIN <axis>                prefer less of a graded axis   (lexicographic tier)
```

Filters run first (set-valued, order-independent); the MAX/MIN stages are the
priority order as written.

## explain=True: machine evidence, not reasoning

```json
{
  "candidate": "haiku",
  "decision_program": ["FILTER tool_support", "FILTER quality>=0.7", "MIN latency", "MIN cost"],
  "evidence": {
    "haiku": {"FILTER tool_support": 1.0, "FILTER quality>=0.7": 0.04, "MIN latency": 200.0, "MIN cost": 0.1}
  },
  "survivors": ["haiku"]
}
```

No generated natural-language chain-of-thought. The output is only the typed
grounded values, the executed program, and the surviving candidate set.

## Semantic Resolution

Natural language yields intervals, not exact floats. Each grounded ordinal value
carries a resolution ε; two values within ε are treated as **semantically tied**
and the decision defers to the next priority. This is why "ample" vs "generous"
does not spuriously decide a choice, while "ample" vs "little" does. Numeric
structured fields use ε = 0 (exact comparison).

## Permutation & rename exactness (CRUX-P)

The decision is a pure function of the unordered set of candidate consequence
texts:

- candidate identity is stripped before grounding (names carry no signal);
- grounded values are quantized (absorb ~1e-6 batched-float noise);
- ties are broken on candidate content, never on position or id.

Validated at **1.000** order- and rename-invariance on 1,500 held-out generic
decisions. See [BENCHMARKS.md](BENCHMARKS.md).

## Trust / abstain

If no candidate satisfies the constraints, `answer` is `None`, `decision_mode`
is `"abstain"`, and `trust.state == "abstain"`. Callers can escalate. When the
top two survivors are within resolution on the primary priority, `trust.state`
is `"tie"`.

## Lanes and cost

`decide()` picks the cheapest lane that can answer the decision:

- **structured**, when every stage resolves against explicit numeric or enum
  candidate fields. No model loads.
- **field**, when every stage is a MAX on an axis the distilled scalar field was
  measured on (`food progress`, `open space`, `headroom`). One forward pass per
  candidate. The field weights are `field.safetensors` on `fazinahamed/vey`,
  pinned revision `3eb1460a82c98fc99c350032b9cf29a74075a4a6`. Override with
  `VEY_CRUX_FIELD` or `VEY_CRUX_FIELD_HF`.
- **crux**, for everything else that needs language grounding: a frozen
  ModernBERT-base-NLI predicate grounder plus the trained pairwise ordinal
  comparator. The comparator weights are `comparator.safetensors` on the same
  repo. Override with `VEY_CRUX_COMPARATOR` or `VEY_CRUX_COMPARATOR_HF`.

A learned lane fails closed only if its own artifact cannot be fetched. Reuse a
`vey.Runtime` to control the device and keep a loaded lane warm.
