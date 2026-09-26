"""CRUX composition + decision.

Executes a compiled program (FILTER / MAX / MIN stages) over grounded evidence
with the permutation-exact rule from CRUX-P:

  * candidate IDENTITY is stripped before grounding (grounder never sees names),
  * grounded values are quantized (absorb ~1e-6 batched-float noise),
  * FILTER stages keep candidates whose predicate margin >= 0,
  * ordinal stages apply epsilon-lexicographic tie bands (Semantic Resolution):
    within resolution of the best, defer to the next priority,
  * the final survivor set is resolved on candidate CONTENT, never on order/name.

The decision is therefore a pure function of the UNORDERED set of candidate
consequence texts: exact order- and rename-invariance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .compiler import Stage
from .resolution import DEFAULT_RESOLUTION, quantize


def strip_identity(text: str) -> str:
    """Remove leading candidate identity ('Option Bravo:', 'route_a:') so
    grounding depends only on consequences."""
    t = re.sub(r"^\s*(option\s+)?[\w-]+\s*[:\-]\s*", "", text, flags=re.I)
    return t.strip()


@dataclass
class Grounded:
    """Per-stage grounded evidence across candidates (aligned to candidate order)."""
    stage: Stage
    values: list[float]
    resolution: float


def ground_program(grounder, texts: list[str], program: list[Stage]) -> list[Grounded]:
    stripped = [strip_identity(t) for t in texts]
    ord_res = getattr(grounder, "ordinal_resolution", DEFAULT_RESOLUTION["ordinal"])
    out: list[Grounded] = []
    for st in program:
        if st.kind == "FILTER":
            vals = [quantize(v) for v in grounder.predicate(stripped, st.key, st.threshold)]
            out.append(Grounded(st, vals, DEFAULT_RESOLUTION["predicate"]))
        else:  # MAX / MIN
            vals = [quantize(v) for v in grounder.ordinal(stripped, st.key)]
            out.append(Grounded(st, vals, ord_res))
    return out


def compose(ids: list[str], texts: list[str], grounded: list[Grounded]):
    """Return (winner_index, survivors_per_stage). Deterministic and
    permutation/rename exact (content tie-break)."""
    K = len(ids)
    cands = list(range(K))
    survivors_trace = []

    # FILTER stages first (order among filters is irrelevant to the set result)
    for g in grounded:
        if g.stage.kind != "FILTER":
            continue
        cands = [i for i in cands if g.values[i] >= 0.0]
        survivors_trace.append((g.stage.label(), [ids[i] for i in cands]))
    if not cands:
        return None, survivors_trace  # abstain: no eligible candidate

    # ordinal lexicographic tiers with epsilon resolution
    ordinal_stages = [g for g in grounded if g.stage.kind in ("MAX", "MIN")]
    for g in ordinal_stages:
        vals = g.values
        if g.stage.kind == "MAX":
            best = max(vals[i] for i in cands)
            cands = [i for i in cands if best - vals[i] <= g.resolution]
        else:
            best = min(vals[i] for i in cands)
            cands = [i for i in cands if vals[i] - best <= g.resolution]
        survivors_trace.append((g.stage.label(), [ids[i] for i in cands]))

    # deterministic final selection: within the tied survivor set, pick the
    # extreme on the FIRST ordinal priority; exact ties broken on content.
    if ordinal_stages:
        first = ordinal_stages[0]
        target = max if first.stage.kind == "MAX" else min
        bestv = target(first.values[i] for i in cands)
        top = [i for i in cands if first.values[i] == bestv]
    else:
        top = cands
    winner = min(top, key=lambda i: strip_identity(texts[i]))
    return winner, survivors_trace
