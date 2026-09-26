"""CRUX decision orchestration: compile -> ground -> compose -> result.

Lane-agnostic: the same pipeline runs with any Grounder (frozen NLI stack for
the CRUX lane, deterministic DictGrounder for the structured/exact lane).
"""
from __future__ import annotations

import math

from ..decision import Certificate, DecisionResult
from .compiler import compile_instruction
from .compose import compose, ground_program, strip_identity


def _softmax(xs, temp=0.5):
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp((x - m) / max(1e-6, temp)) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def run_decision(grounder, request, mode_label: str) -> DecisionResult:
    ids = list(request.candidates.keys())
    texts = [request.candidates[i] for i in ids]
    program = compile_instruction(request.question)
    grounded = ground_program(grounder, texts, program)
    winner_ix, survivors_trace = compose(ids, texts, grounded)

    # trust / abstain
    eligible = survivors_trace[-1][1] if survivors_trace else ids
    if winner_ix is None:
        trust = {"state": "abstain", "reason": "no candidate satisfies the constraints",
                 "eligible": 0}
        cert = _certificate(ids, texts, program, grounded, survivors_trace) if request.explain else None
        return DecisionResult(None, {i: 0.0 for i in ids}, "abstain", trust, cert)

    # probabilities over the FIRST ordinal priority, among constraint-eligible
    ordinal = [g for g in grounded if g.stage.kind in ("MAX", "MIN")]
    filt_ok = set(_eligible_ids(ids, grounded))
    if ordinal:
        g0 = ordinal[0]
        sign = 1.0 if g0.stage.kind == "MAX" else -1.0
        raw = {i: sign * g0.values[k] for k, i in enumerate(ids)}
        elig_vals = [raw[i] for i in ids if i in filt_ok]
        probs_list = _softmax(elig_vals)
        probs = {}
        it = iter(probs_list)
        for i in ids:
            probs[i] = next(it) if i in filt_ok else 0.0
    else:
        probs = {i: (1.0 if i == ids[winner_ix] else 0.0) for i in ids}

    winner_id = ids[winner_ix]
    # margin on the primary priority (winner vs best runner-up among survivors)
    margin = _primary_margin(ids, winner_ix, grounded)
    state = "confident" if margin is None or margin > 0 else "tie"
    trust = {"state": state, "eligible": len(filt_ok), "primary_margin": margin}
    cert = _certificate(ids, texts, program, grounded, survivors_trace) if request.explain else None
    return DecisionResult(winner_id, probs, mode_label, trust, cert)


def _eligible_ids(ids, grounded):
    cands = list(range(len(ids)))
    for g in grounded:
        if g.stage.kind == "FILTER":
            cands = [i for i in cands if g.values[i] >= 0.0]
    return [ids[i] for i in cands]

def _primary_margin(ids, winner_ix, grounded):
    ordinal = [g for g in grounded if g.stage.kind in ("MAX", "MIN")]
    if not ordinal:
        return None
    g0 = ordinal[0]
    vals = g0.values
    elig = {ids.index(i) for i in _eligible_ids(ids, grounded)}
    others = [vals[i] for i in range(len(ids)) if i != winner_ix and i in elig]
    if not others:
        return None
    if g0.stage.kind == "MAX":
        return round(vals[winner_ix] - max(others), 4)
    return round(min(others) - vals[winner_ix], 4)

def _certificate(ids, texts, program, grounded, survivors_trace) -> Certificate:
    evidence = {i: {} for i in ids}
    for g in grounded:
        for k, i in enumerate(ids):
            evidence[i][g.stage.label()] = round(g.values[k], 4)
    survivors = survivors_trace[-1][1] if survivors_trace else list(ids)
    return Certificate(decision_program=[s.label() for s in program],
                       evidence=evidence, survivors=survivors)
