"""Vey 2 capability-matrix tests (deterministic, no model download).

The structured lane and the compose/router logic run without any model. The
CRUX-lane invariants (permutation, rename, Semantic Resolution, abstention) are
exercised through a deterministic MockGrounder injected into the Runtime, so the
whole capability matrix is fast and hermetic. A separate, skipped-by-default
integration test can exercise the real 150M stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import vey
from vey.runtime import Runtime
from vey.crux.compiler import compile_instruction
from vey.crux.compose import strip_identity


# ---- a deterministic qualitative grounder for CRUX-lane logic --------------
_SCALE = {"none": 0, "little": 1, "some": 2, "moderate": 3, "fair": 4,
          "ample": 5, "high": 6, "vast": 7}


class MockGrounder:
    """Grounds qualitative axes from '<axis> <word>' facts in the (identity-
    stripped) candidate text. Values depend ONLY on content, so any residual
    order/name dependence must come from the pipeline, not the grounder."""
    ordinal_resolution = 0.15

    def _fact(self, text, key):
        toks = text.lower().replace(",", " ").replace(";", " ").split()
        for i, t in enumerate(toks):
            if t.startswith(key[:4]) and i + 1 < len(toks) and toks[i + 1] in _SCALE:
                return _SCALE[toks[i + 1]]
        return None

    def predicate(self, texts, concept, threshold=None):
        out = []
        for t in texts:
            low = t.lower()
            out.append(-1.0 if ("not " + concept) in low or ("no " + concept) in low
                       else (1.0 if concept in low else -1.0))
        return out

    def ordinal(self, texts, axis):
        key = axis.split()[0]
        return [float(self._fact(t, key) or 0) for t in texts]

    def evidence(self, texts, axis):
        key = axis.split()[0]
        return [1.0 if self._fact(t, key) is not None else 0.0 for t in texts]


def mock_runtime():
    return Runtime(crux_grounder=MockGrounder())


# ---------------------------------------------------------------- compiler
def test_compiler_program():
    prog = compile_instruction(
        "Use a route that supports tools and meets the quality floor 0.7. "
        "Among those, minimize latency, then cost.")
    labels = [s.label() for s in prog]
    assert "FILTER tool_support" in labels
    assert "FILTER quality>=0.7" in labels
    assert labels[-2:] == ["MIN latency", "MIN cost"]


# ---------------------------------------------------------------- router dispatch
def test_router_structured_lane():
    r = vey.decide(question="minimize latency, then cost",
                   candidates={"a": "latency 30; cost 5", "b": "latency 30; cost 2"})
    assert r.decision_mode == "structured"
    assert r.answer == "b"


def test_router_crux_lane_qualitative():
    rt = mock_runtime()
    r = vey.decide(question="Choose the permitted option with the most headroom.",
                   candidates={"a": "permitted; headroom ample", "b": "permitted; headroom little"},
                   runtime=rt)
    assert r.decision_mode == "crux"
    assert r.answer == "a"


# ---------------------------------------------------------------- structured lexicographic
def test_structured_model_routing():
    r = vey.decide(
        question="Use a route that supports tools and meets the quality floor 0.7. "
                 "Among those, minimize latency, then cost.",
        candidates={
            "gpt": "supports tools; quality 0.92; latency 800; cost 0.9",
            "sonnet": "supports tools; quality 0.88; latency 500; cost 0.6",
            "haiku": "supports tools; quality 0.74; latency 200; cost 0.1",
            "local": "no tool support; quality 0.60; latency 50; cost 0.0",
        }, explain=True)
    assert r.answer == "haiku"
    assert r.trust["state"] == "confident"
    assert r.certificate.survivors == ["haiku"]


# ---------------------------------------------------------------- abstention
def test_abstention_no_eligible():
    r = vey.decide(question="Choose the permitted option with the most headroom.",
                   candidates={"a": "not permitted; headroom ample", "b": "not permitted; headroom vast"},
                   runtime=mock_runtime())
    assert r.answer is None
    assert r.decision_mode == "abstain"
    assert r.trust["state"] == "abstain"


# ---------------------------------------------------------------- permutation / rename exactness
import itertools


def _winner_content(res, cands):
    return None if res.answer is None else strip_identity(cands[res.answer])


def test_permutation_invariance_structured():
    base = {"x": "latency 30; cost 5", "y": "latency 30; cost 2", "z": "latency 90; cost 1"}
    q = "minimize latency, then cost"
    ref = vey.decide(question=q, candidates=base)
    for perm in itertools.permutations(base):
        c = {k: base[k] for k in perm}
        r = vey.decide(question=q, candidates=c)
        assert _winner_content(r, c) == _winner_content(ref, base)


def test_permutation_invariance_crux():
    base = {"a": "permitted; headroom high", "b": "permitted; headroom little", "c": "permitted; headroom ample"}
    q = "Choose the permitted option with the most headroom."
    rt = mock_runtime()
    ref = vey.decide(question=q, candidates=base, runtime=rt)
    for perm in itertools.permutations(base):
        c = {k: base[k] for k in perm}
        r = vey.decide(question=q, candidates=c, runtime=mock_runtime())
        assert _winner_content(r, c) == _winner_content(ref, base)


def test_rename_invariance_crux():
    q = "Choose the permitted option with the most headroom."
    a = {"a": "permitted; headroom high", "b": "permitted; headroom little"}
    b = {"opt_1": "permitted; headroom little", "opt_2": "permitted; headroom high"}  # renamed + reordered
    ra = vey.decide(question=q, candidates=a, runtime=mock_runtime())
    rb = vey.decide(question=q, candidates=b, runtime=mock_runtime())
    assert _winner_content(ra, a) == _winner_content(rb, b) == "permitted; headroom high"


# ---------------------------------------------------------------- exact-tie determinism
def test_exact_tie_deterministic():
    # two options identical on every tier -> deterministic, content-based winner
    a = {"p": "latency 40; cost 3", "q": "latency 40; cost 3", "r": "latency 90; cost 1"}
    q = "minimize latency, then cost"
    winners = {_winner_content(vey.decide(question=q, candidates={k: a[k] for k in perm}),
                               {k: a[k] for k in perm})
               for perm in itertools.permutations(a)}
    assert len(winners) == 1  # identical winner content regardless of order


# ---------------------------------------------------------------- Semantic Resolution
def test_semantic_resolution_defers():
    # headroom within resolution (high vs high) -> tie -> defer to next priority (permitted margin
    # equal too) -> content tie-break; the point: near-equal ordinal values do NOT flip the winner
    q = "Choose the permitted option with the most headroom, then the most fair."
    base = {"a": "permitted; headroom high; fair vast", "b": "permitted; headroom high; fair none"}
    rt = mock_runtime()
    r1 = vey.decide(question=q, candidates=base, runtime=rt)
    # headroom ties (both 'high') so 'fair' decides -> a (vast > none)
    assert r1.answer == "a"


# ---------------------------------------------------------------- certificate shape
def test_certificate_is_machine_evidence():
    r = vey.decide(question="minimize latency",
                   candidates={"a": "latency 10", "b": "latency 20"}, explain=True)
    cert = r.certificate.to_dict()
    assert set(cert) == {"schema_version", "decision_program", "evidence", "survivors"}
    assert cert["schema_version"] == "crux-certificate/1"
    assert cert["decision_program"] == ["MIN latency"]
    assert set(cert["evidence"]) == {"a", "b"}
    # evidence is numeric machine values, not free text
    for ev in cert["evidence"].values():
        assert all(isinstance(v, (int, float)) for v in ev.values())
