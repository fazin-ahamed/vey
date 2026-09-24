"""Trust and abstention policy tests.

The policy is deterministic on purpose: exact schema state decides first, and a
calibrated risk estimate only gates the uncertain remainder. These tests defend
that ordering, because inverting it is how a high-risk decision would leak
through as an ACT.
"""
import numpy as np
import pytest

from vey.trust.observation import from_scores
from vey.trust.policy import (
    ABSTAIN,
    ACT,
    ASK_FOR_INFO,
    ESCALATE,
    REJECT,
    RERUN_CHEAP_LANE,
    PolicyConfig,
    decide,
)


def _obs(**kw):
    kw.setdefault("semantic_top1", 0.8)
    kw.setdefault("semantic_top2", 0.6)
    kw.setdefault("semantic_margin", 0.2)
    kw.setdefault("candidate_count", 3)
    kw.setdefault("n_required_missing", 0)
    kw.setdefault("n_ambiguous", 0)
    kw.setdefault("schema_complete", 1.0)
    return from_scores(
        np.array([0.8, 0.6, 0.4]),
        np.array([9.0, 1.0, 0.5]) if kw.pop("agree", True) else np.array([1.0, 9.0, 0.5]),
    )


def test_missing_required_slot_is_always_ask_for_info():
    o = from_scores(np.array([0.9, 0.5]), np.array([3.0, 1.0]), n_required_missing=2)
    r = decide(o, 0.0, PolicyConfig())
    assert r["decision"] == ASK_FOR_INFO
    assert "missing" in r["reason"]


def test_ambiguous_slot_is_ask_for_info_not_a_guess():
    o = from_scores(np.array([0.9, 0.5]), np.array([3.0, 1.0]), n_ambiguous=1)
    assert decide(o, 0.0, PolicyConfig())["decision"] == ASK_FOR_INFO


def test_low_risk_with_complete_schema_is_act():
    assert decide(_obs(), 0.01, PolicyConfig())["decision"] == ACT


def test_disagreement_never_returns_act_at_high_risk():
    o = from_scores(np.array([0.52, 0.50, 0.48]), np.array([1.0, 9.0, 0.5]))
    r = decide(o, 0.95, PolicyConfig())
    assert r["decision"] == RERUN_CHEAP_LANE
    assert r["decision"] != ACT


def test_high_risk_agrees_escalates_when_available():
    assert decide(_obs(agree=True), 0.9, PolicyConfig())["decision"] == ESCALATE


def test_high_risk_without_escalation_abstains():
    cfg = PolicyConfig(escalation_available=False)
    assert decide(_obs(agree=True), 0.9, cfg)["decision"] == ABSTAIN


def test_mid_risk_abstains_instead_of_acting():
    assert decide(_obs(agree=True), 0.30, PolicyConfig())["decision"] == ABSTAIN


def test_every_outcome_is_one_of_the_declared_set():
    o = _obs()
    allowed = {ACT, ASK_FOR_INFO, REJECT, ESCALATE, ABSTAIN, RERUN_CHEAP_LANE}
    for risk in np.linspace(0, 1, 21):
        assert decide(o, float(risk), PolicyConfig())["decision"] in allowed
