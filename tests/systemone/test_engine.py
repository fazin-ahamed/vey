import random

import numpy as np
import pytest

from vey.systemone import Calibration, Engine, EngineConfig, LazyNLI, select_state
from vey.systemone.schema import ApiError, parse_request

from .fakes import FakeEmbedder, FakeNLI, UniformEmbedder


def engine(embedder=None, nli=None, **cfg):
    return Engine(embedder or FakeEmbedder(), LazyNLI(lambda: nli or FakeNLI()),
                  Calibration({"choice:2": 0.1, "choice:3-5": 0.1, "choice:6-10": 0.1, "choice:11+": 0.1,
                               "score": 0.1, "noul": 1.0, "choice_nli": 1.0}),
                  config=EngineConfig(**cfg))


def ask(eng, state, questions, **kw):
    body = {"model": "m", "state": state, "questions": questions}
    return eng.answer(parse_request(body), raw_body=body, explain=True, **kw)


TEAM = {"type": "choice", "instructions": "Which team?", "criteria": {
    "billing": "invoices payments refunds", "technical": "bugs outages errors", "shipping": "parcels delivery"}}


def test_choice_picks_the_matching_option_and_probabilities_sum_to_one():
    r = ask(engine(rerank_below=0.0), "I was charged twice, please refund the payment", {"team": TEAM})
    a = r["answers"]["team"]
    assert a["choice"] == "billing" and a["type"] == "choice"
    assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)
    assert a["confidence"] == max(a["probabilities"].values())
    assert r["trace"]["questions"]["team"]["lane"] == "matcher"


def test_option_order_does_not_change_any_probability():
    state = "the parcel never arrived, tracking says delivery failed"
    base = ask(engine(), state, {"team": TEAM})["answers"]["team"]["probabilities"]
    for seed in range(5):
        items = list(TEAM["criteria"].items())
        random.Random(seed).shuffle(items)
        shuffled = {"team": {**TEAM, "criteria": dict(items)}}
        got = ask(engine(), state, shuffled)["answers"]["team"]["probabilities"]
        assert got == pytest.approx(base, abs=1e-6)


def test_exact_lane_answers_from_a_state_field_without_running_a_model():
    emb, nli = FakeEmbedder(), FakeNLI()
    eng = engine(emb, nli)
    state = {"invoice": {"status": "Overdue", "is_large": True, "risk": 2}, "note": "pay soon"}
    qs = {"status": {"type": "choice", "criteria": {"paid": "", "overdue": "", "draft": ""}},
          "is_large": {"type": "noul"},
          "risk": {"type": "score", "criteria": ["low", "mid", "high"]}}
    r = ask(eng, state, qs)
    assert r["answers"]["status"]["choice"] == "overdue"
    assert r["answers"]["status"]["probabilities"] == {"paid": 0.0, "overdue": 1.0, "draft": 0.0}
    assert r["answers"]["is_large"]["noul"] == 1.0
    assert r["answers"]["risk"]["score"] == 2.0
    assert {t["lane"] for t in r["trace"]["questions"].values()} == {"exact"}
    assert emb.calls == 0 and nli.calls == 0


def test_exact_lane_ignores_values_that_are_not_valid_answers():
    r = ask(engine(rerank_below=0.0), {"status": "pending review"},
            {"status": {"type": "choice", "criteria": {"paid": "", "open": ""}}})
    assert r["trace"]["questions"]["status"]["lane"] == "matcher"


def test_exact_lane_can_be_disabled():
    r = ask(engine(exact_lane=False), {"is_large": True}, {"is_large": {"type": "noul", "instructions": "Is it large?"}})
    assert r["trace"]["questions"]["is_large"]["lane"] == "nli"


def test_noul_uses_the_nli_lane_on_a_statement_made_from_the_question():
    nli = FakeNLI()
    yes = ask(engine(nli=nli), "I need a refund for the duplicate charge",
              {"q": {"type": "noul", "instructions": "Is the customer asking for a refund?"}})
    no = ask(engine(nli=nli), "What time does the museum open?",
             {"q": {"type": "noul", "instructions": "Is the customer asking for a refund?"}})
    assert yes["answers"]["q"]["noul"] > 0.9 and no["answers"]["q"]["noul"] < 0.1
    assert yes["trace"]["questions"]["q"]["hypothesis"] == "The customer is asking for a refund."


def test_noul_with_both_criteria_compares_them():
    q = {"type": "noul", "criteria": {"true": "angry", "false": "calm"}}
    assert ask(engine(), "I am angry", {"q": q})["answers"]["q"]["noul"] > 0.9
    assert ask(engine(), "I am calm", {"q": q})["answers"]["q"]["noul"] < 0.1


def test_score_is_the_probability_weighted_level():
    qs = {"u": {"type": "score", "instructions": "urgency", "criteria": ["can wait", "this week", "today urgent"]}}
    a = ask(engine(), "urgent please fix today", qs)["answers"]["u"]
    probs = a["probabilities"]
    assert list(probs) == ["0", "1", "2"] and list(a["legend"]) == ["0", "1", "2"]
    assert a["score"] == pytest.approx(sum(int(k) * v for k, v in probs.items()), abs=1e-4)


def test_long_state_keeps_the_sentences_that_matter():
    filler = " ".join(f"Sentence {i} talks about the weather and gardens." for i in range(40))
    state = filler + " The invoice from Acme is overdue by two weeks. " + filler
    kept = select_state(state, "invoice overdue", budget=25, count=FakeEmbedder().count_tokens)
    assert "invoice from Acme is overdue" in kept
    assert FakeEmbedder().count_tokens(kept) <= 25
    assert select_state("short text", "q", budget=25, count=FakeEmbedder().count_tokens) == "short text"


def test_unsure_matcher_is_reranked_by_the_nli_model():
    qs = {"team": TEAM}
    r = ask(engine(UniformEmbedder(), FakeNLI(), rerank_below=0.9), "the parcel delivery is late", qs)
    t = r["trace"]["questions"]["team"]
    assert t["lane"] == "matcher+nli" and r["answers"]["team"]["choice"] == "shipping"
    assert sum(r["answers"]["team"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_no_rerank_beyond_the_option_cap_or_when_disabled():
    big = {"type": "choice", "criteria": {f"opt{i}": f"thing {i}" for i in range(25)}}
    assert ask(engine(UniformEmbedder(), rerank_below=0.99), "x", {"q": big})["trace"]["questions"]["q"]["lane"] == "matcher"
    assert ask(engine(UniformEmbedder(), rerank_below=0.0), "x", {"q": TEAM})["trace"]["questions"]["q"]["lane"] == "matcher"


def test_confidence_floor_flags_and_exact_answers_are_exempt():
    r = ask(engine(UniformEmbedder(), floor=0.99, rerank_below=0.0), {"flag": True},
            {"team": TEAM, "flag": {"type": "noul"}})
    assert r["trace"]["questions"]["team"]["below_floor"] is True
    assert "below_floor" not in r["trace"]["questions"]["flag"]


def test_images_and_box_questions_need_escalation():
    body = {"model": "m", "state": "photo", "assets": {"photo": {"type": "image"}},
            "questions": {"b": {"type": "box", "instructions": "find it"}}}
    with pytest.raises(ApiError) as e:
        engine().answer(parse_request(body), raw_body=body)
    assert e.value.status == 400 and "escalat" in e.value.body["detail"]


def test_noul_without_an_nli_model_is_a_clear_error():
    eng = Engine(FakeEmbedder(), LazyNLI(lambda: (_ for _ in ()).throw(RuntimeError("offline"))))
    with pytest.raises(ApiError) as e:
        ask(eng, "text", {"q": {"type": "noul", "instructions": "Is it?"}})
    assert e.value.status == 400 and "NLI" in e.value.body["detail"]


def test_usage_counts_input_tokens_and_no_output():
    r = ask(engine(), "one two three", {"team": TEAM})
    assert r["usage"]["output_tokens"] == 0 and r["usage"]["input_tokens"] > 3
