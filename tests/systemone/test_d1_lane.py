"""Tier 2 routing, with a fake d1 that records what it was sent. No weights, no network."""
import base64
import json

import pytest

from vey.systemone import Calibration, Engine, EngineConfig, LazyNLI
from vey.systemone.schema import ApiError, parse_request

from .fakes import FakeEmbedder, FakeNLI, UniformEmbedder
from .test_server import Running

PNG = ("iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAHElEQVR4nGP4TyJgoKcGBgbsaFQDfTUQB4aDBgDAen2fpitsVAAAAABJRU5ErkJggg==")


class FakeD1:
    """noul -> [0.9, 0.1]; choice -> 0.7 on the last option; score -> peak on the middle level."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def probabilities(self, state, questions, images=None):
        self.calls.append({"state": state, "questions": questions, "images": images})
        if self.fail:
            raise MemoryError("simulated out of memory")
        out = []
        for q in questions:
            if q["type"] == "noul":
                out.append([0.9, 0.1])
            elif q["type"] == "choice":
                k = len(q["criteria"])
                out.append([0.3 / max(k - 1, 1)] * (k - 1) + [0.7] if k > 1 else [1.0])
            else:
                k = len(q["criteria"])
                out.append([0.1 if i != k // 2 else 0.8 for i in range(k)] if k > 2 else [0.5, 0.5])
        return out

    def info(self):
        return {"repo": "fake", "calls": len(self.calls)}


def engine(d1, embedder=None, **cfg):
    return Engine(embedder or FakeEmbedder(), LazyNLI(lambda: FakeNLI()), Calibration(),
                  config=EngineConfig(**cfg), d1=d1)


TEAM = {"type": "choice", "instructions": "Which team?", "criteria": {
    "billing": "invoices payments refunds", "technical": "bugs outages errors", "shipping": "parcels delivery"}}


def ask(eng, state, questions, assets=None):
    body = {"model": "m", "state": state, "questions": questions}
    if assets:
        body["assets"] = assets
    return eng.answer(parse_request(body), raw_body=body, explain=True)


def test_yes_no_and_score_go_to_d1_in_one_batched_call():
    d1 = FakeD1()
    r = ask(engine(d1, d1_noul=True), "text", {"y": {"type": "noul", "instructions": "Is it?"},
                                  "s": {"type": "score", "criteria": ["a", "b", "c"]}})
    assert len(d1.calls) == 1 and len(d1.calls[0]["questions"]) == 2
    assert r["answers"]["y"]["noul"] == 0.9  # a noul's first probability is yes
    assert r["answers"]["s"]["probabilities"] == {"0": 0.1, "1": 0.8, "2": 0.1}
    assert {t["lane"] for t in r["trace"]["questions"].values()} == {"d1"}


def test_a_confident_matcher_never_calls_d1_and_an_unsure_one_does():
    d1 = FakeD1()
    sure = ask(engine(d1, rerank_below=0.0, d1_choice_below=0.0), "refund the payment", {"t": TEAM})
    assert not d1.calls and sure["trace"]["questions"]["t"]["lane"] == "matcher"
    unsure = ask(engine(d1, UniformEmbedder()), "anything", {"t": TEAM})  # uniform cosines: confidence 1/3 < 0.8
    t = unsure["trace"]["questions"]["t"]
    assert len(d1.calls) == 1 and t["lane"] == "d1" and "matcher_confidence" in t
    assert unsure["answers"]["t"]["choice"] == "shipping"  # the fake's 0.7 option


def test_large_choices_stay_on_the_matcher():
    big = {"type": "choice", "criteria": {f"o{i}": f"thing {i}" for i in range(60)}}
    d1 = FakeD1()
    r = ask(engine(d1, UniformEmbedder(), d1_max_options=50), "x", {"q": big})
    assert not d1.calls and r["trace"]["questions"]["q"]["lane"] == "matcher"


def test_threshold_one_sends_even_a_certain_matcher_to_d1():
    eng = engine(FakeD1(), d1_choice_below=1.0)
    eng.calibration = Calibration({"choice:3-5": 1e-12})
    r = ask(eng, "invoices payments refunds", {"t": TEAM})
    assert r["trace"]["questions"]["t"]["matcher_confidence"] == 1.0
    assert r["answers"]["t"]["choice"] == "shipping" and r["trace"]["questions"]["t"]["lane"] == "d1"


def test_single_option_choices_and_scores_have_forced_answers_even_with_images(pil):
    d1 = FakeD1(fail=True)
    one = {"photo": {"type": "image", "format": "png", "data_base64": PNG}}
    r = ask(engine(d1, exact_lane=False, d1_choice_below=1.0), "photo", {
        "c": {"type": "choice", "criteria": {"only": "the only answer"}},
        "s": {"type": "score", "criteria": ["the only level"]}}, one)
    assert r["answers"]["c"]["probabilities"] == {"only": 1.0}
    assert r["answers"]["s"]["score"] == 0.0 and r["answers"]["s"]["probabilities"] == {"0": 1.0}
    assert not d1.calls


def test_the_exact_lane_comes_first_and_routing_can_be_switched_off():
    d1 = FakeD1()
    r = ask(engine(d1), {"flag": True}, {"flag": {"type": "noul"}})
    assert not d1.calls and r["trace"]["questions"]["flag"]["lane"] == "exact"
    r = ask(engine(d1), "I need a refund", {"q": {"type": "noul", "instructions": "Is it a refund?"}})
    assert not d1.calls and r["trace"]["questions"]["q"]["lane"] == "nli"  # yes/no stays on the NLI model by default


def test_d1_failure_on_text_falls_back_to_the_lighter_lanes():
    d1 = FakeD1(fail=True)
    r = ask(engine(d1, d1_noul=True), "I need a refund for the payment",
            {"y": {"type": "noul", "instructions": "Is the customer asking for a refund?"}})
    t = r["trace"]["questions"]["y"]
    assert t["lane"] == "nli" and "MemoryError" in t["d1_error"] and r["answers"]["y"]["noul"] > 0.5
    r = ask(engine(d1, UniformEmbedder()), "anything", {"t": TEAM})
    t = r["trace"]["questions"]["t"]
    assert t["lane"] == "matcher" and "d1_error" in t and r["answers"]["t"]["type"] == "choice"


@pytest.fixture
def pil():
    return pytest.importorskip("PIL")


def test_images_are_decoded_placeholders_resolved_and_encoded_once_for_all_questions(pil):
    d1 = FakeD1()
    assets = {"img-1": {"type": "image", "format": "png", "data_base64": PNG}}
    qs = {"legible": {"type": "noul", "instructions": "Is the total on img-1 legible?"},
          "kind": {"type": "choice", "criteria": {"receipt": "a receipt", "photo": "a photo"}}}
    r = ask(engine(d1), {"note": "Review img-1 please"}, qs, assets)
    assert len(d1.calls) == 1  # one call: d1 encodes the media once for every question
    call = d1.calls[0]
    assert call["state"] == {"note": "Review the attached image please"}
    assert call["questions"][0]["instructions"] == "Is the total on the attached image legible?"
    assert len(call["images"]) == 1 and call["images"][0].size == (16, 16)
    assert {t["lane"] for t in r["trace"]["questions"].values()} == {"d1"}
    assert r["trace"]["questions"]["kind"]["images"] == 1


def test_a_state_that_is_only_a_placeholder_becomes_none_and_unused_assets_are_ignored(pil):
    d1 = FakeD1()
    assets = {"photo": {"type": "image", "format": "png", "data_base64": PNG},
              "spare": {"type": "image", "format": "png", "data_base64": PNG}}
    ask(engine(d1), "photo", {"q": {"type": "noul", "instructions": "Is it red?"}}, assets)
    assert d1.calls[0]["state"] is None and len(d1.calls[0]["images"]) == 1


def test_placeholders_are_matched_once_longest_first_and_not_inside_words(pil):
    image = {"type": "image", "format": "png", "data_base64": PNG}
    d1 = FakeD1()
    ask(engine(d1), "img-1 then img; shipping is fine", {
        "q": {"type": "noul", "instructions": "Is the shipment red?"}}, {"img": image, "img-1": image, "i": None})
    call = d1.calls[0]
    assert call["state"] == "the attached image then the attached image; shipping is fine"
    assert call["questions"][0]["instructions"] == "Is the shipment red?" and len(call["images"]) == 2


@pytest.mark.parametrize("asset,fragment", [
    (None, "image object"),
    ({"type": "image", "data_base64": 123}, "must be a string"),
    ({"type": "image", "format": "png", "data_base64": "!!notbase64!!"}, "base64"),
    ({"type": "image", "format": "png", "data_base64": base64.b64encode(b"not an image").decode()}, "decode"),
    ({"type": "image", "part": "p1"}, "multipart"),
    ({"type": "audio"}, "type 'image'"),
])
def test_bad_assets_are_a_400(pil, asset, fragment):
    with pytest.raises(ApiError) as e:
        ask(engine(FakeD1()), "a", {"q": {"type": "noul", "instructions": "Is a ok?"}}, {"a": asset})
    assert e.value.status == 400 and fragment in e.value.body["detail"]


def test_too_many_images_and_image_failures(pil):
    assets = {f"i{k}": {"type": "image", "format": "png", "data_base64": PNG} for k in range(5)}
    with pytest.raises(ApiError) as e:
        ask(engine(FakeD1()), " ".join(assets), {"q": {"type": "noul", "instructions": "Is it?"}}, assets)
    assert e.value.status == 400 and "at most 4" in e.value.body["detail"]
    one = {"i": {"type": "image", "format": "png", "data_base64": PNG}}
    with pytest.raises(ApiError) as e:
        ask(engine(FakeD1(fail=True)), "i", {"q": {"type": "noul", "instructions": "Is it?"}}, one)
    assert e.value.status == 503  # no lighter lane can read an image


def test_box_questions_and_images_without_d1_still_need_escalation(pil):
    one = {"i": {"type": "image", "format": "png", "data_base64": PNG}}
    for eng, qs in ((engine(FakeD1()), {"b": {"type": "box", "instructions": "find it"}}),
                    (engine(None), {"q": {"type": "noul", "instructions": "Is it?"}})):
        with pytest.raises(ApiError) as e:
            ask(eng, "i", qs, one)
        assert e.value.status == 400 and "escalat" in e.value.body["detail"]


def test_image_requests_work_over_http_and_info_reports_the_lane(pil):
    s = Running(engine(FakeD1()))
    try:
        body = {"model": "m", "state": "photo", "assets": {"photo": {"type": "image", "format": "png", "data_base64": PNG}},
                "questions": {"red": {"type": "noul", "instructions": "Is the shape red?"}}}
        st, raw, _ = s.call("/v1/systemone", {**body, "explain": True})
        out = json.loads(raw)
        assert st == 200 and out["answers"]["red"]["noul"] == 0.9 and out["trace"]["questions"]["red"]["lane"] == "d1"
        assert json.loads(s.call("/v1/info")[1])["lanes"]["d1"]["repo"] == "fake"
    finally:
        s.close()


@pytest.mark.parametrize("options", [50, 51])
def test_image_choice_limit_is_explicit_not_a_text_fallback(pil, options):
    qs = {"q": {"type": "choice", "criteria": {f"o{i}": f"thing {i}" for i in range(options)}}}
    assets = {"photo": {"type": "image", "format": "png", "data_base64": PNG}}
    eng = engine(FakeD1())
    if options == 50:
        r = ask(eng, "photo", qs, assets)
        assert r["trace"]["questions"]["q"]["lane"] == "d1" and r["answers"]["q"]["choice"] == "o49"
    else:
        with pytest.raises(ApiError) as e:
            ask(eng, "photo", qs, assets)
        assert e.value.status == 400 and "more than 50 options" in e.value.body["detail"]


@pytest.mark.parametrize("args,fragment", [
    (["--d1-batch-tokens", "0"], "must be positive"),
    (["--d1-max-image-side", "-1"], "must be positive"),
    (["--d1-choice-below", "1.1"], "between 0 and 1"),
    (["--d1-choice-below", "nan"], "between 0 and 1"),
    (["--d1-noul"], "requires --d1"),
])
def test_invalid_d1_cli_settings_fail_before_loading_models(args, fragment, capsys):
    from vey.systemone.cli import main

    with pytest.raises(SystemExit) as e:
        main(args)
    assert e.value.code == 2 and fragment in capsys.readouterr().err


def test_unprepared_d1_has_an_actionable_error_without_loading_torch(tmp_path):
    from vey.systemone.d1 import D1Model, D1Unavailable

    with pytest.raises(D1Unavailable, match="vey-d1-prepare"):
        D1Model(tmp_path)
