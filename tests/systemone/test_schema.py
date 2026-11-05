import pytest

from vey.systemone.schema import ApiError, parse_request, state_to_text, to_text


def ok():
    return {"model": "jev-latest", "state": "hello",
            "questions": {"a": {"type": "noul", "instructions": "Is this a greeting?"},
                          "b": {"type": "choice", "criteria": {"x": "first", "y": None}},
                          "c": {"type": "score", "criteria": ["low", "high"]}}}


def errors(body):
    with pytest.raises(ApiError) as e:
        parse_request(body)
    assert e.value.status == 422
    return e.value.body["detail"]


def test_valid_request_parses_in_order():
    r = parse_request(ok())
    assert [q.name for q in r.questions] == ["a", "b", "c"]
    assert [k for k, _ in r.questions[1].options] == ["x", "y"]
    assert [k for k, _ in r.questions[2].options] == ["0", "1"]


def test_missing_fields_use_fastapi_error_shape():
    d = errors({})
    assert {tuple(e["loc"]) for e in d} == {("body", "model"), ("body", "state"), ("body", "questions")}
    assert all(e["type"] == "missing" and e["msg"] == "Field required" for e in d)


def test_every_error_is_collected_not_just_the_first():
    body = ok()
    body["model"] = 3
    body["questions"]["b"] = {"type": "choice"}
    body["questions"]["c"] = {"type": "score", "criteria": []}
    locs = {tuple(e["loc"]) for e in errors(body)}
    assert ("body", "model") in locs
    assert ("body", "questions", "b", "criteria") in locs
    assert ("body", "questions", "c", "criteria") in locs


def test_limits_match_the_published_schema():
    body = ok()
    body["questions"]["b"]["criteria"] = {str(i): "x" for i in range(256)}
    assert errors(body)[0]["type"] == "too_long"
    body = ok()
    body["questions"]["c"]["criteria"] = [str(i) for i in range(11)]
    assert errors(body)[0]["type"] == "too_long"
    body = ok()
    body["questions"] = {}
    assert errors(body)[0]["type"] == "too_short"


def test_unknown_question_type_and_noul_criteria():
    body = ok()
    body["questions"]["a"] = {"type": "ranking"}
    assert errors(body)[0]["type"] == "union_tag_invalid"
    body = ok()
    body["questions"]["a"] = {"type": "noul", "criteria": {"maybe": "x"}}
    assert errors(body)[0]["type"] == "noul_criteria"


def test_state_may_be_string_object_or_array_but_not_null():
    for state in ("text", {"k": 1}, [1, "two"]):
        body = ok(); body["state"] = state
        parse_request(body)
    body = ok(); body["state"] = None
    assert errors(body)[0]["loc"] == ["body", "state"]


def test_non_object_body():
    assert errors([1, 2])[0]["loc"] == ["body"]


def test_text_flattening():
    assert to_text({"task": "Identify spam", "tone": ["calm", "polite"]}) == "task: Identify spam; tone: calm; polite"
    assert to_text(None) == "" and to_text(True) == "true"
    assert state_to_text({"a": "x", "b": {"c": 1}}) == "a: x\nb: c: 1"
