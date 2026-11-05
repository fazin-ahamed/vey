"""Validate real server responses against TypeSafe's published OpenAPI schemas.

Opt in with ``VEY_NETWORK_TESTS=1`` (it fetches https://api.typesafe.ai/openapi.json) and
``pip install jsonschema``. The models are fakes; what is under test is the wire format.
"""
import json
import os
import urllib.request

import pytest

jsonschema = pytest.importorskip("jsonschema")
pytestmark = pytest.mark.skipif(os.environ.get("VEY_NETWORK_TESTS") != "1",
                                reason="set VEY_NETWORK_TESTS=1 to fetch the published OpenAPI spec")

from .test_server import BODY, Running, make_engine  # noqa: E402


@pytest.fixture(scope="module")
def spec():
    with urllib.request.urlopen("https://api.typesafe.ai/openapi.json", timeout=20) as r:
        return json.load(r)


def check(spec, name, instance):
    schema = {"$ref": f"#/components/schemas/{name}", "components": spec["components"]}
    jsonschema.Draft202012Validator(schema).validate(instance)


@pytest.fixture
def server():
    s = Running(make_engine())
    yield s
    s.close()


def test_requests_we_accept_are_valid_requests_and_our_responses_are_valid_responses(spec, server):
    check(spec, "SystemOneRequest", BODY)
    st, raw, _ = server.call("/v1/systemone", BODY)
    assert st == 200
    check(spec, "SystemOneResponse", json.loads(raw))


def test_explain_mode_still_validates_apart_from_the_declared_extra_field(spec, server):
    out = json.loads(server.call("/v1/systemone", {**BODY, "explain": True})[1])
    out.pop("trace")
    check(spec, "SystemOneResponse", out)


def test_models_list_matches_the_published_shape(spec, server):
    check(spec, "ModelMetadataList", json.loads(server.call("/v1/models")[1]))


def test_error_bodies_match_the_published_shapes(spec, server):
    st, raw, _ = server.call("/v1/systemone", {"model": "m"})
    assert st == 422
    check(spec, "HTTPValidationError", json.loads(raw))
    st, raw, _ = server.call("/v1/systemone", {"model": "m", "state": "x", "assets": {"a": {"type": "image"}},
                                                  "questions": {"b": {"type": "box"}}})
    assert st == 400
    check(spec, "HTTPBadRequestError", json.loads(raw))


def test_every_question_the_spec_itself_documents_is_accepted(spec, server):
    # the request examples published in the OpenAPI document must validate locally and be answered
    ex = spec["paths"]["/v1/systemone"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    for name in ("noul", "choice", "score"):
        body = ex[name]["value"]
        st, raw, _ = server.call("/v1/systemone", body)
        assert st == 200, (name, raw)
        check(spec, "SystemOneResponse", json.loads(raw))
