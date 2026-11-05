import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vey.systemone import Calibration, Engine, EngineConfig, Escalator, LazyNLI, make_server

from .fakes import FakeEmbedder, FakeNLI, UniformEmbedder

BODY = {"model": "jev-latest", "state": "I was charged twice, please refund the duplicate payment",
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {"type": "choice", "instructions": "Which team?", "criteria": {
                "billing": "invoices payments refunds", "technical": "bugs outages errors"}},
            "urgency": {"type": "score", "criteria": ["can wait", "today urgent"]}}}


class Running:
    def __init__(self, engine, **kw):
        self.srv = make_server(engine, "127.0.0.1", 0, quiet=True, **kw)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()

    def call(self, path, body=None, headers=None, raw=None, method=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method or ("POST" if data is not None else "GET"))
        req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers


def make_engine(embedder=None, escalator=None, **cfg):
    return Engine(embedder or FakeEmbedder(), LazyNLI(lambda: FakeNLI()), Calibration(), escalator=escalator,
                  config=EngineConfig(**cfg))


@pytest.fixture
def server():
    s = Running(make_engine())
    yield s
    s.close()


def j(resp):
    return json.loads(resp[1])


def test_all_three_primitives_in_the_published_response_shape(server):
    st, raw, h = server.call("/v1/systemone", BODY)
    out = json.loads(raw)
    assert st == 200 and h["Content-Type"] == "application/json"
    assert set(out) == {"model", "answers", "usage"}
    a = out["answers"]
    assert set(a["refund"]) == {"type", "noul"}
    assert set(a["team"]) == {"type", "choice", "confidence", "probabilities"}
    assert set(a["urgency"]) == {"type", "score", "confidence", "legend", "probabilities"}
    assert set(out["usage"]) == {"input_tokens", "output_tokens"}
    assert "trace" not in out  # the trace is opt-in, so strict clients see exactly the published fields


def test_explain_adds_a_trace_and_is_not_an_error(server):
    out = j(server.call("/v1/systemone", {**BODY, "explain": True}))
    assert out["trace"]["questions"]["team"]["lane"] in ("matcher", "matcher+nli")
    out2 = j(server.call("/v1/systemone", BODY, headers={"X-Vey-Explain": "1"}))
    assert "trace" in out2


def test_validation_errors_are_422_in_fastapi_shape(server):
    st, raw, _ = server.call("/v1/systemone", {"model": "x", "questions": {"q": {"type": "choice"}}})
    d = json.loads(raw)["detail"]
    assert st == 422 and {tuple(e["loc"]) for e in d} == {("body", "state"), ("body", "questions", "q", "criteria")}


def test_invalid_json_and_missing_length(server):
    st, raw, _ = server.call("/v1/systemone", raw=b"{nope")
    assert st == 422 and json.loads(raw)["detail"][0]["type"] == "json_invalid"


def test_images_without_escalation_are_a_400_with_an_explanation(server):
    body = {"model": "m", "state": "photo", "assets": {"photo": {"type": "image", "format": "png", "data_base64": "AAAA"}},
            "questions": {"b": {"type": "box", "instructions": "find the red square"}}}
    st, raw, _ = server.call("/v1/systemone", body)
    assert st == 400 and "escalat" in json.loads(raw)["detail"]


def test_routes_models_health_info_and_console(server):
    assert j(server.call("/healthz")) == {"status": "ok"}
    models = j(server.call("/v1/models"))["models"]
    assert [m["name"] for m in models] == ["vey-latest", "vey-1"]
    assert all(set(m) == {"name", "description", "release_date"} for m in models)
    info = j(server.call("/v1/info"))
    assert info["lanes"]["matcher"]["embedder"] == "FakeEmbedder" and info["auth_required"] is False
    st, raw, h = server.call("/")
    assert st == 200 and b"Vey Console" in raw and h["Content-Type"].startswith("text/html")
    assert "default-src 'self'" in h["Content-Security-Policy"]
    assert server.call("/nope")[0] == 404
    assert server.call("/nope", {"a": 1})[0] == 404


def test_bearer_auth_protects_v1_but_not_health_or_console():
    s = Running(make_engine(), api_key="s3cret")
    try:
        assert s.call("/v1/systemone", BODY)[0] == 401
        assert s.call("/v1/systemone", BODY, headers={"Authorization": "Bearer wrong"})[0] == 401
        assert s.call("/v1/systemone", BODY, headers={"Authorization": "Bearer s3cret"})[0] == 200
        assert s.call("/v1/models")[0] == 401 and s.call("/v1/info")[0] == 401
        assert s.call("/healthz")[0] == 200 and s.call("/")[0] == 200
        # a rejected POST must not poison the next request on the same connection
        assert s.call("/v1/systemone", BODY, headers={"Authorization": "Bearer s3cret"})[0] == 200
    finally:
        s.close()


def test_oversized_body_is_413():
    s = Running(make_engine(), max_body=200)
    try:
        st, raw, _ = s.call("/v1/systemone", BODY)
        assert st == 413 and "too large" in json.loads(raw)["detail"]
    finally:
        s.close()


class Remote:
    """A fake System One target that records what it was sent."""

    def __init__(self, status=200):
        outer = self
        outer.received = []
        outer.status = status

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                outer.received.append({"body": body, "headers": dict(self.headers.items())})
                req = json.loads(body) if self.headers["Content-Type"] == "application/json" else {"questions": {"b": {}}}
                answers = {n: ({"type": "box", "box": {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.6}, "confidence": None}
                               if q.get("type") == "box" else {"type": "noul", "noul": 0.77})
                           for n, q in req["questions"].items()}
                out = json.dumps({"model": "remote-1", "answers": answers, "usage": {"input_tokens": 9, "output_tokens": 1}}).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown(); self.srv.server_close()


def test_low_confidence_questions_escalate_and_merge_with_local_answers():
    remote = Remote()
    s = Running(make_engine(UniformEmbedder(), Escalator(remote.url, model="target-model"),
                            floor=0.9, rerank_below=0.0))
    try:
        out = j(s.call("/v1/systemone", {**BODY, "explain": True},
                       headers={"TypeSafe-Danger-Zone-Images": "true"}))
        tr = out["trace"]["questions"]
        assert tr["team"]["lane"] == "escalated" and out["answers"]["team"] == {"type": "noul", "noul": 0.77}
        assert tr["refund"]["lane"] == "nli"  # confident locally, so it stayed local
        sent = json.loads(remote.received[0]["body"])
        assert sent["model"] == "target-model" and "explain" not in sent
        assert set(sent["questions"]) == {n for n, t in tr.items() if t["lane"] == "escalated"}
        got = {k.lower(): v for k, v in remote.received[0]["headers"].items()}
        assert got.get("typesafe-danger-zone-images") == "true"  # forwarded whatever the case it arrived in
    finally:
        s.close(); remote.close()


def test_images_and_box_questions_are_answered_by_the_escalation_target():
    remote = Remote()
    s = Running(make_engine(escalator=Escalator(remote.url)))
    try:
        body = {"model": "m", "state": "photo", "assets": {"photo": {"type": "image", "format": "png", "data_base64": "AAAA"}},
                "questions": {"red": {"type": "box", "instructions": "red square"}}}
        out = j(s.call("/v1/systemone", {**body, "explain": True}))
        assert out["answers"]["red"]["box"]["top"] == 0.1 and out["model"] == "remote-1"
        assert out["trace"]["questions"]["red"]["lane"] == "escalated"
        assert json.loads(remote.received[0]["body"])["assets"]["photo"]["format"] == "png"
    finally:
        s.close(); remote.close()


def test_multipart_requests_are_proxied_byte_for_byte():
    remote = Remote()
    s = Running(make_engine(escalator=Escalator(remote.url)))
    try:
        payload = b"--B\r\nContent-Disposition: form-data; name=request\r\n\r\n{}\r\n--B--\r\n"
        st, raw, _ = s.call("/v1/systemone", raw=payload, headers={"Content-Type": "multipart/form-data; boundary=B"})
        assert st == 200 and remote.received[0]["body"] == payload
    finally:
        s.close(); remote.close()


def test_an_unreachable_escalation_target_is_a_502_for_unreadable_requests_and_a_flag_otherwise():
    dead = Escalator("http://127.0.0.1:9", timeout=1)
    s = Running(make_engine(UniformEmbedder(), dead, floor=0.9, rerank_below=0.0))
    try:
        st, raw, _ = s.call("/v1/systemone", {"model": "m", "state": "x", "assets": {"a": {"type": "image"}},
                                                  "questions": {"b": {"type": "box"}}})
        assert st == 502
        out = j(s.call("/v1/systemone", {**BODY, "explain": True}))
        assert "escalation_error" in out["trace"]["questions"]["team"]
        assert out["answers"]["team"]["type"] == "choice"  # the local answer is kept
    finally:
        s.close()


def test_escalator_rejects_non_http_urls():
    with pytest.raises(ValueError):
        Escalator("file:///etc/passwd")
