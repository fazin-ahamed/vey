"""Fail-closed behavior of the published-Laya peer capture (no model load)."""
import copy
import json
from pathlib import Path

import pytest

from research.endgame import native_peer_capture as capture


class _FakeAgent:
    """Records the exact request and returns a caller-chosen prediction."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return self.reply


class _FakeRuntime:
    """Stand-in for the ``laya`` module: ``.Agent(path, ...)`` returns the fake agent."""

    def __init__(self, agent):
        self.agent = agent
        self.constructs = []

    def Agent(self, path, **kwargs):
        self.constructs.append((path, kwargs))
        return self.agent


def _rows(limit):
    return list(capture.dev_rows(capture.protocol()))[:limit]


def _patched(monkeypatch, tmp_path, agent, rows):
    """Real protocol, but no environment/dependency/interpreter custody and no model."""
    cfg = copy.deepcopy(capture.protocol())
    cfg["data"]["root"] = str(tmp_path)
    cfg["laya"]["environment_variables"] = {}
    cfg["laya"]["packages"] = {}
    cfg["laya"]["executable"] = capture.sys.executable
    monkeypatch.setattr(capture, "protocol", lambda: cfg)
    monkeypatch.setattr(capture, "dev_rows", lambda _cfg: iter(rows))
    monkeypatch.setattr(capture, "custody", lambda _cfg: (None, None, None, tmp_path, {"files": {}}))
    monkeypatch.setattr(capture, "materialize", lambda root, route: (tmp_path, {}))
    monkeypatch.setattr(capture, "load_agent", lambda *_a: (_FakeRuntime(agent), None))
    monkeypatch.setattr(capture, "checked", lambda entry: Path(entry["path"]))


def test_dev_rows_are_choice_only_and_carry_native_inputs():
    rows = _rows(3)
    assert rows and all(r["endpoint"] in capture.DEV_ENDPOINTS for r in rows)
    for row in rows:
        assert set(row["candidates"]) == set(row["candidate_ids"])
        assert row["gold"] in row["candidate_ids"]
        assert row["state"] and row["question"]


def test_valid_prediction_is_persisted_with_every_candidate(monkeypatch, tmp_path):
    row = _rows(1)[0]
    probabilities = {cid: 0.0 for cid in row["candidate_ids"]}
    probabilities[row["gold"]] = 1.0
    agent = _FakeAgent({"answers": {"q": {"choice": row["gold"], "probabilities": probabilities}}})
    _patched(monkeypatch, tmp_path, agent, [row])

    out = tmp_path / "capture.jsonl"
    assert capture.main(["--out", str(out), "--limit", "1"]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["schema"] == capture.CAPTURE_SCHEMA
    record = lines[1]
    assert record["correct"] is True and record["answer"] == row["gold"]
    assert set(record["probabilities"]) == set(row["candidate_ids"])
    assert record["probability_decimals"] == 4
    # The model must receive the exact rendered state and every candidate text.
    state, questions = agent.calls[0]
    assert state == {"text": row["state"]}
    assert questions["q"]["criteria"] == {cid: row["candidates"][cid] for cid in row["candidate_ids"]}


@pytest.mark.parametrize("choice", ["not_a_candidate", None])
def test_illegal_choice_is_rejected_and_failure_retained(monkeypatch, tmp_path, choice):
    row = _rows(1)[0]
    probabilities = {cid: 0.0 for cid in row["candidate_ids"]}
    probabilities[row["candidate_ids"][0]] = 1.0
    agent = _FakeAgent({"answers": {"q": {"choice": choice, "probabilities": probabilities}}})
    _patched(monkeypatch, tmp_path, agent, [row])

    out = tmp_path / "capture.jsonl"
    with pytest.raises(RuntimeError):
        capture.main(["--out", str(out), "--limit", "1"])
    tail = [json.loads(line) for line in out.read_text().splitlines()]
    assert tail[-1]["schema"] == "vey.native-peer.laya-capture-failure.v1"
    assert tail[-1]["rows_before_failure"] == 0


def test_unnormalized_distribution_is_rejected(monkeypatch, tmp_path):
    row = _rows(1)[0]
    probabilities = {cid: 0.5 for cid in row["candidate_ids"]}
    agent = _FakeAgent({"answers": {"q": {"choice": row["candidate_ids"][0], "probabilities": probabilities}}})
    _patched(monkeypatch, tmp_path, agent, [row])

    out = tmp_path / "capture.jsonl"
    with pytest.raises(RuntimeError, match="Serialization bound"):
        capture.main(["--out", str(out), "--limit", "1"])


def test_missing_probability_key_is_rejected(monkeypatch, tmp_path):
    row = _rows(1)[0]
    probabilities = {cid: 0.0 for cid in row["candidate_ids"][:-1]}
    probabilities[row["candidate_ids"][0]] = 1.0
    agent = _FakeAgent({"answers": {"q": {"choice": row["candidate_ids"][0], "probabilities": probabilities}}})
    _patched(monkeypatch, tmp_path, agent, [row])

    out = tmp_path / "capture.jsonl"
    with pytest.raises(RuntimeError, match="Probability keys differ"):
        capture.main(["--out", str(out), "--limit", "1"])


def test_checked_rejects_a_tampered_digest(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"payload")
    good = capture.sha(path)
    capture.checked({"path": str(path), "bytes": 7, "sha256": good})
    with pytest.raises(RuntimeError, match="identity differs"):
        capture.checked({"path": str(path), "bytes": 7, "sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="size differs"):
        capture.checked({"path": str(path), "bytes": 8, "sha256": good})


def test_custody_binds_the_published_release_and_english_payloads():
    manifest, compat, release, source_root, route = capture.custody(capture.protocol())
    assert manifest["release_commit"] == compat["release_commit"]
    assert manifest["package_version"] == capture.protocol()["laya"]["published_version"]
    assert route["repository"] == "convaiinnovations/laya"
    assert (source_root / "laya/agent.py").is_file()


def test_capture_refuses_an_existing_output(monkeypatch, tmp_path):
    row = _rows(1)[0]
    probabilities = {cid: 0.0 for cid in row["candidate_ids"]}
    probabilities[row["gold"]] = 1.0
    agent = _FakeAgent({"answers": {"q": {"choice": row["gold"], "probabilities": probabilities}}})
    _patched(monkeypatch, tmp_path, agent, [row])

    out = tmp_path / "capture.jsonl"
    out.write_text("existing\n")
    with pytest.raises(FileExistsError):
        capture.main(["--out", str(out), "--limit", "1"])
