"""Independent reconstruction checks for the NATIVE-P1 peer verifier."""
import json

import pytest

from research.endgame import native_peer_verify as V


def _refs():
    return V.projection(V.protocol())


def _write(path, header, records):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def _peer_record(ref, answer):
    ids = ref["candidate_ids"]
    values = {cid: 0.0 for cid in ids}
    values[answer] = 1.0
    return {"id": ref["id"], "group_id": ref["group_id"], "component_id": ref["component_id"],
            "locale": ref["locale"], "endpoint": ref["endpoint"], "answer": answer,
            "probabilities": values, "candidate_ids": ids}


def test_projection_matches_registered_counts():
    refs = _refs()
    cfg = V.protocol()
    assert len(refs) == sum(v["rows"] for v in cfg["data"]["expected_counts"].values())
    assert {r["endpoint"] for r in refs.values()} == set(V.DEV_ENDPOINTS)
    for ref in refs.values():
        assert ref["gold"] in ref["candidate_ids"]


def test_verifier_reconstructs_gold_from_projection_not_the_arm(tmp_path):
    refs = _refs()
    sample = sorted(refs)[:20]
    # A capture that lies about gold in a "gold" field must be ignored: gold
    # comes from the verified projection.
    peer = [dict(_peer_record(refs[i], refs[i]["gold"]), gold="forged") for i in sample]
    vey = [dict(_peer_record(refs[i], refs[i]["gold"])) for i in sample]
    laya_path, vey_path = tmp_path / "laya.jsonl", tmp_path / "vey.jsonl"
    _write(laya_path, {"schema": V.CAPTURE_SCHEMA, "route": "english",
                       "release_commit": V.protocol()["laya"]["release_commit"],
                       "peer_protocol_sha256": V.PROTOCOL_SHA256}, peer)
    _write(vey_path, {"schema": "vey.native-field.predictions.v1"}, vey)
    out = tmp_path / "receipt.json"
    # Partial arms fail the registered full-coverage gate; the run reports the
    # missing rows rather than raising or fabricating a full-coverage PASS.
    assert V.main(["--vey", str(vey_path), "--laya", str(laya_path), "--out", str(out)]) == 1
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "FAIL"
    assert any("missing" in problem for problem in receipt["membership_problems"])
    assert receipt["vey"]["rows"] == len(sample) and receipt["vey"]["top1_accuracy"] == 1.0
    assert receipt["laya"]["top1_accuracy"] == 1.0
    assert receipt["discordance"]["both_correct"] == len(sample)


def test_forged_answer_is_a_membership_problem(tmp_path):
    refs = _refs()
    sample = sorted(refs)[:10]
    peer = [_peer_record(refs[i], refs[i]["gold"]) for i in sample]
    # Vey claims a wrong answer while its distribution maximizes the gold.
    bad = _peer_record(refs[sample[0]], refs[sample[0]]["gold"])
    gold = refs[sample[0]]["gold"]
    bad["answer"] = next(cid for cid in refs[sample[0]]["candidate_ids"] if cid != gold)
    vey = [bad] + [_peer_record(refs[i], refs[i]["gold"]) for i in sample[1:]]
    laya_path, vey_path = tmp_path / "laya.jsonl", tmp_path / "vey.jsonl"
    _write(laya_path, {"schema": V.CAPTURE_SCHEMA, "route": "english",
                       "release_commit": V.protocol()["laya"]["release_commit"],
                       "peer_protocol_sha256": V.PROTOCOL_SHA256}, peer)
    _write(vey_path, {"schema": "vey.native-field.predictions.v1"}, vey)
    out = tmp_path / "receipt.json"
    assert V.main(["--vey", str(vey_path), "--laya", str(laya_path), "--out", str(out)]) == 1
    receipt = json.loads(out.read_text())
    assert receipt["status"] == "FAIL"
    assert any("not a maximizing candidate" in problem for problem in receipt["membership_problems"])


def test_paired_delta_is_zero_for_identical_arms():
    refs = _refs()
    sample = sorted(refs)[:50]
    ids = {rid: refs[rid]["candidate_ids"] for rid in refs}
    vey = {i: {"id": i, "component_id": refs[i]["component_id"], "gold": refs[i]["gold"],
               "answer": refs[i]["gold"], "correct": True, "probs": [1.0] * len(refs[i]["candidate_ids"]),
               "endpoint": refs[i]["endpoint"]} for i in sample}
    delta = V.paired_delta(vey, dict(vey))
    assert delta["observed_delta"] == 0.0
    assert delta["ci95_bootstrap"] == [0.0, 0.0]
    assert delta["shared_rows"] == len(sample)


def test_metrics_matches_hand_computation():
    ids = {"a": ["x", "y"], "b": ["x", "y"]}
    records = [
        {"id": "a", "gold": "x", "answer": "x", "correct": True, "probs": [0.75, 0.25]},
        {"id": "b", "gold": "y", "answer": "x", "correct": False, "probs": [0.6, 0.4]},
    ]
    result = V.metrics(records, ids)
    assert result["rows"] == 2 and result["top1_accuracy"] == 0.5
    assert result["nll"] == pytest.approx((-V.math.log(0.75) - V.math.log(0.4)) / 2)
    assert 0 <= result["ece15"] <= 1


def test_clopper_pearson_upper_is_a_valid_bound():
    assert V.clopper_pearson_upper(0, 100) < 0.05
    assert V.clopper_pearson_upper(100, 100) == 1.0
    assert V.clopper_pearson_upper(5, 0) == 1.0


def test_peer_header_mismatch_is_rejected(tmp_path):
    refs = _refs()
    sample = sorted(refs)[:5]
    peer = [_peer_record(refs[i], refs[i]["gold"]) for i in sample]
    vey = [_peer_record(refs[i], refs[i]["gold"]) for i in sample]
    laya_path, vey_path = tmp_path / "laya.jsonl", tmp_path / "vey.jsonl"
    _write(laya_path, {"schema": "wrong.schema"}, peer)
    _write(vey_path, {"schema": "vey.native-field.predictions.v1"}, vey)
    with pytest.raises(RuntimeError, match="peer capture schema"):
        V.main(["--vey", str(vey_path), "--laya", str(laya_path)])


def test_confidence_ties_cannot_be_split_into_selective_successes():
    result = V.confidence_statistics([0.9] * 4, [True, False, True, False])
    assert result["adaptive_ece15"] == pytest.approx(0.4)
    assert result["confidence_correctness_auroc"] == 0.5
    assert result["selective_risk_coverage"] == [
        {"threshold": 0.9, "accepted": 4, "coverage": 1.0, "errors": 2, "risk": 0.5}
    ]
    assert [(row["rows"], row["accuracy"]) for row in result["adaptive_reliability15"]] == [(4, 0.5)]


def test_confidence_ranking_counts_cross_class_ties_as_half_wins():
    result = V.confidence_statistics([0.9, 0.9, 0.3, 0.1], [True, False, True, False])
    assert result["confidence_correctness_auroc"] == 0.625
    curve = result["selective_risk_coverage"]
    assert [row["accepted"] for row in curve] == [2, 3, 4]
    assert [row["risk"] for row in curve] == pytest.approx([0.5, 1 / 3, 0.5])


def test_adaptive_reliability_preserves_large_equal_confidence_groups():
    result = V.confidence_statistics([0.1] * 8 + [0.9] * 8, [False] * 8 + [True] * 8)
    assert result["adaptive_ece15"] == pytest.approx(0.1)
    assert [(row["rows"], row["accuracy"]) for row in result["adaptive_reliability15"]] == [
        (8, 0.0), (8, 1.0)
    ]
    assert result["confidence_correctness_auroc"] == 1.0
    populated = [row for row in result["reliability15"] if row["rows"]]
    assert [row["mean_confidence"] for row in populated] == pytest.approx([0.1, 0.9])


@pytest.mark.parametrize("correct", [True, False])
def test_single_correctness_class_has_no_confidence_ranking_credit(correct):
    result = V.confidence_statistics([0.6], [correct])
    assert result["confidence_correctness_auroc"] is None
    accepted, empty = result["fixed_confidence_risk_coverage"][0], result["fixed_confidence_risk_coverage"][-1]
    assert accepted["threshold"] == 0.5 and accepted["accepted"] == 1
    assert accepted["risk"] == int(not correct)
    assert empty["threshold"] == 0.99 and empty["accepted"] == 0
    assert empty["coverage"] == 0.0 and empty["risk"] is None


@pytest.mark.parametrize("confidence,label", [
    ([0.5], []), ([float("nan")], [True]), ([1.01], [True]), ([True], [False]), ([0.5], [1]),
])
def test_invalid_confidence_observations_fail_closed(confidence, label):
    with pytest.raises(RuntimeError):
        V.confidence_statistics(confidence, label)
