#!/usr/bin/env python3
"""Exercise the ECA reader through real encoder/cache calls on development IR."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ephemeral_pages_capture import CORPUS, load_phase, rows_for
from ephemeral_pages_features import CachedRuntime, FeatureNormalizer, PROTOCOL_SHA256
from ephemeral_pages_train import load_control, predict_control


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def choice(row, scores, known_logits, threshold: float) -> dict:
    ids = [item["id"] for item in row["candidates"] if item["id"] != "__unknown__"]
    masses = 1.0 / (1.0 + np.exp(-np.clip(known_logits, -700, 700)))
    if not np.isfinite(scores).all() or (masses < threshold).any():
        winner = "__unknown__"
    else:
        maximum = float(np.max(scores))
        tied = [cid for cid, score in zip(ids, scores) if abs(float(score) - maximum) <= 1e-12]
        winner = min(tied, key=lambda cid: row["metadata"]["stable_ordinals"][cid])
    return {"winner": winner, "correct": winner in row["gold"],
            "scores": dict(zip(ids, map(float, scores))),
            "known_mass": dict(zip(ids, map(float, masses))),
            "source": "semantic", "certificate": "unavailable",
            "fallback": winner == "__unknown__"}


def run(run_root: Path, threshold: float, output: Path, device: str = "cuda") -> dict:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be the frozen calibration value in (0,1)")
    arrays = load_phase("development")
    records = arrays["records"]
    bases = []
    changed = None
    world = None
    for row in rows_for("development"):
        md = row["metadata"]
        if world is None:
            world = md["world_id"]
        if md["world_id"] != world:
            break
        if md["variant"] == "base" and md["query_kind"] == "atomic":
            bases.append(row)
        if bases and md["variant"] == "relevant_page_grade_change" and md["parent_decision_id"] == bases[0]["id"]:
            changed = row
    if len(bases) < 2 or changed is None:
        raise RuntimeError("development world lacks paired questions/page intervention")
    first, second = bases[:2]
    candidates = [c["id"] for c in first["candidates"] if c["id"] != "__unknown__"]
    lookup = {(r["row_id"], r["candidate_id"], r["term_index"]): i for i, r in enumerate(records)}
    reader, stats = load_control("pages", run_root, device)
    normalizer = FeatureNormalizer(stats["pages"]["mean"], stats["pages"]["std"], mode="pages")
    events = []
    with CachedRuntime(normalizer, device=device) as runtime:
        state = runtime.encode_state(first["state_blocks"], first["metadata"]["page_owners"])
        cold_state = dict(runtime.counters)
        for label, row in (("first_question", first), ("different_question", second), ("repeated_question", first)):
            before = dict(runtime.counters)
            result = runtime.score_question(state, row["question"], reader, candidates)
            after = dict(runtime.counters)
            assert after["page_encoded_examples"] == cold_state["page_encoded_examples"]
            if label == "repeated_question":
                assert before["query_encoded_examples"] == after["query_encoded_examples"]
            direct = result["output"]
            scores = direct.score.detach().cpu().numpy()
            known = direct.known_logits.detach().cpu().numpy()
            indices = [lookup[(row["id"], cid, 0)] for cid in candidates]
            cached = predict_control("pages", reader, stats, arrays, indices=indices)
            np.testing.assert_allclose(scores, cached.score, atol=1e-4, rtol=1e-4)
            np.testing.assert_allclose(known, cached.known_logits, atol=1e-4, rtol=1e-4)
            events.append({"event": label, "row_id": row["id"], "before": before, "after": after,
                           "decision": choice(row, scores, known, threshold),
                           "packed_score_max_error": float(np.max(np.abs(scores-cached.score)))})
        before = dict(runtime.counters)
        changed_state = runtime.encode_state(changed["state_blocks"], changed["metadata"]["page_owners"])
        result = runtime.score_question(changed_state, changed["question"], reader, candidates)
        assert runtime.counters["query_encoded_examples"] == before["query_encoded_examples"]
        assert runtime.counters["page_encoded_examples"] - before["page_encoded_examples"] <= 1
        events.append({"event": "relevant_page_change", "row_id": changed["id"], "before": before,
                       "after": dict(runtime.counters), "decision": choice(
                           changed, result["output"].score.detach().cpu().numpy(),
                           result["output"].known_logits.detach().cpu().numpy(), threshold)})
        del reader
        cross, cross_stats = load_control("cross", run_root, device)
        runtime.normalizer = FeatureNormalizer(cross_stats["cross"]["mean"], cross_stats["cross"]["std"], mode="cross")
        for row in (first, second):
            before = dict(runtime.counters)
            result = runtime.score_cross(state, row["question"], cross, candidates)
            assert runtime.counters["cross_encoded_examples"] > before["cross_encoded_examples"]
            events.append({"event": "joint_cross_question", "row_id": row["id"],
                           "before": before, "after": dict(runtime.counters)})
        lineage = runtime.encoder.finalize_lineage()
    report = {"protocol_sha256": PROTOCOL_SHA256, "phase": "development", "threshold": threshold,
              "state_sha256": digest(first["state_blocks"]), "events": events, "encoder_lineage": lineage,
              "claim": "Observed encoder reuse and packed/live parity; no latency or capability promotion."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run(args.run_root, args.threshold, args.output, args.device)
    print(json.dumps({"status": "cached_runtime_verified", "events": report["events"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
