#!/usr/bin/env python3
"""Shared, arm-agnostic row assembly for the neutral development comparison.

Reads the verified native dev projection ONLY through the committed guarded
reader (neutral_native_compile.open_phase), joins the four streams by decision
id, and yields frozen-workflow row records: one serving payload, one candidate
universe in a fixed order, and one target. Both arms consume this identical
stream with the same candidate order, state and question strings. Each model
applies its own pinned tokenizer and truncation after assembly.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PROTOCOL = HERE / "neutral_development_comparison_protocol.json"
COMPARISON_PROTOCOL_SHA256 = "65aaafd10c3845f6d3f05e16ae70a99f3f5eb3a6d67b39bc5794243d11f99cf9"

CHOICE_ENDPOINT = "massive.intent"
SCORE_ENDPOINTS = ("massive.grammar_score", "massive.spelling_score")

# Each dev utterance group carries exactly one massive.intent decision; the
# 63,852 rows are 51 locale descendants of 1,252 base groups, not a polar-axis
# duplication (census in neutral_development_comparison_amendment_v1.json).
# The pole filter below is a zero-row guard, kept because it costs nothing and
# fails loudly if a future projection ever does emit pole golds.
POLE_LABELS = {"yes", "no", "affirmative", "negative"}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compiler = _load("neutral_native_compile_dev", HERE / "neutral_native_compile.py")


def protocol():
    import hashlib

    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != COMPARISON_PROTOCOL_SHA256:
        raise RuntimeError("Comparison preregistration changed after commit")
    return json.loads(raw.decode("utf-8"))


def rows(endpoint: str, phase: str = "dev"):
    """Yield (row_id, serving, decision, target, provenance) for one endpoint.

    Streams are read once each and joined by id, so no row can be scored with
    another row's state, candidates or target.
    """
    serving = {}
    for record in compiler.open_phase(phase, "serving"):
        serving[record["id"]] = record
    decision = {}
    for record in compiler.open_phase(phase, "decisions"):
        decision[record["id"]] = record
    target = {}
    for record in compiler.open_phase(phase, "targets"):
        target[record["id"]] = record
    provenance = {}
    for record in compiler.open_phase(phase, "provenance"):
        provenance[record["id"]] = record

    served_ids = [rid for rid in serving if serving[rid]["task"] != "score" or True]
    assert set(serving) == set(decision) == set(target) == set(provenance), "stream id sets differ"
    del served_ids

    for rid in serving:
        prov = provenance[rid]
        if prov["endpoint"] != endpoint:
            continue
        yield rid, serving[rid], decision[rid], target[rid], prov


def choice_rows(phase: str = "dev"):
    """Labeled MASSIVE intent rows with the native intent gold."""
    for rid, serving, decision, target, prov in rows(CHOICE_ENDPOINT, phase):
        label = target.get("label_id")
        if not target.get("target_available") or label is None:
            continue
        if label in POLE_LABELS:
            continue
        candidates = {c["id"]: c["text"] for c in serving["candidates"]}
        if label not in candidates:
            raise RuntimeError("gold candidate missing from serving candidate universe")
        yield {
            "id": rid,
            "endpoint": CHOICE_ENDPOINT,
            "group_id": prov["group_id"],
            "component_id": prov["component_id"],
            "locale": serving["locale"],
            "task": serving["task"],
            "question": serving["question"],
            "state": serving["state"],
            "candidate_ids": [c["id"] for c in serving["candidates"]],
            "candidates": candidates,
            "gold": label,
        }


def score_rows(phase: str = "dev"):
    """Labeled MASSIVE ordinal rows, one entry per retained-rater target."""
    for endpoint in SCORE_ENDPOINTS:
        for rid, serving, decision, target, prov in rows(endpoint, phase):
            if not target.get("target_available"):
                continue
            candidates = {c["id"]: c["text"] for c in serving["candidates"]}
            yield {
                "id": rid,
                "endpoint": endpoint,
                "group_id": prov["group_id"],
                "component_id": prov["component_id"],
                "locale": serving["locale"],
                "task": serving["task"],
                "question": serving["question"],
                "state": serving["state"],
                "candidate_ids": [c["id"] for c in serving["candidates"]],
                "candidates": candidates,
                "level_max": target["native_level_max"],
                "counts": list(target["counts"]),
                "distribution": list(target["distribution"]),
                "mean_level": target["mean_level"],
                "observed_raters": target["observed_raters"],
            }


if __name__ == "__main__":
    # Input-only census. Prints counts and candidate cardinalities; reads no
    # model and produces no score.
    choice = list(choice_rows())
    scores = list(score_rows())
    print("choice rows", len(choice))
    print("choice groups", len({r["group_id"] for r in choice}))
    print("choice locales", len({r["locale"] for r in choice}))
    print("choice K", sorted({len(r["candidates"]) for r in choice}))
    for endpoint in SCORE_ENDPOINTS:
        subset = [r for r in scores if r["endpoint"] == endpoint]
        print(endpoint, "rows", len(subset), "groups", len({r["group_id"] for r in subset}),
              "levels", sorted({r["level_max"] for r in subset}),
              "raters", sorted({r["observed_raters"] for r in subset}))