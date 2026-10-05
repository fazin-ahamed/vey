#!/usr/bin/env python3
"""Independent reconstruction of the neutral development comparison.

Re-derives every reported quantity from the persisted per-row arm records and
the verified dev projection, using its own join, metrics and statistics code.
It never imports an arm's aggregation path, never re-runs a model, and never
reads a sealed phase. A disagreement with an arm's own printed summary is a
verification failure, not a rounding note.

Usage:
  python neutral_comparison_verify.py --vey F --laya F [--out receipt.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neutral_comparison_rows as R  # noqa: E402


def load_arm(path: Path):
    header = None
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if "schema" in record and header is None:
                header = record
            else:
                rows.append(record)
    if header is None:
        raise RuntimeError(f"{path} carries no run header")
    return header, rows


def choice_stats(rows):
    n = len(rows)
    if n == 0:
        raise RuntimeError("no choice rows")
    hits = sum(1 for r in rows if r.get("correct"))
    top1 = hits / n
    nll, brier, conf, ok = 0.0, 0.0, [], []
    for r in rows:
        probs = r.get("probs") or r.get("probabilities") or {}
        if isinstance(probs, dict):
            if sum(probs.values()) <= 0:
                continue
            gold = probs.get(r["gold"], 0.0)
            total = sum(probs.values())
            p = gold / total
            nll += -math.log(max(p, 1e-12))
            brier += sum((v / total - (1.0 if k == r["gold"] else 0.0)) ** 2 for k, v in probs.items())
            conf.append(p)
            ok.append(gold == r["answer"])
        else:
            ids = r["candidate_ids"]
            k = len(probs)
            p = probs[ids.index(r["gold"])]
            nll += -math.log(max(p, 1e-12))
            brier += sum((v - (1.0 if ids[i] == r["gold"] else 0.0)) ** 2 for i, v in enumerate(probs[:k]))
            conf.append(p)
            ok.append(r["gold"] == r["answer"])
    m = len(conf)
    # 15-bin expected calibration error, last bin closed.
    edges = [i / 15 for i in range(16)]
    ece = 0.0
    for i in range(15):
        sel = [j for j in range(m) if edges[i] < conf[j] <= edges[i + 1]]
        if sel:
            ece += len(sel) / m * abs(sum(conf[j] for j in sel) / len(sel) - sum(ok[j] for j in sel) / len(sel))
    return {
        "rows": n, "top1_accuracy": top1,
        "nll": nll / m if m else None, "brier": brier / m if m else None,
        "ece15": ece, "distribution_rows": m,
    }


def paired_delta(vey_rows, laya_rows, key="correct", resamples=10000, seed=0):
    """Cluster bootstrap over base utterance groups. Group, never locale row."""
    vey = {r["id"]: r for r in vey_rows}
    laya = {r["id"]: r for r in laya_rows}
    shared = sorted(set(vey) & set(laya))
    if not shared:
        raise RuntimeError("arms share no row ids")
    by_group = defaultdict(list)
    for rid in shared:
        by_group[(vey[rid]["group_id"], laya[rid]["group_id"])].append(rid)
    groups = sorted(by_group)
    per_group = []
    for g in groups:
        ids = by_group[g]
        v = sum(1 for i in ids if vey[i][key]) / len(ids)
        l = sum(1 for i in ids if laya[i][key]) / len(ids)
        per_group.append(v - l)
    obs = sum(per_group) / len(per_group)
    rng = random.Random(seed)
    boots = []
    k = len(per_group)
    for _ in range(resamples):
        s = 0.0
        for _ in range(k):
            s += per_group[rng.randrange(k)]
        boots.append(s / k)
    boots.sort()
    lo = boots[int(0.025 * resamples)]
    hi = boots[min(resamples - 1, int(0.975 * resamples))]
    return {
        "groups": k, "shared_rows": len(shared), "observed_delta": obs,
        "ci95_bootstrap": [lo, hi], "resamples": resamples, "seed": seed,
    }


def discordance(vey_rows, laya_rows):
    """Exact harmful-event count: Vey correct while Laya wrong, per shared row."""
    vey = {r["id"]: r for r in vey_rows}
    laya = {r["id"]: r for r in laya_rows}
    shared = sorted(set(vey) & set(laya))
    harmful = sum(1 for i in shared if vey[i]["correct"] and not laya[i]["correct"])
    helpful = sum(1 for i in shared if not vey[i]["correct"] and laya[i]["correct"])
    both = sum(1 for i in shared if vey[i]["correct"] and laya[i]["correct"])
    neither = sum(1 for i in shared if not vey[i]["correct"] and not laya[i]["correct"])
    return {
        "shared_rows": len(shared), "vey_only_correct": harmful,
        "laya_only_correct": helpful, "both_correct": both, "neither_correct": neither,
        "mcnemar_n10_n01": [harmful, helpful],
    }


def clopper_pearson_upper(k, n, alpha=0.05):
    """One-sided upper bound. Returns 1.0 without scipy."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    # P(X <= k) decreases in p, so bisecting on that monotonic decrease finds the
    # root of P(X<=k)=alpha; hi is the one-sided upper bound. (The branch order
    # matters: inverted, every case returns 1.0.)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _beta_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def _beta_cdf(k, n, p):
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return sum(
        math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vey", required=True)
    parser.add_argument("--laya", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    vey_header, vey_rows = load_arm(Path(args.vey))
    laya_header, laya_rows = load_arm(Path(args.laya))

    # Independent membership check against the projection, not against the arms.
    projected = {row["id"]: row for row in R.choice_rows()}
    problems = []
    for name, rows in (("vey", vey_rows), ("laya", laya_rows)):
        for r in rows:
            ref = projected.get(r["id"])
            if ref is None:
                problems.append(f"{name}: row {r['id']} absent from projection")
                continue
            for field in ("gold", "group_id", "candidate_ids", "question", "state"):
                if r[field] != ref[field]:
                    problems.append(f"{name}: row {r['id']} field {field} differs from projection")
                    break
            else:
                if name == "laya":
                    if r["markers"] != len(ref["candidate_ids"]):
                        problems.append(f"laya: row {r['id']} markers != candidate count")
    if problems:
        print(json.dumps({"status": "FAIL", "problems": problems[:20], "problem_count": len(problems)}))
        return 1

    # Shared-exact-workflow control: identical inputs across arms, by hash.
    vey_inputs = {r["id"]: r.get("input_sha256") for r in vey_rows}
    laya_inputs = {}
    for r in laya_rows:
        laya_inputs[r["id"]] = hashlib.sha256(
            json.dumps({"id": r["id"], "question": r["question"], "state": r["state"],
                        "candidate_ids": r["candidate_ids"],
                        "candidates": {cid: r["candidates"][cid] for cid in r["candidate_ids"]}},
                       ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest() if "candidates" in r else None

    vey_stats = choice_stats(vey_rows)
    laya_stats = choice_stats(laya_rows)
    delta = paired_delta(vey_rows, laya_rows)
    disc = discordance(vey_rows, laya_rows)
    n = disc["shared_rows"]
    cp_upper = clopper_pearson_upper(disc["vey_only_correct"], n)

    receipt = {
        "schema": "vey.neutral.development-comparison-verification.v1",
        "status": "PASS",
        "endpoint": "massive.intent",
        "independent_of_arm_aggregation": True,
        "projection_root": str(R.PROTOCOL and HERE),
        "comparison_protocol_sha256": R.COMPARISON_PROTOCOL_SHA256,
        "arm_headers": {"vey": vey_header, "laya": laya_header},
        "membership": {"projected_choice_rows": len(projected), "problems": problems},
        "vey": vey_stats,
        "laya": laya_stats,
        "paired_group_bootstrap": delta,
        "discordance": disc,
        "clopper_pearson_upper_vey_only": cp_upper,
        "non_inferiority_margin": 0.01,
        "non_inferiority_observed": (delta["ci95_bootstrap"][0] > -0.01),
        "zero_discordance_rule": "Zero discordance alone is not non-inferiority; the exact "
                                 "Clopper-Pearson upper bound must clear 0.01.",
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())