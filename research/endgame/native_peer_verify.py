#!/usr/bin/env python3
"""Independently reconstruct the NATIVE-P1 English development peer contrast.

Re-derives every reported quantity from the persisted published-Laya capture and
the persisted NATIVE-1 arm records, using its own join, metrics and statistics
code. It never imports an arm's aggregation path, never re-runs a model, never
reads a sealed phase, and never trusts an arm's own gold or correctness flag:
gold comes from the verified DEV projection.

Usage:
  python native_peer_verify.py --vey F --laya F [--control F ...] [--out receipt.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "native_peer_protocol.json"
PROTOCOL_SHA256 = "8ca37c2a64f22af5846c698a16ae1269e76f988770226aff4446d7abac3b31ca"
CAPTURE_SCHEMA = "vey.native-peer.laya-capture.v1"
DEV_ENDPOINTS = ("banking77.intent", "massive.intent")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked(entry):
    path = Path(entry["path"])
    if "bytes" in entry:
        require(path.stat().st_size == entry["bytes"], "Pinned artifact size differs: " + str(path))
    require(sha(path) == entry["sha256"], "Pinned artifact identity differs: " + str(path))
    return path


def protocol():
    require(sha(PROTOCOL) == PROTOCOL_SHA256, "Peer protocol identity differs")
    cfg = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(cfg["schema"] == "vey.native-field.current-peer-protocol.v1", "Unregistered peer schema")
    return cfg


def load_arm(path):
    header = None
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if header is None and "schema" in record and "id" not in record:
                header = record
                continue
            rows.append(record)
    require(header is not None, str(path) + " carries no run header")
    return header, rows


def projection(cfg):
    """Independent DEV row universe: id -> exact serving inputs and native gold."""
    rows = {}
    for entry in (cfg["data"]["dev"],):
        path = checked(entry)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                decision = record["decisions"][0]
                require(decision["endpoint"] in DEV_ENDPOINTS, "Unregistered DEV endpoint")
                rid = record["id"]
                require(rid not in rows, "Duplicate DEV row id")
                rows[rid] = {
                    "id": rid, "group_id": record["group_id"], "component_id": record["component_id"],
                    "locale": record["locale"], "endpoint": decision["endpoint"],
                    "state": record["state"], "question": decision["question"],
                    "candidate_ids": list(decision["candidate_ids"]),
                    "candidates": dict(decision["candidates"]),
                    "gold": decision["gold"],
                }
    return rows


def input_digest(ref):
    return hashlib.sha256(json.dumps({
        "id": ref["id"], "question": ref["question"], "state": ref["state"],
        "candidate_ids": ref["candidate_ids"], "candidates": ref["candidates"],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _distribution(row, ids, label, problems):
    probabilities = row.get("probabilities", row.get("probs"))
    if isinstance(probabilities, dict):
        if set(probabilities) != set(ids):
            problems.append(label + ": row " + row["id"] + " probability keys differ")
            return None
        values = [float(probabilities[cid]) for cid in ids]
    elif isinstance(probabilities, list) and len(probabilities) == len(ids):
        values = [float(p) for p in probabilities]
    else:
        problems.append(label + ": row " + row["id"] + " distribution does not cover candidates")
        return None
    if any((not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p)
            or not 0 <= p <= 1) for p in values):
        problems.append(label + ": row " + row["id"] + " invalid probability value")
        return None
    if abs(sum(values) - 1) > 1e-6:
        problems.append(label + ": row " + row["id"] + " distribution does not sum to one")
        return None
    return values


def reconstruct(rows, label, refs, problems):
    """Return per-id records with independently derived gold, correctness and inputs."""
    seen = set()
    out = {}
    for row in rows:
        rid = row.get("id")
        if rid in seen:
            problems.append(label + ": duplicate row " + str(rid))
            continue
        seen.add(rid)
        ref = refs.get(rid)
        if ref is None:
            problems.append(label + ": row " + str(rid) + " absent from projection")
            continue
        if row.get("candidate_ids") != ref["candidate_ids"]:
            problems.append(label + ": row " + rid + " candidate order differs from projection")
            continue
        for field in ("group_id", "component_id", "locale", "endpoint"):
            if field in row and row[field] != ref[field]:
                problems.append(label + ": row " + rid + " field " + field + " differs from projection")
        values = _distribution(row, ref["candidate_ids"], label, problems)
        if values is None:
            continue
        answer = row.get("answer")
        if answer not in ref["candidate_ids"]:
            problems.append(label + ": row " + rid + " answer is not a candidate")
            continue
        if values[ref["candidate_ids"].index(answer)] < max(values) - 2e-8:
            problems.append(label + ": row " + rid + " answer is not a maximizing candidate")
            continue
        out[rid] = {
            "id": rid, "group_id": ref["group_id"], "component_id": ref["component_id"],
            "endpoint": ref["endpoint"], "locale": ref["locale"],
            "gold": ref["gold"], "answer": answer, "correct": answer == ref["gold"],
            "probs": values, "input_sha256": input_digest(ref),
        }
    missing = set(refs) - seen
    if missing:
        problems.append(label + ": missing " + str(len(missing)) + " projected rows")
    return out


def metrics(records, ids):
    """Top-1, NLL, Brier, ECE15 and confidence ranking for one arm.

    An empty arm returns null metrics: the registered coverage gate requires
    every DEV row, so an empty arm is already a membership problem, and a
    crash here would hide the row-level diagnostics that explain why.
    """
    n = len(records)
    if n == 0:
        return {"rows": 0, "top1_accuracy": None, "nll": None, "brier": None, "ece15": None}
    hits = 0
    nll = brier = 0.0
    conf, ok = [], []
    for r in records:
        position = ids[r["id"]].index(r["gold"])
        values = r["probs"]
        total = sum(values)
        p = values[position] / total
        nll += -math.log(max(p, 1e-12))
        brier += sum((v / total - (1.0 if i == position else 0.0)) ** 2 for i, v in enumerate(values))
        conf.append(max(values) / total)
        ok.append(bool(r["correct"]))
        hits += int(r["correct"])
    edges = [i / 15 for i in range(16)]
    ece = 0.0
    for i in range(15):
        sel = [j for j in range(n) if edges[i] <= conf[j] and (conf[j] < edges[i + 1] or i == 14)]
        if sel:
            ece += len(sel) / n * abs(sum(conf[j] for j in sel) / len(sel)
                                      - sum(ok[j] for j in sel) / len(sel))
    return {"rows": n, "top1_accuracy": hits / n, "nll": nll / n, "brier": brier / n, "ece15": ece}


def paired_delta(vey, laya, resamples=10000, seed=0):
    """Row-weighted paired component-ratio bootstrap of Vey minus Laya top-1."""
    shared = sorted(set(vey) & set(laya))
    if not shared:
        return {"unit": "input-only connected component", "components": 0, "shared_rows": 0,
                "estimand": "row-weighted top1 accuracy difference", "observed_delta": None,
                "ci95_bootstrap": None, "resamples": resamples, "seed": seed,
                "interval_scope": "Not computed: the arms share no rows."}
    by_component = defaultdict(list)
    for rid in shared:
        require(vey[rid]["component_id"] == laya[rid]["component_id"], "paired component mismatch")
        by_component[vey[rid]["component_id"]].append(rid)
    aggregates = []
    for component in sorted(by_component):
        ids = by_component[component]
        delta = sum(int(vey[i]["correct"]) - int(laya[i]["correct"]) for i in ids)
        aggregates.append((delta, len(ids)))
    observed = sum(d for d, _ in aggregates) / len(shared)
    rng = random.Random(seed)
    boots = []
    for _ in range(resamples):
        numerator = denominator = 0
        for _ in range(len(aggregates)):
            delta, count = aggregates[rng.randrange(len(aggregates))]
            numerator += delta
            denominator += count
        boots.append(numerator / denominator)
    boots.sort()
    return {
        "unit": "input-only connected component", "components": len(aggregates),
        "shared_rows": len(shared), "estimand": "row-weighted top1 accuracy difference",
        "observed_delta": observed,
        "ci95_bootstrap": [boots[int(0.025 * resamples)], boots[min(resamples - 1, int(0.975 * resamples))]],
        "resamples": resamples, "seed": seed,
        "interval_scope": "Nominal descriptive interval; not a simultaneous family-wise gate.",
    }


def discordance(vey, laya):
    shared = sorted(set(vey) & set(laya))
    harmful = sum(1 for i in shared if vey[i]["correct"] and not laya[i]["correct"])
    helpful = sum(1 for i in shared if not vey[i]["correct"] and laya[i]["correct"])
    both = sum(1 for i in shared if vey[i]["correct"] and laya[i]["correct"])
    neither = sum(1 for i in shared if not vey[i]["correct"] and not laya[i]["correct"])
    return {"shared_rows": len(shared), "vey_only_correct": harmful, "laya_only_correct": helpful,
            "both_correct": both, "neither_correct": neither, "mcnemar_n10_n01": [harmful, helpful]}


def clopper_pearson_upper(k, n, alpha=0.05):
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
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
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * math.log(p) + (n - i) * math.log1p(-p) for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vey", required=True)
    parser.add_argument("--laya", required=True)
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    cfg = protocol()
    refs = projection(cfg)
    require(len(refs) == sum(v["rows"] for v in cfg["data"]["expected_counts"].values()),
            "DEV projection row count differs from the registered expectation")
    problems = []

    laya_header, laya_rows = load_arm(args.laya)
    require(laya_header.get("schema") == CAPTURE_SCHEMA, "Unexpected peer capture schema")
    require(laya_header.get("route") == cfg["laya"]["route"], "Peer capture route differs")
    require(laya_header.get("release_commit") == cfg["laya"]["release_commit"], "Peer release commit differs")
    require(laya_header.get("peer_protocol_sha256") == PROTOCOL_SHA256, "Peer capture protocol differs")
    laya = reconstruct(laya_rows, "laya", refs, problems)

    vey_header, vey_rows = load_arm(args.vey)
    vey = reconstruct(vey_rows, "vey", refs, problems)

    controls = {}
    for path in args.control:
        header, rows = load_arm(path)
        name = Path(path).stem
        controls[name] = (header, reconstruct(rows, "vey-control:" + name, refs, problems))

    ids = {rid: refs[rid]["candidate_ids"] for rid in refs}
    shared = sorted(set(vey) & set(laya))
    mismatched = [i for i in shared if vey[i]["input_sha256"] != laya[i]["input_sha256"]]
    if mismatched:
        problems.append("shared-exact-workflow control failed on " + str(len(mismatched)) + " rows")

    receipt = {
        "schema": "vey.native-peer.verification.v1",
        "status": "PASS" if not problems else "FAIL",
        "independent_of_arm_aggregation": True,
        "peer_protocol_sha256": PROTOCOL_SHA256,
        "projection_rows": len(refs),
        "membership_problems": problems,
        "shared_exact_workflow": {"shared_rows": len(shared), "mismatched_input_digests": len(mismatched)},
        "endpoints": {endpoint: {"rows": sum(1 for r in refs.values() if r["endpoint"] == endpoint),
                                  "components": len({r["component_id"] for r in refs.values() if r["endpoint"] == endpoint})}
                      for endpoint in DEV_ENDPOINTS},
        "laya": metrics([laya[i] for i in sorted(laya)], ids),
        "vey": metrics([vey[i] for i in sorted(vey)], ids),
        "vey_controls": {name: metrics([records[i] for i in sorted(records)], ids)
                         for name, (_h, records) in controls.items()},
        "paired_component_bootstrap": paired_delta(vey, laya),
        "discordance": discordance(vey, laya),
        "per_endpoint": {},
        "promotion": False,
        "quality_claim": False,
        "performance_claim": False,
        "note": "Descriptive measured DEV peer contrast only; no Pareto, release, OOD, runtime, "
                "certification or direct-Jev credit. Published Laya probabilities are rounded to 4 "
                "decimals and normalized only for secondary proper-score arithmetic.",
    }
    disc = receipt["discordance"]
    receipt["clopper_pearson_upper_laya_only"] = clopper_pearson_upper(disc["laya_only_correct"], disc["shared_rows"])
    receipt["nominal_interval_above_minus_one_point"] = receipt["paired_component_bootstrap"]["ci95_bootstrap"][0] > -0.01
    for endpoint in DEV_ENDPOINTS:
        v = {i: r for i, r in vey.items() if r["endpoint"] == endpoint}
        l = {i: r for i, r in laya.items() if r["endpoint"] == endpoint}
        receipt["per_endpoint"][endpoint] = {
            "rows": len({i for i in refs if refs[i]["endpoint"] == endpoint}),
            "vey": metrics([v[i] for i in sorted(v)], ids),
            "laya": metrics([l[i] for i in sorted(l)], ids),
            "paired": paired_delta(v, l),
        }

    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
