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
    hits = sum(r["answer"] == r["gold"] for r in rows)
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
            conf.append(max(v / total for v in probs.values()))
            ok.append(r["gold"] == r["answer"])
        else:
            ids = r["candidate_ids"]
            k = len(probs)
            p = probs[ids.index(r["gold"])]
            nll += -math.log(max(p, 1e-12))
            brier += sum((v - (1.0 if ids[i] == r["gold"] else 0.0)) ** 2 for i, v in enumerate(probs[:k]))
            conf.append(max(probs))
            ok.append(r["gold"] == r["answer"])
    m = len(conf)
    # 15-bin expected calibration error, last bin closed.
    edges = [i / 15 for i in range(16)]
    ece = 0.0
    for i in range(15):
        sel = [j for j in range(m) if edges[i] <= conf[j] and
               (conf[j] < edges[i + 1] or i == 14)]
        if sel:
            ece += len(sel) / m * abs(sum(conf[j] for j in sel) / len(sel) - sum(ok[j] for j in sel) / len(sel))
    locales = defaultdict(list)
    for r in rows:
        locales[r["locale"]].append(r)
    locale_accuracy = {
        locale: {"rows": len(rs),
                 "top1_accuracy": sum(r["answer"] == r["gold"] for r in rs) / len(rs)}
        for locale, rs in sorted(locales.items())}
    return {
        "rows": n, "top1_accuracy": top1,
        "nll": nll / m if m else None, "brier": brier / m if m else None,
        "per_locale": locale_accuracy,
        "macro_locale_accuracy": sum(v["top1_accuracy"] for v in locale_accuracy.values()) / len(locales),
        "ece15": ece, "distribution_rows": m,
    }


def score_stats(rows):
    """Ordinal metrics against the retained-rater mean level.

    Targets are retained per-rater arrays, not population probabilities, so this
    reports ordinal distance only. No population-calibration claim is derived
    here; NLL/Brier against a retained-rater distribution is reported separately
    and only for rows with at least three observed raters, per the preregistration.
    """
    n = len(rows)
    if n == 0:
        raise RuntimeError("no score rows")
    xs = [r["expected_level"] for r in rows]
    ys = [r["mean_level"] for r in rows]
    signed_errors = [x - y for x, y in zip(xs, ys)]
    target_mean = sum(ys) / n
    error_mean = sum(signed_errors) / n
    target_variance = sum((y - target_mean) ** 2 for y in ys)
    rps, nll, brier, eligible = 0.0, 0.0, 0.0, 0
    for r in rows:
        p, t = r["probs"], r["target_distribution"]
        cp, ct = 0.0, 0.0
        for predicted, target in zip(p[:-1], t[:-1]):
            cp += predicted
            ct += target
            rps += (cp - ct) ** 2 / r["native_level_max"]
        if r["observed_raters"] >= 3:
            eligible += 1
            nll += -sum(v * math.log(max(q, 1e-12)) for q, v in zip(p, t))
            brier += sum((q - v) ** 2 for q, v in zip(p, t))
    return {
        "rows": n,
        "mean_abs_error": sum(abs(e) for e in signed_errors) / n,
        "normalized_mean_abs_error": sum(abs(e) / r["native_level_max"]
                                         for e, r in zip(signed_errors, rows)) / n,
        "mean_signed_error": error_mean,
        "signed_error_slope_against_rater_mean": (
            sum((y - target_mean) * (e - error_mean) for y, e in zip(ys, signed_errors)) /
            target_variance if target_variance else None),
        "normalized_ranked_probability_score": rps / n,
        "spearman_rank_correlation": _spearman(xs, ys),
        "three_or_more_raters_rows": eligible,
        "retained_rater_nll": nll / eligible if eligible else None,
        "retained_rater_brier": brier / eligible if eligible else None,
    }


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(ra)
    if n < 3:
        return None
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else None


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
    """Count both discordance directions on shared rows."""
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
    """Exact P(X <= k) for X ~ Binomial(n, p), evaluated in log space.

    The naive sum overflows to a float error once n reaches tens of thousands,
    so each term is formed with lgamma and the total with a max-shift
    log-sum-exp. This stays exact rather than falling back to a normal or beta
    approximation.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)
        for i in range(k + 1)
    ]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vey", help="Choice endpoint: Vey arm file")
    parser.add_argument("--laya", help="Choice endpoint: competitor arm file")
    parser.add_argument("--laya-alt", help="Choice endpoint: second competitor route")
    parser.add_argument("--alt-name", default="laya_alt", help="Label for --laya-alt")
    parser.add_argument("--score", help="Ordinal endpoint: single competitor arm file")
    parser.add_argument("--score-endpoint", default=R.SCORE_ENDPOINTS[0])
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if not args.score and not args.laya:
        parser.error("--laya is required unless --score is given")

    if args.score:
        return verify_score(Path(args.score), args.score_endpoint, args.out)

    vey_header, vey_rows = (load_arm(Path(args.vey)) if args.vey else (None, []))
    laya_header, laya_rows = load_arm(Path(args.laya))
    alt_header, alt_rows = (load_arm(Path(args.laya_alt)) if args.laya_alt else (None, []))

    # Independent membership check against the projection, not against the arms.
    projected = {row["id"]: row for row in R.choice_rows()}

    def membership_of(rows, label):
        """Per-arm membership check. Returns the recomputed input digests."""
        digests, bad, seen = {}, [], set()
        for r in rows:
            if r["id"] in seen:
                bad.append(f"{label}: duplicate row {r['id']}")
                continue
            seen.add(r["id"])
            ref = projected.get(r["id"])
            if ref is None:
                bad.append(f"{label}: row {r['id']} absent from projection")
                continue
            # An ABSENT field is a verification failure in its own right. Reading
            # it with .get() yields None, which compares unequal and manufactures
            # a false "differs from projection" for every row of that arm.
            required = ["gold", "group_id", "locale"] + (
                ["question", "state", "candidate_ids"] if label != "vey" else [])
            missing = [f for f in required if f not in r]
            if missing:
                bad.append(f"{label}: row {r['id']} missing persisted fields {missing}")
                continue
            differing = [f for f in required if r[f] != ref[f]]
            if differing:
                bad.append(f"{label}: row {r['id']} fields {differing} differ from projection")
                continue
            digests[ref["id"]] = hashlib.sha256(json.dumps({
                "id": ref["id"], "question": ref["question"], "state": ref["state"],
                "candidate_ids": ref["candidate_ids"], "candidates": ref["candidates"],
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
            if label == "vey" and r.get("input_sha256") != digests[ref["id"]]:
                bad.append(f"vey: row {r['id']} input_sha256 differs from independent recompute")
            if label != "vey" and r.get("markers") != len(ref["candidate_ids"]):
                bad.append(f"{label}: row {r['id']} markers != candidate count")
            ids = ref["candidate_ids"]
            probabilities = r.get("probs", r.get("probabilities"))
            if isinstance(probabilities, dict) and set(probabilities) == set(ids):
                values = [probabilities[cid] for cid in ids]
            elif isinstance(probabilities, list) and len(probabilities) == len(ids):
                values = probabilities
            else:
                bad.append(f"{label}: row {r['id']} distribution does not cover candidates")
                continue
            if (any(not isinstance(p, (int, float)) or isinstance(p, bool)
                    or not math.isfinite(p) or not 0 <= p <= 1 for p in values)
                    or abs(sum(values) - 1) > 1e-6):
                bad.append(f"{label}: row {r['id']} invalid probability distribution")
                continue
            answer = r.get("answer")
            if answer not in ids or values[ids.index(answer)] < max(values) - 2e-8:
                bad.append(f"{label}: row {r['id']} answer is not a maximizing candidate")
            if type(r.get("correct")) is not bool or r["correct"] != (answer == ref["gold"]):
                bad.append(f"{label}: row {r['id']} correctness flag disagrees with native gold")
        missing = set(projected) - seen
        if missing:
            bad.append(f"{label}: missing {len(missing)} projected rows")
        return digests, bad

    vey_inputs, bad_vey = membership_of(vey_rows, "vey") if args.vey else ({}, [])
    laya_inputs, bad_laya = membership_of(laya_rows, "laya")
    alt_inputs, bad_alt = membership_of(alt_rows, "alt") if args.laya_alt else ({}, [])
    problems = bad_vey + bad_laya + bad_alt

    # Shared-exact-workflow control: the two arms' independently recomputed input
    # digests must be identical on every shared row. If they differ, the arms did
    # not receive identical inputs and no paired statistic below is valid.
    shared_ids = sorted(set(vey_inputs) & set(laya_inputs))
    mismatched = [i for i in shared_ids if vey_inputs[i] != laya_inputs[i]]
    if mismatched:
        problems.append(f"shared-exact-workflow control failed on {len(mismatched)} rows")
    shared_digest = hashlib.sha256(
        json.dumps({i: vey_inputs[i] for i in shared_ids}, sort_keys=True).encode("utf-8")
    ).hexdigest()


    receipt = {
        "schema": "vey.neutral.development-comparison-verification.v2",
        "status": "PASS" if not problems else "FAIL",
        "endpoint": "massive.intent",
        "independent_of_arm_aggregation": True,
        "metric_definitions": {
            "ece15": "maximum predicted probability versus native-gold correctness",
            "correctness": "answer equals native gold; persisted flags validated",
            "membership": "complete unique native rows; probabilities and argmax validated",
            "nll_probability_floor": 1e-12,
        },
        "projection_root": str(HERE),
        "comparison_protocol_sha256": R.COMPARISON_PROTOCOL_SHA256,
        "arm_headers": {"vey": vey_header, "laya": laya_header, "alt": alt_header},
        "membership": {"projected_choice_rows": len(projected), "problems": problems},
        "shared_exact_workflow": {
            "shared_rows": len(shared_ids),
            "mismatched_input_digests": len(mismatched),
            "shared_input_digest_sha256": shared_digest,
            "note": "Recomputed here from the projection, not copied from either arm.",
        },
        "laya": choice_stats(laya_rows),
        "non_inferiority_margin": 0.01,
        "zero_discordance_rule": "Zero discordance alone is not non-inferiority; the exact "
                                 "Clopper-Pearson upper bound must clear 0.01.",
    }
    if vey_rows:
        delta = paired_delta(vey_rows, laya_rows)
        disc = discordance(vey_rows, laya_rows)
        receipt["vey"] = choice_stats(vey_rows)
        receipt["paired_group_bootstrap"] = delta
        receipt["discordance"] = disc
        receipt["clopper_pearson_upper_vey_only"] = clopper_pearson_upper(
            disc["vey_only_correct"], disc["shared_rows"])
        receipt["clopper_pearson_upper_laya_only"] = clopper_pearson_upper(
            disc["laya_only_correct"], disc["shared_rows"])
        receipt["discordance_bound_limit"] = (
            "Row-binomial bounds assume independent rows, violated by 51 locale descendants "
            "per base group. They are descriptive calculations, not valid clustered risk guarantees. "
            "Laya-only-correct is the harmful direction for Vey.")
        receipt["non_inferiority_observed"] = delta["ci95_bootstrap"][0] > -0.01
    if alt_rows:
        alt_stats = choice_stats(alt_rows)
        shared_alt = sorted(set(laya_inputs) & set(alt_inputs))
        mismatched_alt = [i for i in shared_alt if laya_inputs[i] != alt_inputs[i]]
        if mismatched_alt:
            problems.append(f"route-comparison input control failed on {len(mismatched_alt)} rows")
        alt_by = {r["id"]: r for r in alt_rows}
        laya_by = {r["id"]: r for r in laya_rows}
        by_group = defaultdict(list)
        for i in shared_alt:
            by_group[laya_by[i]["group_id"]].append(i)
        diffs = [sum(alt_by[i]["correct"] for i in ids) / len(ids)
                 - sum(laya_by[i]["correct"] for i in ids) / len(ids)
                 for ids in by_group.values()]
        rng = random.Random(0)
        boots = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))) / len(diffs)
                       for _ in range(10000))
        receipt[args.alt_name] = alt_stats
        receipt["route_comparison"] = {
            "reference": "laya",
            "shared_rows": len(shared_alt),
            "mismatched_input_digests": len(mismatched_alt),
            "groups": len(diffs),
            "observed_delta": sum(diffs) / len(diffs),
            "ci95_bootstrap": [boots[250], boots[9749]],
            "resamples": 10000, "seed": 0,
        }
        receipt["vendor_claim_context"] = {
            "claim": "Laya multilingual MASSIVE-51 macro accuracy 0.4008",
            "status": "ADVERTISED, NOT OUR MEASUREMENT",
            "rule": "Report macro-over-locales explicitly. Equal locale counts make macro "
                    "equal to micro here, but different splits/configurations remain unresolved.",
        }
    receipt["status"] = "PASS" if not problems else "FAIL"
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if not problems else 1


def verify_score(path: Path, endpoint: str, out):
    """Reconstruct ordinal results from probabilities and guarded native targets."""
    R.protocol()
    header, rows = load_arm(path)
    projected = {r["id"]: r for r in R.score_rows() if r["endpoint"] == endpoint}
    problems, reconstructed, digests, seen = [], [], {}, set()
    route = header.get("route")
    for field, want in (("schema", "vey.neutral.laya-arm-run.v1"),
                        ("arm", "laya"), ("qtype", "score"), ("endpoint", endpoint),
                        ("bundle_revision", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")):
        if header.get(field) != want:
            problems.append(f"header {field} differs")
    if route not in ("english", "multilingual"):
        problems.append("unregistered route")
    if route == "english" and header.get("comparison_protocol_sha256") != R.COMPARISON_PROTOCOL_SHA256:
        problems.append("English comparison protocol digest differs")
    amendment_sha = "247cf2e3f6c0c488803d91deffca3c842d8ffe7372d09f323e22036c98e6f67c"
    historical_multilingual = (route == "multilingual" and
                              "score_amendment_sha256" not in header)
    if historical_multilingual:
        if header.get("protocol_sha256") != "73f823b543581ebd5ba4ff46dd38985df3e6fd9c5b40acde5c6fd736f3183b52":
            problems.append("historical multilingual protocol digest differs")
    elif "score_amendment_sha256" in header:
        if header["score_amendment_sha256"] != amendment_sha:
            problems.append("Score amendment digest differs")
    for r in rows:
        rid = r.get("id")
        if rid in seen:
            problems.append(f"duplicate row {rid}")
            continue
        seen.add(rid)
        ref = projected.get(rid)
        if ref is None:
            problems.append(f"row {rid} absent from projection")
            continue
        required = ("group_id", "locale", "question", "state", "candidate_ids", "endpoint")
        if any(r.get(f) != ref[f] for f in required):
            problems.append(f"row {rid} serving input differs")
            continue
        if "candidates" in r and r["candidates"] != ref["candidates"]:
            problems.append(f"row {rid} rubric descriptions differ")
            continue
        if r.get("route") != route or r.get("arm") != "laya":
            problems.append(f"row {rid} arm differs")
            continue
        k = ref["level_max"] + 1
        if (r.get("markers") != k or r.get("options") != k or
                r.get("native_level_max") != ref["level_max"] or
                r.get("observed_raters") != ref["observed_raters"] or
                r.get("mean_level") != ref["mean_level"] or
                r.get("target_counts") != ref["counts"] or
                r.get("target_distribution") != ref["distribution"]):
            problems.append(f"row {rid} native target or cardinality differs")
            continue
        p = r.get("probs")
        # Eight-decimal serialization plus FP32 softmax accumulation.
        if (not isinstance(p, list) or len(p) != k or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or
                    v < 0 or v > 1 for v in p) or abs(sum(p) - 1) > 1e-6):
            problems.append(f"row {rid} invalid ordinal probabilities")
            continue
        if r.get("probs_full") != p:
            problems.append(f"row {rid} probability copies differ")
            continue
        expected = sum(i * v for i, v in enumerate(p))
        error = abs(expected - ref["mean_level"])
        if any(not isinstance(r.get(f), (int, float)) or not math.isfinite(r[f]) or
               abs(r[f] - want) > 1e-6 for f, want in (
                   ("expected_level", expected), ("abs_error", error),
                   ("normalized_abs_error", error / ref["level_max"]))):
            problems.append(f"row {rid} derived metric differs")
            continue
        reconstructed.append({
            "probs": p, "expected_level": expected, "mean_level": ref["mean_level"],
            "native_level_max": ref["level_max"], "observed_raters": ref["observed_raters"],
            "target_distribution": ref["distribution"], "locale": ref["locale"],
        })
        digests[rid] = hashlib.sha256(json.dumps(
            {f: ref[f] for f in required}, sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode()).hexdigest()
    missing = set(projected) - seen
    if missing:
        problems.append(f"{len(missing)} projected rows missing")
    locales = defaultdict(list)
    for r in reconstructed:
        locales[r["locale"]].append(r)
    receipt = {
        "schema": "vey.neutral.ordinal-score-verification.v2",
        "status": "PASS" if not problems else "FAIL",
        "endpoint": endpoint,
        "vey_arm": "Vey-not-applicable: measured frozen decide() has no rubric-relative score surface",
        "arm_header": header,
        "source_file": str(path),
        "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "comparison_protocol_sha256": R.COMPARISON_PROTOCOL_SHA256,
        "independent_of_arm_aggregation": True,
        "input_validation_limit": (
            "Historical rows omit rubric description strings; only persisted serving fields "
            "are checked. Guarded native rubric mappings are reconstructed, not encoded-byte proof."
            if any("candidates" not in r for r in rows) else None),
        "registration_limit": ("The historical multilingual Score run cites a Choice-only "
                               "follow-up protocol; its Score execution was not prospectively "
                               "covered by that route protocol." if historical_multilingual else None),
        "projected_rows": len(projected), "verified_rows": len(reconstructed),
        "input_digest_sha256": hashlib.sha256(json.dumps(
            digests, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "membership_problems": problems[:20], "membership_problem_count": len(problems),
        "ordinal": score_stats(reconstructed) if reconstructed else None,
        "per_locale": {locale: score_stats(rs) for locale, rs in sorted(locales.items())},
        "calibration_limit": "Retained per-rater targets are not population probabilities; "
                             "no population calibration claim is derived.",
    }
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n", encoding="utf-8")
    return 0 if not problems else 1



if __name__ == "__main__":
    raise SystemExit(main())