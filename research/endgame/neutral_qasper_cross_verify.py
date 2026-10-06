#!/usr/bin/env python3
"""Reconstruct QNATIVE-2 cross evidence without loading a neural model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import random

HERE = Path(__file__).resolve().parent
AMENDMENT = HERE / "neutral_qasper_cross_study_amendment.json"
PHASES = ("fit", "selection", "calibration", "dev")
BOOL = ("qasper.yes_no", "qasper.answerability")
RETRIEVAL = "qasper.evidence_retrieval"
EXTRACTION = "qasper.extractive_answer"
INTENT = ("banking77.intent", "massive.intent")
ENDPOINTS = INTENT + BOOL + (RETRIEVAL, EXTRACTION)
K_GRID = (2, 4, 8, 16, 32, 64, 128, 255, 512, 1000)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
EMPTY_STATE = "[EMPTY EVIDENCE]"
REQUIRED = ("metadata.json", "liveness.json", "history.json", "selected.safetensors", "selection.jsonl.gz",
            "calibration.jsonl.gz", "dev.jsonl.gz", "baselines.jsonl.gz", "question_masked.jsonl.gz",
            "state_masked.jsonl.gz", "question_swapped.jsonl.gz", "reordered.jsonl.gz",
            "candidate_count.jsonl.gz", "repeated_state.jsonl", "support.jsonl.gz", "runtime.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _nonfinite(value):
    raise ValueError("Nonfinite JSON denied: " + value)


def decode(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def numeric(text):
        value = float(text)
        require(math.isfinite(value), "Nonfinite JSON numeric literal")
        return value
    return json.loads(text, parse_constant=_nonfinite, parse_float=numeric, object_pairs_hook=unique)


def load_json(path):
    return decode(Path(path).read_text(encoding="utf-8"))


def json_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            require(line.strip(), "Blank capture row")
            row = decode(line)
            require(isinstance(row, dict), "Capture row is not an object")
            yield row


def finite(value, label="number"):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid finite " + label)
    return float(value)


def integer(value, label="count"):
    require(type(value) is int and value >= 0, "Invalid nonnegative " + label)
    return value


def close(actual, expected, label="numeric reconstruction", atol=1e-6):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + " keys")
        for key in expected:
            close(actual[key], expected[key], label + "." + str(key), atol)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label + " length")
        for index, value in enumerate(expected):
            close(actual[index], value, label + "." + str(index), atol)
    elif type(expected) in (int, float):
        require(abs(finite(actual, label) - expected) <= atol, label + " differs")
    else:
        require(actual == expected, label + " differs")


def distribution(values, size):
    require(isinstance(values, list) and len(values) == size and size > 0, "Probability dimensions")
    result = [finite(x, "probability") for x in values]
    require(all(0 <= x <= 1 for x in result) and abs(sum(result) - 1) <= 1e-6, "Probability mass")
    return result


def softmax(logits, temperature=1.0):
    require(logits and finite(temperature) > 0, "Nonempty logits and positive temperature required")
    scaled = [finite(x, "logit") / temperature for x in logits]
    maximum = max(scaled)
    masses = [math.exp(x - maximum) for x in scaled]
    total = math.fsum(masses)
    return [mass / total for mass in masses]


def ranking(ids, scores):
    require(len(ids) == len(scores) and len(set(ids)) == len(ids), "Ranking coordinates")
    require(all(type(item) is str for item in ids), "Ranking IDs")
    by_id = dict(zip(ids, (finite(score) for score in scores)))
    return sorted(ids, key=lambda item: (-by_id[item], item))


def ratio(numerator, denominator):
    require(finite(denominator) >= 0, "Negative denominator")
    return finite(numerator) / denominator if denominator else None


def checked(entry, root=None):
    require(isinstance(entry, dict) and {"path", "bytes", "sha256"} <= set(entry), "Artifact pin schema")
    path = Path(entry["path"])
    if root is not None:
        require(not path.is_absolute() and len(path.parts) == 1, "Unrooted artifact")
        path = Path(root) / path
    require(path.is_file() and not path.is_symlink(), "Missing/link artifact: " + str(path))
    require(path.stat().st_size == integer(entry["bytes"]) and digest(path) == entry["sha256"],
            "Artifact custody differs: " + str(path))
    return path


def member_index(rows, expected=None):
    result = {}
    for row in rows:
        identity = row.get("id")
        require(type(identity) is str and identity not in result, "Invalid/duplicate member ID")
        result[identity] = row
    if expected is not None:
        require(set(result) == set(expected), "Incomplete/foreign member universe")
    return result


def split_component(component):
    bucket = int(hashlib.sha256(("native-field-v1|7|" + component).encode()).hexdigest(), 16) % 100
    return "fit" if bucket < 70 else "selection" if bucket < 85 else "calibration"


def native_target(ledger, blocks, endpoint):
    """Rebuild targets from retained native values, not stored match summaries."""
    if endpoint in BOOL:
        counts = [0, 0]
        for annotation in ledger["ratings"]:
            require(type(annotation["present"]) is bool and
                    (annotation["native_value"] is None or type(annotation["native_value"]) is bool), "Native Bool field type")
            observed = annotation["present"] is True and type(annotation["native_value"]) is bool
            if observed:
                value = annotation["native_value"]
                counts[int(not value if endpoint == BOOL[1] else value)] += 1
        return {"kind": "bool", "counts": counts, "observed_raters": sum(counts),
                "distribution": [x / sum(counts) for x in counts] if sum(counts) else None}
    require(endpoint in (RETRIEVAL, EXTRACTION), "Unknown native QASPER endpoint")
    ratings, items = [], []
    for annotation in ledger["ratings"]:
        present, value = annotation["present"], annotation["native_value"]
        require(type(present) is bool and (value is None or isinstance(value, list)), "Native annotation schema")
        observed = present and value is not None
        require(not observed or all(type(text) is str for text in value), "Native array item schema")
        matched = []
        for index, text in enumerate(value if observed else []):
            occurrences = []
            for block in blocks:
                if endpoint == RETRIEVAL:
                    if block["text"] == text:
                        occurrences.append({"block_id": block["id"], "start": 0, "end": len(text)})
                elif text:
                    start = block["text"].find(text)
                    while start != -1:
                        occurrences.append({"block_id": block["id"], "start": start, "end": start + len(text)})
                        start = block["text"].find(text, start + 1)
            item = {"source_index": index, "text": text, "occurrences": occurrences}
            matched.append(item)
            items.append(item)
        ratings.append({"annotation_ordinal": annotation["annotation_ordinal"], "present": present,
                        "null": present and value is None, "observed": observed, "items": matched})
    count = len(items)
    unmatched = sum(not item["occurrences"] for item in items)
    return {"kind": "retrieval" if endpoint == RETRIEVAL else "extract", "ratings": ratings,
            "native_item_count": count, "unmatched_item_count": unmatched,
            "support_stratum": "no_native_items" if not count else "none" if unmatched == count else
                               "mixed" if unmatched else "full"}


def categorical_target(record, ids=None):
    endpoint, target = record["endpoint"], record["target"]
    ids = ids if ids is not None else record["candidate_ids"]
    if endpoint in BOOL + INTENT:
        return target["distribution"]
    if endpoint != RETRIEVAL:
        return None
    masses, matched = dict.fromkeys(ids, 0.0), 0
    for annotation in target["ratings"]:
        for item in annotation["items"]:
            found = list(dict.fromkeys(x["block_id"] for x in item["occurrences"]))
            if found:
                matched += 1
                for identity in found:
                    require(identity in masses, "Matched item outside full catalogue")
                    masses[identity] += 1 / len(found)
    return [masses[identity] / matched for identity in ids] if matched else None


def decode_bio(blocks, windows):
    """Greedy BIO, orphan I starts, strict intersection merge, verbatim source slices."""
    sources = {block["id"]: block["text"] for block in blocks}
    order = {block["id"]: index for index, block in enumerate(blocks)}
    intervals = []
    for window in windows:
        block = window["block_id"]
        require(block in sources, "BIO foreign source block")
        offsets, logits = window["offsets"], window["bio_logits"]
        indices = window["source_token_indices"]
        require(len(offsets) == len(logits) == len(indices), "BIO trace dimensions")
        current = []
        def finish():
            if current:
                intervals.append((block, current[0][0], current[-1][1], {(x[0], x[1]): x[2] for x in current}))
                current.clear()
        previous = -1
        for index, offset, raw in zip(indices, offsets, logits):
            require(type(index) is int and index >= 0 and (previous < 0 or index == previous + 1),
                    "BIO source token continuity")
            previous = index
            require(isinstance(offset, list) and len(offset) == 2 and all(type(x) is int for x in offset), "BIO offset schema")
            start, end = offset
            require(0 <= start <= end <= len(sources[block]), "BIO source offset bounds")
            require(len(raw) == 3, "BIO class count")
            probs = softmax(raw)
            tag = max(range(3), key=lambda i: probs[i])
            if start == end:
                continue
            if tag == 0:
                finish()
            elif tag == 1:
                finish()
                current.append((start, end, probs[tag]))
            else:
                current.append((start, end, probs[tag]))
        finish()
    unique = {}
    for block, start, end, confidences in intervals:
        saved = unique.setdefault((block, start, end), {})
        for offset, confidence in confidences.items():
            saved[offset] = max(saved.get(offset, 0), confidence)
    merged = []
    for (block, start, end), confidences in sorted(unique.items(), key=lambda item: (order[item[0][0]], item[0][1], item[0][2])):
        if merged and merged[-1]["block_id"] == block and start < merged[-1]["end"]:
            previous = merged[-1]
            previous["end"] = max(previous["end"], end)
            for offset, confidence in confidences.items():
                previous["_token_confidences"][offset] = max(previous["_token_confidences"].get(offset, 0), confidence)
        else:
            merged.append({"block_id": block, "start": start, "end": end, "_token_confidences": dict(confidences)})
    for span in merged:
        values = span.pop("_token_confidences").values()
        span["confidence"] = sum(values) / len(values)
        span["text"] = sources[span["block_id"]][span["start"]:span["end"]]
    return merged


def supervision(target, blocks, source_offsets, windows):
    """Partial annotations label known positive tokens only; absence never means O."""
    sources = {block["id"]: block["text"] for block in blocks}
    counts = {bid: [[0, 0, 0] for _ in offsets] for bid, offsets in source_offsets.items()}
    annotations = []
    for annotation in target["ratings"]:
        supported_items, positive = [], defaultdict(dict)
        for item in annotation["items"]:
            aligned, occurrence_support = [], []
            for occurrence in item["occurrences"]:
                bid, start, end = (occurrence[key] for key in ("block_id", "start", "end"))
                require(sources[bid][start:end] == item["text"], "Support source string")
                indices = [index for index, (a, b) in enumerate(source_offsets[bid]) if start <= a < b <= end]
                boundary = bool(indices) and source_offsets[bid][indices[0]][0] == start and source_offsets[bid][indices[-1]][1] == end
                contained = boundary and any(window["block_id"] == bid and
                    set(indices) <= set(window["source_token_indices"]) for window in windows)
                occurrence_support.append({**occurrence, "source_token_indices": indices,
                                           "token_boundary_supported": boundary, "window_supported": contained})
                if contained:
                    aligned.append({**occurrence, "source_token_indices": indices})
                    for position, index in enumerate(indices):
                        tag = 1 if position == 0 else 2
                        positive[bid][index] = min(positive[bid].get(index, tag), tag)
            supported_items.append({"source_index": item["source_index"], "text": item["text"],
                                    "source_occurrences": item["occurrences"], "supported_occurrences": aligned,
                                    "occurrence_support": occurrence_support})
        full = bool(supported_items) and all(item["supported_occurrences"] for item in supported_items)
        usable = bool(positive)
        if annotation["observed"] and usable:
            for bid, block_counts in counts.items():
                for index, token_counts in enumerate(block_counts):
                    if full or index in positive[bid]:
                        token_counts[positive[bid].get(index, 0)] += 1
        annotations.append({"annotation_ordinal": annotation["annotation_ordinal"], "observed": annotation["observed"],
                            "native_item_count": len(annotation["items"]), "full_support": full,
                            "positive_only": usable and not full, "items": supported_items})
    denominators = {bid: [sum(values) for values in block] for bid, block in counts.items()}
    distributions = {bid: [[x / sum(values) for x in values] if sum(values) else None for values in block]
                     for bid, block in counts.items()}
    items = [item for annotation in annotations for item in annotation["items"]]
    support_counts = {"native_item_count": len(items),
                      "supported_items": sum(bool(item["supported_occurrences"]) for item in items),
                      "source_unmatched_items": sum(not item["source_occurrences"] for item in items),
                      "token_boundary_unsupported_items": sum(bool(item["source_occurrences"]) and
                          not any(x["token_boundary_supported"] for x in item["occurrence_support"]) for item in items),
                      "window_length_unsupported_items": sum(any(x["token_boundary_supported"] for x in item["occurrence_support"])
                          and not item["supported_occurrences"] for item in items)}
    return {"annotations": annotations, "token_distributions": distributions, "token_annotation_counts": denominators,
            "supervised_source_tokens": sum(value > 0 for block in denominators.values() for value in block),
            "support_counts": support_counts}


def tokens(text):
    return Counter(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def token_f1(predicted, native):
    p, g = Counter(), Counter()
    for text in predicted:
        p.update(tokens(text))
    for text in native:
        g.update(tokens(text))
    common = sum((p & g).values())
    return 2 * common / (sum(p.values()) + sum(g.values())) if p or g else 0.0


def score_row(record, prediction):
    target, endpoint = record["target"], record["endpoint"]
    output = {"id": record["id"], "component_id": record["component_id"], "endpoint": endpoint,
              "row_count": 1, "status": prediction["status"], "native_items": 0, "annotations": 0,
              "missing_annotations": 0, "null_annotations": 0, "empty_annotations": 0}
    available = prediction["status"] == "OK"
    if endpoint in BOOL + INTENT:
        gold = target["distribution"]
        if gold is None:
            output.update(target_unavailable=1, accuracy=[0, 0], nll=[0, 0], raw_nll=[0, 0], brier=[0, 0], raw_brier=[0, 0])
            return output
        predicted = prediction["probabilities"] if available else None
        raw = prediction["raw_probabilities"] if available else None
        ids = prediction["candidate_ids"]
        gold_index = max(range(len(gold)), key=lambda index: (gold[index], -index))
        guess = ids.index(ranking(ids, predicted)[0]) if available else None
        output.update(accuracy=[float(guess == gold_index), 1],
                      observed_raters=target.get("observed_raters", 1), target_unavailable=0)
        if available:
            output.update(nll=[-math.fsum(x * math.log(max(y, 1e-12)) for x, y in zip(gold, predicted)), 1],
                          raw_nll=[-math.fsum(x * math.log(max(y, 1e-12)) for x, y in zip(gold, raw)), 1],
                          brier=[math.fsum((x - y) ** 2 for x, y in zip(gold, predicted)), 1],
                          raw_brier=[math.fsum((x - y) ** 2 for x, y in zip(gold, raw)), 1],
                          confidence=max(predicted), raw_confidence=max(raw), correct=float(guess == gold_index))
        return output
    ratings = target["ratings"]
    output.update(native_items=target["native_item_count"], unmatched_items=target["unmatched_item_count"],
                  support_stratum=target["support_stratum"], annotations=len(ratings),
                  missing_annotations=sum(not a["present"] for a in ratings),
                  null_annotations=sum(a["null"] for a in ratings),
                  empty_annotations=sum(a["observed"] and not a["items"] for a in ratings))
    output["per_annotation"] = {}
    for annotation in ratings:
        native = annotation["items"]
        item_count = len(native)
        unmatched = sum(not item["occurrences"] for item in native)
        annotation_metric = {"observed": annotation["observed"], "native_items": item_count, "unmatched_items": unmatched,
                             "support_stratum": "no_native_items" if not item_count else "none" if unmatched == item_count else
                                                "mixed" if unmatched else "full"}
        if item_count:
            if endpoint == RETRIEVAL:
                for k in (1, 5, 10):
                    selected = prediction["ranking"][:k] if available else []
                    annotation_metric["recall@" + str(k)] = ratio(sum(any(x["block_id"] in selected for x in item["occurrences"])
                                                                     for item in native), item_count)
            elif annotation["observed"]:
                strings = [span["text"] for span in prediction["spans"]] if available else []
                annotation_metric["f1"] = token_f1(strings, [item["text"] for item in native]) if available else 0
                annotation_metric["exact_set_match"] = int(set(strings) == {item["text"] for item in native}) if available else 0
        output["per_annotation"][str(annotation["annotation_ordinal"])] = annotation_metric
    if endpoint == RETRIEVAL:
        for k in (1, 5, 10):
            picked = set(prediction["ranking"][:k]) if available else set()
            hits = sum(any(occurrence["block_id"] in picked for occurrence in item["occurrences"])
                       for a in ratings for item in a["items"])
            output["recall@" + str(k)] = [hits, output["native_items"]]
        gold = categorical_target(record)
        if gold is not None and available:
            predicted = prediction["probabilities"]
            output["nll"] = [-math.fsum(x * math.log(max(y, 1e-12)) for x, y in zip(gold, predicted)), 1]
            output["brier"] = [math.fsum((x - y) ** 2 for x, y in zip(gold, predicted)), 1]
            raw = prediction["raw_probabilities"]
            output["raw_nll"] = [-math.fsum(x * math.log(max(y, 1e-12)) for x, y in zip(gold, raw)), 1]
            output["raw_brier"] = [math.fsum((x - y) ** 2 for x, y in zip(gold, raw)), 1]
        output["matched_target_rows"] = int(gold is not None)
    else:
        strings = [span["text"] for span in prediction["spans"]] if available else []
        observed = [a for a in ratings if a["observed"] and a["items"]]
        output["f1"] = [sum(token_f1(strings, [x["text"] for x in a["items"]]) if available else 0 for a in observed), len(observed)]
        output["exact_set_match"] = [sum(set(strings) == {x["text"] for x in a["items"]} if available else 0 for a in observed), len(observed)]
        output["exact_item_recall"] = [sum(item["text"] in strings for a in ratings for item in a["items"]), output["native_items"]]
        output["position_coverage"] = [sum(any(span["block_id"] == occurrence["block_id"] and
                                                span["start"] <= occurrence["start"] and span["end"] >= occurrence["end"]
                                                for span in prediction["spans"])
                                           for a in ratings for item in a["items"] for occurrence in item["occurrences"]),
                                       sum(len(item["occurrences"]) for a in ratings for item in a["items"])]
    return output


def calibration_metrics(confidences, correctness):
    require(len(confidences) == len(correctness), "Reliability dimensions")
    count = len(confidences)
    if not count:
        return {"ECE15": None, "adaptive_ECE15": None, "confidence_AUROC": None,
                "selective": [{"threshold": threshold, "selected": 0, "coverage": None, "risk": None} for threshold in THRESHOLDS]}
    pairs = [(finite(c), finite(y)) for c, y in zip(confidences, correctness)]
    bins = [[] for _ in range(15)]
    for confidence, correct in pairs:
        require(0 <= confidence <= 1 and correct in (0, 1), "Reliability range")
        bins[min(int(confidence * 15), 14)].append((confidence, correct))
    def ece(groups):
        return sum(abs(sum(c - y for c, y in group)) / count for group in groups if group)
    ordered = sorted(enumerate(pairs), key=lambda item: (item[1][0], item[0]))
    adaptive = [[pair for _, pair in ordered[index * count // 15:(index + 1) * count // 15]] for index in range(15)]
    positives = sum(y for _, y in pairs)
    negatives = count - positives
    auc = None
    if positives and negatives:
        wins, lower = 0.0, 0
        by_confidence = defaultdict(lambda: [0, 0])
        for confidence, correct in pairs:
            by_confidence[confidence][int(correct)] += 1
        for confidence in sorted(by_confidence):
            n, p = by_confidence[confidence]
            wins += p * (lower + n / 2)
            lower += n
        auc = wins / (positives * negatives)
    selective = []
    for threshold in THRESHOLDS:
        selected = [y for c, y in pairs if c >= threshold]
        selective.append({"threshold": threshold, "selected": len(selected), "coverage": len(selected) / count,
                          "risk": 1 - sum(selected) / len(selected) if selected else None})
    return {"ECE15": ece(bins), "adaptive_ECE15": ece(adaptive), "confidence_AUROC": auc, "selective": selective}


def summarize(samples):
    pairs = defaultdict(lambda: [0.0, 0])
    numerators = defaultdict(list)
    counts = Counter()
    strata = Counter()
    confidences, raw_confidences, correctness = [], [], []
    for sample in samples:
        counts["rows"] += 1
        counts[sample["status"]] += 1
        for key, value in sample.items():
            if isinstance(value, list) and len(value) == 2:
                if key == "f1" and "per_annotation" in sample:
                    numerators[key].extend(annotation["f1"] for annotation in sample["per_annotation"].values() if "f1" in annotation)
                else:
                    numerators[key].append(finite(value[0]))
                pairs[key][1] += integer(value[1])
            elif key in ("native_items", "unmatched_items", "annotations", "missing_annotations", "null_annotations",
                         "empty_annotations", "target_unavailable", "observed_raters", "matched_target_rows"):
                counts[key] += integer(value)
        if "support_stratum" in sample:
            strata[sample["support_stratum"]] += 1
        if "confidence" in sample:
            confidences.append(sample["confidence"])
            raw_confidences.append(sample["raw_confidence"])
            correctness.append(sample["correct"])
    for key, values in numerators.items():
        pairs[key][0] = math.fsum(values)
    return {"counts": dict(counts), "support_strata": dict(strata),
            "metrics": {key: {"numerator": value[0], "denominator": value[1], "value": ratio(*value)} for key, value in sorted(pairs.items())},
            "calibrated_reliability": calibration_metrics(confidences, correctness),
            "raw_reliability": calibration_metrics(raw_confidences, correctness)}


def paired_interval(left, right, metric, universe, resamples=10000, seed=0):
    """Resample frozen components, then divide paired sums by descendant counts."""
    require(set(left) == set(right), "Paired row membership differs")
    components = sorted(set(universe))
    require(components and len(components) == len(universe), "Frozen component universe is empty/duplicated")
    totals = {component: [0.0, 0] for component in components}
    for identity in left:
        a, b = left[identity], right[identity]
        require(a["component_id"] == b["component_id"] and a["component_id"] in totals, "Paired component mismatch")
        av, bv = a.get(metric), b.get(metric)
        require(av is not None and bv is not None and av[1] == bv[1], "Missing/mismatched paired denominator")
        totals[a["component_id"]][0] += finite(av[0]) - finite(bv[0])
        totals[a["component_id"]][1] += integer(av[1])
    total_denominator = sum(value[1] for value in totals.values())
    total_numerator = math.fsum(value[0] for value in totals.values())
    if not total_denominator:
        return {"status": "UNMEASURED", "estimate": None, "ci95": None, "components": len(components), "denominator": 0,
                "resamples": resamples, "seed": seed}
    values = [(totals[component][0], totals[component][1]) for component in components]
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        numerator = denominator = 0.0
        for _ in range(len(components)):
            delta, count = values[rng.randrange(len(components))]
            numerator += delta
            denominator += count
        require(denominator > 0, "Zero-denominator component bootstrap draw")
        draws.append(numerator / denominator)
    ordered = sorted(draws)
    bounds = [ordered[int(math.floor(0.025 * (len(ordered) - 1)))],
              ordered[int(math.ceil(0.975 * (len(ordered) - 1)))]]
    require(bounds[0] <= bounds[1], "Reversed bootstrap interval")
    return {"status": "MEASURED", "estimate": total_numerator / total_denominator,
            "ci95": bounds, "components": len(components), "denominator": total_denominator, "resamples": resamples, "seed": seed}


def interval_pass(receipt, threshold=0, strict=True):
    if not isinstance(receipt, dict) or receipt.get("status") != "MEASURED" or type(receipt.get("denominator")) is not int or receipt["denominator"] <= 0:
        return False
    estimate = receipt.get("estimate")
    if type(estimate) not in (int, float) or not math.isfinite(estimate):
        return False
    interval = receipt.get("ci95")
    if not isinstance(interval, list) or len(interval) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) for x in interval):
        return False
    if interval[0] > interval[1]:
        return False
    return interval[0] > threshold if strict else interval[0] >= threshold


def authority():
    cfg = load_json(AMENDMENT)
    require(cfg["schema"] == "vey.neutral.qasper.cross-study-amendment.v1", "Scientific authority schema")
    pins = {**cfg["authority"], **cfg["current_authority"],
            "historical_native_arch_protocol": cfg["historical_native_arch_protocol"]}
    for entry in pins.values():
        checked(entry)
    historical = pins["historical_native_arch_protocol"]
    original = load_json(checked(pins["deferred_question_protocol"]))
    for entry in original["authority"].values():
        checked(historical if entry["sha256"] == historical["sha256"] else entry)
    require(cfg["training"]["epochs"] == 10 and cfg["training"]["include_epoch0"] is True and
            cfg["sealed_access"] is False and cfg["promotion"] is False, "Registered authority boundary")
    transport = load_json(HERE / "neutral_qasper_cross_transport_input_amendment.json")
    checked(transport["scientific_amendment"])
    require(transport["scientific_changes"] is False and transport["sealed_labels_accessed"] is False,
            "Operational authority changes science")
    return cfg, pins, transport


def reconstruct_sources():
    """Use old guarded readers solely for custody-bound source bytes."""
    cfg, pins, transport = authority()
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import neutral_qasper_native_compile as projection
    import native_field_data as native_reader
    phases = {phase: {} for phase in PHASES}
    derived = {phase: {} for phase in PHASES}
    split_manifest = load_json(checked(pins["question_split_manifest"]))
    components, all_ids = {}, set()
    for outer in ("train", "dev"):
        provenance = member_index(projection.open_phase(outer, "provenance"))
        state_pool, candidate_pool = {}, {}
        def intern_views():
            for view in projection.open_phase(outer, "serving"):
                view["state"] = state_pool.setdefault(view["state"], view["state"])
                key = canonical(view["candidates"])
                view["candidates"] = candidate_pool.setdefault(key, view["candidates"])
                yield view
        serving = member_index(intern_views(), provenance)
        targets = member_index(projection.open_phase(outer, "targets"), provenance)
        decision_ids = set()
        for decision in projection.open_phase(outer, "decisions"):
            require(decision["id"] in provenance and decision["id"] not in decision_ids and decision["split"] == outer,
                    "Original decision source membership/phase")
            decision_ids.add(decision["id"])
        require(decision_ids == set(provenance), "Original decision member completeness")
        counts = Counter(row["endpoint"] for row in provenance.values())
        require(dict(counts) == {e: value["rows"][outer] for e, value in cfg["data"]["question_study"]["endpoints"].items()},
                "Original QASPER outer endpoint population")
        catalogues = {}
        for identity, prov in provenance.items():
            if prov["endpoint"] != RETRIEVAL:
                continue
            blocks = [{"id": c["id"], "text": c["text"]} for c in serving[identity]["candidates"]]
            require(len(blocks) == len({b["id"] for b in blocks}), "Duplicate source catalogue coordinate")
            paper = (prov["group_id"], prov["paper_id"])
            require(catalogues.setdefault(paper, blocks) == blocks, "Same-paper source catalogue drift")
        for identity in sorted(provenance):
            prov, view, ledger = provenance[identity], serving[identity], targets[identity]
            endpoint = prov["endpoint"]
            phase = "dev" if outer == "dev" else split_component(prov["component_id"])
            require(identity not in all_ids and ledger["endpoint"] == endpoint and prov["phase"] == outer,
                    "Original source identity/phase")
            all_ids.add(identity)
            require(components.setdefault(prov["component_id"], phase) == phase, "Cross-phase paper component")
            blocks = catalogues[(prov["group_id"], prov["paper_id"])]
            target = native_target(ledger, blocks, endpoint)
            require(view["id"] == identity and view["state"] and type(view["question"]) is str,
                    "Source-only serving mapping")
            raw = {"id": identity, "endpoint": endpoint, "component_id": prov["component_id"], "phase": phase,
                   "serving": view, "blocks": blocks, "target": ledger,
                   "paper_id": prov["paper_id"], "group_id": prov["group_id"], "question_ordinal": prov["question_ordinal"],
                   "input_sha256": value_digest(view)}
            derived[phase][identity] = raw
            phases[phase][identity] = {**raw, "target": target,
                                      "candidate_ids": [c["id"] for c in view["candidates"]]}
        for phase in (("dev",) if outer == "dev" else PHASES[:3]):
            membership = member_index(json_rows(checked(split_manifest["phases"][phase]["file"])))
            require(set(membership) == set(phases[phase]), "Frozen question membership universe")
            for identity, registered in membership.items():
                require(all(provenance[identity].get(key) == value for key, value in registered.items()),
                        "Frozen question membership fields")
            rows = list(phases[phase].values())
            census = {"rows": len(rows), "components": len({r["component_id"] for r in rows}),
                      "endpoint_counts": dict(Counter(r["endpoint"] for r in rows))}
            require(census == cfg["data"]["question_phase_counts"][phase], "Frozen question phase census")
    native_cfg = load_json(HERE / "native_field_protocol.json")
    native_manifest, native_phases = native_reader.load_dataset(native_cfg["output_root"])
    native_counts = Counter()
    for phase in PHASES:
        for state in native_phases[phase]:
            require(components.setdefault(state["component_id"], phase) == phase, "Cross-study component phase overlap")
            if phase != "dev":
                require(split_component(state["component_id"]) == phase, "Native input component split")
            for decision in state["decisions"]:
                if decision["endpoint"] not in INTENT:
                    continue
                identity, ids = decision["id"], decision["candidate_ids"]
                require(identity not in all_ids and decision["gold"] in ids, "Native decision universe/gold")
                all_ids.add(identity)
                target = [float(cid == decision["gold"]) for cid in ids]
                close(decision["target_distribution"], target, "Native intent target")
                view = {"id": identity, "task": decision["task"], "locale": state["locale"], "state": state["state"],
                        "question": decision["question"],
                        "candidates": [{"id": cid, "text": decision["candidates"][cid]} for cid in ids]}
                raw = {"id": identity, "endpoint": decision["endpoint"], "component_id": state["component_id"],
                       "phase": phase, "serving": view, "blocks": [],
                       "target": {"target_available": True, "distribution": target, "label_id": decision["gold"]},
                       "paper_id": None, "group_id": state["group_id"], "question_ordinal": None,
                       "input_sha256": value_digest(view)}
                derived[phase][identity] = raw
                phases[phase][identity] = {**raw, "target": {"kind": "intent", "distribution": target}, "candidate_ids": ids}
                native_counts[("dev" if phase == "dev" else "train", decision["endpoint"])] += 1
    require(native_counts == Counter({("train", INTENT[0]): 4105, ("dev", INTENT[0]): 1027,
                                     ("train", INTENT[1]): 3971, ("dev", INTENT[1]): 1252}),
            "Complete original intent replay population")
    return cfg, pins, transport, phases, derived, native_manifest


def verify_dataset(root):
    cfg, pins, transport, phases, derived, native_manifest = reconstruct_sources()
    root = Path(root)
    require(not (root / "failure.json").exists(), "Failed derived dataset")
    manifest = load_json(root / "dataset_manifest.json")
    require(manifest["schema"] == transport["derived_schema"] and set(manifest["phases"]) == set(PHASES),
            "Derived dataset manifest")
    require(manifest["sealed_access"] is False and manifest["model_loads"] == manifest["model_forwards"] == 0,
            "Derived input construction boundary")
    expected_sources = {"neutral_qasper_cross_data.py", "native_field_data.py", "neutral_qasper_native_compile.py"}
    require(set(manifest["construction_sources"]) == expected_sources, "Derived construction source inventory")
    for name, entry in manifest["construction_sources"].items():
        require(checked(entry).resolve() == (HERE / name).resolve(), "Derived current construction source")
    paper_path = checked(manifest["papers"])
    require(paper_path.resolve() == (root / "papers.jsonl.gz").resolve(), "Derived paper rooting")
    papers = member_index(json_rows(paper_path))
    require(list(papers) == sorted(papers), "Derived stable paper ordering")
    for identity, paper in papers.items():
        require(set(paper) == {"id", "state", "blocks"} and identity == value_digest([paper["state"], paper["blocks"]]),
                "Derived deduplicated source-paper identity")
    used_papers = set()
    for phase in PHASES:
        entry = manifest["phases"][phase]
        path = checked(entry["file"] if "file" in entry else entry)
        require(path.resolve() == (root / (phase + ".jsonl.gz")).resolve(), "Derived phase rooting")
        rows = member_index(json_rows(path), derived[phase])
        require(list(rows) == sorted(rows), "Derived stable member ordering")
        for identity, row in rows.items():
            if row["endpoint"] not in INTENT:
                reference = row.pop("paper_ref")
                require(reference in papers and "state" not in row["serving"] and "blocks" not in row, "Derived QASPER paper reference")
                used_papers.add(reference)
                row["serving"]["state"] = papers[reference]["state"]
                row["blocks"] = papers[reference]["blocks"]
            require(set(row) == set(transport["record_fields"]), "Registered hydrated record fields")
            require(row == derived[phase][identity], "Independent derived source row differs: " + identity)
        require(entry["rows"] == len(rows) and entry["endpoint_counts"] == dict(Counter(r["endpoint"] for r in rows.values()))
                and entry["components"] == len({r["component_id"] for r in rows.values()}), "Derived phase denominators")
    require(used_papers == set(papers) and manifest["papers"]["rows"] == len(papers), "Complete derived paper universe")
    require(manifest["source_pins"] == pins and manifest["protocol_sha256"] == digest(AMENDMENT) and
            manifest["transport_protocol_sha256"] == digest(HERE / "neutral_qasper_cross_transport_input_amendment.json"),
            "Derived original scientific/transport custody")
    return {"schema": "vey.qnative.cross.dataset-verification.v1", "status": "PASS",
            "manifest_sha256": digest(root / "dataset_manifest.json"), "protocol_sha256": digest(AMENDMENT),
            "manifest": {"path": str(root / "dataset_manifest.json"), "bytes": (root / "dataset_manifest.json").stat().st_size,
                         "sha256": digest(root / "dataset_manifest.json")},
            "transport_protocol_sha256": digest(HERE / "neutral_qasper_cross_transport_input_amendment.json"),
            "phases": manifest["phases"], "source_pins": pins, "all_rows_independently_reconstructed": True,
            "sealed_labels_accessed": False, "model_loads": 0, "model_forwards": 0, "quality_credit": False}


def donor_plan(records):
    groups = defaultdict(list)
    for record in records.values():
        if record["endpoint"] not in INTENT:
            groups[(record["group_id"], record["paper_id"], record["endpoint"])].append(record)
    donors = {}
    for group in groups.values():
        group.sort(key=lambda row: row["id"])
        for index, record in enumerate(group):
            donor = next((group[(index + offset) % len(group)] for offset in range(1, len(group))
                          if group[(index + offset) % len(group)]["serving"]["question"] != record["serving"]["question"]), None)
            donors[record["id"]] = donor["id"] if donor is not None else None
    return donors


def changed_record(record, condition, donor=None, subset=None):
    view = {**record["serving"]}
    blocks = [dict(block) for block in record["blocks"]]
    if condition == "question_masked":
        view["question"] = ""
    elif condition == "state_masked":
        view["state"] = EMPTY_STATE
        blocks = [{**block, "text": EMPTY_STATE} for block in blocks]
        if record["endpoint"] == RETRIEVAL:
            view["candidates"] = [{"id": block["id"], "text": block["text"]} for block in blocks]
    elif condition == "question_swapped":
        require(donor is not None, "Missing eligible donor")
        view["question"] = donor["serving"]["question"]
    elif condition == "reordered":
        blocks.reverse()
        if record["endpoint"] == RETRIEVAL:
            view["candidates"] = list(reversed(view["candidates"]))
    elif condition == "candidate_count":
        require(subset is not None, "Missing prefix catalogue")
        blocks = [block for block in blocks if block["id"] in subset]
        if record["endpoint"] == RETRIEVAL:
            view["candidates"] = [{"id": block["id"], "text": block["text"]} for block in blocks]
    return {**record, "serving": view, "blocks": blocks, "input_sha256": value_digest(view),
            "candidate_ids": [c["id"] for c in view["candidates"]]}


def nested_prefix(ids, component, k):
    require(type(k) is int and k > 0 and len(ids) == len(set(ids)), "Prefix catalogue dimensions")
    ordered = sorted(ids, key=lambda cid: (hashlib.sha256(("qnative-cross-v1|7|" + component + "|" + cid).encode()).hexdigest(), cid))
    return ordered[:min(k, len(ids))]


def validate_windows(record, prediction):
    blocks, windows, coverage = record["blocks"], prediction["windows"], prediction["coverage"]
    if record["endpoint"] in INTENT:
        require(windows == [] and coverage == [] and prediction["source_token_offsets"] == {}, "Intent window payload")
        return
    offsets = prediction["source_token_offsets"]
    ids = [b["id"] for b in blocks]
    require(set(offsets) == set(ids) and len(coverage) == len(blocks), "Full source coverage universe")
    census_by_id = {row["block_id"]: row for row in coverage}
    require(set(census_by_id) == set(ids), "Coverage census source membership")
    question_ids = set()
    grouped = defaultdict(list)
    qcounts = set()
    for window in windows:
        bid = window["block_id"]
        require(bid in offsets, "Foreign source window")
        grouped[bid].append(window)
        input_ids, attention = window["input_ids"], window["attention_mask"]
        require(isinstance(input_ids, list) and 0 < len(input_ids) <= 512 and
                all(type(x) is int for x in input_ids) and attention == [1] * len(input_ids), "Exact input/mask trace")
        require(window["input_sha256"] == value_digest([input_ids, attention]), "Window input IDs/mask digest")
        qcount, special, capacity = (integer(window[key]) for key in
                                    ("question_token_count", "pair_special_tokens", "source_capacity"))
        require(capacity > 0 and capacity == 512 - qcount - special, "Untruncated question source capacity")
        qcounts.add(qcount)
        qids, qpositions = window["question_input_ids"], window["question_positions"]
        require(len(qids) == qcount and all(type(token) is int for token in qids) and
                len(qpositions) == qcount and qpositions == sorted(set(qpositions)) and
                all(type(position) is int and 0 <= position < len(input_ids) for position in qpositions),
                "Complete question token coordinates")
        require([input_ids[position] for position in qpositions] == qids, "Untruncated question tokens in joint input")
        question_ids.add(tuple(qids))
        indices = window["source_token_indices"]
        positions = window["source_positions"]
        require(len(indices) == len(positions) == len(window["offsets"]) and
                len(input_ids) == qcount + special + len(indices), "Question/special/source layout")
        require(positions == sorted(set(positions)) and all(type(x) is int and 0 <= x < len(input_ids) for x in positions),
                "Source-only decode positions")
        require(window["offsets"] == [offsets[bid][index] for index in indices], "Source token offset projection")
        require(not set(qpositions).intersection(positions), "Question tokens cannot be source BIO tokens")
        source_ids = census_by_id[bid]["source_token_ids"]
        require(len(source_ids) == len(offsets[bid]) and all(type(token) is int for token in source_ids), "Complete source token ID census")
        require([input_ids[position] for position in positions] == [source_ids[index] for index in indices],
                "Joint input source token IDs match whole-source census")
    require(len(qcounts) == 1, "Question token count differs across source windows")
    require(len(question_ids) == 1, "Question tokens drift across source windows")
    for position, (block, census) in enumerate(zip(blocks, coverage)):
        bid = block["id"]
        require(census["block_id"] == bid and census["block_position"] == position and
                census["source_token_offsets"] == offsets[bid], "Source coverage coordinate")
        previous = -1
        for start, end in offsets[bid]:
            require(type(start) is int and type(end) is int and 0 <= start <= end <= len(block["text"]) and start >= previous,
                    "Source character offset census")
            previous = start
        planned = grouped[bid]
        if condition == "state_masked":
            require(offsets[bid] in ([[0, len(EMPTY_STATE)]], []), "Empty-evidence mask exposes no verbatim source coordinates")
        require(planned and census["window_count"] == len(planned) and
                census["source_token_count"] == census["covered_source_tokens"] == len(offsets[bid]), "Full source window/token census")
        start = 0
        covered = set()
        for index, window in enumerate(planned):
            capacity = window["source_capacity"]
            end = min(start + capacity, len(offsets[bid]))
            require(window["window_index"] == index and window["block_position"] == position and
                    window["source_token_indices"] == list(range(start, end)) and
                    window["overlap"] == min(64, capacity - 1), "Deterministic complete overlapping windows")
            covered.update(window["source_token_indices"])
            require(end != len(offsets[bid]) or index == len(planned) - 1, "Extra post-coverage window")
            start += capacity - min(64, capacity - 1)
        require(covered == set(range(len(offsets[bid]))), "Unrepresented source tokens")


def validate_prediction(saved, record, condition, temperature, checkpoint, baseline=False):
    for key in ("id", "endpoint", "component_id", "phase"):
        require(saved.get(key) == record[key], "Prediction native " + key)
    require(saved["kind"] == "prediction" and saved["condition"] == condition and
            saved["input_sha256"] == value_digest(record["serving"]), "Prediction condition/source input")
    require(saved["status"] in ("OK", "UNSUPPORTED", "ERROR"), "Prediction status")
    if saved["status"] != "OK":
        require(type(saved["error"]) is str and saved["error"] and
                saved["raw_logits"] == saved["raw_probabilities"] == saved["probabilities"] == [],
                "Failure must retain explicit reason and no fabricated neural values")
        return saved
    require(saved["error"] is None, "Successful prediction has error")
    endpoint = record["endpoint"]
    ids = [b["id"] for b in record["blocks"]] if endpoint == RETRIEVAL else record["candidate_ids"] if endpoint != EXTRACTION else []
    require(saved["candidate_ids"] == ids, "Complete candidate coordinate universe")
    if endpoint != EXTRACTION:
        require(len(saved["raw_logits"]) == len(ids), "Full raw categorical logits")
        close(saved["raw_probabilities"], softmax(saved["raw_logits"]), "Raw softmax")
        close(saved["temperature"], temperature, "Calibration temperature", 1e-12)
        close(saved["probabilities"], softmax(saved["raw_logits"], temperature), "Calibrated softmax")
        require(saved["ranking"] == ranking(ids, saved["probabilities"]) and saved["spans"] == [], "Stable categorical output")
    else:
        require(saved["raw_logits"] == saved["raw_probabilities"] == saved["probabilities"] == saved["ranking"] == [],
                "Extraction is not a string probability field")
    if baseline:
        return saved
    require(saved["weight_identity"] == checkpoint and
            saved["cache_key"] == value_digest([checkpoint, record["serving"], record["blocks"]]), "Input/weight cache identity")
    validate_windows(record, saved, condition)
    for window in saved["windows"]:
        finite(window["relevance_logit"], "window relevance")
        require(len(window["bool_logits"]) == 2 and all(math.isfinite(finite(x)) for x in window["bool_logits"]),
                "Raw window Bool class trace")
        require(len(window["bio_logits"]) == len(window["source_token_indices"]) and
                all(len(logits) == 3 and all(math.isfinite(finite(x)) for x in logits) for logits in window["bio_logits"]),
                "Raw source-only BIO class trace")
    if endpoint == EXTRACTION and condition == "state_masked":
        require(saved["spans"] == [], "Masked-evidence extraction must not predict source spans")
        return saved
    if endpoint == EXTRACTION:
        semantic_blocks = list(reversed(record["blocks"])) if condition == "reordered" else record["blocks"]
        close(saved["spans"], decode_bio(semantic_blocks, saved["windows"]), "Independent BIO/source decode")
        if condition in ("dev", "selection", "calibration", "reordered", "question_masked", "question_swapped"):
            close(saved["support"], supervision(record["target"], record["blocks"], saved["source_token_offsets"], saved["windows"]),
                  "Independent partial BIO supervision")
    elif endpoint == RETRIEVAL:
        scores = {block["id"]: max(finite(w["relevance_logit"]) for w in saved["windows"] if w["block_id"] == block["id"])
                  for block in record["blocks"]}
        close(saved["raw_logits"], [scores[identity] for identity in ids], "Block max window relevance")
    elif endpoint in BOOL:
        weights = softmax([w["relevance_logit"] for w in saved["windows"]])
        logits = [sum(weight * finite(w["bool_logits"][index]) for weight, w in zip(weights, saved["windows"])) for index in range(2)]
        close(saved["raw_logits"], logits, "Relevance-weighted joint Bool logits", 2e-5)
    counters = saved["runtime_counters"]
    for value in counters.values():
        integer(value, "runtime counter")
    return saved


def capture(path, natives, condition, pins, checkpoint, temperatures, plans=None, baseline=False, weight=None):
    rows = json_rows(path)
    header = next(rows, None)
    require(header is not None and header.get("kind") == "header" and
            header["schema"] == "vey.qnative.cross.capture.v1" and header["condition"] == condition and
            header["protocol_sha256"] == digest(AMENDMENT) and header["source_pins"] == pins and
            header["checkpoint_sha256"] == checkpoint, "Capture header custody")
    require(natives and {row["phase"] for row in natives.values()} == {header["phase"]}, "Capture phase custody")
    result, samples = {}, {}
    for saved in rows:
        identity = saved["id"]
        require(identity in natives and identity not in result, "Capture member universe")
        native = natives[identity]
        record = native
        if condition == "question_swapped":
            donor_id = plans[identity]
            require(saved.get("donor_id") == donor_id, "Registered same-paper donor")
            if donor_id is None:
                require(saved["status"] == "UNSUPPORTED" and saved["error"], "Unpaired swap row is not successful")
            else:
                record = changed_record(native, condition, natives[donor_id])
        elif condition in ("question_masked", "state_masked", "reordered"):
            record = changed_record(native, condition)
        temperature = temperatures.get(native["endpoint"], 1.0)
        validate_prediction(saved, record, condition, temperature, weight or checkpoint, baseline)
        if native["endpoint"] == EXTRACTION and condition == "state_masked" and saved["status"] == "OK":
            require(saved["spans"] == [], "Masked-evidence extraction must not predict source spans")
        if native["endpoint"] == EXTRACTION and saved["status"] == "OK" and condition != "state_masked":
            saved["_bio_reliability"] = bio_reliability(native, saved)
        saved["_window_count"] = len(saved.get("windows", []))
        saved["_block_scores"] = {block["id"]: max(finite(window["relevance_logit"]) for window in saved.get("windows", [])
            if window["block_id"] == block["id"]) for block in record["blocks"]} if saved["status"] == "OK" else {}
        for bulky in ("windows", "coverage", "source_token_offsets", "support"):
            saved.pop(bulky, None)
        result[identity] = saved
        samples[identity] = score_row(native, saved)
    require(set(result) == set(natives), "Missing registered capture members")
    return result, samples


def checkpoint_fingerprint(path):
    types = {"F32": ("torch.float32", 4), "I64": ("torch.int64", 8)}
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "Truncated checkpoint")
        length = int.from_bytes(prefix, "little")
        require(0 < length < min(path.stat().st_size - 8, 100_000_000), "Checkpoint header length")
        header = decode(stream.read(length).decode("utf-8"))
        tensors = {name: item for name, item in header.items() if name != "__metadata__"}
        require(tensors, "Empty checkpoint tensor universe")
        payload = path.stat().st_size - 8 - length
        ranges, fingerprints, model_hash = [], {}, hashlib.sha256()
        subgroup_hashes = {"encoder": hashlib.sha256(), "cross": hashlib.sha256()}
        for name in sorted(tensors):
            tensor = tensors[name]
            require(set(tensor) == {"dtype", "shape", "data_offsets"} and tensor["dtype"] in types, "Checkpoint tensor schema/dtype")
            shape, offsets = tensor["shape"], tensor["data_offsets"]
            require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), "Tensor shape")
            require(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets), "Tensor offsets")
            start, end = offsets
            require(0 <= start <= end <= payload and end - start == math.prod(shape) * types[tensor["dtype"]][1], "Tensor byte extent")
            ranges.append((start, end))
            rendered = canonical([name, types[tensor["dtype"]][0], shape]).encode("utf-8")
            model_hash.update(len(rendered).to_bytes(8, "little"))
            model_hash.update(rendered)
            active = []
            for group, group_prefix in (("encoder", "cross.scorer.pool.encoder."), ("cross", "cross.")):
                if name.startswith(group_prefix):
                    rendered_group = canonical([name[len(group_prefix):], types[tensor["dtype"]][0], shape]).encode("utf-8")
                    subgroup_hashes[group].update(len(rendered_group).to_bytes(8, "little"))
                    subgroup_hashes[group].update(rendered_group)
                    active.append(subgroup_hashes[group])
            tensor_hash = hashlib.sha256()
            stream.seek(8 + length + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(remaining, 1 << 20))
                require(block, "Truncated checkpoint tensor payload")
                model_hash.update(block)
                tensor_hash.update(block)
                for group_hash in active:
                    group_hash.update(block)
                remaining -= len(block)
            fingerprints[name] = {"shape": shape, "dtype": tensor["dtype"], "sha256": tensor_hash.hexdigest()}
        previous = 0
        for start, end in sorted(ranges):
            require(start == previous, "Checkpoint payload gap/overlap")
            previous = end
        require(previous == payload, "Unaccounted checkpoint bytes")
        result = model_hash.hexdigest()
        if "__metadata__" in header and "state_dict_sha256" in header["__metadata__"]:
            require(header["__metadata__"]["state_dict_sha256"] == result, "Tensor fingerprint metadata mismatch")
    return {"state_dict_fingerprint": result, "tensors": fingerprints, "sha256": digest(path),
            "encoder_state_dict_fingerprint": subgroup_hashes["encoder"].hexdigest(),
            "cross_state_dict_fingerprint": subgroup_hashes["cross"].hexdigest()}


def tfidf_fit(records):
    documents = sorted({block["text"] for record in records.values() for block in record["blocks"]})
    require(documents, "Empty FIT lexical corpus")
    df = Counter()
    for document in documents:
        df.update(tokens(document).keys())
    return {term: math.log((len(documents) + 1) / (count + 1)) + 1 for term, count in df.items()}, len(documents)


def tfidf_scores(question, blocks, idf, document_count):
    def vector(text):
        values = {term: count * idf.get(term, math.log(document_count + 1) + 1) for term, count in tokens(text).items()}
        length = math.sqrt(sum(x * x for x in values.values()))
        return {term: value / length for term, value in values.items()} if length else {}
    query = vector(question)
    return [sum(value * query.get(term, 0) for term, value in vector(block["text"]).items()) for block in blocks]


def fit_priors(records):
    result = {}
    for endpoint in BOOL:
        masses = [record["target"]["distribution"] for record in records.values()
                  if record["endpoint"] == endpoint and record["target"]["distribution"] is not None]
        require(masses, "Missing FIT native Bool prior support")
        result[endpoint] = [sum(row[index] for row in masses) / len(masses) for index in range(2)]
    return result


def baseline_expected(record, priors, idf, document_count):
    endpoint = record["endpoint"]
    if endpoint in BOOL:
        probs = priors[endpoint]
        return {"probabilities": probs, "raw_probabilities": probs, "candidate_ids": ["false", "true"],
                "ranking": ranking(["false", "true"], probs), "spans": []}
    scores = tfidf_scores(record["serving"]["question"], record["blocks"], idf, document_count)
    ids = [block["id"] for block in record["blocks"]]
    ordered = ranking(ids, scores)
    if endpoint == RETRIEVAL:
        return {"raw_logits": scores, "raw_probabilities": softmax(scores), "probabilities": softmax(scores),
                "candidate_ids": ids, "ranking": ordered, "spans": []}
    require(endpoint == EXTRACTION, "Unknown baseline endpoint")
    text = next(block["text"] for block in record["blocks"] if block["id"] == ordered[0])
    return {"candidate_ids": [], "ranking": ordered, "probabilities": [], "raw_probabilities": [], "raw_logits": [],
            "spans": [{"block_id": ordered[0], "start": 0, "end": len(text), "text": text, "confidence": None}]}


def reconstruct_baselines(path, records, pins, checkpoint, priors, idf, document_count, original):
    rows = json_rows(path)
    header = next(rows, None)
    require(header and header["kind"] == "header" and header["protocol_sha256"] == digest(AMENDMENT) and
            header["source_pins"] == pins and header["checkpoint_sha256"] == checkpoint, "Baseline header custody")
    predictions, samples = {}, {}
    for saved in rows:
        identity = saved["id"]
        require(identity in records and identity not in predictions, "Baseline member universe")
        record = records[identity]
        require(saved["endpoint"] == record["endpoint"] and saved["component_id"] == record["component_id"] and
                saved["phase"] == "dev" and saved["input_sha256"] == record["input_sha256"] and saved["status"] == "OK",
                "Baseline source identity/status")
        if record["endpoint"] in INTENT:
            reference = original[identity]
            expected = {key: reference[key] for key in ("candidate_ids", "raw_logits", "raw_probabilities", "probabilities", "ranking", "spans")}
        else:
            expected = baseline_expected(record, priors, idf, document_count)
        for key, value in expected.items():
            close(saved[key], value, "Independent FIT-only lexical/prior " + key)
        predictions[identity] = saved
        samples[identity] = score_row(record, saved)
    require(set(predictions) == set(records), "Incomplete baseline universe")
    return predictions, samples


def metric_value(summary, name):
    if not isinstance(summary, dict) or not isinstance(summary.get("metrics"), dict):
        return None
    item = summary["metrics"].get(name)
    if not isinstance(item, dict) or type(item.get("denominator")) is not int or item["denominator"] <= 0:
        return None
    value = item.get("value")
    return value if type(value) in (int, float) and math.isfinite(value) else None


def selection_key(samples):
    summaries = {endpoint: summarize([row for row in samples.values() if row["endpoint"] == endpoint]) for endpoint in ENDPOINTS}
    primaries = [metric_value(summaries[endpoint], "accuracy" if endpoint in BOOL + INTENT else "recall@10" if endpoint == RETRIEVAL else "f1")
                 for endpoint in ENDPOINTS]
    require(all(value is not None for value in primaries), "Missing inner selection primary denominator")
    nll = [metric_value(summaries[endpoint], "raw_nll") for endpoint in BOOL + INTENT]
    require(all(value is not None for value in nll), "Missing inner selection NLL denominator")
    return math.fsum(primaries) / len(primaries), math.fsum(nll) / len(nll), summaries


def reconstruct_selection(path, records, pins, history):
    rows = json_rows(path)
    header = next(rows, None)
    require(header and header["kind"] == "header" and header["protocol_sha256"] == digest(AMENDMENT) and
            header["source_pins"] == pins and header["phase"] == "selection", "Selection header custody")
    samples = {epoch: {} for epoch in range(11)}
    probe_rows = {}
    probe_ids = {min(identity for identity, row in records.items() if row["endpoint"] == endpoint) for endpoint in ENDPOINTS}
    for row in rows:
        epoch, identity = row["epoch"], row["id"]
        require(type(epoch) is int and epoch in samples and identity in records and identity not in samples[epoch],
                "Selection epoch/member universe")
        checkpoint = row["weight_identity"]
        require(checkpoint == history["epochs"][epoch]["state_dict_fingerprint"], "Epoch selection tensor identity")
        validate_prediction(row, records[identity], "selection", 1.0, checkpoint)
        require(row["status"] == "OK", "Failed selection prediction")
        samples[epoch][identity] = score_row(records[identity], row)
        if identity in probe_ids:
            probe_rows[(epoch, identity)] = {key: row[key] for key in ("raw_logits", "raw_probabilities", "spans")}
    metrics = []
    for epoch in range(11):
        require(set(samples[epoch]) == set(records), "Missing epoch selection rows")
        macro, nll, summaries = selection_key(samples[epoch])
        metrics.append({"epoch": epoch, "macro_primary": macro, "mean_nll": nll, "endpoints": summaries})
    chosen = max(metrics, key=lambda item: (item["macro_primary"], -item["mean_nll"], -item["epoch"]))["epoch"]
    return chosen, metrics, probe_rows


def reconstruct_calibration(path, records, pins, checkpoint, declared, weight):
    predictions, samples = capture(path, records, "calibration", pins, checkpoint, {e: 1.0 for e in ENDPOINTS}, weight=weight)
    result = {}
    grid = [math.exp(math.log(0.5) + index * (math.log(10) - math.log(0.5)) / 100) for index in range(101)]
    for endpoint in ENDPOINTS:
        if endpoint == EXTRACTION:
            continue
        eligible = [(predictions[identity]["raw_logits"], categorical_target(record)) for identity, record in records.items()
                    if record["endpoint"] == endpoint and categorical_target(record) is not None]
        require(eligible, "No calibration target denominator: " + endpoint)
        losses = []
        for temperature in grid:
            nll = math.fsum(-math.fsum(target * math.log(max(mass, 1e-12)) for target, mass in zip(gold, softmax(logits, temperature)))
                           for logits, gold in eligible) / len(eligible)
            losses.append((nll, abs(temperature - 1), temperature))
        nll, _, temperature = min(losses)
        close(declared[endpoint], temperature, "Disjoint calibration grid optimum", 1e-12)
        result[endpoint] = {"temperature": temperature, "NLL": nll, "rows": len(eligible), "grid_points": 101,
                            "grid": [{"temperature": candidate, "nll": loss} for loss, _, candidate in losses],
                            "selected": {"temperature": temperature, "nll": nll}}
    require(set(declared) == set(result), "Registered categorical temperature endpoint inventory")
    extraction = [predictions[identity] for identity, r in records.items() if r["endpoint"] == EXTRACTION]
    confidence = [value for row in extraction for value in row["_bio_reliability"]["confidences"]]
    correct = [value for row in extraction for value in row["_bio_reliability"]["correctness"]]
    result[EXTRACTION] = {"rows": len(extraction), "raw_BIO_reliability": calibration_metrics(confidence, correct),
                          "unique_supported_tokens": len(confidence), "string_probability_certificate": False}
    return result


def compare_reorder(real, reordered, records):
    for identity, original in real.items():
        other = reordered[identity]
        require(original["status"] == other["status"] == "OK", "Failed stable-order control")
        left = dict(zip(original["candidate_ids"], original["raw_logits"]))
        right = dict(zip(other["candidate_ids"], other["raw_logits"]))
        close(left, right, "Reorder score-by-ID parity", 1e-6)
        require(original["ranking"] == other["ranking"], "Reorder stable semantic ranking")
        if records[identity]["endpoint"] == EXTRACTION:
            a = sorted(original["spans"], key=lambda x: (x["block_id"], x["start"], x["end"]))
            b = sorted(other["spans"], key=lambda x: (x["block_id"], x["start"], x["end"]))
            close(a, b, "Reorder semantic source spans", 1e-6)
        close(original["_block_scores"], other["_block_scores"], "Reorder all-endpoint block score-by-ID parity", 1e-6)
    return {"status": "PASS", "rows": len(real), "raw_score_by_ID_tolerance": 1e-6}


def swap_results(records, swapped, plans):
    result = {}
    for endpoint in BOOL + (RETRIEVAL, EXTRACTION):
        counts = Counter()
        correct_new, paired = [], []
        for identity, record in records.items():
            if record["endpoint"] != endpoint:
                continue
            donor_id = plans[identity]
            counts["rows"] += 1
            if donor_id is None:
                counts["unpaired"] += 1
                continue
            donor = records[donor_id]
            if endpoint in BOOL:
                a, b = record["target"]["distribution"], donor["target"]["distribution"]
                if a is None or b is None:
                    counts["unsupported_target"] += 1
                    continue
                changed = max(range(2), key=lambda i: (a[i], -i)) != max(range(2), key=lambda i: (b[i], -i))
            else:
                def signature(target):
                    return sorted(sorted(item["text"] for item in annotation["items"]) for annotation in target["ratings"]
                                  if annotation["observed"] and annotation["items"])
                if not signature(record["target"]) or not signature(donor["target"]):
                    counts["unsupported_target"] += 1
                    continue
                changed = signature(record["target"]) != signature(donor["target"])
            counts["paired"] += 1
            counts["teacher_changing" if changed else "teacher_stable"] += 1
            sample = score_row(donor, swapped[identity])
            metric = "accuracy" if endpoint in BOOL else "recall@10" if endpoint == RETRIEVAL else "f1"
            paired.append(sample[metric])
            if changed:
                correct_new.append(sample[metric])
        new_value = ratio(sum(x[0] for x in correct_new), sum(x[1] for x in correct_new))
        result[endpoint] = {"counts": dict(counts), "correct_new_changing": new_value,
                            "correct_new_denominator": sum(x[1] for x in correct_new),
                            "correct_new_paired": ratio(sum(x[0] for x in paired), sum(x[1] for x in paired))}
    return result


def bio_reliability(record, prediction):
    if record["target"]["kind"] != "extract" or not prediction.get("windows"):
        return {"confidences": [], "correctness": []}
    supported = supervision(record["target"], record["blocks"], prediction["source_token_offsets"], prediction["windows"])
    observations = {}
    for window in prediction["windows"]:
        bid = window["block_id"]
        for index, logits in zip(window["source_token_indices"], window["bio_logits"]):
            gold = supported["token_distributions"][bid][index]
            if gold is None:
                continue
            masses = softmax(logits)
            tag = max(range(3), key=lambda i: masses[i])
            confidence = masses[tag]
            correct = int(tag == max(range(3), key=lambda i: gold[i]))
            key = (bid, index)
            if key not in observations or confidence > observations[key][0]:
                observations[key] = (confidence, correct)
    ordered = [observations[key] for key in sorted(observations)]
    return {"confidences": [x[0] for x in ordered], "correctness": [x[1] for x in ordered],
            "unit": "unique supported source token; maximum raw-confidence window, retained-label argmax",
            "string_probability_certificate": False}


def check_support(path, phases, derived, pins):
    rows = json_rows(path)
    header = next(rows, None)
    require(header and header["kind"] == "header" and header["condition"] == "support" and
            header["protocol_sha256"] == digest(AMENDMENT) and header["source_pins"] == pins, "Support custody header")
    expected = {(phase, identity) for phase in PHASES for identity in phases[phase]}
    seen, counts, denominators = set(), Counter(), Counter()
    report = {phase: {"rows": 0, "windows": 0, "source_tokens": 0, "native_items": 0, "unmatched_items": 0,
                      "unsupported_token_boundary_or_window_items": 0, "token_boundary_unsupported_items": 0,
                      "window_length_unsupported_items": 0, "supervised_source_tokens": 0} for phase in PHASES}
    for saved in rows:
        key = (saved["phase"], saved["id"])
        require(key in expected and key not in seen and saved["kind"] == "support", "Support membership")
        seen.add(key)
        phase, identity = key
        record, raw = phases[phase][identity], derived[phase][identity]
        require(saved["endpoint"] == record["endpoint"] and saved["component_id"] == record["component_id"] and
                saved["target"] == raw["target"] and saved["target_sha256"] == value_digest(raw["target"]), "Native support ledger")
        validate_windows(record, saved)
        summary = report[phase]
        summary["rows"] += 1
        summary["windows"] += len(saved["windows"])
        summary["source_tokens"] += sum(len(x) for x in saved["source_token_offsets"].values())
        endpoint, target = record["endpoint"], record["target"]
        denominator = 0
        if endpoint == EXTRACTION:
            expected_support = supervision(target, record["blocks"], saved["source_token_offsets"], saved["windows"])
            close(saved["support"], expected_support, "Independent all-phase annotation supervision")
            denominator = sum(sum(x * weight for x, weight in zip(values, (1, 10, 10)))
                              for block in expected_support["token_distributions"].values() for values in block if values is not None)
            summary["supervised_source_tokens"] += expected_support["supervised_source_tokens"]
            summary["unsupported_token_boundary_or_window_items"] += sum(not item["supported_occurrences"]
                for annotation in expected_support["annotations"] for item in annotation["items"] if item["source_occurrences"])
            for key in ("token_boundary_unsupported_items", "window_length_unsupported_items"):
                summary[key] += expected_support["support_counts"][key]
        else:
            require(saved["support"] is None, "Unexpected BIO supervision on another endpoint")
            denominator = int(categorical_target(record) is not None)
        if endpoint in (RETRIEVAL, EXTRACTION):
            summary["native_items"] += target["native_item_count"]
            summary["unmatched_items"] += target["unmatched_item_count"]
        if phase == "fit" and denominator:
            counts[endpoint] += 1
            denominators[endpoint] += denominator
    require(seen == expected, "Missing native support rows")
    return report, dict(counts), dict(denominators)


def check_candidates(path, records, pins, checkpoint_sha, weight, temperatures):
    rows = json_rows(path)
    header = next(rows, None)
    require(header and header["condition"] == "candidate_count" and header["protocol_sha256"] == digest(AMENDMENT) and
            header["source_pins"] == pins and header["checkpoint_sha256"] == checkpoint_sha, "Prefix header custody")
    expected = {(identity, size) for identity, row in records.items()
                for size in {min(k, len(row["blocks"])) for k in K_GRID + (len(row["blocks"]),)}}
    seen, summaries = set(), defaultdict(lambda: Counter())
    for saved in rows:
        identity, k = saved["id"], integer(saved["K"])
        require((identity, k) in expected and (identity, k) not in seen, "Prefix K/member universe")
        seen.add((identity, k))
        native = records[identity]
        ids = [block["id"] for block in native["blocks"]]
        prefix = nested_prefix(ids, native["component_id"], k)
        require(saved["subset_ids"] == prefix and saved["full_candidate_count"] == len(ids) and
                saved["input_only_prefix"] is True and saved["kind"] == "candidate_count", "Input-only nested prefix membership")
        record = changed_record(native, "candidate_count", subset=prefix)
        validate_prediction({**saved, "kind": "prediction"}, record, "candidate_count",
                            temperatures.get(native["endpoint"], 1.0), weight)
        require(saved["status"] == "OK", "Failed prefix control")
        summary = summaries[(native["endpoint"], k)]
        summary["rows"] += 1
        metric = "accuracy" if native["endpoint"] in BOOL else "recall@10" if native["endpoint"] == RETRIEVAL else "f1"
        sample = score_row(native, saved)
        summary["learned_primary_numerator"] += sample[metric][0]
        summary["learned_primary_denominator"] += sample[metric][1]
        summary["encoded_sequences"] += saved["runtime_counters"]["encoded_sequences"]
        summary["encoded_windows"] += saved["runtime_counters"]["encoded_windows"]
        for annotation in native["target"].get("ratings", []):
            for item in annotation.get("items", []):
                summary["native_items"] += 1
                summary["native_membership_hits"] += int(any(x["block_id"] in prefix for x in item["occurrences"]))
                if native["endpoint"] == RETRIEVAL:
                    summary["recall10_hits"] += int(any(x["block_id"] in saved["ranking"][:10] for x in item["occurrences"]))
    require(seen == expected, "Incomplete prefix grid")
    return [{"endpoint": endpoint, "K": k, **dict(value),
             "membership_frequency": ratio(value["native_membership_hits"], value["native_items"]),
             "learned_primary": ratio(value["learned_primary_numerator"], value["learned_primary_denominator"]),
             "learned_recall@10": ratio(value["recall10_hits"], value["native_items"]) if endpoint == RETRIEVAL else None}
            for (endpoint, k), value in sorted(summaries.items())]


def repeat_plan(records):
    native_questions = {(row["paper_id"], row["question_ordinal"]): row["serving"]["question"]
                        for row in records.values() if row["endpoint"] == EXTRACTION}
    papers = defaultdict(list)
    for record in sorted(records.values(), key=lambda row: (row["paper_id"], row["id"])):
        key = (record["paper_id"], record["question_ordinal"])
        require(key in native_questions, "Repeat plan missing original native question")
        questions = papers[record["paper_id"]]
        if all(other["question_ordinal"] != record["question_ordinal"] and
               native_questions[(other["paper_id"], other["question_ordinal"])] != native_questions[key] for other in questions):
            questions.append(record)
    return papers[min(papers)][:2]


def check_repeat(path, records, pins, checkpoint_sha, weight, temperatures):
    plan = repeat_plan(records)
    require(len(plan) == 2, "Unavailable second distinct question diagnostic")
    ids = [record["id"] for record in plan]
    require(plan[0]["question_ordinal"] != plan[1]["question_ordinal"], "Repeat native questions are not distinct")
    rows = json_rows(path)
    header = next(rows, None)
    require(header and header["condition"] == "repeat" and header["source_pins"] == pins and
            header["protocol_sha256"] == digest(AMENDMENT) and header["checkpoint_sha256"] == checkpoint_sha,
            "Repeat header custody")
    results, first, cumulative = [], {}, None
    for saved in rows:
        position = len(results)
        require(position < 6 and saved["round"] == position // 2 and saved["id"] == ids[position % 2] and
                saved["question_ids"] == ids, "Three-round fresh-cache repeat plan")
        record = plan[position % 2]
        validate_prediction(saved, record, "repeat", temperatures.get(record["endpoint"], 1.0), weight)
        require(saved["status"] == "OK", "Failed repeat prediction")
        counters = saved["runtime_counters"]
        if position < 2:
            require(counters["encoded_sequences"] == counters["encoded_windows"] == counters["state_containing_sequences"] == len(saved["windows"])
                    and counters["cache_hits"] == 0 and counters["encoder_forward_calls"] == math.ceil(len(saved["windows"]) / 8),
                    "Distinct question must pay full joint encoding")
            first[record["id"]] = {key: saved[key] for key in ("raw_logits", "raw_probabilities", "spans", "cache_key")}
        else:
            require(counters["encoded_sequences"] == counters["encoded_windows"] == counters["state_containing_sequences"] ==
                    counters["encoder_forward_calls"] == 0 and counters["cache_hits"] == 1, "Exact repeat must encode zero")
            close({key: saved[key] for key in first[record["id"]]}, first[record["id"]], "Exact repeat output/cache identity", 0)
        current = saved["cumulative_counters"]
        if cumulative is not None:
            close({key: current[key] - cumulative[key] for key in counters}, counters, "Repeat cumulative counter reconciliation", 0)
        cumulative = current
        results.append({"round": saved["round"], "id": saved["id"], "cache_key": saved["cache_key"], "runtime_counters": counters})
    require(len(results) == 6 and first[ids[0]]["cache_key"] != first[ids[1]]["cache_key"], "Complete distinct-input repeat evidence")
    return {"status": "PASS", "question_ids": ids, "rounds": results,
            "scope": "Counted recorded encoding work, not independent forwards or latency measurements"}


def original_reference(records, pins):
    metadata_path = checked(pins["selected_initial_cross_metadata"])
    metadata = load_json(metadata_path)
    entry = metadata["artifact_hashes"]["dev.jsonl"]
    path = metadata_path.parent / "dev.jsonl"
    require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Original DEV prediction custody")
    predictions, samples = {}, {}
    for saved in json_rows(path):
        if saved["endpoint"] not in INTENT:
            continue
        identity = saved["id"]
        require(identity in records and identity not in predictions, "Original intent reference universe")
        record = records[identity]
        require(saved["candidate_ids"] == record["candidate_ids"] and saved["component_id"] == record["component_id"],
                "Original intent source/candidate coordinates")
        close(saved["raw_probs"], softmax(saved["logits"]), "Original raw cross probabilities")
        close(saved["probs"], softmax(saved["logits"], metadata["temperatures"][saved["endpoint"]]), "Original calibration-only probabilities")
        converted = {**saved, "status": "OK", "raw_logits": saved["logits"], "raw_probabilities": saved["raw_probs"],
                     "probabilities": saved["probs"], "ranking": ranking(saved["candidate_ids"], saved["probs"]), "spans": []}
        predictions[identity] = converted
        samples[identity] = score_row(record, converted)
    require(set(predictions) == set(records), "Complete original cross intent DEV")
    return predictions, samples


def quality_gates(summaries, intervals, swaps, completion, structural):
    gates = {}
    for endpoint in BOOL + (RETRIEVAL, EXTRACTION):
        metric = "accuracy" if endpoint in BOOL else "recall@10" if endpoint == RETRIEVAL else "f1"
        value = metric_value(summaries[endpoint], metric)
        comparisons = intervals.get(endpoint, {})
        quality = value is not None and value >= 0.80
        quality = quality and interval_pass(comparisons.get("model_minus_baseline")) and interval_pass(comparisons.get("model_minus_question_masked"))
        if endpoint in BOOL:
            quality = quality and interval_pass(comparisons.get("baseline_minus_model_brier"))
        swap = swaps.get(endpoint, {})
        changing = swap.get("counts", {}).get("teacher_changing", 0)
        denominator, paired = swap.get("correct_new_denominator"), swap.get("counts", {}).get("paired")
        correct_new = swap.get("correct_new_changing")
        swap_measured = all(type(count) is int and count > 0 for count in (changing, denominator, paired)) and \
                        type(correct_new) in (int, float) and math.isfinite(correct_new)
        swap_pass = swap_measured and (endpoint not in BOOL or correct_new >= 0.80)
        gates[endpoint] = {"status": "PASS" if completion and structural and quality and swap_pass else
                          "UNMEASURED" if value is None or not swap_measured else "FAIL",
                          "quality": bool(quality), "question_swap": "PASS" if swap_pass else "UNMEASURED" if not swap_measured else "FAIL"}
    for endpoint in INTENT:
        receipt = intervals.get(endpoint, {}).get("model_minus_original_cross")
        passed = interval_pass(receipt, -0.01, strict=False)
        gates[endpoint] = {"status": "PASS" if completion and structural and passed else "UNMEASURED" if
                          not isinstance(receipt, dict) or receipt.get("status") != "MEASURED" else "FAIL",
                          "accuracy_nonregression": bool(passed)}
    return gates


def output_pin(entry, root, name):
    require(Path(entry["path"]).name == name, "Output artifact name")
    path = Path(root) / name
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == entry["bytes"] and
            digest(path) == entry["sha256"], "Output artifact bytes: " + name)
    return path


def check_checkpoints(root, metadata, history, pins):
    require(len(history["epochs"]) == 11 and [r["epoch"] for r in history["epochs"]] == list(range(11)),
            "Complete ten training epochs plus epoch0")
    candidates = metadata["candidate_checkpoints"]
    require(len(candidates) == 11 and [r["epoch"] for r in candidates] == list(range(11)), "Complete candidate tensor bytes")
    initial = checkpoint_fingerprint(checked(pins["selected_initial_cross_checkpoint"]))
    tensors, hashes = None, []
    for epoch, entry in enumerate(candidates):
        actual = checkpoint_fingerprint(output_pin(entry, root, "epoch%02d.safetensors" % epoch))
        require(actual["state_dict_fingerprint"] == entry["state_dict_fingerprint"] ==
                history["epochs"][epoch]["state_dict_fingerprint"], "Epoch actual tensor fingerprints")
        require(history["epochs"][epoch]["checkpoint"] == entry, "Epoch checkpoint custody receipt")
        if tensors is None:
            tensors = actual["tensors"]
            expected = {"cross." + name for name in initial["tensors"]} | {"bool_head.weight", "bool_head.bias", "bio_head.weight", "bio_head.bias"}
            require(set(tensors) == expected, "Exactly one shared original cross and two new readouts")
            for name, tensor in initial["tensors"].items():
                require(tensors["cross." + name] == tensor, "Strict initial original checkpoint tensor bytes")
            for name, shape in (("bool_head.weight", [2, 384]), ("bool_head.bias", [2]),
                                ("bio_head.weight", [3, 384]), ("bio_head.bias", [3])):
                require(tensors[name]["shape"] == shape and tensors[name]["dtype"] == "F32", "Declared new readout tensor")
            require(actual["state_dict_fingerprint"] == metadata["custody"]["initial_model_fingerprint"] and
                    actual["cross_state_dict_fingerprint"] == metadata["custody"]["initial_cross_fingerprint"] ==
                    initial["state_dict_fingerprint"], "Actual strict cross initialization")
        else:
            require(set(actual["tensors"]) == set(tensors) and
                    all(actual["tensors"][name]["shape"] == tensors[name]["shape"] and
                        actual["tensors"][name]["dtype"] == tensors[name]["dtype"] for name in tensors), "Shared model tensor universe changed")
        hashes.append(actual["state_dict_fingerprint"])
    selected = checkpoint_fingerprint(output_pin(metadata["checkpoint"], root, "selected.safetensors"))
    epoch = history["selected_epoch"]
    require(type(epoch) is int and 0 <= epoch <= 10 and
            metadata["checkpoint"]["selected_epoch"] == epoch and selected["state_dict_fingerprint"] ==
            metadata["checkpoint"]["state_dict_fingerprint"] == history["selected_fingerprint"] == hashes[epoch],
            "Selected actual checkpoint tensor fingerprint")
    require(selected["encoder_state_dict_fingerprint"] == metadata["checkpoint"]["encoder_state_dict_fingerprint"] and
            metadata["checkpoint"]["encoder_prefix"] == "cross.scorer.pool.encoder.", "Single shared selected encoder")
    require(set(selected["tensors"]) == set(tensors), "Selected tensor universe")
    return selected


def check_liveness(liveness, history, selected, probes, chosen, records, supervised, denominators):
    smoke = liveness["smoke"]
    require(smoke["status"] == "PASS" and smoke["initial_model_sha256"] != smoke["after_step_sha256"] and
            smoke["effective_block_decisions"] == 32, "Actual numerical smoke/effective block receipt")
    for field in ("named_gradients", "named_parameter_deltas"):
        values = smoke[field]
        require(values and any(name.startswith("cross.scorer.pool.encoder.") for name in values) and
                {"bool_head.weight", "bio_head.weight", "cross.scorer.head.0.weight"} <= set(values) and
                all(finite(value) > 0 for value in values.values()), "Active shared encoder/output " + field)
    require(set(smoke["losses"]) == set(ENDPOINTS) and all(math.isfinite(finite(x)) for x in smoke["losses"].values()),
            "All endpoint finite numerical smoke")
    restore = liveness["selected_restore"]
    require(restore["strict"] is True and restore["missing_keys"] == [] and restore["unexpected_keys"] == [] and
            restore["selected_epoch"] == chosen and restore["expected_fingerprint"] == restore["restored_fingerprint"] ==
            restore["fingerprint"] == selected["state_dict_fingerprint"] and restore["checkpoint_sha256"] == selected["sha256"] and
            restore["before_restore_fingerprint"] == history["epochs"][-1]["state_dict_fingerprint"], "Strict selected restore receipt")
    before = member_index(restore["selected_rows"])
    after = member_index(restore["restored_rows"], before)
    required_probes = {min(identity for identity, row in records.items() if row["endpoint"] == endpoint) for endpoint in ENDPOINTS}
    require(set(before) == required_probes, "Selected restore all endpoint probe universe")
    for identity in before:
        expected = probes[(chosen, identity)]
        for saved in (before[identity], after[identity]):
            require(saved["endpoint"] == records[identity]["endpoint"], "Restore native endpoint")
            close({key: saved[key] for key in expected}, expected, "Recorded selected/restored raw outputs", 0)
    close(restore["max_abs_difference"], 0, "Recorded strict restore maximum difference", 0)
    for epoch in history["epochs"]:
        if epoch["epoch"] == 0:
            require(epoch["supervised_counts"] == epoch["loss_denominators"] == {}, "Epoch0 has no optimizer supervision")
        else:
            require(epoch["supervised_counts"] == supervised, "Independent supervised native decision counts")
            close(epoch["loss_denominators"], denominators, "Independent native/token weighted loss denominators", 1e-3)
    return {"status": "PASS", "selected_epoch": chosen, "selected_fingerprint": selected["state_dict_fingerprint"],
            "named_smoke_gradients_and_deltas": "validated recorded receipts",
            "restore_outputs": "matched selected raw captures and recorded strict restored captures",
            "independent_neural_forwards": 0}


def verify(root):
    root = Path(root)
    if not (root / "metadata.json").is_file() and (root / "train" / "metadata.json").is_file():
        root = root / "train"
    errors = []
    report = {"schema": "vey.qnative.cross.verification.v1", "root": str(root),
              "protocol_sha256": digest(AMENDMENT), "independent_neural_forwards": 0,
              "scope": "Independent raw-source/target, tensor-byte, recorded-logit/decoder/metric reconstruction; no rerun model forwards",
              "promotion": False, "Pareto_credit": False, "endgame_complete": False}
    def stage(name, function):
        try:
            return function()
        except Exception as exc:
            errors.append({"stage": name, "error_type": type(exc).__name__, "error": str(exc)})
            return None
    source = stage("source_authority_and_membership", reconstruct_sources)
    if source is None:
        report.update(status="FAIL", evidence_reconstruction={"status": "FAIL", "errors": errors},
                      mechanism_gates={e: {"status": "UNMEASURED"} for e in ENDPOINTS}, whole_study_capability=False)
        return report
    cfg, pins, transport, phases, derived, native_manifest = source
    report["source_phase_counts"] = {phase: dict(Counter(r["endpoint"] for r in rows.values())) for phase, rows in phases.items()}
    metadata = stage("complete_metadata", lambda: load_json(root / "metadata.json"))
    if metadata is None:
        report.update(status="FAIL", evidence_reconstruction={"status": "FAIL", "errors": errors},
                      mechanism_gates={e: {"status": "UNMEASURED"} for e in ENDPOINTS}, whole_study_capability=False)
        return report
    def validate_metadata():
        require(not (root / "failure.json").exists() and metadata["mode"] == "train" and metadata["status"] == "train_complete" and
                metadata["protocol_sha256"] == digest(AMENDMENT) and metadata["source_pins"] == pins, "Complete registered run custody")
        for name in REQUIRED:
            output_pin(metadata["artifact_hashes"][name], root, name)
        for name, entry in metadata["artifact_hashes"].items():
            output_pin(entry, root, name)
        require(metadata["promotion"] is metadata["Pareto_credit"] is metadata["endgame_complete"] is False,
                "No development promotion")
        for phase, counts in metadata["phase_counts"].items():
            require(counts["rows"] == len(phases[phase]) and counts["endpoint_counts"] ==
                    dict(Counter(r["endpoint"] for r in phases[phase].values())), "All-phase denominator custody")
        require(set(metadata["phase_counts"]) == set(PHASES), "All-phase metadata inventory")
        return True
    stage("artifact_custody", validate_metadata)
    history = stage("history", lambda: load_json(root / "history.json"))
    liveness = stage("liveness", lambda: load_json(root / "liveness.json"))
    runtime = stage("runtime", lambda: load_json(root / "runtime.json"))
    selected = stage("checkpoint_tensor_bytes", lambda: check_checkpoints(root, metadata, history, pins)) if history is not None else None
    if selected:
        report["checkpoint"] = selected
        report["candidate_checkpoints"] = metadata["candidate_checkpoints"]
    report["source_pins"] = pins
    support = stage("support_and_supervision", lambda: check_support(root / "support.jsonl.gz", phases, derived, pins))
    if support:
        report["support"] = support[0]
        def window_census():
            require(all(metadata["phase_counts"][phase]["windows"] == support[0][phase]["windows"] for phase in PHASES),
                    "Independent full phase window counts")
            return True
        stage("complete_phase_window_census", window_census)
    selection = stage("all_epoch_selection", lambda: reconstruct_selection(root / "selection.jsonl.gz", phases["selection"], pins, history)) if history else None
    if selection is not None:
        chosen, metrics, probes = selection
        report["selection"] = {"selected_epoch": chosen, "epochs": metrics}
        def selection_receipt():
            require(chosen == history["selected_epoch"], "Independent selection chooses a different epoch")
            for epoch, independent in enumerate(metrics):
                saved = history["epochs"][epoch]["selection"]
                expected_primary = {endpoint: metric_value(independent["endpoints"][endpoint],
                    "accuracy" if endpoint in BOOL + INTENT else "recall@10" if endpoint == RETRIEVAL else "f1") for endpoint in ENDPOINTS}
                close(saved["primary"], expected_primary, "Selection endpoint primaries")
                close(saved["macro_primary"], independent["macro_primary"], "Selection macro")
                close(saved["mean_nll"], independent["mean_nll"], "Selection tie NLL")
                expected_counts = {endpoint: independent["endpoints"][endpoint]["metrics"][
                    "accuracy" if endpoint in BOOL + INTENT else "recall@10" if endpoint == RETRIEVAL else "f1"]["denominator"]
                    for endpoint in ENDPOINTS}
                require(saved["counts"] == expected_counts, "Selection native-item/annotation denominators")
            return True
        stage("selection_receipt", selection_receipt)
        if selected and support and liveness:
            report["liveness"] = stage("strict_restore_and_training_denominators", lambda: check_liveness(
                liveness, history, selected, probes, chosen, phases["selection"], support[1], support[2]))
    checkpoint_sha = metadata.get("checkpoint", {}).get("sha256")
    weight = selected["state_dict_fingerprint"] if selected else metadata.get("checkpoint", {}).get("state_dict_fingerprint")
    temperatures = metadata.get("temperatures", {})
    if checkpoint_sha and weight:
        report["calibration"] = stage("disjoint_temperature_and_BIO_calibration", lambda: reconstruct_calibration(
            root / "calibration.jsonl.gz", phases["calibration"], pins, checkpoint_sha, temperatures, weight))
        if report["calibration"]:
            def temperature_receipt():
                require(set(metadata["temperature_evidence"]) == set(temperatures), "Temperature evidence endpoint inventory")
                for endpoint in temperatures:
                    result = report["calibration"][endpoint]
                    close(metadata["temperature_evidence"][endpoint],
                          {key: result[key] for key in ("rows", "grid", "selected")}, "Independent full temperature grid", 1e-12)
                return True
            stage("registered_temperature_grid_receipts", temperature_receipt)
    qrows = {identity: row for identity, row in phases["dev"].items() if row["endpoint"] not in INTENT}
    native_rows = {identity: row for identity, row in phases["dev"].items() if row["endpoint"] in INTENT}
    plans = donor_plan(qrows)
    captures = {}
    if checkpoint_sha and weight:
        for condition in ("dev", "question_masked", "state_masked", "question_swapped", "reordered"):
            natives = phases["dev"] if condition == "dev" else qrows
            captures[condition] = stage(condition + "_capture", lambda condition=condition, natives=natives: capture(
                root / (condition + ".jsonl.gz"), natives, condition, pins, checkpoint_sha, temperatures, plans, weight=weight))
        report["candidate_count"] = stage("input_only_nested_K", lambda: check_candidates(
            root / "candidate_count.jsonl.gz", qrows, pins, checkpoint_sha, weight, temperatures))
        report["repeated_state"] = stage("counted_distinct_question_repeats", lambda: check_repeat(
            root / "repeated_state.jsonl", qrows, pins, checkpoint_sha, weight, temperatures))
    reference = stage("fixed_original_cross_reference", lambda: original_reference(native_rows, pins))
    idf, documents = tfidf_fit(phases["fit"])
    priors = fit_priors(phases["fit"])
    def baseline_custody():
        close(metadata["baseline"]["priors"], priors, "FIT-only prior")
        close(metadata["baseline"]["idf"], idf, "FIT-only IDF")
        require(metadata["baseline"]["deduplicated_fit_texts"] == documents and metadata["baseline"]["idf_sha256"] == value_digest(idf),
                "FIT-only deduplicated lexical document count")
        return True
    stage("FIT_only_baseline_custody", baseline_custody)
    baseline = stage("all_native_baselines", lambda: reconstruct_baselines(root / "baselines.jsonl.gz", phases["dev"], pins,
        checkpoint_sha, priors, idf, documents, reference[0])) if reference else None
    summaries = {endpoint: summarize([]) for endpoint in ENDPOINTS}
    intervals, swaps = {}, {}
    if captures.get("dev"):
        real, samples = captures["dev"]
        summaries = {e: summarize([sample for sample in samples.values() if sample["endpoint"] == e]) for e in ENDPOINTS}
        report["metrics"] = {"dev": summaries}
        report["per_row_metrics"] = samples
        if baseline:
            report["metrics"]["baselines"] = {e: summarize([sample for sample in baseline[1].values() if sample["endpoint"] == e])
                                               for e in ENDPOINTS}
        if reference:
            report["metrics"]["original_cross"] = {e: summarize([sample for sample in reference[1].values() if sample["endpoint"] == e])
                                                   for e in INTENT}
        for condition, captured in captures.items():
            if condition != "dev" and captured:
                report["metrics"][condition] = {e: summarize([sample for sample in captured[1].values() if sample["endpoint"] == e])
                                                for e in BOOL + (RETRIEVAL, EXTRACTION)}
        for condition, captured in captures.items():
            if captured:
                failed = [identity for identity, row in captured[0].items() if row["status"] != "OK" and
                          not (condition == "question_swapped" and plans[identity] is None)]
                if failed:
                    errors.append({"stage": condition + "_completion", "error": "Failed registered source predictions", "ids": failed})
        if captures.get("reordered"):
            report["reorder"] = stage("stable_semantic_order", lambda: compare_reorder(
                {identity: real[identity] for identity in qrows}, captures["reordered"][0], qrows))
        if captures.get("question_swapped"):
            swaps = swap_results(qrows, captures["question_swapped"][0], plans)
        for endpoint in ENDPOINTS:
            left = {identity: sample for identity, sample in samples.items() if sample["endpoint"] == endpoint}
            universe = sorted({row["component_id"] for row in phases["dev"].values() if row["endpoint"] == endpoint})
            endpoint_intervals = {}
            comparisons = []
            if endpoint in INTENT and reference:
                comparisons.append(("model_minus_original_cross", left, reference[1], "accuracy"))
            elif endpoint not in INTENT:
                metric = "accuracy" if endpoint in BOOL else "recall@10" if endpoint == RETRIEVAL else "f1"
                if baseline:
                    comparisons.append(("model_minus_baseline", left, baseline[1], metric))
                    if endpoint in BOOL:
                        comparisons.append(("baseline_minus_model_brier", baseline[1], left, "brier"))
                if captures.get("question_masked"):
                    comparisons.append(("model_minus_question_masked", left, captures["question_masked"][1], metric))
            for name, a, b, metric in comparisons:
                a = {identity: a[identity] for identity in left}
                b = {identity: b[identity] for identity in left}
                interval = stage(endpoint + ":" + name, lambda a=a, b=b, metric=metric, universe=universe: paired_interval(a, b, metric, universe))
                endpoint_intervals[name] = interval
                if interval is None or interval["status"] != "MEASURED":
                    errors.append({"stage": endpoint + ":" + name, "error": "Required paired interval absent/unmeasured"})
            intervals[endpoint] = endpoint_intervals
    report["intervals"] = intervals
    report["question_swap"] = swaps
    peak = stage("finite_peak_GPU_allocation", lambda: integer(runtime["peak_allocated_gpu_bytes"])) if runtime else None
    structural = peak is not None and 0 < peak <= 12 * 2**30 and bool(report.get("reorder")) and bool(report.get("repeated_state"))
    report["structural"] = {"status": "PASS" if structural else "FAIL", "peak_allocated_gpu_bytes": peak,
                            "max_allocated_gpu_bytes": 12 * 2**30, "source_only_input_and_windows": not any(
                                "source" in error["stage"] or "support" in error["stage"] for error in errors)}
    complete = not errors
    gates = quality_gates(summaries, intervals, swaps, complete, structural)
    earned = complete and structural and all(gate["status"] == "PASS" for gate in gates.values())
    report.update(status="PASS" if complete else "FAIL",
                  evidence_reconstruction={"status": "PASS" if complete else "FAIL", "errors": errors},
                  mechanism_gates=gates, mechanism_status="PASS" if earned else "FAIL", whole_study_capability=earned,
                  completion={"status": "PASS" if complete else "FAIL", "ten_epochs": selection is not None,
                              "strict_selected_tensor_restore": report.get("liveness") is not None},
                  statistics={"unit": "frozen input-only component ratio", "resamples": 10000, "seed": 0,
                              "intervals": "nominal descriptive95%; no simultaneous release claim"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=load_json(AMENDMENT)["output_root"])
    parser.add_argument("--mode", choices=("run", "dataset"), default="run")
    args = parser.parse_args(argv)
    destination = args.root / ("verification_receipt.json" if args.mode == "dataset" else "verification.json")
    if args.mode == "run" and not (args.root / "metadata.json").is_file() and (args.root / "train").is_dir():
        destination = args.root / "train" / "verification.json"
    require(not destination.exists(), "Refusing to overwrite independent verification evidence")
    try:
        receipt = verify_dataset(args.root) if args.mode == "dataset" else verify(args.root)
    except Exception as exc:
        receipt = {"schema": "vey.qnative.cross.dataset-verification.v1" if args.mode == "dataset" else "vey.qnative.cross.verification.v1",
                   "status": "FAIL", "evidence_reconstruction": {"status": "FAIL", "errors": [
                       {"error_type": type(exc).__name__, "error": str(exc)}]},
                   "all_rows_independently_reconstructed": False, "sealed_labels_accessed": False,
                   "whole_study_capability": False, "independent_neural_forwards": 0, "model_loads": 0,
                   "model_forwards": 0, "promotion": False, "Pareto_credit": False, "endgame_complete": False}
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(canonical(receipt) + "\n")
    print(canonical({"path": str(destination), "status": receipt["status"],
                     "whole_study_capability": receipt.get("whole_study_capability", False)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
