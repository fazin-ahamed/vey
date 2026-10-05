#!/usr/bin/env python3
"""Independent reconstruction of the NATIVE-2 native architecture comparison.

Reads the committed NATIVE-1 dataset, the frozen NATIVE-1 pooled full-arm
reference, and the cross/dual/pages prediction directories. It rebuilds native
membership, DecisionIR IDs, catalogue order, native gold and finite normalized
distributions itself, then recomputes every screen metric from logits and
native targets. No arm metric, no model and no sealed phase is trusted.

Persisted row schemas consumed here (coordinated with the NATIVE-2 trainer):

  <root>/<arch>/dev.jsonl, masked.jsonl, swapped.jsonl
      id, endpoint, task, component_id, group_id, locale, candidate_ids,
      logits, raw_probs, probs, answer, ties, raw_answer, raw_ties.
      swapped rows add donor_id; donor-unavailable rows carry donor_id=null,
      a reason string and no logits/probs keys.
  <root>/<arch>/candidate_count.jsonl
      the dev fields plus k, subset_ids, gold_rank, gold_in_shortlist;
      candidate_ids == subset_ids. Entire-catalogue permutations depend only
      on component_id and seed7; absent gold has rank null.
  <root>/<arch>/coordinate_reversal.jsonl
      the dev fields with candidate_ids reversed and reversed=true.
  <root>/<arch>/repeated_state.jsonl
      three rounds of the native and other endpoint's public catalogue on
      one DEV state. Source/native decision IDs, probe eligibility and before/
      after state_encode_calls, encoder_forward_calls, encoded_states are explicit.
  <root>/<arch>/selection.jsonl, calibration.jsonl, selected.safetensors
      the trained epoch-selection and disjoint-calibration logits plus the
      strict-restore checkpoint (selection/calibration rows carry the native
      target distribution; gold is present only if the trainer persists it).

The selected checkpoint's real tensor schema and bytes are independently
hashed, including its architecture-specific encoder. Unrecognized schemas
fail closed. Restore-forward equality remains a persisted numerical receipt,
not an independent model-forward replay.

Usage:
  python research/endgame/native_arch_verify.py --root <registered-root>
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
sys.path.insert(0, str(HERE))

import native_field_data as DATA  # noqa: E402  committed NATIVE-1 dataset reader
import native_field_verify as NF  # noqa: E402  corrected metrics/custody helpers

ARCHS = ("cross", "dual", "pages")
PHASES = ("fit", "selection", "calibration", "dev")
ARCH_FILES = ("dev", "masked", "swapped", "candidate_count",
              "coordinate_reversal", "repeated_state")
TRAINED_FILES = ("selection.jsonl", "calibration.jsonl", "selected.safetensors")
NESTED_K = (2, 4, 8, 16, 32)
REPEATED_STATE_REPEATS = 3
EMPTY_EVIDENCE = "[EMPTY EVIDENCE]"
NATIVE1_COUNTS = {"banking_train": 4105, "banking_dev": 1027,
                  "massive_en_train": 3971, "massive_en_dev": 1252}
DEV_KEYS = frozenset({"id", "endpoint", "task", "component_id", "group_id", "locale",
                      "candidate_ids", "logits", "raw_probs", "probs", "answer", "ties",
                      "raw_answer", "raw_ties"})

require = NF.require
finite = NF.finite
digest = NF.digest
load_json = NF.load_json
json_rows = NF.json_rows
close = NF.close
distribution = NF.distribution
softmax = NF.softmax


def choice_winner(ids, probs):
    require(len(ids) == len(probs) and len(set(ids)) == len(ids) and bool(ids),
            "invalid choice coordinates")
    top = max(probs)
    ties = sorted(cid for cid, value in zip(ids, probs) if value == top)
    return ties[0], ties


def nested_subset(candidate_ids, component_id, k):
    """Reconstruct an input-only prefix of the whole catalogue permutation."""
    pool = list(candidate_ids)
    seed = int.from_bytes(hashlib.sha256(
        ("native-arch-candidate-count|7|" + component_id).encode()).digest()[:8], "big")
    random.Random(seed).shuffle(pool)
    return pool[:k]


def gold_rank(subset, probs, gold):
    """1-based rank of gold among the subset, ties broken by lexicographic ID."""
    if gold not in subset:
        return None
    by_id = dict(zip(subset, probs))
    return 1 + sum(1 for cid in subset if cid != gold
                   and (by_id[cid] > by_id[gold]
                        or (by_id[cid] == by_id[gold] and cid < gold)))


def reconstruct_natives(manifest, phases):
    """Independently rebuild membership, IDs, catalogues and native gold."""
    shim = {"model": {"fields": sorted(manifest["catalogues"])},
            "data": {"known_intent_counts": dict(NATIVE1_COUNTS)}}
    records, census, components, eligibility = NF.reconstruct_dataset(
        manifest, phases, shim)
    intents = sorted(e for e, a in eligibility.items()
                     if a["eligible"] and e.endswith(".intent"))
    require(intents == ["banking77.intent", "massive.intent"],
            "registered intent eligibility: " + repr(intents))
    return records, census, components, eligibility


def build_priors(records, endpoints):
    return {endpoint: NF.empirical_prior(records, endpoint) for endpoint in endpoints}


def build_donors(records, endpoints):
    plans = {endpoint: NF.donor_plan(list(records["dev"][endpoint].values()))
             for endpoint in endpoints}
    for endpoint, plan in plans.items():
        require(set(plan) == set(records["dev"][endpoint]),
                "donor plan membership: " + endpoint)
    return plans


def validate_field(saved, native, label, temperatures, endpoint):
    count = len(native["candidate_ids"])
    logits = saved.get("logits")
    require(isinstance(logits, list) and len(logits) == count, label + ": logits shape")
    logits = [finite(value, label + ":logits") for value in logits]
    raw = distribution(saved.get("raw_probs"), count, label + ": raw probabilities")
    probs = distribution(saved.get("probs"), count, label + ": probabilities")
    close(raw, softmax(logits), label + ": raw softmax", 1e-12, 1e-12)
    close(probs, softmax(logits, temperatures[endpoint]), label + ": temperature softmax",
          1e-12, 1e-12)
    answer, ties = choice_winner(native["candidate_ids"], probs)
    raw_answer, raw_ties = choice_winner(native["candidate_ids"], raw)
    require(saved.get("answer") == answer and saved.get("ties") == ties,
            label + ": calibrated semantic argmax/ties")
    require(saved.get("raw_answer") == raw_answer and saved.get("raw_ties") == raw_ties,
            label + ": raw semantic argmax/ties")
    return logits, raw, probs, answer, ties


def read_dev_predictions(path, natives, label, temperatures, control, plans):
    """Validate one dev/masked/swapped file against native membership.

    Returns (rows, unavailable) where rows is endpoint -> rid -> row-or-None and
    unavailable is the set of (endpoint, rid) donor-unavailable decisions.
    """
    expected = {rid: row for rows in natives.values() for rid, row in rows.items()}
    rows = {endpoint: {} for endpoint in natives}
    unavailable = set()
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(isinstance(rid, str) and rid in expected and rid not in seen,
                str(path) + ": extra/duplicate prediction ID")
        seen.add(rid)
        native = expected[rid]
        endpoint = native["endpoint"]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale",
                    "candidate_ids"):
            require(saved.get(key) == native[key], str(path) + ": native " + key)
        if control == "swapped":
            donor = plans[endpoint][rid]
            if donor is None:
                require(saved.get("donor_id") is None and isinstance(saved.get("reason"), str)
                        and "logits" not in saved and "probs" not in saved,
                        str(path) + ": donor-unavailable row shape")
                require(set(saved) == {"id", "endpoint", "task", "component_id", "group_id",
                                       "locale", "candidate_ids", "donor_id", "reason"},
                        str(path) + ": donor-unavailable row keys")
                rows[endpoint][rid] = None
                unavailable.add((endpoint, rid))
                continue
            require(saved.get("donor_id") == donor, str(path) + ": deterministic donor ID")
        else:
            require("donor_id" not in saved, str(path) + ": donor on non-swapped path")
        expected_keys = DEV_KEYS | ({"donor_id"} if control == "swapped" else set())
        require(set(saved) == expected_keys, str(path) + ": exact prediction row keys")
        logits, raw, probs, answer, ties = validate_field(
            saved, native, str(path), temperatures, endpoint)
        rows[endpoint][rid] = {
            **{key: native[key] for key in ("id", "endpoint", "task", "component_id",
                                            "group_id", "locale", "candidate_ids", "gold")},
            "logits": logits, "raw_probs": raw, "probs": probs,
            "answer": answer, "ties": ties, "donor_id": saved.get("donor_id"),
        }
    require(seen == set(expected), str(path) + ": missing native prediction rows")
    return rows, unavailable


def check_masked_constant(masked, natives):
    """An empty-evidence state yields one constant field per endpoint."""
    checked = 0
    for endpoint, values in natives.items():
        reference = None
        for rid in sorted(values):
            row = masked[endpoint][rid]
            require(row is not None, "masked row present for every decision")
            if reference is None:
                reference = row
            for key in ("logits", "raw_probs", "probs"):
                close(row[key], reference[key], "constant empty-evidence " + key, 0, 0)
            require(row["answer"] == reference["answer"] and row["ties"] == reference["ties"],
                    "constant empty-evidence readout")
            checked += 1
    return checked


def check_swap_control(dev, swapped, natives, plans):
    """Swapped rows re-read the donor state with the same catalogue.

    Within an endpoint the candidate catalogue and question text are constant, so
    a swapped row equals the donor's own dev row exactly (NATIVE-1 convention).
    """
    checked = 0
    changed = 0
    for endpoint, values in natives.items():
        for rid in sorted(values):
            donor = plans[endpoint][rid]
            if donor is None:
                require(swapped[endpoint][rid] is None, "donor-unavailable row missing")
                continue
            row = swapped[endpoint][rid]
            require(row is not None, "available donor row present")
            require(row["candidate_ids"] == natives[endpoint][rid]["candidate_ids"],
                    "swap preserves the original candidate set")
            original_donor = dev[endpoint][donor]
            for key in ("logits", "raw_probs", "probs"):
                close(row[key], original_donor[key], "swap/donor exact " + key, 0, 0)
            require(row["answer"] == original_donor["answer"]
                    and row["ties"] == original_donor["ties"],
                    "swap/donor readout")
            changed += int(row["probs"] != dev[endpoint][rid]["probs"])
            checked += 1
    return {"rows": checked, "probability_changed_rows": changed,
            "unchanged_rows": checked - changed}


def read_candidate_count(path, natives, label, temperatures):
    """Validate input-only nested-K rows and recompute membership and gold rank."""
    expected = {rid: row for rows in natives.values() for rid, row in rows.items()}
    by_decision = defaultdict(list)
    for saved in json_rows(path):
        require(set(saved) == DEV_KEYS | {"k", "subset_ids", "gold_rank", "gold_in_shortlist"},
                str(path) + ": candidate-count row keys")
        rid = saved.get("id")
        require(isinstance(rid, str) and rid in expected, str(path) + ": unknown decision ID")
        native = expected[rid]
        endpoint = native["endpoint"]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale"):
            require(saved.get(key) == native[key], str(path) + ": native " + key)
        k = saved.get("k")
        require(type(k) is int and not isinstance(k, bool) and k >= 2,
                str(path) + ": subset size")
        gold = native["gold"]
        subset = saved.get("subset_ids")
        require(isinstance(subset, list) and len(subset) == k and len(set(subset)) == k
                and set(subset) <= set(native["candidate_ids"]),
                str(path) + ": subset membership")
        require(saved.get("candidate_ids") == subset, str(path) + ": candidate order")
        require(subset == nested_subset(native["candidate_ids"], native["component_id"], k),
                str(path) + ": seeded nested subset")
        logits = saved.get("logits")
        require(isinstance(logits, list) and len(logits) == k, str(path) + ": logits shape")
        logits = [finite(value, str(path) + ":logits") for value in logits]
        raw = distribution(saved.get("raw_probs"), k, str(path) + ": raw probabilities")
        probs = distribution(saved.get("probs"), k, str(path) + ": probabilities")
        close(raw, softmax(logits), str(path) + ": raw softmax", 1e-12, 1e-12)
        close(probs, softmax(logits, temperatures[endpoint]), str(path) + ": temperature softmax",
              1e-12, 1e-12)
        answer, ties = choice_winner(subset, probs)
        require(saved.get("answer") == answer and saved.get("ties") == ties,
                str(path) + ": subset argmax/ties")
        raw_answer, raw_ties = choice_winner(subset, raw)
        require(saved.get("raw_answer") == raw_answer and saved.get("raw_ties") == raw_ties,
                str(path) + ": subset raw argmax/ties")
        in_subset = gold in subset
        require(saved.get("gold_in_shortlist") is in_subset,
                str(path) + ": input-only gold membership")
        rank = gold_rank(subset, probs, gold)
        require("gold_rank" in saved and saved["gold_rank"] == rank
                and (rank is None or type(saved["gold_rank"]) is int),
                str(path) + ": gold rank accounting")
        by_decision[(endpoint, rid)].append((k, subset, answer, in_subset))
    require(by_decision, str(path) + ": empty candidate-count rows")
    require({rid for _, rid in by_decision} == set(expected),
            str(path) + ": missing candidate-count decisions")
    summary = {}
    for (endpoint, rid), entries in by_decision.items():
        entries.sort(key=lambda item: item[0])
        sizes = [item[0] for item in entries]
        require(len(sizes) == len(set(sizes)), str(path) + ": duplicate K for one decision")
        native = expected[rid]
        k_max = len(native["candidate_ids"])
        expected_sizes = sorted({k for k in NESTED_K if k <= k_max}
                                | ({k_max} if k_max >= 2 else set()))
        require(sizes == expected_sizes, str(path) + ": registered K grid")
        previous = None
        for k, subset, _, _ in entries:
            if previous is not None:
                require(set(previous) < set(subset), str(path) + ": nested subsets")
            previous = subset
        summary[(endpoint, rid)] = entries
    return summary


def summarize_candidate_count(summary, natives):
    """Separate model accuracy from random input-only catalogue membership."""
    per_endpoint = defaultdict(lambda: defaultdict(
        lambda: {"rows": 0, "correct": 0, "gold_present": 0}))
    for (endpoint, rid), entries in summary.items():
        native = natives[endpoint][rid]
        for k, _, answer, in_subset in entries:
            slot = per_endpoint[endpoint][k]
            slot["rows"] += 1
            slot["correct"] += int(answer == native["gold"])
            slot["gold_present"] += int(in_subset)
    result = {}
    for endpoint, by_k in per_endpoint.items():
        ks = sorted(by_k)
        table = {}
        for k in ks:
            slot = by_k[k]
            table[str(k)] = {
                "rows": slot["rows"], "gold_present_rows": slot["gold_present"],
                "overall_accuracy": slot["correct"] / slot["rows"],
                "conditional_accuracy_gold_present": (
                    slot["correct"] / slot["gold_present"] if slot["gold_present"] else None),
                "membership_frequency": slot["gold_present"] / slot["rows"],
            }
        frequencies = [table[str(k)]["membership_frequency"] for k in ks]
        monotone = all(b >= a - 1e-12 for a, b in zip(frequencies, frequencies[1:]))
        result[endpoint] = {
            "grid": ks, "by_k": table, "monotone_membership_frequency": monotone,
            "input_only_random_subsets": True, "learned_retrieval_credit": False,
        }
    return result


def read_coordinate_reversal(path, natives, dev, temperatures, label):
    expected = {rid: row for rows in natives.values() for rid, row in rows.items()}
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(isinstance(rid, str) and rid in expected and rid not in seen,
                str(path) + ": extra/duplicate reversal ID")
        seen.add(rid)
        native = expected[rid]
        endpoint = native["endpoint"]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale"):
            require(saved.get(key) == native[key], str(path) + ": native " + key)
        require(saved.get("reversed") is True, str(path) + ": reversal flag")
        require(set(saved) == DEV_KEYS | {"reversed", "state_encode_calls_before",
                                          "state_encode_calls_after"},
                str(path) + ": reversal row keys")
        require(type(saved["state_encode_calls_before"]) is int
                and saved["state_encode_calls_after"] == saved["state_encode_calls_before"],
                str(path) + ": reversal called the encoder")
        catalogue = native["candidate_ids"]
        require(saved.get("candidate_ids") == list(reversed(catalogue)),
                str(path) + ": reversed coordinate order")
        original = dev[endpoint][rid]
        count = len(catalogue)
        logits = saved.get("logits")
        require(isinstance(logits, list) and len(logits) == count, str(path) + ": logits shape")
        logits = [finite(value, str(path) + ":logits") for value in logits]
        raw = distribution(saved.get("raw_probs"), count, str(path) + ": raw probabilities")
        probs = distribution(saved.get("probs"), count, str(path) + ": probabilities")
        close(logits, list(reversed(original["logits"])), str(path) + ": reversed logits", 0, 0)
        close(raw, list(reversed(original["raw_probs"])), str(path) + ": reversed raw", 0, 0)
        close(probs, list(reversed(original["probs"])), str(path) + ": reversed probs", 0, 0)
        require(saved.get("answer") == original["answer"]
                and saved.get("ties") == original["ties"],
                str(path) + ": reversal changed the semantic winner")
    require(seen == set(expected), str(path) + ": missing coordinate reversals")
    return {"status": "passed", "rows": len(seen),
            "scope": "Reversed coordinate gathers reproduce the native winner by ID"}


def repeated_probe_plan(natives, fit_natives):
    """Choose both public catalogues without consulting either decision's target."""
    dev_rows = [row for rows in natives.values() for row in rows.values()]
    require(dev_rows and len(natives) == 2, "two eligible repeated-probe endpoints")
    native = min(dev_rows, key=lambda row: (row["state_id"], row["id"]))
    foreign_rows = [row for endpoint, rows in fit_natives.items()
                    if endpoint in natives and endpoint != native["endpoint"]
                    for row in rows.values()]
    require(foreign_rows, "other endpoint FIT catalogue")
    foreign = min(foreign_rows, key=lambda row: row["id"])
    return native, foreign


def read_repeated_state(path, natives, fit_natives, temperatures, arch):
    """Reconcile fresh-cache exposure and identical-repeat memoization."""
    require(arch in ARCHS, str(path) + ": architecture")
    native, foreign = repeated_probe_plan(natives, fit_natives)
    plan = (native, foreign)
    counters = ("state_encode_calls", "encoder_forward_calls", "encoded_states")
    keys = DEV_KEYS | {
        "state_id", "native_state_decision_id", "catalogue_source_decision_id",
        "probe_kind", "semantic_metric_eligible", "repeat_index",
    } | {name + suffix for name in counters for suffix in ("_before", "_after")}
    rows = list(json_rows(path))
    require(len(rows) == len(plan) * REPEATED_STATE_REPEATS,
            str(path) + ": complete two-catalogue repeat schedule")
    previous = dict.fromkeys(counters, 0)
    first_outputs = {}
    first_round = None
    first_costs = []
    for position, saved in enumerate(rows):
        repeat, slot = divmod(position, len(plan))
        source = plan[slot]
        endpoint = source["endpoint"]
        require(set(saved) == keys, str(path) + ": unlabelled probe row keys")
        require(type(saved["repeat_index"]) is int and saved["repeat_index"] == repeat,
                str(path) + ": round-major repeat schedule")
        require(saved["state_id"] == native["state_id"]
                and saved["native_state_decision_id"] == native["id"]
                and saved["catalogue_source_decision_id"] == source["id"],
                str(path) + ": native state/public catalogue source identity")
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale",
                    "candidate_ids"):
            require(saved[key] == source[key], str(path) + ": catalogue source " + key)
        require(saved["probe_kind"] == ("native" if slot == 0 else "unlabelled_foreign_catalogue")
                and saved["semantic_metric_eligible"] is (slot == 0),
                str(path) + ": foreign catalogue is unlabelled")
        logits, raw, probs, answer, ties = validate_field(
            saved, source, str(path), temperatures, endpoint)
        output = {"logits": logits, "raw_probs": raw, "probs": probs,
                  "answer": answer, "ties": ties,
                  "raw_answer": saved["raw_answer"], "raw_ties": saved["raw_ties"]}
        current, deltas = {}, {}
        for name in counters:
            before, after = saved[name + "_before"], saved[name + "_after"]
            require(type(before) is int and type(after) is int
                    and before == previous[name] and after >= before,
                    str(path) + ": fresh chained " + name)
            current[name], deltas[name] = after, after - before
        if repeat == 0:
            expected = {
                "state_encode_calls": 1 if arch == "cross" or slot == 0 else 0,
                "encoder_forward_calls": 1 if arch == "cross" or slot == 1 else 2,
                "encoded_states": len(source["candidate_ids"]) if arch == "cross"
                else int(slot == 0),
            }
            require(deltas == expected, str(path) + ": first-round architecture exposure")
            first_costs.append({"catalogue_source_decision_id": source["id"],
                                "endpoint": endpoint, **deltas})
            first_outputs[slot] = output
            if slot == len(plan) - 1:
                first_round = current.copy()
        else:
            require(all(delta == 0 for delta in deltas.values()),
                    str(path) + ": identical-repeat encoder memoization")
            close(output, first_outputs[slot], str(path) + ": identical repeat output", 0, 0)
        previous = current
    extra = {name: previous[name] - first_round[name] for name in counters}
    return {
        "status": "passed", "state_id": native["state_id"],
        "native_state_decision_id": native["id"],
        "catalogue_source_decision_ids": [source["id"] for source in plan],
        "catalogue_endpoints": [source["endpoint"] for source in plan],
        "catalogue_count": len(plan), "repeats": REPEATED_STATE_REPEATS,
        **{"first_round_" + name: value for name, value in first_round.items()},
        "state_encodes_total": previous["state_encode_calls"],
        "encoder_forwards_total": previous["encoder_forward_calls"],
        "encoded_states_total": previous["encoded_states"],
        "extra_state_encodes": extra["state_encode_calls"],
        "extra_encoder_forwards": extra["encoder_forward_calls"],
        "extra_encoded_states": extra["encoded_states"],
        "first_round_catalogue_costs": first_costs,
        "novel_catalogue_state_reuse": True,
        "foreign_catalogue_semantic_metrics": False, "learned_retrieval_credit": 0,
        "scope": "Unlabelled computational probe; exact repeats prove generic memoization only",
    }


def verify_artifacts(directory, metadata, required, label):
    artifacts = metadata["artifact_hashes"]
    require(isinstance(artifacts, dict), label + ": artifact manifest")
    require(set(required) <= set(artifacts), label + ": omitted mandatory artifact")
    verified = {}
    for name, entry in artifacts.items():
        require(not Path(name).is_absolute(), label + ": absolute artifact name")
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()),
                label + ": artifact escapes directory")
        expected = entry["sha256"]
        require(isinstance(expected, str) and len(expected) == 64, label + ": malformed hash")
        require(path.is_file() and path.stat().st_size == entry["bytes"]
                and digest(path) == expected, label + ": artifact hash mismatch " + name)
        verified[name] = expected
    return verified


def read_pooled_reference(root, natives):
    """Frozen NATIVE-1 full-arm dev predictions as the pooled reference."""
    directory = root / "full"
    metadata = load_json(directory / "metadata.json")
    require(metadata["schema"] == "vey.native-field.arm.v1" and metadata["arm"] == "full"
            and metadata["mode"] == "train" and metadata["status"] == "train_complete",
            "pooled reference run identity")
    require(metadata["protocol_sha256"] == digest(HERE / "native_field_protocol.json"),
            "pooled reference NATIVE-1 protocol binding")
    require(metadata["dataset_manifest_sha256"] == digest(root / "dataset_manifest.json"),
            "pooled reference dataset binding")
    verify_artifacts(directory, metadata, required={"dev.jsonl"}, label="pooled")
    temperatures = metadata["temperatures"]
    require(set(temperatures) == set(natives), "pooled reference temperature endpoints")
    rows, _ = read_dev_predictions(directory / "dev.jsonl", natives, "pooled", temperatures,
                                   "dev", {})
    return {"metadata_sha256": digest(directory / "metadata.json"),
            "temperatures": temperatures, "dev": rows,
            "scope": "Frozen NATIVE-1 full-arm dev predictions; artifact and dataset hashes "
                     "verified and predictions re-derived from logits."}


def checkpoint_fingerprint(path, arch):
    """Hash actual safetensors model and architecture-specific encoder payloads."""
    prefixes = {"cross": "scorer.pool.encoder.", "dual": "scorer.pool.encoder.",
                "pages": "encoder."}
    require(arch in prefixes, "checkpoint architecture")
    encoder_prefix = prefixes[arch]
    types = {"F32": ("torch.float32", 4), "F64": ("torch.float64", 8),
             "F16": ("torch.float16", 2), "BF16": ("torch.bfloat16", 2),
             "I64": ("torch.int64", 8), "I32": ("torch.int32", 4),
             "I16": ("torch.int16", 2), "I8": ("torch.int8", 1),
             "U8": ("torch.uint8", 1), "BOOL": ("torch.bool", 1)}
    path = Path(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, "truncated safetensors header length")
        length = int.from_bytes(prefix, "little")
        require(0 < length <= min(path.stat().st_size - 8, 100_000_000),
                "invalid safetensors header size")
        header = NF.decode(stream.read(length).decode("utf-8"))
        require(isinstance(header, dict), "safetensors header object")
        metadata = header.get("__metadata__")
        require(isinstance(metadata, dict)
                and metadata.get("schema") == "vey.native-field.checkpoint.v1",
                "native.save_checkpoint safetensors schema")
        tensors = {name: item for name, item in header.items() if name != "__metadata__"}
        require(tensors, "empty selected checkpoint")
        payload_size = path.stat().st_size - 8 - length
        ranges, model_hash, encoder_hash = [], hashlib.sha256(), hashlib.sha256()
        encoder_count = 0
        for name in sorted(tensors):
            item = tensors[name]
            require(isinstance(item, dict) and set(item) == {"dtype", "shape", "data_offsets"},
                    "safetensors tensor schema")
            dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
            require(isinstance(dtype, str) and dtype in types and isinstance(shape, list)
                    and all(type(dim) is int and dim >= 0 for dim in shape),
                    "invalid tensor dtype/shape")
            require(isinstance(offsets, list) and len(offsets) == 2
                    and all(type(offset) is int for offset in offsets), "invalid tensor offsets")
            start, end = offsets
            require(0 <= start <= end <= payload_size
                    and end - start == math.prod(shape) * types[dtype][1],
                    "tensor payload extent")
            ranges.append((start, end))
            encoder_tensor = name.startswith(encoder_prefix)
            for hashed, tensor_name in ((model_hash, name), (encoder_hash, name[len(encoder_prefix):])):
                if hashed is encoder_hash and not encoder_tensor:
                    continue
                rendered = json.dumps([tensor_name, types[dtype][0], shape],
                                      sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False, allow_nan=False).encode("utf-8")
                hashed.update(len(rendered).to_bytes(8, "little"))
                hashed.update(rendered)
            encoder_count += int(encoder_tensor)
            stream.seek(8 + length + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(remaining, 1 << 20))
                require(block, "truncated tensor payload")
                model_hash.update(block)
                if encoder_tensor:
                    encoder_hash.update(block)
                remaining -= len(block)
        previous = 0
        for start, end in sorted(ranges):
            require(start == previous, "overlapping/gapped safetensors tensors")
            previous = end
        require(previous == payload_size, "unaccounted checkpoint payload bytes")
        require(encoder_count > 0, "missing architecture encoder tensors")
    actual = model_hash.hexdigest()
    require(metadata.get("state_dict_sha256") == actual, "checkpoint metadata/tensor fingerprint")
    return actual, encoder_hash.hexdigest(), encoder_prefix


def check_restore(restore, probe_natives, endpoints, checkpoint, checkpoint_sha, arch):
    expected = restore["expected_fingerprint"]
    require(isinstance(expected, str) and len(expected) == 64
            and restore["restored_fingerprint"] == expected
            and restore["missing_keys"] == [] and restore["unexpected_keys"] == [],
            "strict restore fingerprint/keys")
    require(probe_natives and restore.get("probe_state_ids") ==
            list(dict.fromkeys(row["state_id"] for row in probe_natives.values())),
            "restore probe native states")
    before, after = restore["selected_rows"], restore["restored_rows"]
    require(isinstance(before, list) and before and isinstance(after, list)
            and len(after) == len(before), "restore probe rows")
    seen = set()
    for selected, restored in zip(before, after):
        rid = selected.get("id")
        require(rid in probe_natives and rid not in seen, "restore probe native decision")
        seen.add(rid)
        native = probe_natives[rid]
        for row in (selected, restored):
            for key in ("id", "endpoint", "task", "component_id", "group_id", "locale",
                        "candidate_ids"):
                require(row.get(key) == native[key], "restore probe native " + key)
            count = len(native["candidate_ids"])
            logits = row.get("logits")
            require(isinstance(logits, list) and len(logits) == count, "restore probe logits")
            logits = [finite(value, "restore probe logits") for value in logits]
            raw = distribution(row.get("raw_probs"), count, "restore probe raw probabilities")
            close(raw, softmax(logits), "restore probe raw softmax", 1e-12, 1e-12)
    require(seen == set(probe_natives), "restore probe covers the probe states")
    require({row["endpoint"] for row in before} == set(endpoints),
            "restore probe intent heads")
    difference = NF.numerical_max_difference(
        [{key: row[key] for key in ("logits", "raw_probs")} for row in before],
        [{key: row[key] for key in ("logits", "raw_probs")} for row in after])
    close(restore["max_abs_difference"], difference, "reconstructed restore drift", 1e-12, 1e-12)
    require(difference == 0, "strict restore numerical equality")
    require(checkpoint is not None and Path(checkpoint).is_file()
            and checkpoint_sha == digest(checkpoint)
            and restore.get("checkpoint_sha256") == checkpoint_sha,
            "restore checkpoint artifact identity")
    reconstructed, encoder, encoder_prefix = checkpoint_fingerprint(checkpoint, arch)
    require(reconstructed == expected, "independent selected checkpoint tensor bytes")
    return {"state_dict_fingerprint": reconstructed,
            "encoder_state_dict_fingerprint": encoder, "encoder_prefix": encoder_prefix,
            "numerical_max_abs_difference": difference,
            "tensor_fingerprint_independently_reconstructed": True,
            "scope": "Actual model/encoder bytes independently hashed; "
                     "restore-forward equality is a receipt, not an independent forward replay"}


def check_selected_checkpoint(metadata, history, liveness, restored, checkpoint, checkpoint_sha):
    """Bind independently reconstructed bytes to every selected-model receipt."""
    selected = metadata["checkpoint"]
    strict = liveness["selected_restore"]
    epoch, model_hash = history["selected_epoch"], restored["state_dict_fingerprint"]
    require(selected.get("schema") == "vey.native-field.checkpoint.v1"
            and Path(selected["path"]).resolve() == Path(checkpoint).resolve()
            and selected["sha256"] == checkpoint_sha
            and strict["checkpoint_sha256"] == checkpoint_sha,
            "selected checkpoint schema/path/file identity")
    require(selected["selected_epoch"] == strict["selected_epoch"] == epoch
            and selected["state_dict_fingerprint"] == history["selected_fingerprint"] == model_hash
            and history["epochs"][epoch]["state_dict_fingerprint"] == model_hash,
            "selected checkpoint actual model/history fingerprint")
    require(strict["before_restore_fingerprint"] == history["epochs"][-1]["state_dict_fingerprint"],
            "actual pre-restore final epoch fingerprint")
    require(selected["encoder_prefix"] == restored["encoder_prefix"]
            and selected["encoder_state_dict_fingerprint"] ==
            liveness["encoder_selected_sha256"] == restored["encoder_state_dict_fingerprint"],
            "selected checkpoint actual architecture encoder bytes")


def reconstruct_selection(path, selection_natives, endpoints, cfg):
    """Independently recompute per-epoch selection metrics and the chosen epoch."""
    expected = {rid: row for rows in selection_natives.values() for rid, row in rows.items()}
    per_epoch = defaultdict(lambda: {endpoint: {"nll": [], "acc": []} for endpoint in endpoints})
    seen = set()
    for saved in json_rows(path):
        epoch, rid = saved.get("epoch"), saved.get("id")
        require(type(epoch) is int and rid in expected and (epoch, rid) not in seen,
                str(path) + ": extra/duplicate selection row")
        seen.add((epoch, rid))
        native = expected[rid]
        endpoint = native["endpoint"]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale",
                    "candidate_ids"):
            require(saved.get(key) == native[key], str(path) + ": selection native " + key)
        count = len(native["candidate_ids"])
        logits = saved.get("logits")
        require(isinstance(logits, list) and len(logits) == count, str(path) + ": selection logits")
        logits = [finite(value, str(path) + ":selection logits") for value in logits]
        raw = distribution(saved.get("raw_probs"), count, str(path) + ": selection raw")
        close(raw, softmax(logits), str(path) + ": selection raw softmax", 1e-12, 1e-12)
        close(saved.get("target_distribution"), native["target_distribution"],
              str(path) + ": selection target", 0, 0)
        if "gold" in saved:
            require(saved.get("gold") == native.get("gold"), str(path) + ": selection gold")
        nll = -math.fsum(t * math.log(max(p, 1e-12))
                         for t, p in zip(native["target_distribution"], raw))
        answer, _ = choice_winner(native["candidate_ids"], raw)
        per_epoch[epoch][endpoint]["nll"].append(nll)
        per_epoch[epoch][endpoint]["acc"].append(int(answer == native["gold"]))
    epochs = sorted(per_epoch)
    require(epochs == list(range(cfg["training"]["epochs"] + 1)),
            str(path) + ": complete epoch0..N selection")
    require(seen == {(epoch, rid) for epoch in epochs for rid in expected},
            str(path) + ": selection row membership")
    metrics = {}
    for epoch in epochs:
        by_endpoint = per_epoch[epoch]
        require(set(by_endpoint) == set(endpoints), str(path) + ": selection endpoint membership")
        accuracy = {endpoint: math.fsum(v["acc"]) / len(v["acc"])
                    for endpoint, v in by_endpoint.items()}
        nll = {endpoint: math.fsum(v["nll"]) / len(v["nll"])
               for endpoint, v in by_endpoint.items()}
        metrics[epoch] = {
            "macro_source_intent_accuracy": math.fsum(accuracy.values()) / len(accuracy),
            "mean_endpoint_nll": math.fsum(nll.values()) / len(nll),
            "source_intent_accuracy": accuracy, "endpoint_nll": nll}
    chosen = min(epochs, key=lambda e: (-metrics[e]["macro_source_intent_accuracy"],
                                        metrics[e]["mean_endpoint_nll"], e))
    return chosen, metrics


def reconstruct_calibration(path, calibration_natives, endpoints, temperatures):
    """Independently recompute the bounded temperature grid optimum per endpoint."""
    expected = {rid: row for rows in calibration_natives.values() for rid, row in rows.items()}
    logits_by = {endpoint: [] for endpoint in endpoints}
    targets = {endpoint: [] for endpoint in endpoints}
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(rid in expected and rid not in seen, str(path) + ": extra/duplicate calibration row")
        seen.add(rid)
        native = expected[rid]
        endpoint = native["endpoint"]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale",
                    "candidate_ids"):
            require(saved.get(key) == native[key], str(path) + ": calibration native " + key)
        count = len(native["candidate_ids"])
        logits = saved.get("logits")
        require(isinstance(logits, list) and len(logits) == count, str(path) + ": calibration logits")
        logits = [finite(value, str(path) + ":calibration logits") for value in logits]
        raw = distribution(saved.get("raw_probs"), count, str(path) + ": calibration raw")
        close(raw, softmax(logits), str(path) + ": calibration raw softmax", 1e-12, 1e-12)
        close(saved.get("target_distribution"), native["target_distribution"],
              str(path) + ": calibration target", 0, 0)
        logits_by[endpoint].append(logits)
        targets[endpoint].append(native["target_distribution"])
    require(seen == set(expected), str(path) + ": calibration row membership")
    grid = [math.exp(math.log(.5) + i * (math.log(10.) - math.log(.5)) / 100) for i in range(101)]
    grid[0], grid[-1] = .5, 10.
    evidence = {}
    for endpoint in endpoints:
        require(bool(logits_by[endpoint]), str(path) + ": empty calibration endpoint")
        losses = []
        for temperature in grid:
            values = [-math.fsum(t * math.log(max(p, 1e-12))
                                 for p, t in zip(softmax(z, temperature), target))
                      for z, target in zip(logits_by[endpoint], targets[endpoint])]
            losses.append(math.fsum(values) / len(values))
        index = min(range(len(grid)), key=lambda i: (losses[i], abs(grid[i] - 1.), grid[i]))
        close(temperatures[endpoint], grid[index], str(path) + ": independent temperature", 1e-12, 1e-12)
        evidence[endpoint] = {"rows": len(logits_by[endpoint]), "temperature": grid[index],
                              "nll": losses[index]}
    return evidence


def check_history(history, selection, endpoints, cfg):
    """Compare independently recomputed epoch metrics and selected epoch to the receipt."""
    epochs = history["epochs"]
    require([row["epoch"] for row in epochs] == list(range(cfg["training"]["epochs"] + 1)),
            "complete epoch0..N history")
    for entry in epochs:
        rebuilt = selection[entry["epoch"]]
        saved = entry["selection"]
        close(saved["macro_source_intent_accuracy"], rebuilt["macro_source_intent_accuracy"],
              "independent macro intent accuracy", 1e-12, 1e-12)
        close(saved["mean_endpoint_nll"], rebuilt["mean_endpoint_nll"],
              "independent mean endpoint NLL", 1e-12, 1e-12)
        close(saved["source_intent_accuracy"], rebuilt["source_intent_accuracy"],
              "independent per-source intent accuracy", 1e-12, 1e-12)
        close(saved["endpoint_nll"], rebuilt["endpoint_nll"],
              "independent per-endpoint NLL", 1e-12, 1e-12)
        require(isinstance(entry["state_dict_fingerprint"], str)
                and len(entry["state_dict_fingerprint"]) == 64, "epoch tensor fingerprint")
    chosen = min(epochs, key=lambda row: (-selection[row["epoch"]]["macro_source_intent_accuracy"],
                                          selection[row["epoch"]]["mean_endpoint_nll"],
                                          row["epoch"]))["epoch"]
    require(history["selected_epoch"] == chosen
            and history["selected_fingerprint"] == epochs[chosen]["state_dict_fingerprint"],
            "independent selected epoch/fingerprint")
    return {"epochs_including0": len(epochs), "selected_epoch": chosen,
            "scope": "Selection metrics independently re-derived from selection logits; "
                     "state_dict fingerprints remain persisted receipts"}


def verify_arch(root, arch, records, phases, natives, plans, cfg):
    """Return (report, dev_rows) for one architecture directory."""
    directory = root / arch
    if not directory.is_dir():
        return {"status": "no-data", "reason": "missing architecture directory"}, None
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return {"status": "no-data", "reason": "missing metadata.json"}, None
    metadata = load_json(metadata_path)
    if metadata.get("mode") != "train" or metadata.get("status") != "train_complete":
        return {"status": "failed",
                "reason": "run not train_complete: " + repr(metadata.get("status"))}, None
    required = ({name + ".jsonl" for name in ARCH_FILES} | set(TRAINED_FILES)
                | {"history.json", "liveness.json"})
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        return {"status": "failed",
                "reason": "missing prediction artifacts: " + repr(missing)}, None
    require(metadata["schema"] == "vey.native-arch.arm.v1" and metadata["arm"] == arch,
            arch + ": arm identity")
    require(metadata["protocol_sha256"] == digest(HERE / "native_arch_protocol.json"),
            arch + ": protocol hash")
    require(metadata["dataset_manifest_sha256"] == digest(root / "dataset_manifest.json"),
            arch + ": dataset binding")
    require(metadata["sealed_phases_accessed"] is False, arch + ": sealed phases")
    require(metadata.get("state_mask_text") == EMPTY_EVIDENCE, arch + ": empty-evidence mask text")
    for name in ("native_arch_model.py", "native_arch_train.py"):
        path = HERE / name
        require(metadata["source_files"].get(name) == digest(path),
                arch + ": trainer source identity " + name)
    eligible = set(metadata["eligible_endpoints"])
    require(eligible == set(natives), arch + ": eligible endpoints")
    temperatures = metadata["temperatures"]
    require(set(temperatures) == eligible, arch + ": complete endpoint temperatures")
    for value in temperatures.values():
        require(0.5 <= finite(value, "temperature") <= 10.0, arch + ": bounded temperature")
    artifacts = verify_artifacts(directory, metadata, required=required, label=arch)
    history = load_json(directory / "history.json")
    liveness = load_json(directory / "liveness.json")
    require(history.get("schema") == "vey.native-arch.history.v1" and history.get("arm") == arch,
            arch + ": history identity")
    require(liveness.get("schema") == "vey.native-arch.liveness.v1" and liveness.get("arm") == arch,
            arch + ": liveness identity")
    report = {"status": "passed", "artifact_hashes": artifacts,
              "metadata_sha256": digest(metadata_path), "temperatures": temperatures}

    smoke = liveness["smoke"]
    require(smoke.get("smoke_weights_discarded") is True, arch + ": discarded smoke")
    require(finite(smoke["loss"], "smoke loss") >= 0, arch + ": smoke loss")
    require(smoke.get("encoder_before_sha256") != smoke.get("encoder_after_sha256"),
            arch + ": smoke encoder liveness")
    report["smoke"] = {"loss": smoke["loss"], "encoder_changed": True,
                       "scope": "Persisted finite smoke receipt; smoke is discarded, not fit quality"}

    epochs = history["epochs"]
    selection_natives = {endpoint: records["selection"][endpoint] for endpoint in natives}
    selected_epoch, selection_metrics = reconstruct_selection(
        directory / "selection.jsonl", selection_natives, set(natives), cfg)
    report["history"] = check_history(history, selection_metrics, set(natives), cfg)
    require(history["selected_epoch"] == selected_epoch, arch + ": selected epoch")

    checkpoint = directory / "selected.safetensors"
    covered, probe_natives = set(), {}
    for state in phases["selection"]:
        decisions = [decision for decision in state["decisions"] if decision["endpoint"] in natives]
        added = {decision["endpoint"] for decision in decisions} - covered
        if added:
            for decision in decisions:
                probe_natives[decision["id"]] = records["selection"][decision["endpoint"]][decision["id"]]
            covered.update(added)
        if covered == set(natives):
            break
    require(covered == set(natives), arch + ": restore probe covers eligible endpoints")
    report["selected_restore"] = check_restore(
        liveness["selected_restore"], probe_natives, set(natives), checkpoint,
        artifacts["selected.safetensors"], arch)
    check_selected_checkpoint(metadata, history, liveness, report["selected_restore"],
                              checkpoint, artifacts["selected.safetensors"])
    require(liveness["encoder_training_before_sha256"] == metadata["custody"]["encoder_initial_sha256"],
            arch + ": fresh encoder training custody")
    require(liveness["encoder_training_before_sha256"] != liveness["encoder_training_after_sha256"],
            arch + ": trained encoder liveness")

    dev, _ = read_dev_predictions(directory / "dev.jsonl", natives, arch, temperatures,
                                  "dev", plans)
    masked, _ = read_dev_predictions(directory / "masked.jsonl", natives, arch, temperatures,
                                     "masked", plans)
    swapped, unavailable = read_dev_predictions(directory / "swapped.jsonl", natives, arch,
                                                temperatures, "swapped", plans)
    report["masked_constant_rows"] = check_masked_constant(masked, natives)
    report["swap_control"] = check_swap_control(dev, swapped, natives, plans)
    report["field_controls"] = {}
    for endpoint in natives:
        masked_rows = masked[endpoint]
        swapped_rows = {rid: row for rid, row in swapped[endpoint].items() if row is not None}
        report["field_controls"][endpoint] = {
            "masked_accuracy": sum(int(row["answer"] == row["gold"])
                                   for row in masked_rows.values()) / len(masked_rows),
            "swapped_accuracy_original_target": (
                sum(int(row["answer"] == natives[endpoint][rid]["gold"])
                    for rid, row in swapped_rows.items()) / len(swapped_rows)
                if swapped_rows else None),
            "swapped_rows": len(swapped_rows),
        }
    expected_denominators = {}
    for endpoint, plan in plans.items():
        available = [donor for donor in plan.values() if donor is not None]
        expected_denominators[endpoint] = {
            "total": len(natives[endpoint]), "available": len(available),
            "unavailable": len(natives[endpoint]) - len(available),
            "unique_donors": len(set(available))}
    require(metadata["swap_denominators"] == expected_denominators,
            arch + ": independent donor denominators")
    require({endpoint for endpoint, _ in unavailable}
            == {endpoint for endpoint, plan in plans.items()
                for donor in plan.values() if donor is None},
            arch + ": donor-unavailable membership")
    report["swap_denominators"] = expected_denominators

    candidate = read_candidate_count(directory / "candidate_count.jsonl", natives, arch,
                                     temperatures)
    report["candidate_count"] = summarize_candidate_count(candidate, natives)
    report["coordinate_reversal"] = read_coordinate_reversal(
        directory / "coordinate_reversal.jsonl", natives, dev, temperatures, arch)
    fit_natives = {endpoint: records["fit"][endpoint] for endpoint in natives}
    probe_plan = repeated_probe_plan(natives, fit_natives)
    require(sum(len(source["candidate_ids"]) for source in probe_plan) == 137,
            arch + ": registered two-catalogue exposure count")
    report["repeated_state"] = read_repeated_state(
        directory / "repeated_state.jsonl", natives, fit_natives, temperatures, arch)
    saved_probe = liveness["controls"]["repeated_state"]
    require(isinstance(saved_probe, dict) and set(saved_probe) ==
            set(report["repeated_state"]) -
            {"status", "first_round_catalogue_costs", "novel_catalogue_state_reuse", "scope"},
            arch + ": complete repeated-state receipt")
    for key, value in saved_probe.items():
        require(key in report["repeated_state"], arch + ": unexpected repeated-state receipt " + key)
        close(value, report["repeated_state"][key], arch + ": repeated-state receipt " + key, 0, 0)
    require(liveness["controls"]["candidate_count_learned_retrieval_credit"] == 0,
            arch + ": random subset membership earns no learned retrieval credit")
    calibration_natives = {endpoint: records["calibration"][endpoint] for endpoint in natives}
    report["calibration"] = reconstruct_calibration(
        directory / "calibration.jsonl", calibration_natives, set(natives), temperatures)
    evidence = metadata.get("temperature_evidence")
    require(isinstance(evidence, dict) and set(evidence) == set(natives),
            arch + ": temperature evidence membership")
    for endpoint in natives:
        close(evidence[endpoint]["selected_temperature"], temperatures[endpoint],
              arch + ": persisted temperature", 1e-12, 1e-12)
        close(evidence[endpoint]["selected_nll"], report["calibration"][endpoint]["nll"],
              arch + ": persisted calibration NLL", 1e-12, 1e-12)
    dev_metrics = metadata.get("dev_metrics")
    require(isinstance(dev_metrics, dict) and set(dev_metrics.get("calibrated", {})) == set(natives),
            arch + ": dev_metrics membership")
    for endpoint in natives:
        accuracy = sum(int(dev[endpoint][rid]["answer"] == dev[endpoint][rid]["gold"])
                       for rid in natives[endpoint]) / len(natives[endpoint])
        close(dev_metrics["calibrated"][endpoint], accuracy,
              arch + ": persisted calibrated dev accuracy", 1e-12, 1e-12)
    report["mechanism_scope"] = (
        "Prediction accuracy is reconstructed from native gold; random subset membership and "
        "the foreign-catalogue computational probe earn zero learned retrieval credit. "
        "Actual selected model/encoder bytes are independently hashed; no model or sealed phase runs.")
    return report, dev


def paired_comparison(endpoint, rows, correct, cfg):
    names = {"cross_minus_pooled_accuracy": ("cross", "pooled"),
             "pages_minus_cross_accuracy": ("pages", "cross"),
             "dual_minus_cross_accuracy": ("dual", "cross")}
    differences = {}
    for name, (left, right) in names.items():
        if left in correct and right in correct:
            differences[name] = [float(correct[left][row["id"]]) - float(correct[right][row["id"]])
                                 for row in rows]
    if not differences:
        return {}
    return NF.component_ratio_bootstrap(rows, differences,
                                        resamples=cfg["statistics"]["resamples"],
                                        seed=cfg["statistics"]["seed"])


def screen_arch(arch, accuracy, uncertainty, candidate, reversal, repeated, cfg):
    gates = cfg["screen_gates"]
    checks = {
        "candidate_count_monotone_membership_frequency": candidate["monotone_membership_frequency"],
        "reorder_winner_invariance": reversal["status"] == "passed",
        "repeated_state_no_extra_state_encode": (
            repeated["extra_state_encodes"] == repeated["extra_encoder_forwards"] ==
            repeated["extra_encoded_states"] == 0),
        "novel_catalogue_state_reuse": repeated["novel_catalogue_state_reuse"],
    }
    if arch == "cross":
        checks["cross_intent_accuracy_each_source"] = (
            accuracy >= gates["cross_intent_accuracy_each_source"])
    comparison, gate = {
        "cross": ("cross_minus_pooled_accuracy",
                  "cross_vs_pooled_accuracy_lower_nominal_bound"),
        "dual": ("dual_minus_cross_accuracy",
                 "dual_minus_cross_accuracy_lower_nominal_bound"),
        "pages": ("pages_minus_cross_accuracy",
                  "pages_minus_cross_accuracy_lower_nominal_bound"),
    }[arch]
    evidence = uncertainty.get(comparison) if isinstance(uncertainty, dict) else None
    interval = evidence.get("ci95_bootstrap") if isinstance(evidence, dict) else None
    checks[gate] = (
        isinstance(interval, (list, tuple)) and len(interval) == 2
        and all(type(value) in (int, float) and math.isfinite(value) for value in interval)
        and interval[0] <= interval[1]
        and interval[0] >= gates[gate])
    return {"status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "failed_checks": [name for name, ok in checks.items() if not ok],
            "scope": gates["scope"]}


def select_architecture(passing, architectures):
    """Choose by measured first-round state exposure, then declared arm order."""
    if not passing:
        return {"architectures": [], "earned": False, "passing": [],
                "reason": "no architecture passed its gates"}
    ordered = [arch for arch in ARCHS if arch in passing]
    exposures = {arch: architectures[arch]["repeated_state"]["first_round_encoded_states"]
                 for arch in ordered}
    require(all(type(value) is int and value > 0 for value in exposures.values()),
            "measured first-round encoded_states")
    winner = min(ordered, key=lambda arch: (exposures[arch], ARCHS.index(arch)))
    return {"architectures": [winner], "earned": True, "passing": ordered,
            "first_round_encoded_states": exposures[winner],
            "passing_first_round_encoded_states": exposures,
            "learned_retrieval_credit": 0,
            "scope": "Quality-passing development arms; exact repeats earn no selection credit"}


def verify(root: Path):
    cfg = load_json(HERE / "native_arch_protocol.json")
    require(cfg["schema"] == "vey.native-arch.protocol.v1", "registered protocol schema")
    require(cfg["statistics"]["resamples"] == 10000 and cfg["statistics"]["seed"] == 0,
            "registered bootstrap recipe")
    require(tuple(cfg["arms"]) == ARCHS, "registered architectures")
    root = root.resolve()
    require((root / "dataset_manifest.json").is_file(), "registered NATIVE-1 dataset root")
    manifest, phases = DATA.load_dataset(root)
    records, census, components, eligibility = reconstruct_natives(manifest, phases)
    endpoints = sorted(e for e, a in eligibility.items() if a["eligible"] and e.endswith(".intent"))
    require(endpoints == sorted(cfg["data"]["fields"]), "protocol/eligible intent fields")
    natives = {endpoint: records["dev"][endpoint] for endpoint in endpoints}
    priors = build_priors(records, endpoints)
    plans = build_donors(records, endpoints)
    pooled = read_pooled_reference(root, natives)

    architectures = {}
    devs = {}
    for arch in ARCHS:
        report, dev = verify_arch(root, arch, records, phases, natives, plans, cfg)
        architectures[arch] = report
        if dev is not None:
            devs[arch] = dev

    paired = {}
    screens = {}
    for endpoint in endpoints:
        rows = [{"id": rid, "component_id": natives[endpoint][rid]["component_id"],
                 "group_id": natives[endpoint][rid]["group_id"]}
                for rid in sorted(natives[endpoint])]
        correct = {}
        for arch in ARCHS:
            if arch in devs:
                correct[arch] = {rid: int(devs[arch][endpoint][rid]["answer"]
                                         == devs[arch][endpoint][rid]["gold"])
                                 for rid in natives[endpoint]}
        correct["pooled"] = {rid: int(pooled["dev"][endpoint][rid]["answer"]
                                      == pooled["dev"][endpoint][rid]["gold"])
                             for rid in natives[endpoint]}
        uncertainty = paired_comparison(endpoint, rows, correct, cfg)
        controls = {arch: architectures[arch]["field_controls"][endpoint]
                    for arch in ARCHS if arch in devs}
        paired[endpoint] = {"uncertainty": uncertainty, "controls": controls,
                            "accuracies": {name: sum(values.values()) / len(values)
                                           for name, values in correct.items()}}
        screens[endpoint] = {}
        for arch in ARCHS:
            if arch not in devs:
                screens[endpoint][arch] = {"status": "no-data",
                                           "reason": "architecture predictions absent"}
                continue
            screens[endpoint][arch] = screen_arch(
                arch, paired[endpoint]["accuracies"][arch], uncertainty,
                architectures[arch]["candidate_count"][endpoint],
                architectures[arch]["coordinate_reversal"],
                architectures[arch]["repeated_state"], cfg)

    failed = [arch for arch in ARCHS if architectures[arch]["status"] == "failed"]
    missing = [arch for arch in ARCHS if architectures[arch]["status"] == "no-data"]
    selection = {}
    for endpoint in endpoints:
        passing = [arch for arch in ARCHS
                   if screens[endpoint].get(arch, {}).get("status") == "passed"]
        selection[endpoint] = select_architecture(passing, architectures)
        if failed or missing:
            selection[endpoint].update(
                architectures=[], earned=False,
                reason="registered architecture study is failed or incomplete",
                scope="Local passing-arm diagnostics only; no architecture selection credit")
            selection[endpoint].pop("first_round_encoded_states", None)

    status = "FAIL" if failed else ("INCOMPLETE" if missing else "PASS")
    return {"schema": "vey.native-arch.verification.v1", "status": status,
            "completed_architectures": [arch for arch in ARCHS if arch not in failed + missing],
            "missing_architectures": missing, "failed_architectures": failed,
            "custody": {"protocol_sha256": digest(HERE / "native_arch_protocol.json"),
                        "dataset_manifest_sha256": digest(root / "dataset_manifest.json"),
                        "verifier_source_sha256": digest(HERE / "native_arch_verify.py"),
                        "sealed_phases_accessed": False,
                        "scope": "Committed NATIVE-1 dataset and protocol; no sealed phase read"},
            "counts": census, "components": components, "eligibility": eligibility,
            "registered_architectures": list(ARCHS),
            "fit_only_priors": priors,
            "pooled_reference": pooled,
            "architectures": architectures,
            "paired_comparisons": paired,
            "screens": screens,
            "selection": selection,
            "screen_scope": cfg["screen_gates"]["scope"],
            "screen_thresholds": cfg["screen_gates"],
            "statistics": cfg["statistics"],
            "all_negative_architectures_and_controls_retained": True,
            "claim_limits": ["English selected native dev only",
                             "fixed canonical native catalogues",
                             "nominal descriptive component intervals",
                             "restore-forward equality is not an independent model-forward replay",
                             "no sealed-final, competitor, many-axis, certificate or shipping credit"],
            "model_executed_by_verifier": False,
            "B_STEF_allowed": False, "Pareto_credit": False, "promotion": False,
            "final_credit": False, "endgame_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        receipt = verify(root)
    except Exception as error:
        receipt = {"schema": "vey.native-arch.verification.v1", "status": "FAIL",
                   "error_type": type(error).__name__, "error": str(error),
                   "registered_architectures": list(ARCHS),
                   "architectures": {arch: {"status": "verification-failed"}
                                     for arch in ARCHS},
                   "selection": {}, "model_executed_by_verifier": False,
                   "B_STEF_allowed": False, "Pareto_credit": False, "promotion": False,
                   "final_credit": False, "endgame_complete": False}
    output = root / "native_arch_verification.json"
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_suffix(".json.pending")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": receipt["status"], "verification": str(output),
                      "earned": [endpoint for endpoint, row in receipt["selection"].items()
                                 if row.get("earned")]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
