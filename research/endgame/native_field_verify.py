#!/usr/bin/env python3
"""Reconstruct NATIVE-1 development screens without loading or running a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from neutral_comparison_verify import choice_stats, score_stats as ordinal_stats

ARMS = ("prior", "frozen", "full")
PATHS = ("dev", "masked", "swapped")
PHASES = ("fit", "selection", "calibration", "dev")
ABS_TOL = 2e-5
REL_TOL = 2e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), label + ": nonfinite/non-numeric value")
    return float(value)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "duplicate JSON key: " + key)
        obj[key] = value
    return obj


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def decode(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def load_json(path):
    value = decode(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), str(path) + ": expected object")
    return value


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"{path}:{number}: blank row")
            row = decode(line)
            require(isinstance(row, dict), f"{path}:{number}: expected object")
            yield row


def close(actual, expected, label, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), label + ": keys")
        for key in expected:
            close(actual[key], expected[key], label + "." + key, abs_tol, rel_tol)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label + ": shape")
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, f"{label}[{i}]", abs_tol, rel_tol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        a, b = finite(actual, label), finite(expected, label)
        require(math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol), label + ": numeric mismatch")
    else:
        require(actual == expected, label + ": mismatch")


def distribution(value, count, label):
    require(isinstance(value, list) and len(value) == count and count > 1, label + ": shape")
    result = [finite(v, label) for v in value]
    require(all(0 <= v <= 1 for v in result), label + ": mass outside [0,1]")
    require(math.isclose(math.fsum(result), 1., abs_tol=1e-6, rel_tol=0), label + ": not normalized")
    return result


def softmax(logits, temperature=1.):
    require(logits and temperature > 0 and math.isfinite(temperature), "invalid softmax")
    scaled = [finite(v, "logit") / temperature for v in logits]
    maximum = max(scaled)
    masses = [math.exp(v - maximum) for v in scaled]
    total = math.fsum(masses)
    return [v / total for v in masses]


def semantic_winner(ids, probs):
    require(len(ids) == len(probs) and len(set(ids)) == len(ids) and bool(ids), "invalid coordinates")
    top = max(probs)
    return min(cid for cid, value in zip(ids, probs) if value == top)


def cdf(probs):
    return [math.fsum(probs[:i + 1]) for i in range(len(probs))]


def expectation(probs):
    return math.fsum(i * value for i, value in enumerate(probs))


def nmae(row):
    return abs(row["expected_level"] - row["mean_level"]) / row["native_level_max"]


def nrps(row):
    predicted, target = cdf(row["probs"]), cdf(row["target_distribution"])
    return math.fsum((p - t) ** 2 for p, t in zip(predicted[:-1], target[:-1])) / row["native_level_max"]


def component_ratio_bootstrap(rows, differences, resamples=10000, seed=0):
    """Bootstrap component sums/counts, never unweighted component means."""
    require(bool(rows) and bool(differences) and resamples > 0, "empty paired population")
    ids = [r["id"] for r in rows]
    require(len(set(ids)) == len(ids), "duplicate paired IDs")
    components = sorted({r["component_id"] for r in rows})
    positions = {component: i for i, component in enumerate(components)}
    slots = np.asarray([positions[r["component_id"]] for r in rows], dtype=np.int64)
    counts = np.bincount(slots, minlength=len(components)).astype(np.float64)
    names = list(differences)
    sums = []
    for name in names:
        values = differences[name]
        require(len(values) == len(rows), "paired metric population mismatch: " + name)
        values = [finite(v, name) for v in values]
        sums.append(np.bincount(slots, weights=values, minlength=len(components)))
    sums = np.stack(sums, axis=1)
    rng = np.random.default_rng(seed)
    samples = np.empty((resamples, len(names)), dtype=np.float64)
    for start in range(0, resamples, 128):
        end = min(resamples, start + 128)
        indices = rng.integers(0, len(components), size=(end - start, len(components)))
        samples[start:end] = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
    samples.sort(axis=0)
    result = {}
    for column, name in enumerate(names):
        result[name] = {
            "observed_delta": math.fsum(differences[name]) / len(rows),
            "ci95_bootstrap": [float(samples[int(.025 * resamples), column]),
                               float(samples[min(resamples - 1, int(.975 * resamples)), column])],
            "rows": len(rows), "components": len(components),
            "lineage_groups": len({r["group_id"] for r in rows}),
            "resamples": resamples, "seed": seed,
            "unit": "input-only connected component",
            "estimand": "ratio of summed paired row differences to summed row counts",
            "interval_scope": "Nominal descriptive 95%; not a simultaneous non-inferiority gate",
        }
    return result


def donor_plan(rows):
    """The cyclic search is not a permutation; donor reuse is recorded explicitly."""
    ordered = sorted(rows, key=lambda r: r["id"])
    require(len({r["id"] for r in ordered}) == len(ordered), "duplicate donor population IDs")
    require(not ordered or all(other["endpoint"] == ordered[0]["endpoint"] for other in ordered),
            "mixed donor endpoints")
    plan = {}
    for i, row in enumerate(ordered):
        target = row["gold"] if row["task"] == "choice" else row["mean_level"]
        for offset in range(1, len(ordered)):
            other = ordered[(i + offset) % len(ordered)]
            other_target = other["gold"] if other["task"] == "choice" else other["mean_level"]
            if other["component_id"] != row["component_id"] and other_target != target:
                plan[row["id"]] = other["id"]
                break
        plan.setdefault(row["id"], None)
    return plan


def reconstruct_dataset(manifest, phases, cfg):
    require(set(phases) == set(PHASES), "exact dataset phase membership")
    fields = cfg["model"]["fields"]
    require(len(fields) == len(set(fields)) and set(manifest["catalogues"]) == set(fields), "registered catalogues")
    require(set(manifest["phases"]) == set(PHASES), "manifest phase membership")
    records, census, components = {}, {}, {}
    all_ids, all_states = set(), set()
    component_phases = defaultdict(set)
    for phase in PHASES:
        records[phase] = {endpoint: {} for endpoint in fields}
        census[phase] = {}
        states = phases[phase]
        require(isinstance(states, list), "dataset phase is not a state list")
        phase_state_ids = set()
        for state in states:
            sid = state["id"]
            require(isinstance(sid, str) and sid and sid not in all_states, "duplicate/invalid native state ID")
            all_states.add(sid)
            phase_state_ids.add(sid)
            require(isinstance(state["state"], str), "canonical state text")
            for key in ("component_id", "group_id", "locale", "source_id"):
                require(isinstance(state[key], str) and state[key], "invalid native " + key)
            component = state["component_id"]
            component_phases[component].add(phase)
            if phase != "dev":
                bucket = int(hashlib.sha256(("native-field-v1|7|" + component).encode()).hexdigest(), 16) % 100
                expected_phase = "fit" if bucket < 70 else "selection" if bucket < 85 else "calibration"
                require(phase == expected_phase, "independent inner component split mismatch")
            require(isinstance(state["decisions"], list) and bool(state["decisions"]), "empty native state decisions")
            state_endpoints = set()
            for decision in state["decisions"]:
                endpoint, rid = decision["endpoint"], decision["id"]
                require(endpoint in fields and endpoint not in state_endpoints, "unsupported/repeated native endpoint")
                state_endpoints.add(endpoint)
                require(isinstance(rid, str) and rid and rid not in all_ids, "duplicate/invalid native decision ID")
                expected_id = hashlib.sha256(json.dumps([sid, endpoint], ensure_ascii=False, sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
                require(rid == expected_id, "independently reconstructed native DecisionIR ID")
                all_ids.add(rid)
                task = "choice" if endpoint.endswith(".intent") else "score"
                require(decision["task"] == task, "native task mismatch")
                ids = manifest["catalogues"][endpoint]
                require(isinstance(ids, list) and len(ids) > 1 and len(set(ids)) == len(ids)
                        and all(isinstance(cid, str) and cid for cid in ids), "invalid native catalogue")
                require(decision["candidate_ids"] == ids and set(decision["candidates"]) == set(ids), "native catalogue order")
                require(isinstance(decision["question"], str) and all(isinstance(v, str)
                        for v in decision["candidates"].values()), "native descriptions")
                target = distribution(decision["target_distribution"], len(ids), "native target")
                row = {**decision, **{key: state[key] for key in ("component_id", "group_id", "locale", "source_id")},
                       "state_id": sid, "state": state["state"]}
                if task == "choice":
                    require(decision["gold"] in ids, "native gold absent")
                    close(target, [float(cid == decision["gold"]) for cid in ids], "native intent target", 0, 0)
                    expected_source = "PolyAI/banking77" if endpoint == "banking77.intent" else "AmazonScience/massive"
                    require(state["source_id"] == expected_source, "native source mismatch")
                    if endpoint == "massive.intent":
                        require(state["locale"] == "en-US", "nonselected MASSIVE locale")
                else:
                    level_max = decision["level_max"]
                    require(isinstance(level_max, int) and not isinstance(level_max, bool)
                            and level_max > 0 and ids == [str(i) for i in range(level_max + 1)],
                            "native ordinal support")
                    counts = decision["counts"]
                    require(isinstance(counts, list) and len(counts) == len(ids)
                            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts), "native rater counts")
                    observed = decision["observed_raters"]
                    require(type(observed) is int and observed >= 1 and sum(counts) == observed, "native observed-rater count")
                    close(target, [v / observed for v in counts], "native retained-rater distribution", 1e-12, 1e-12)
                    close(decision["mean_level"], expectation(target), "native retained-rater mean", 1e-12, 1e-12)
                    require(state["source_id"] == "AmazonScience/massive" and state["locale"] == "en-US", "ordinal selected provenance")
                    row["native_level_max"] = level_max
                records[phase][endpoint][rid] = row
        expected = manifest["phases"][phase]
        require(expected["states"] == len(states)
                and expected["decisions"] == sum(len(v) for v in records[phase].values()), "manifest phase census")
        components[phase] = sorted({s["component_id"] for s in states})
        require(expected["components"] == len(components[phase]), "manifest component census")
        require(expected["endpoint_counts"] == {endpoint: len(values) for endpoint, values in records[phase].items()
                                               if values}, "manifest endpoint census")
        for endpoint in fields:
            values = list(records[phase][endpoint].values())
            census[phase][endpoint] = {"decisions": len(values),
                "components": len({r["component_id"] for r in values}),
                "observed_raters_at_least3": sum(r.get("observed_raters", 0) >= 3 for r in values)}
            census[phase][endpoint]["native_ID_component_membership_sha256"] = hashlib.sha256(
                json.dumps(sorted((r["id"], r["component_id"], r["group_id"]) for r in values),
                           ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    require(all(len(phases_seen) == 1 for phases_seen in component_phases.values()), "component leaked across phases")
    counts = cfg["data"]["known_intent_counts"]
    for source, endpoint in (("banking", "banking77.intent"), ("massive_en", "massive.intent")):
        require(sum(len(records[p][endpoint]) for p in PHASES[:-1]) == counts[source + "_train"], "native train intent census")
        require(len(records["dev"][endpoint]) == counts[source + "_dev"], "native dev intent census")
    eligibility = {}
    for endpoint in fields:
        fit, dev = len(records["fit"][endpoint]), len(records["dev"][endpoint])
        eligible = (fit > 0 and dev > 0) if endpoint.endswith(".intent") else fit >= 250 and dev >= 100
        eligibility[endpoint] = {"eligible": eligible, "fit": fit, "dev": dev,
            "status": "eligible" if eligible else "data-insufficient",
            "minimum_fit": 1 if endpoint.endswith(".intent") else 250,
            "minimum_dev": 1 if endpoint.endswith(".intent") else 100}
    require(set(manifest["eligible_endpoints"]) == {e for e, v in eligibility.items() if v["eligible"]}
            and len(manifest["eligible_endpoints"]) == len(set(manifest["eligible_endpoints"])), "independent rubric eligibility")
    require(set(manifest["ineligible_endpoints"]) == {e for e, v in eligibility.items() if not v["eligible"]},
            "independent ineligible endpoint census")
    for endpoint, assessment in eligibility.items():
        if assessment["eligible"]:
            require(all(records[phase][endpoint] for phase in PHASES), "eligible endpoint has an empty phase")
    return records, census, components, eligibility


def empirical_prior(records, endpoint):
    rows = list(records["fit"][endpoint].values())
    require(bool(rows), "empty fit prior: " + endpoint)
    count = len(rows[0]["candidate_ids"])
    return [math.fsum(r["target_distribution"][i] for r in rows) / len(rows) for i in range(count)]


def validate_prediction(row, native, arm, temperature, prior=None, calibration=False):
    label = arm + ":" + native["id"]
    for key in ("id", "endpoint", "task", "component_id", "group_id", "locale", "candidate_ids"):
        require(row.get(key) == native[key], label + ": native " + key)
    require(not ({"state", "question", "gold", "mean_level", "counts", "target_distribution"} & set(row))
            or calibration, label + ": target/input fields in predictions")
    count = len(native["candidate_ids"])
    logits = row.get("logits")
    if arm == "prior":
        require(logits is None, label + ": state-blind logits must be null")
        raw = prior
    else:
        require(isinstance(logits, list) and len(logits) == count, label + ": logits shape")
        logits = [finite(value, label + ":logits") for value in logits]
        raw = softmax(logits)
    if calibration:
        close(row["target_distribution"], native["target_distribution"], label + ": calibration target", 0, 0)
        close(row["raw_probs"], raw, label + ": calibration raw softmax", 1e-12, 1e-12)
        require(row.get("gold") == native.get("gold")
                and row.get("observed_raters") == native.get("observed_raters"), label + ": calibration target metadata")
        return logits
    raw_saved = distribution(row.get("raw_probs"), count, label + ": raw probabilities")
    calibrated = distribution(row.get("probs"), count, label + ": calibrated probabilities")
    close(raw_saved, raw, label + ": raw softmax", 1e-12, 1e-12)
    close(calibrated, raw if arm == "prior" else softmax(logits, temperature),
          label + ": temperature softmax", 1e-12, 1e-12)
    result = {}
    for scope, probs, prefix in (("raw", raw_saved, "raw_"), ("calibrated", calibrated, "")):
        rebuilt = {**native, "probs": probs, "logits": logits}
        if native["task"] == "choice":
            answer = semantic_winner(native["candidate_ids"], probs)
            require(row.get(prefix + "answer") == answer, label + ": semantic argmax")
            require(row.get(prefix + "ties") == sorted(cid for cid, value in zip(native["candidate_ids"], probs)
                                                       if value == max(probs)), label + ": deterministic semantic ties")
            rebuilt.update(answer=answer, correct=answer == native["gold"])
        else:
            expected = expectation(probs)
            close(row.get(prefix + "expected_level"), expected, label + ": expectation", 1e-12, 1e-12)
            close(row.get(prefix + "cdf"), cdf(probs), label + ": ordinal CDF", 1e-12, 1e-12)
            rebuilt["expected_level"] = expected
        result[scope] = rebuilt
    return result


def read_predictions(path, natives, arm, temperatures, priors, control, plans):
    expected = {rid: row for endpoint, values in natives.items() for rid, row in values.items()}
    if control == "swapped":
        expected = {rid: row for rid, row in expected.items() if plans[row["endpoint"]][rid] is not None}
    result = {scope: {endpoint: {} for endpoint in natives} for scope in ("raw", "calibrated")}
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(isinstance(rid, str) and rid in expected and rid not in seen, str(path) + ": extra/duplicate ID")
        seen.add(rid)
        native = expected[rid]
        endpoint = native["endpoint"]
        if control == "swapped":
            require(saved.get("donor_id") == plans[endpoint][rid], str(path) + ": deterministic donor mismatch")
        else:
            require("donor_id" not in saved, str(path) + ": donor on nonswapped path")
        rebuilt = validate_prediction(saved, native, arm, temperatures[endpoint], priors[endpoint])
        for scope in result:
            result[scope][endpoint][rid] = rebuilt[scope]
    require(seen == set(expected), str(path) + ": missing native prediction rows")
    return result


def read_unavailable(path, natives, plans):
    expected = {rid: row for endpoint, values in natives.items() for rid, row in values.items()
                if plans[endpoint][rid] is None}
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(rid in expected and rid not in seen, "extra/duplicate unavailable donor ID")
        native = expected[rid]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale", "candidate_ids"):
            require(saved.get(key) == native[key], "unavailable donor native metadata")
        require(saved.get("reason") == "no different-component different-native-target dev donor",
                "unavailable donor reason")
        seen.add(rid)
    require(seen == set(expected), "missing unavailable donor IDs")
    return len(seen)


def reconstruct_calibration(path, natives, arm, temperatures):
    expected = {rid: row for values in natives.values() for rid, row in values.items()}
    logits = {endpoint: [] for endpoint in natives}
    targets = {endpoint: [] for endpoint in natives}
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(rid in expected and rid not in seen, "extra/duplicate calibration ID")
        seen.add(rid)
        native = expected[rid]
        endpoint = native["endpoint"]
        values = validate_prediction(saved, native, arm, 1., calibration=True)
        logits[endpoint].append(values)
        targets[endpoint].append(native["target_distribution"])
    require(seen == set(expected), "missing inner calibration rows")
    grid = [math.exp(math.log(.5) + i * (math.log(10.) - math.log(.5)) / 100) for i in range(101)]
    grid[0], grid[-1] = .5, 10.
    evidence = {}
    for endpoint in natives:
        require(bool(logits[endpoint]), "empty eligible inner calibration field")
        losses = []
        for temperature in grid:
            values = [-math.fsum(t * math.log(max(p, 1e-12))
                                 for p, t in zip(softmax(z, temperature), target))
                      for z, target in zip(logits[endpoint], targets[endpoint])]
            losses.append(math.fsum(values) / len(values))
        index = min(range(len(grid)), key=lambda i: (losses[i], abs(grid[i] - 1.), grid[i]))
        close(temperatures[endpoint], grid[index], "independent temperature optimum", 1e-12, 1e-12)
        evidence[endpoint] = {"rows": len(logits[endpoint]), "grid_index": index,
                              "temperature": grid[index], "nll": losses[index], "grid_nll": losses}
    return evidence


def reconstruct_selection(path, natives, cfg, probe_state_id):
    expected = {rid: row for values in natives.values() for rid, row in values.items()}
    epochs = {epoch: {endpoint: [] for endpoint in natives}
              for epoch in range(cfg["training"]["epochs"] + 1)}
    probes = {epoch: {} for epoch in epochs}
    seen = set()
    for saved in json_rows(path):
        epoch, rid = saved.get("epoch"), saved.get("id")
        require(type(epoch) is int and epoch in epochs and rid in expected and (epoch, rid) not in seen,
                "extra/duplicate epoch selection row")
        seen.add((epoch, rid))
        native = expected[rid]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale", "candidate_ids"):
            require(saved.get(key) == native[key], "selection native metadata")
        require(isinstance(saved.get("logits"), list) and len(saved["logits"]) == len(native["candidate_ids"]),
                "selection logits shape")
        probs = softmax(saved["logits"])
        close(saved["raw_probs"], probs, "selection raw softmax", 1e-12, 1e-12)
        close(saved["target_distribution"], native["target_distribution"], "selection native targets", 0, 0)
        require(saved.get("gold") == native.get("gold")
                and saved.get("observed_raters") == native.get("observed_raters"), "selection target metadata")
        nll = -math.fsum(t * math.log(max(p, 1e-12)) for p, t in zip(probs, native["target_distribution"]))
        hit = semantic_winner(native["candidate_ids"], probs) == native.get("gold")
        epochs[epoch][native["endpoint"]].append((hit, nll))
        if native["state_id"] == probe_state_id:
            probes[epoch][native["endpoint"]] = saved["logits"]
    require(seen == {(epoch, rid) for epoch in epochs for rid in expected}, "missing epoch selection rows")
    history = []
    for epoch, endpoints in epochs.items():
        stats = {endpoint: {"rows": len(values), "accuracy": sum(v[0] for v in values) / len(values),
                           "nll": math.fsum(v[1] for v in values) / len(values)}
                 for endpoint, values in endpoints.items()}
        intents = [stats[e]["accuracy"] for e in stats if e.endswith(".intent")]
        history.append({"epoch": epoch, "macro_intent_accuracy": math.fsum(intents) / len(intents),
                        "mean_endpoint_nll": math.fsum(v["nll"] for v in stats.values()) / len(stats),
                        "endpoints": stats})
    chosen = min(history, key=lambda row: (-row["macro_intent_accuracy"], row["mean_endpoint_nll"], row["epoch"]))
    require(all(probes[epoch] for epoch in epochs), "selected restore probe absent from epoch predictions")
    return chosen["epoch"], history, probes


def endpoint_metrics(rows, catalogue):
    if not rows:
        return {"status": "data-insufficient", "rows": 0}
    if rows[0]["task"] == "choice":
        result = choice_stats(rows)
        result["per_class"] = {
            cid: {"target_rows": sum(row["gold"] == cid for row in rows),
                  "predicted_rows": sum(row["answer"] == cid for row in rows),
                  "correct_rows": sum(row["gold"] == cid and row["answer"] == cid for row in rows)}
            for cid in catalogue}
        return result
    result = ordinal_stats(rows)
    retained3 = [row for row in rows if row["observed_raters"] >= 3]
    result["rps_all_observed_raters"] = result["normalized_ranked_probability_score"]
    result["three_or_more_raters_normalized_RPS"] = (
        ordinal_stats(retained3)["normalized_ranked_probability_score"] if retained3 else None)
    result["distribution_scope"] = "Retained-rater agreement; no population-calibration claim"
    return result


def endpoint_comparison(endpoint, native, predictions, prior, plan, catalogue, cfg):
    ids = sorted(native)
    real = [predictions["dev"][rid] for rid in ids]
    masked = [predictions["masked"][rid] for rid in ids]
    baseline = [prior[rid] for rid in ids]
    available = [rid for rid in ids if plan[rid] is not None]
    swapped = [predictions["swapped"][rid] for rid in available]
    correct_new = []
    for row in swapped:
        donor = native[plan[row["id"]]]
        revised = {**row, "gold": donor.get("gold"), "target_distribution": donor["target_distribution"]}
        if donor["task"] == "choice":
            revised["correct"] = row["answer"] == donor["gold"]
        else:
            revised.update(mean_level=donor["mean_level"], observed_raters=donor["observed_raters"])
        correct_new.append(revised)
    donors = Counter(plan[rid] for rid in available)
    controls = {
        "teacher_changing_rows": len(available), "unavailable_rows": len(ids) - len(available),
        "denominator": "Only lawful different-component, different-native-target donors",
        "unique_donors": len(donors), "donor_reuse_rows": sum(max(0, n - 1) for n in donors.values()),
        "donor_use_counts": dict(sorted(donors.items())),
        "real_vs_mask_probability_changed_rows": sum(
            r["probs"] != m["probs"] for r, m in zip(real, masked)),
        "real_vs_swap_probability_changed_rows": sum(
            predictions["dev"][rid]["probs"] != predictions["swapped"][rid]["probs"] for rid in available),
    }
    report = {"metrics": {"real": endpoint_metrics(real, catalogue),
                          "masked": endpoint_metrics(masked, catalogue),
                          "swapped_against_original": endpoint_metrics(swapped, catalogue),
                          "swapped_correct_new": endpoint_metrics(correct_new, catalogue)}, "controls": controls}
    kwargs = {"resamples": cfg["statistics"]["resamples"], "seed": cfg["statistics"]["seed"]}
    if endpoint.endswith(".intent"):
        differences = {
            "model_minus_prior_accuracy": [float(r["correct"]) - float(p["correct"]) for r, p in zip(real, baseline)],
            "real_minus_masked_accuracy": [float(r["correct"]) - float(m["correct"]) for r, m in zip(real, masked)]}
        controls["real_vs_mask_decision_flips"] = sum(r["answer"] != m["answer"] for r, m in zip(real, masked))
        controls["real_vs_swap_decision_flips"] = sum(
            predictions["dev"][rid]["answer"] != predictions["swapped"][rid]["answer"] for rid in available)
        report["uncertainty"] = component_ratio_bootstrap(real, differences, **kwargs)
    else:
        differences = {
            "model_minus_prior_nMAE": [nmae(r) - nmae(p) for r, p in zip(real, baseline)],
            "masked_minus_real_nMAE": [nmae(m) - nmae(r) for r, m in zip(real, masked)],
            "prior_minus_model_normalized_RPS_all_raters": [nrps(p) - nrps(r) for r, p in zip(real, baseline)],
            "masked_minus_real_normalized_RPS_all_raters": [nrps(m) - nrps(r) for r, m in zip(real, masked)]}
        report["uncertainty"] = component_ratio_bootstrap(real, differences, **kwargs)
        indices3 = [i for i, row in enumerate(real) if row["observed_raters"] >= 3]
        if indices3:
            report["uncertainty"].update(component_ratio_bootstrap(
                [real[i] for i in indices3],
                {"prior_minus_model_normalized_RPS_at_least3": [nrps(baseline[i]) - nrps(real[i]) for i in indices3],
                 "masked_minus_real_normalized_RPS_at_least3": [nrps(masked[i]) - nrps(real[i]) for i in indices3]}, **kwargs))
        else:
            report["uncertainty"]["prior_minus_model_normalized_RPS_at_least3"] = None
        controls["real_vs_mask_expectation_changed_rows"] = sum(
            r["expected_level"] != m["expected_level"] for r, m in zip(real, masked))
        controls["real_vs_swap_expectation_changed_rows"] = sum(
            predictions["dev"][rid]["expected_level"] != predictions["swapped"][rid]["expected_level"] for rid in available)
    return report


def screen_field(endpoint, report, invariant_pass, cfg):
    gates = cfg["screen_gates"]
    metrics, uncertainty, controls = report["metrics"], report["uncertainty"], report["controls"]
    checks = {"exact_cache_and_coordinate_invariance": invariant_pass,
              "all_three_controls_complete": controls["unavailable_rows"] == 0}
    insufficient = []
    if controls["unavailable_rows"] > 0:
        insufficient.append("native rows lack lawful teacher-changing different-component donors")
    if endpoint.endswith(".intent"):
        checks.update({
            "intent_accuracy_each_source": metrics["real"]["top1_accuracy"] >= gates["intent_accuracy_each_source"],
            "intent_vs_prior_lower_nominal_bound": (
                uncertainty["model_minus_prior_accuracy"]["ci95_bootstrap"][0]
                >= gates["intent_vs_prior_lower_nominal_bound"]),
            "intent_real_minus_masked_accuracy_lower_nominal_bound": (
                uncertainty["real_minus_masked_accuracy"]["ci95_bootstrap"][0]
                >= gates["intent_real_minus_masked_accuracy_lower_nominal_bound"]),
            "intent_swapped_correct_new_each_source": (
                metrics["swapped_correct_new"].get("top1_accuracy", -1)
                >= gates["intent_swapped_correct_new_each_source"]),
        })
    else:
        rps = uncertainty["prior_minus_model_normalized_RPS_at_least3"]
        if rps is None:
            insufficient.append("no dev records with at least three retained raters")
        checks.update({
            "ordinal_prior_minus_model_normalized_RPS_lower_nominal_bound": (
                rps is not None and rps["ci95_bootstrap"][0]
                >= gates["ordinal_prior_minus_model_normalized_RPS_lower_nominal_bound"]),
            "ordinal_model_minus_prior_nMAE_upper_nominal_bound": (
                uncertainty["model_minus_prior_nMAE"]["ci95_bootstrap"][1]
                <= gates["ordinal_model_minus_prior_nMAE_upper_nominal_bound"]),
        })
    status = "data-insufficient" if insufficient else "passed" if all(checks.values()) else "failed"
    return {"status": status, "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "data_insufficient_reasons": insufficient,
            "scope": gates["scope"]}


def validate_controls(predictions, natives, plans):
    checked = {"masked_constant_rows": 0, "swap_matches_donor_rows": 0}
    for scope in ("raw", "calibrated"):
        for endpoint, rows in natives.items():
            reference = None
            for rid in sorted(rows):
                masked = predictions["masked"][scope][endpoint][rid]
                if reference is None:
                    reference = masked
                for key in ("logits", "probs"):
                    close(masked[key], reference[key], "exact constant empty-evidence " + key, 0, 0)
                checked["masked_constant_rows"] += 1
                donor = plans[endpoint][rid]
                if donor is not None:
                    swapped = predictions["swapped"][scope][endpoint][rid]
                    original_donor = predictions["dev"][scope][endpoint][donor]
                    for key in ("logits", "probs"):
                        close(swapped[key], original_donor[key], "exact cached swap donor path " + key, 0, 0)
                    checked["swap_matches_donor_rows"] += 1
    checked["counts_include_raw_and_calibrated"] = True
    return checked


def artifact_files(directory, metadata, arm):
    artifacts = metadata["artifact_hashes"]
    require(isinstance(artifacts, dict), arm + ": artifact manifest")
    required = {control + ".jsonl" for control in PATHS} | {
        "swapped_unavailable.jsonl", "history.json", "liveness.json", "coordinate_reversal.jsonl"}
    if arm != "prior":
        required |= {"selection.jsonl", "calibration.jsonl", "selected.safetensors"}
    require(required <= set(artifacts), arm + ": omitted mandatory artifact")
    verified = {}
    for name, entry in artifacts.items():
        expected = entry["sha256"]
        path = directory / name
        require(not Path(name).is_absolute() and path.resolve().is_relative_to(directory.resolve()),
                arm + ": artifact escapes arm directory")
        require(isinstance(expected, str) and len(expected) == 64, arm + ": malformed artifact hash")
        require(path.is_file() and path.stat().st_size == entry["bytes"]
                and digest(path) == expected, arm + ": artifact hash mismatch " + name)
        verified[name] = expected
    return verified


def model_custody(custody, cfg):
    spec = cfg["model"]
    for key in ("repo", "revision", "weights_sha256"):
        require(custody[key] == spec[key], "pinned stock " + key)
    require(custody["dtype"] == cfg["training"]["dtype"], "stock dtype custody")
    artifacts = custody["artifacts"]
    require({"model.safetensors", "config.json", "spm.model"} <= set(artifacts), "stock weights/config/tokenizer custody")
    snapshot = Path(custody["snapshot"])
    require(snapshot.name == spec["revision"], "stock snapshot revision")
    for name, item in artifacts.items():
        path = Path(item["path"])
        require(path.parent == snapshot and path.name == name and path.is_file(), "stock artifact membership")
        require(str(path.resolve()) == item["resolved_path"] and path.stat().st_size == item["bytes"]
                and digest(path) == item["sha256"], "stock artifact byte custody")
    require(artifacts["model.safetensors"]["sha256"] == spec["weights_sha256"], "stock weight bytes")
    require(custody["loading"].get("missing_keys", []) == []
            and custody["loading"].get("mismatched_keys", []) == []
            and custody["loading"].get("error_msgs", []) == []
            and custody["unexpected_base_keys"] == [], "complete stock load receipt")
    require(custody["tokenizer_is_fast"] is True, "tokenizer load receipt")
    return {"repo": spec["repo"], "revision": spec["revision"],
            "verified_artifacts": {name: item["sha256"] for name, item in artifacts.items()},
            "encoder_initial_sha256": custody["encoder_initial_sha256"],
            "scope": "Artifact bytes verified; loader/config/parameter-count receipts are not a replayed model load"}


def canonical_readout(ids, probs, task):
    if task == "choice":
        answer = semantic_winner(ids, probs)
        return {"answer": answer, "ties": sorted(cid for cid, value in zip(ids, probs) if value == max(probs))}
    by_id = dict(zip(ids, probs))
    ordered = [by_id[str(level)] for level in range(len(ids))]
    return {"expected_level": expectation(ordered), "cdf": cdf(ordered)}


def check_readout(saved, native, arm, temperature, prior, reverse=False):
    ids = native["candidate_ids"]
    requested = list(reversed(ids)) if reverse else ids
    require(saved.get("candidate_ids") == requested, "actual coordinate order")
    count = len(ids)
    raw = distribution(saved.get("raw_probs"), count, "actual raw coordinate masses")
    probs = distribution(saved.get("probs"), count, "actual calibrated coordinate masses")
    if arm == "prior":
        require(saved.get("logits") is None, "coordinate prior logits")
        expected_raw = list(reversed(prior)) if reverse else prior
    else:
        require(isinstance(saved.get("logits"), list) and len(saved["logits"]) == count, "coordinate logits")
        expected_raw = softmax(saved["logits"])
    close(raw, expected_raw, "coordinate raw softmax", 1e-12, 1e-12)
    close(probs, expected_raw if arm == "prior" else softmax(saved["logits"], temperature),
          "coordinate calibrated softmax", 1e-12, 1e-12)
    for prefix, masses in (("", probs), ("raw_", raw)):
        for key, value in canonical_readout(requested, masses, native["task"]).items():
            close(saved.get(prefix + key), value, "coordinate native " + prefix + key, 0, 0)
    return {key: saved[key] for key in ("candidate_ids", "logits", "raw_probs", "probs")}


def check_reversal(path, natives, predictions, arm, temperatures, priors):
    expected = {rid: row for rows in natives.values() for rid, row in rows.items()}
    seen = set()
    for saved in json_rows(path):
        rid = saved.get("id")
        require(rid in expected and rid not in seen, "extra/duplicate coordinate reversal ID")
        seen.add(rid)
        native = expected[rid]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale", "candidate_ids"):
            require(saved.get(key) == native[key], "coordinate native metadata")
        require(type(saved["encoder_calls_before"]) is int and saved["encoder_calls_before"] >= 0
                and saved["encoder_calls_after"] == saved["encoder_calls_before"], "coordinate gather called encoder")
        endpoint = native["endpoint"]
        original = saved["original"]
        reversed_output = saved["reversed"]
        check_readout(original, native, arm, temperatures[endpoint], priors[endpoint])
        check_readout(reversed_output, native, arm, temperatures[endpoint], priors[endpoint], reverse=True)
        for key in ("logits", "raw_probs", "probs"):
            value = original[key]
            close(reversed_output[key], None if value is None else list(reversed(value)),
                  "exact coordinate gather " + key, 0, 0)
        close(original["logits"], predictions["raw"][endpoint][rid]["logits"], "actual original coordinate path", 0, 0)
        close(original["raw_probs"], predictions["raw"][endpoint][rid]["probs"], "original raw coordinates", 0, 0)
        close(original["probs"], predictions["calibrated"][endpoint][rid]["probs"], "original calibrated coordinates", 0, 0)
    require(seen == set(expected), "missing actual coordinate reversals")
    return {"status": "passed", "rows": len(seen), "raw_and_calibrated": True,
            "scope": "Canonical native IDs and exact coordinate gathers; no candidate-alias semantics"}


def dataset_custody(root, manifest, cfg):
    protocol_hash = digest(HERE / "native_field_protocol.json")
    require(manifest["protocol_sha256"] == protocol_hash, "dataset protocol hash")
    require(set(manifest["source_exports"]) == {"train", "dev"}, "source export phase membership")
    require(manifest["sealed_phases_accessed"] is False, "sealed phase custody")
    for phase, entry in manifest["source_exports"].items():
        native_directory = Path(cfg["native_root"]) / phase
        source_manifest = Path(entry["manifest_path"])
        source_receipt = Path(entry["verification_path"])
        require(source_manifest == native_directory / "manifest.json"
                and source_receipt == native_directory / "verification_receipt.json", "registered source custody paths")
        require(digest(source_manifest) == entry["manifest_sha256"]
                and digest(source_receipt) == entry["verification_sha256"], "native export/verification hashes")
        exported = load_json(source_manifest)
        receipt = load_json(source_receipt)
        require(exported["phase"] == phase and exported["protocol_sha256"] == cfg["native_compiler_protocol_sha256"],
                "native export protocol/phase")
        require(receipt["status"] == "PASS" and receipt["phase"] == phase
                and receipt["protocol_sha256"] == cfg["native_compiler_protocol_sha256"]
                and receipt["manifest_sha256"] == entry["manifest_sha256"]
                and receipt["source_payloads_of_sealed_phases_read"] is False, "native source verification receipt")
    source_files = manifest["source_files"]
    correction_path = HERE / "native_field_execution_correction_v1.json"
    correction = None
    if correction_path.exists():
        correction = load_json(correction_path)
        relative = correction_path.relative_to(HERE.parents[1]).as_posix()
        require(subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=HERE.parents[1])
                == correction_path.read_bytes(), "committed execution correction")
        require(correction["schema"] == "vey.native-field.execution-correction.v1"
                and correction["dataset_manifest_sha256"] == digest(root / "dataset_manifest.json")
                and correction["protocol_sha256"] == protocol_hash
                and correction["original_source_files"] == source_files
                and set(correction["source_files"]) == set(source_files), "bounded code-only execution transition")
        for entry in correction["retained_artifacts"]:
            require(digest(entry["path"]) == entry["sha256"], "retained failed/prior evidence")
        source_files = correction["source_files"]
    require({str(HERE / name) for name in (
        "native_field_data.py", "native_field_model.py", "native_field_train.py", "native_field_verify.py")}
            <= set(source_files), "native implementation custody membership")
    for mapping in (source_files, manifest["semantic_model_files"]):
        for path, expected in mapping.items():
            require(digest(path) == expected, "dataset implementation source hash")
    for phase in PHASES:
        entry = manifest["phases"][phase]
        require(Path(entry["path"]) == root / (phase + ".jsonl")
                and digest(entry["path"]) == entry["sha256"], "selected dataset artifact hash")
    return {"protocol_sha256": protocol_hash, "dataset_manifest_sha256": digest(root / "dataset_manifest.json"),
            "source_exports": manifest["source_exports"], "source_files": source_files,
            "original_source_files": manifest["source_files"],
            "execution_correction": correction,
            "execution_correction_sha256": digest(correction_path) if correction else None,
            "semantic_model_files": manifest["semantic_model_files"],
            "phase_hashes": {phase: manifest["phases"][phase]["sha256"] for phase in PHASES},
            "scope": "Guarded selected records and native projection receipts; no raw source/target payload replay"}


def checkpoint_fingerprint(path):
    """Hash safetensors payloads in the trainer's named-tensor canonical order."""
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
        require(0 < length <= min(path.stat().st_size - 8, 100_000_000), "invalid safetensors header size")
        header = decode(stream.read(length).decode("utf-8"))
        require(isinstance(header, dict), "safetensors header object")
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        require(bool(tensors), "empty selected checkpoint")
        payload_size = path.stat().st_size - 8 - length
        ranges, h, encoder_h = [], hashlib.sha256(), hashlib.sha256()
        for name in sorted(tensors):
            item = tensors[name]
            dtype, shape, offsets = item["dtype"], item["shape"], item["data_offsets"]
            require(dtype in types and isinstance(shape, list)
                    and all(type(dim) is int and dim >= 0 for dim in shape), "invalid tensor dtype/shape")
            require(isinstance(offsets, list) and len(offsets) == 2
                    and all(type(offset) is int for offset in offsets), "invalid tensor data offsets")
            start, end = offsets
            require(0 <= start <= end <= payload_size
                    and end - start == math.prod(shape) * types[dtype][1], "tensor payload extent")
            ranges.append((start, end))
            rendered = json.dumps([name, types[dtype][0], shape], sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode("utf-8")
            h.update(len(rendered).to_bytes(8, "little"))
            h.update(rendered)
            encoder_tensor = name.startswith("pool.encoder.")
            if encoder_tensor:
                rendered_encoder = json.dumps([name.removeprefix("pool.encoder."), types[dtype][0], shape],
                                              sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=False, allow_nan=False).encode("utf-8")
                encoder_h.update(len(rendered_encoder).to_bytes(8, "little"))
                encoder_h.update(rendered_encoder)
            stream.seek(8 + length + start)
            remaining = end - start
            while remaining:
                block = stream.read(min(remaining, 1 << 20))
                require(bool(block), "truncated tensor payload")
                h.update(block)
                if encoder_tensor:
                    encoder_h.update(block)
                remaining -= len(block)
        previous = 0
        for start, end in sorted(ranges):
            require(start == previous, "overlapping/gapped safetensors tensors")
            previous = end
        require(previous == payload_size, "unaccounted checkpoint payload bytes")
    actual = h.hexdigest()
    require(header["__metadata__"]["schema"] == "vey.native-field.checkpoint.v1"
            and header["__metadata__"]["state_dict_sha256"] == actual, "checkpoint header/tensor fingerprint")
    return actual, encoder_h.hexdigest()


def numerical_max_difference(left, right):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), "numeric evidence keys")
        return max((numerical_max_difference(left[key], right[key]) for key in left), default=0.)
    if isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), "numeric evidence shape")
        return max((numerical_max_difference(a, b) for a, b in zip(left, right)), default=0.)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        return abs(finite(left, "numeric evidence") - finite(right, "numeric evidence"))
    require(left == right, "numeric evidence native value")
    return 0.


def check_logit_probe(values, catalogues):
    require(isinstance(values, dict) and set(values) == set(catalogues), "restore probe eligible head membership")
    for endpoint, ids in catalogues.items():
        require(isinstance(values[endpoint], list) and len(values[endpoint]) == len(ids), "restore logits support")
        for value in values[endpoint]:
            finite(value, "restore actual logits")


def check_restore(restore, catalogues, checkpoint=None, smoke=False):
    expected = restore["expected_fingerprint"]
    require(isinstance(expected, str) and len(expected) == 64
            and restore["restored_fingerprint"] == expected
            and restore["missing_keys"] == [] and restore["unexpected_keys"] == [], "strict restore fingerprints/keys")
    before = restore["saved_logits"] if smoke else restore["selected_logits"]
    after = restore["restored_logits"]
    check_logit_probe(before, catalogues)
    check_logit_probe(after, catalogues)
    close(after, before, "numerical selected save/restore logits")
    difference = numerical_max_difference(before, after)
    close(restore["max_abs_difference"], difference, "reconstructed restore numeric drift", 1e-12, 1e-12)
    if smoke:
        require(restore["perturbed_fingerprint"] != expected, "smoke restore did not exercise a different state")
    else:
        require(checkpoint is not None and digest(checkpoint) == restore["checkpoint_sha256"], "selected checkpoint file bytes")
        reconstructed, selected_encoder = checkpoint_fingerprint(checkpoint)
        require(reconstructed == expected, "independent selected checkpoint tensor bytes")
        check_logit_probe(restore["before_restore_logits"], catalogues)
    return {"state_dict_fingerprint": expected, "numerical_max_abs_difference": difference,
            "tensor_fingerprint_independently_reconstructed": not smoke,
            "encoder_state_dict_fingerprint": None if smoke else selected_encoder,
            "scope": "Saved/restored finite logits compared; no independent model-forward replay"}


def check_cache_controls(controls, natives, catalogues, arm, temperatures, priors, predictions=None):
    expected = {rid: row for values in natives.values() for rid, row in values.items()}
    seen_endpoints = set()
    state_texts = set()
    for control in controls:
        rid, endpoint = control["id"], control["endpoint"]
        require(rid in expected and endpoint not in seen_endpoints, "cache probe native membership/duplicate endpoint")
        native = expected[rid]
        for key in ("id", "endpoint", "task", "component_id", "group_id", "locale", "candidate_ids"):
            require(control.get(key) == native[key], "cache probe native metadata")
        require(control["state_sha256"] == hashlib.sha256(native["state"].encode()).hexdigest(), "exact cache state bytes")
        seen_endpoints.add(endpoint)
        first, repeated, uncached = control["first"], control["repeated"], control["uncached"]
        for value in (first, repeated, uncached):
            check_readout(value, native, arm, temperatures[endpoint], priors.get(endpoint))
        close(first, repeated, "exact repeated cached values", 0, 0)
        close(first, uncached, "actual uncached/cached agreement")
        close(control["cached_uncached_max_abs_difference"], numerical_max_difference(first, uncached),
              "reconstructed cached/uncached drift", 1e-12, 1e-12)
        keys = ("encoder_calls_before", "encoder_calls_after_first", "encoder_calls_after_repeat",
                "encoder_calls_after_membership", "encoder_calls_after_uncached")
        counters = [control[key] for key in keys]
        require(all(type(value) is int and value >= 0 for value in counters), "cache actual integer counters")
        first_delta = 0 if arm == "prior" or native["state"] in state_texts else 1
        require(counters[1] - counters[0] == first_delta
                and counters[2] == counters[1] and counters[3] == counters[2]
                and counters[4] - counters[3] == (0 if arm == "prior" else 1), "repeated/membership actual encoder-call invariants")
        state_texts.add(native["state"])
        member = control["membership"]
        if native["task"] == "choice":
            members = sorted(catalogues[endpoint])[:max(1, len(catalogues[endpoint]) // 2)]
            by_id = dict(zip(catalogues[endpoint], first["probs"]))
            mass = math.fsum(by_id[cid] for cid in members)
            close(member, {"members": members, "true_mass": mass, "false_mass": 1 - mass, "answer": mass >= .5},
                  "exact canonical membership Boolean", 0, 0)
        else:
            require(member is None, "ordinal cache probe has intent membership")
        if predictions is not None:
            close(first["logits"], predictions["raw"][endpoint][rid]["logits"], "cache actual original prediction", 0, 0)
            close(first["probs"], predictions["calibrated"][endpoint][rid]["probs"], "cache calibrated original", 0, 0)
    require(seen_endpoints == set(catalogues), "cache probes missing an eligible native field")
    return {"status": "passed", "endpoints": sorted(seen_endpoints), "probe_rows": len(controls),
            "prewarmed_unique_states": len(state_texts), "prewarmed_states": state_texts,
            "scope": "Actual persisted repeated/uncached outputs and call counters on one native probe per field"}


def check_active_statistics(stats, label):
    require(type(stats["nonzero"]) is int and stats["nonzero"] > 0
            and finite(stats["l2"], label) > 0 and finite(stats["max_abs"], label) > 0,
            label + ": inactive gradient/parameter delta")


def check_smoke_liveness(smoke, natives, catalogues, arm, cfg):
    require(smoke["smoke_weights_discarded"] is True, "discarded smoke state receipt")
    if arm == "prior":
        expected_counts = {e: len(rows) for e, rows in natives.items()}
        require(smoke["kind"] == "state_blind_fit_prior" and smoke["fit_endpoint_counts"] == expected_counts,
                "prior fit-only smoke census")
        return {"kind": "state-blind prior", "neural_mechanism_credit": False}
    state_ids = smoke["actual_fit_state_ids"]
    require(isinstance(state_ids, list) and state_ids and len(set(state_ids)) == len(state_ids), "smoke native state IDs")
    expected = {rid: row for rows in natives.values() for rid, row in rows.items() if row["state_id"] in state_ids}
    require(set(state_ids) == {r["state_id"] for r in expected.values()}
            and len(smoke["actual_fit_decision_ids"]) == len(set(smoke["actual_fit_decision_ids"]))
            and set(smoke["actual_fit_decision_ids"]) == set(expected)
            and smoke["valid_decisions"] == len(expected), "actual smoke native fit membership")
    require({r["endpoint"] for r in expected.values()} == set(catalogues), "smoke covers eligible heads")
    require(finite(smoke["loss"], "actual smoke loss") >= 0, "invalid smoke loss")
    names = smoke["active_head_parameter_names"]
    require(set(names) == set(catalogues), "named active smoke head population")
    for endpoint, name in names.items():
        index = sorted(catalogues).index(endpoint)
        require(name.startswith(f"heads.field_{index}."), "active gradient assigned to wrong head")
        check_active_statistics(smoke["gradient_statistics"][name], "actual named head gradient")
        check_active_statistics(smoke["parameter_deltas"][name], "actual named head optimizer delta")
    encoder_name = smoke["active_encoder_parameter_name"]
    if arm == "full":
        require(isinstance(encoder_name, str) and encoder_name.startswith("pool.encoder.")
                and smoke["encoder_before_sha256"] != smoke["encoder_after_sha256"], "full encoder smoke liveness")
        check_active_statistics(smoke["gradient_statistics"][encoder_name], "actual named encoder gradient")
        check_active_statistics(smoke["parameter_deltas"][encoder_name], "actual named encoder optimizer delta")
    else:
        require(encoder_name is None and smoke["encoder_before_sha256"] == smoke["encoder_after_sha256"]
                and not any(name.startswith("pool.encoder.") for name in smoke["gradient_statistics"]), "frozen encoder smoke invariance")
    restored = check_restore(smoke["strict_restore"], catalogues, smoke=True)
    cache = check_cache_controls(smoke["cache_controls"], natives, catalogues, arm,
                                 {endpoint: 1. for endpoint in catalogues}, {})
    cache.pop("prewarmed_states")
    custody = model_custody(smoke["custody"], cfg)
    require(smoke["encoder_before_sha256"] == custody["encoder_initial_sha256"], "smoke encoder starting custody")
    return {"kind": "actual native-fit optimizer step", "states": len(state_ids), "decisions": len(expected),
            "loss": smoke["loss"], "active_head_parameter_names": names,
            "active_encoder_parameter_name": encoder_name,
            "encoder_parameter_delta_proved": arm == "full",
            "frozen_encoder_hash_unchanged": arm == "frozen",
            "restore": restored, "cache": cache,
            "scope": "Finite numerical gradient/delta and output receipts; smoke is discarded, not fit performance"}


def check_history(history, selection, selected_epoch, records, arm, cfg):
    epochs = history["epochs"]
    require([row["epoch"] for row in epochs] == list(range(cfg["training"]["epochs"] + 1)), "complete epoch0..10 history")
    expected_fit = sum(len(rows) for rows in records["fit"].values())
    expected_states = len({row["state_id"] for rows in records["fit"].values() for row in rows.values()})
    updates = 0
    best = None
    for saved, rebuilt in zip(epochs, selection):
        native_stats = {
            "macro_source_intent_accuracy": rebuilt["macro_intent_accuracy"],
            "mean_endpoint_nll": rebuilt["mean_endpoint_nll"],
            "source_intent_accuracy": {e: v["accuracy"] for e, v in rebuilt["endpoints"].items() if e.endswith(".intent")},
            "endpoint_nll": {e: v["nll"] for e, v in rebuilt["endpoints"].items()},
            "endpoint_counts": {e: v["rows"] for e, v in rebuilt["endpoints"].items()},
        }
        close(saved["selection"], native_stats, "independent epoch selection metrics", 1e-12, 1e-12)
        rank = (-rebuilt["macro_intent_accuracy"], rebuilt["mean_endpoint_nll"], rebuilt["epoch"])
        if best is None or rank < best:
            best = rank
        require(saved["selected_epoch_so_far"] == best[2], "independent running selected epoch")
        require(isinstance(saved["state_dict_fingerprint"], str) and len(saved["state_dict_fingerprint"]) == 64, "epoch tensor fingerprint")
        fit = saved["fit"]
        if rebuilt["epoch"] == 0:
            require(fit is None, "epoch0 must precede fitting")
            continue
        require(fit["valid_decisions"] == expected_fit and fit["states"] == expected_states, "actual fit epoch census")
        blocks = fit["effective_batches"]
        require(len(blocks) == fit["optimizer_updates"]
                == math.ceil(expected_states / cfg["training"]["effective_batch_states"]), "actual optimizer update census")
        require(sum(block["states"] for block in blocks) == expected_states
                and sum(block["valid_decisions"] for block in blocks) == expected_fit, "effective batch census")
        for i, block in enumerate(blocks):
            require(type(block["states"]) is int and 0 < block["states"] <= cfg["training"]["effective_batch_states"]
                    and type(block["valid_decisions"]) is int and block["valid_decisions"] > 0
                    and type(block["microbatch_states"]) is int
                    and 1 <= block["microbatch_states"] <= cfg["training"]["batch_states"], "lawful effective/micro batch sizes")
            if i + 1 < len(blocks):
                require(block["states"] == cfg["training"]["effective_batch_states"], "short nonfinal effective batch")
            require(finite(block["normalized_loss"], "actual block loss") >= 0, "negative native CE loss")
        reconstructed_loss = math.fsum(block["normalized_loss"] * block["valid_decisions"] for block in blocks) / expected_fit
        close(fit["mean_decision_loss"], reconstructed_loss, "actual decision-weighted epoch loss", 1e-10, 1e-10)
        updates += len(blocks)
    require(history["selected_epoch"] == selected_epoch
            and history["selected_fingerprint"] == epochs[selected_epoch]["state_dict_fingerprint"], "selected checkpoint history")
    return {"epochs_including0": len(epochs), "optimizer_updates": updates,
            "selected_epoch": selected_epoch, "selected_fingerprint": history["selected_fingerprint"],
            "selection_reconstructed_from_native_logits": True,
            "fit_loss_scope": "Persisted batch-weighted losses/counts, not an independent gradient replay"}


def check_path_counts(controls, natives, plans, cache_proof, arm):
    prewarmed = cache_proof.pop("prewarmed_states")
    entry_list = controls["entries_at_prediction_start"]
    expected_entries = {(controls["model_identity"], hashlib.sha256(state.encode()).hexdigest()) for state in prewarmed}
    require(isinstance(entry_list, list) and len(entry_list) == len(expected_entries)
            and {(entry["model_identity"], entry["state_sha256"]) for entry in entry_list} == expected_entries,
            "prewarmed exact-state/model-identity cache census")
    require(set(controls["path_counts"]) == set(PATHS), "all actual neural control path counters")
    require(controls["path_counts"]["dev"]["encoder_calls_before"]
            == controls["cache_controls"][-1]["encoder_calls_after_uncached"], "cache probes/prediction counter continuity")
    require(controls["path_counts"]["dev"]["cache_misses_before"]
            == len(prewarmed) + len(controls["cache_controls"]), "cached and uncached probe miss census")
    state_sets = {"dev": {row["state"] for rows in natives.values() for row in rows.values()},
                  "masked": {"[EMPTY EVIDENCE]"},
                  "swapped": {natives[endpoint][donor]["state"] for endpoint, plan in plans.items()
                              for donor in plan.values() if donor is not None}}
    seen = set(prewarmed)
    previous = None
    for name in PATHS:
        counts = controls["path_counts"][name]
        for key in ("encoder_calls_before", "encoder_calls_after", "cache_misses_before", "cache_misses_after",
                    "entries_before", "entries_after"):
            require(type(counts[key]) is int and counts[key] >= 0, "integer control path counters")
        if previous is not None:
            for key in ("encoder_calls", "cache_misses", "entries"):
                require(counts[key + "_before"] == previous[key + "_after"], "continuous shared cached paths")
        missing = state_sets[name] - seen
        require(counts["entries_before"] == len(seen) and counts["entries_after"] == len(seen | state_sets[name])
                and counts["cache_misses_after"] - counts["cache_misses_before"] == len(missing)
                and counts["encoder_calls_after"] - counts["encoder_calls_before"] == (0 if arm == "prior" else len(missing)),
                "exact-state encoded/cache-miss path reconstruction: " + name)
        seen |= state_sets[name]
        previous = counts
    require(controls["encoder_calls"] == previous["encoder_calls_after"]
            and controls["cache_misses"] == previous["cache_misses_after"], "final actual path counters")
    return {"paths": controls["path_counts"], "exact_state_cache_entries": len(seen),
            "prewarmed_unique_states": len(prewarmed),
            "scope": "Receipt-backed calls/counts reconciled with exact selected state bytes; no latency claim"}


def verify_arm(root, arm, manifest, records, eligibility, priors, plans, cfg, custody):
    directory = root / arm
    metadata = load_json(directory / "metadata.json")
    require(metadata["schema"] == "vey.native-field.arm.v1" and metadata["arm"] == arm
            and metadata["mode"] == "train" and metadata["status"] == "train_complete", arm + ": complete registered run")
    for key, expected in (
        ("protocol_sha256", custody["protocol_sha256"]),
        ("dataset_manifest_sha256", custody["dataset_manifest_sha256"]),
        ("dataset_root", str(root)), ("dataset_phases", manifest["phases"]),
        ("source_exports", manifest["source_exports"]), ("catalogues", manifest["catalogues"]),
        ("eligible_endpoints", manifest["eligible_endpoints"]),
        ("ineligible_endpoints", manifest["ineligible_endpoints"]), ("recipe", cfg["training"]),
        ("state_mask_text", "[EMPTY EVIDENCE]"), ("semantic_scope", cfg["model"]["semantic_scope"]),
        ("no_dev_selection_or_calibration", True), ("sealed_phases_accessed", False),
    ):
        require(metadata[key] == expected, arm + ": custody/recipe " + key)
    expected_sources = custody["source_files"]
    correction = custody["execution_correction"]
    if arm == "prior" and correction:
        require(digest(directory / "metadata.json") == correction["reused_prior_metadata_sha256"],
                "exact retained pre-correction prior receipt")
        expected_sources = custody["original_source_files"]
    else:
        require(metadata.get("execution_correction_sha256") == custody["execution_correction_sha256"],
                arm + ": execution correction identity")
    require(metadata["source_files"] == {str(HERE / name): expected_sources[str(HERE / name)]
            for name in ("native_field_model.py", "native_field_train.py", "native_field_data.py")},
            arm + ": trainer source identity")
    hashes = artifact_files(directory, metadata, arm)
    eligible = [endpoint for endpoint, assessment in eligibility.items() if assessment["eligible"]]
    catalogues = {e: manifest["catalogues"][e] for e in eligible}
    native = {phase: {endpoint: records[phase][endpoint] for endpoint in eligible} for phase in PHASES}
    temperatures = metadata["temperatures"]
    require(set(temperatures) == set(eligible), arm + ": complete endpoint temperatures")
    for value in temperatures.values():
        require(.5 <= finite(value, "temperature") <= 10., "bounded temperature")
    history = load_json(directory / "history.json")
    liveness = load_json(directory / "liveness.json")
    require(history["schema"] == "vey.native-field.history.v1" and history["arm"] == arm
            and liveness["schema"] == "vey.native-field.liveness.v1" and liveness["arm"] == arm, "history/liveness native identity")
    report = {"artifact_hashes": hashes, "metadata_sha256": digest(directory / "metadata.json"),
              "temperatures": temperatures}
    report["smoke"] = check_smoke_liveness(liveness["smoke"], native["fit"], catalogues, arm, cfg)
    if arm == "prior":
        require(all(value == 1. for value in temperatures.values()) and metadata["checkpoint"] is None
                and metadata["custody"] is None and history["epochs"] == [], "state-blind prior has no fit/calibration/model")
        close(metadata["prior"], priors, "independent fit-only prior", 1e-12, 1e-12)
        close(history["prior"], priors, "history fit-only prior", 1e-12, 1e-12)
        close(liveness["smoke"]["prior"], priors, "smoke fit-only prior", 1e-12, 1e-12)
        require(history["fit_endpoint_counts"] == {e: len(native["fit"][e]) for e in eligible}, "prior native fit census")
    else:
        report["model_custody"] = model_custody(metadata["custody"], cfg)
        require(liveness["fresh_training_initial_sha256"] == liveness["smoke"]["initial_model_sha256"], "fresh seeded arm after discarded smoke")
        restored = liveness["selected_restore"]
        selected_epoch, selected_history, selection_probes = reconstruct_selection(
            directory / "selection.jsonl", native["selection"], cfg, restored["probe_state_id"])
        report["history"] = check_history(history, selected_history, selected_epoch, native, arm, cfg)
        checkpoint = metadata["checkpoint"]
        require(checkpoint["path"] == str(directory / "selected.safetensors")
                and checkpoint["sha256"] == hashes["selected.safetensors"]
                and checkpoint["selected_epoch"] == selected_epoch
                and checkpoint["state_dict_fingerprint"] == history["selected_fingerprint"], "selected checkpoint artifact identity")
        require(restored["selected_epoch"] == selected_epoch
                and restored["expected_fingerprint"] == history["selected_fingerprint"]
                and restored["probe_state_id"] in {row["state_id"] for rows in native["selection"].values() for row in rows.values()},
                "selected restore native state/epoch")
        require(restored["before_restore_fingerprint"] == history["epochs"][-1]["state_dict_fingerprint"],
                "actual pre-restore final epoch fingerprint")
        for endpoint, logits in selection_probes[selected_epoch].items():
            close(restored["selected_logits"][endpoint], logits, "selected actual native selection logits")
            close(restored["restored_logits"][endpoint], logits, "restored actual native selection logits")
        for endpoint, logits in selection_probes[cfg["training"]["epochs"]].items():
            close(restored["before_restore_logits"][endpoint], logits, "pre-restore actual native final epoch logits")
        report["selected_restore"] = check_restore(restored, catalogues, directory / "selected.safetensors")
        require(liveness["encoder_selected_sha256"] == report["selected_restore"]["encoder_state_dict_fingerprint"],
                "independently reconstructed selected encoder bytes")
        initial_encoder = metadata["custody"]["encoder_initial_sha256"]
        require(liveness["encoder_training_before_sha256"] == initial_encoder, "fresh training encoder identity")
        final_encoder = liveness["encoder_training_after_sha256"]
        require((final_encoder == initial_encoder) == (arm == "frozen"), "actual trained-final encoder freeze/liveness")
        if arm == "frozen":
            require(liveness["encoder_selected_sha256"] == initial_encoder, "selected frozen encoder invariant")
        report["encoder_adaptation"] = {
            "trained_final_encoder_changed": final_encoder != initial_encoder,
            "selected_encoder_changed": liveness["encoder_selected_sha256"] != initial_encoder,
            "selected_epoch0": selected_epoch == 0,
            "scope": "Selected encoder bytes independently hashed; final-fit/smoke hashes and gradients are persisted receipts"}
        report["checkpoint_selection"] = selected_history
        report["calibration"] = reconstruct_calibration(directory / "calibration.jsonl", native["calibration"], arm, temperatures)
        evidence = metadata["temperature_evidence"]
        require(set(evidence) == set(eligible), "complete persisted endpoint calibration evidence")
        for endpoint, rebuilt in report["calibration"].items():
            saved = evidence[endpoint]
            require(saved["phase"] == "calibration" and saved["decisions"] == rebuilt["rows"], "calibration disjoint inner population")
            close(saved["selected_temperature"], rebuilt["temperature"], "persisted grid optimum", 1e-12, 1e-12)
            close(saved["selected_nll"], rebuilt["nll"], "persisted calibration NLL", 1e-12, 1e-12)
            require(len(saved["grid"]) == 101, "complete temperature grid receipt")
            close([entry["nll"] for entry in saved["grid"]], rebuilt["grid_nll"], "independent101 grid NLL", 1e-12, 1e-12)
    predictions = {path: read_predictions(directory / (path + ".jsonl"), native["dev"], arm, temperatures,
                                         priors, path, plans) for path in PATHS}
    report["unavailable_donor_rows"] = read_unavailable(directory / "swapped_unavailable.jsonl", native["dev"], plans)
    report["numerical_control_reconstruction"] = validate_controls(predictions, native["dev"], plans)
    controls = liveness["controls"]
    require(controls["state_mask_text"] == "[EMPTY EVIDENCE]", "actual empty evidence control")
    expected_identity = history.get("selected_fingerprint") if arm != "prior" else hashlib.sha256(
        json.dumps(metadata["prior"], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False).encode()).hexdigest()
    require(controls["model_identity"] == expected_identity, "original/masked/swapped selected model identity")
    cache = check_cache_controls(controls["cache_controls"], native["dev"], catalogues, arm, temperatures, priors,
                                 predictions["dev"])
    report["actual_path_counters"] = check_path_counts(controls, native["dev"], plans, cache, arm)
    report["cache_invariance"] = cache
    report["coordinate_invariance"] = check_reversal(directory / "coordinate_reversal.jsonl", native["dev"],
                                                    predictions["dev"], arm, temperatures, priors)
    require(controls["coordinate_reversal_rows"] == report["coordinate_invariance"]["rows"], "all-row coordinate census")
    expected_denominators = {}
    for endpoint in eligible:
        available = [donor for donor in plans[endpoint].values() if donor is not None]
        expected_denominators[endpoint] = {"total": len(native["dev"][endpoint]), "available": len(available),
                                          "unavailable": len(native["dev"][endpoint]) - len(available),
                                          "unique_donors": len(set(available))}
    require(controls["swap_denominators"] == expected_denominators
            and metadata["swap_denominators"] == expected_denominators, "independent teacher-changing denominators")
    report["prediction_counts"] = {path: {endpoint: len(predictions[path]["raw"][endpoint]) for endpoint in eligible}
                                   for path in PATHS}
    report["mechanism_scope"] = (
        "Fit-only empirical mass; no neural or semantic credit" if arm == "prior" else
        "State encoder feeds canonical native fields. Original/masked outputs, donor-aligned swapped outputs, "
        "cache counters and interventions are reconstructed. Reuse of encoded donor fields need not call the encoder again. "
        "No independent encoder forward/backward replay, free-form question interpretation or generic semantic claim.")
    return report, predictions


def verify(root):
    import native_field_data as data

    cfg = data.protocol()
    root = Path(root).resolve()
    require(root == Path(cfg["output_root"]).resolve(), "unregistered native field root")
    require(cfg["statistics"]["resamples"] == 10000 and cfg["statistics"]["seed"] == 0, "registered bootstrap recipe")
    manifest, phases = data.load_dataset(root)
    custody = dataset_custody(root, manifest, cfg)
    records, census, components, eligibility = reconstruct_dataset(manifest, phases, cfg)
    eligible = [endpoint for endpoint, assessment in eligibility.items() if assessment["eligible"]]
    priors = {endpoint: empirical_prior(records, endpoint) for endpoint in eligible}
    plans = {endpoint: donor_plan(list(records["dev"][endpoint].values())) for endpoint in eligible}
    arms, predictions = {}, {}
    for arm in ARMS:
        arms[arm], predictions[arm] = verify_arm(root, arm, manifest, records, eligibility, priors, plans, cfg, custody)
    for arm in ARMS:
        arms[arm]["fields"] = {}
        for endpoint in cfg["model"]["fields"]:
            if not eligibility[endpoint]["eligible"]:
                arms[arm]["fields"][endpoint] = {"status": "data-insufficient", "eligibility": eligibility[endpoint],
                                               "trained_head": False, "semantic_credit": False}
                continue
            scopes = {}
            for scope in ("raw", "calibrated"):
                paths = {path: predictions[arm][path][scope][endpoint] for path in PATHS}
                report = endpoint_comparison(endpoint, records["dev"][endpoint], paths,
                    predictions["prior"]["dev"][scope][endpoint], plans[endpoint], manifest["catalogues"][endpoint], cfg)
                report["screen"] = screen_field(endpoint, report, True, cfg)
                if arm == "prior":
                    report["screen"]["semantic_credit"] = False
                scopes[scope] = report
            arms[arm]["fields"][endpoint] = {"eligibility": eligibility[endpoint], "scopes": scopes,
                "status": "state-blind-no-credit" if arm == "prior" else scopes["calibrated"]["screen"]["status"]}
    selected = {}
    for endpoint in cfg["model"]["fields"]:
        winner = next((arm for arm in ("frozen", "full") if arms[arm]["fields"][endpoint]["status"] == "passed"), None)
        selected[endpoint] = {"arm": winner, "earned": winner is not None,
                              "scope": "calibrated development mechanism screen",
                              "reason": "smallest declared passing learned arm" if winner else "no learned arm earned this native field"}
    return {"schema": "vey.native-field.verification.v1", "status": "PASS", "custody": custody,
            "registered_fields": cfg["model"]["fields"], "registered_arms": list(ARMS),
            "counts": census, "components": components, "eligibility": eligibility,
            "fit_only_priors": priors, "arms": arms, "field_selection": selected,
            "selection_order": ["frozen", "full"], "screen_scope": cfg["screen_gates"]["scope"],
            "screen_thresholds": cfg["screen_gates"], "statistics": cfg["statistics"],
            "all_negative_arms_and_fields_retained": True,
            "claim_limits": ["English selected native train/dev only", "fixed canonical native ontologies",
                             "nominal descriptive component intervals", "retained raters, not population calibration",
                             "cached/coordinate exactness is code behavior, not semantic alias recognition",
                             "receipt-backed numerical liveness; verifier does not run a model"],
            "sealed_phases_accessed": False, "model_executed_by_verifier": False,
            "B_STEF_allowed": False, "Pareto_credit": False, "promotion": False,
            "final_credit": False, "endgame_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_json(HERE / "native_field_protocol.json")
    root = args.root.resolve()
    require(root == Path(cfg["output_root"]).resolve(), "unregistered output root")
    try:
        receipt = verify(root)
    except Exception as error:
        receipt = {"schema": "vey.native-field.verification.v1", "status": "FAIL",
                   "error_type": type(error).__name__, "error": str(error),
                   "registered_fields": cfg["model"]["fields"], "registered_arms": list(ARMS),
                   "arms": {arm: {"status": "verification-failed",
                                  "fields": {endpoint: {"status": "unverified", "semantic_credit": False}
                                             for endpoint in cfg["model"]["fields"]}} for arm in ARMS},
                   "field_selection": {endpoint: {"arm": None, "earned": False} for endpoint in cfg["model"]["fields"]},
                   "B_STEF_allowed": False, "Pareto_credit": False, "promotion": False,
                   "final_credit": False, "endgame_complete": False, "model_executed_by_verifier": False}
    output = root / "verification.json"
    text = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_suffix(".json.pending")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": receipt["status"], "verification": str(output),
                      "earned_fields": [endpoint for endpoint, selection in receipt["field_selection"].items()
                                        if selection["earned"]]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
