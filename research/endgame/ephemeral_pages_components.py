#!/usr/bin/env python3
"""Privileged ECA-1 component factorial and finite cached grade-linear diagnostic.

Only train/validation/calibration/opened-development are accessible. No reader
or encoder forward is performed: frozen raw predictions are reconstructed in
FP32 before any component is substituted. Privileged labels are not deployable.
"""
from __future__ import annotations

import os
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"
# The CPU-only device mask belongs to this module's own CLI run. Masking the
# device at import time also hid a live GPU from every other module that
# imports the truth/component helpers, so scope it to direct execution.
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import gc
import hashlib
import itertools
import json
import math
import platform
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ephemeral_pages_capture as capture
import ephemeral_pages_evaluate as evaluate
import ephemeral_pages_train as train
import ephemeral_pages_build as build
from ephemeral_pages_features import (
    _apply_process_priority, _parse_meminfo, _resource_guard, sha256_file,
)
from ephemeral_pages_model import ReaderOutput, _output

DATA_ROOT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1")
DEFAULT_RUN = DATA_ROOT / "runs/seed7-grade-corrected"
DEFAULT_OUTPUT = DATA_ROOT / "runs/interface-audit-v1/components"
AUDIT_PROTOCOL = HERE / "ephemeral_pages_audit_protocol.json"
AUDIT_SHA256 = "7ff11075c9e8298100a29756354951017a71c85dff34193b63a811d7ac61c567"
PARENT_PROTOCOL_SHA256 = "4c0c1efe8ad8ffdde79004536a44fc6d034b7701cdedde8e2034dc3f4e52b489"
SOURCE_SHA256 = "1fb1f6609cbaa899bbf3f836917027421d2ca6b5543d972a13969d945d51dff4"
ALLOWED_PHASES = ("train", "validation", "calibration", "development")
FIELDS = ReaderOutput._fields
UNKNOWN = evaluate.UNKNOWN


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def save_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(evaluate.plain(row), ensure_ascii=False,
                                    sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return evaluate.artifact(path)


def save_array(path, values):
    with Path(path).open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return evaluate.artifact(path)


def fingerprint(value):
    payload = json.dumps(evaluate.plain(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def record_identity_hash(records, indices):
    digest = hashlib.sha256()
    for i in indices:
        digest.update((fingerprint(records[int(i)]) + "\n").encode())
    return digest.hexdigest()


def stable_id(prefix, *parts, length):
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()
    return prefix + "_" + digest[:length]


def source_catalogue(experiment=capture.DEFAULT_EXPERIMENT):
    root = Path(experiment.corpus_root).parent
    borrowed_provenance = None
    if experiment.protocol()[1] != PARENT_PROTOCOL_SHA256:
        import ephemeral_pages_verify as verify
        audit = verify.Audit()
        original_root, _ = verify.prefinal_provenance(audit, experiment)
        borrowed_provenance = evaluate.artifact(root / "corpus/eca2_prefinal_provenance_manifest.json")
        root = original_root
    preparation = read_json(root / "source/prepare_manifest_v1.json")
    provenance = {}
    if borrowed_provenance is not None:
        provenance["prefinal_borrow"] = borrowed_provenance
    for relative, entry in preparation["files"].items():
        path = root / relative
        require(sha256_file(path) == entry["sha256"], "authored truth-control hash changed: " + relative)
        provenance[relative] = evaluate.artifact(path)
    source_path = root / "source/eca_authored_source_v1.json"
    require(sha256_file(source_path) == SOURCE_SHA256, "audit's authored source changed")
    source = read_json(source_path)
    inventory = read_json(root / "source/eca_source_inventory_v1.json")
    receipt_path = root / "audit/accepted_receipt_v1.json"
    receipt = read_json(receipt_path)
    require(inventory["source_sha256"] == SOURCE_SHA256 == receipt["source_sha256"],
            "source/inventory/review lineage mismatch")
    require(receipt["status"] == "accepted" and receipt["item_count"] == 6300
            and not receipt["ambiguous_review_ids"] and not receipt["rejected_review_ids"],
            "independent opaque-review provenance is incomplete")
    require(receipt["packet_sha256"] == provenance["audit/opaque_review_packet_v1.json"]["sha256"],
            "accepted opaque packet hash mismatch")
    reviews = root / receipt["raw_reviews_file"]
    require(sha256_file(reviews) == receipt["raw_reviews_sha256"], "independent reviews changed")
    provenance["independent_receipt"] = evaluate.artifact(receipt_path)
    provenance["independent_reviews"] = evaluate.artifact(reviews)
    provenance["preparation"] = evaluate.artifact(root / "source/prepare_manifest_v1.json")
    pages, questions, properties = {}, {}, {}
    for prop in source["properties"]:
        properties[prop["property_id"]] = prop
        for key, destination, label in (("page_wordings", pages, "grade"),
                                        ("questions", questions, "orientation")):
            for item in prop[key]:
                # The authored inventory is provenance; closed-phase entries never
                # enter a lookup, diagnostic prediction, or fitted design matrix.
                if item["split"] not in ALLOWED_PHASES:
                    continue
                identity = (item["split"], item["text"])
                target = (prop["property_id"], item[label])
                require(identity not in destination or destination[identity] == target,
                        "authored text has inconsistent private targets")
                destination[identity] = target
    return pages, questions, properties, provenance


def load_phase(phase, experiment=capture.DEFAULT_EXPERIMENT):
    require(phase in ALLOWED_PHASES, "component audit cannot open closed phases")
    data = capture.load_phase(phase, experiment)
    ir = {}
    for row in capture.rows_for(phase, experiment):
        require(row["split"] == phase and row["id"] not in ir, "IR phase/identity mismatch")
        ir[row["id"]] = row
    require({r["row_id"] for r in data["records"]} == set(ir), "capture/IR coverage mismatch")
    manifest_path = experiment.cache_root / (phase + "_manifest.json")
    manifest = read_json(manifest_path)
    evidence = {"manifest": evaluate.artifact(manifest_path),
                "corpus": evaluate.artifact(experiment.corpus_root / (phase + ".jsonl")),
                "files": manifest["files"], "lineage": data["lineage"],
                "capture_token_receipts": data["token_receipts"],
                "historical_capture_counters": data["counters"]}
    return data, ir, evidence


def verify_truth(phase, data, ir, catalogue, initial_swap):
    pages, questions, properties, _ = catalogue
    world_properties = defaultdict(set)
    for row in ir.values():
        meta = row["metadata"]
        if meta["query_kind"] == "atomic" and meta["variant"] == "base":
            require(len(meta["terms"]) == 1, "base atomic query is not a single authored property")
            field = meta["terms"][0]["field_key"]
            require(field in properties, "base atomic property absent from authored source")
            world_properties[meta["world_id"]].add(field)
    worlds = {}
    field_code = "B" if phase == "development" else "A"
    for world_id, fields in world_properties.items():
        selected = tuple(sorted((properties[field] for field in fields),
                                key=lambda prop: build.FAMILIES.index(prop["family"])))
        require(len(selected) == 4 and len({prop["family"] for prop in selected}) == 4
                and all(prop["field_code"] == field_code for prop in selected),
                "base atomic world source inventory mismatch")
        worlds[world_id] = build.World(world_id, phase, 0, 0,
                                      tuple(prop["family"] for prop in selected), selected, {})
    by_family = defaultdict(list)
    for prop in properties.values():
        by_family[prop["family"]].append(prop)
    counts = Counter()
    truth_hash = hashlib.sha256()
    for row in ir.values():
        meta = row["metadata"]
        candidates = [c["id"] for c in row["candidates"] if c["id"] != UNKNOWN]
        require(len(candidates) == len(set(candidates)), "duplicate evidence candidate")
        require(set(candidates) == set(meta["stable_ordinals"]), "ordinal coverage mismatch")
        blocks = {b["id"]: b for b in row["state_blocks"]}
        require(len(blocks) == len(row["state_blocks"]), "duplicate evidence block")
        require(set(meta["page_owners"]) == set(meta["page_grades"]) == set(meta["page_fields"]),
                "page provenance maps disagree")
        mapped = meta.get("identity_map", {})
        for cid in candidates:
            ordinal = meta["stable_ordinals"][cid]
            require(type(ordinal) is int and ordinal >= 0, "invalid stable ordinal")
            source_cid = mapped.get(cid, cid)
            expected = (stable_id("cand", meta["world_id"], ordinal, length=12) if ordinal < 4 else
                        stable_id("cand", meta["world_id"], "probe", ordinal, length=12))
            require(source_cid == expected, "authored candidate ownership identity changed")
        for bid, owner in meta["page_owners"].items():
            require(bid in blocks and not blocks[bid]["exact"] and owner in candidates,
                    "absent/exact/unowned block used as semantic page")
            prop, grade = pages.get((phase, blocks[bid]["text"]), (None, None))
            require(prop == meta["page_fields"][bid] and type(grade) is int
                    and grade == meta["page_grades"][bid] and 0 <= grade <= 4,
                    "authored page text/field/grade mismatch")
            source_cid = mapped.get(owner, owner)
            primary = stable_id("page", meta["world_id"], source_cid, prop, "primary", length=18)
            contradiction = stable_id("page", meta.get("parent_decision_id"), prop,
                                      source_cid, "contradiction", length=18)
            require(bid == primary or (meta["variant"] == "relevant_page_contradiction"
                                      and bid == contradiction), "authored page owner hash mismatch")
            counts["page_ownership_grade_checks"] += 1
        scores, known = {}, {}
        for term in meta["terms"]:
            require(questions.get((phase, term["question"])) ==
                    (term["field_key"], term["orientation"]), "authored criterion orientation mismatch")
            counts["question_orientation_checks"] += 1
        require(all(term["field_key"] in properties for term in meta["terms"]),
                "required question property absent from authored source")
        authored_families = {properties[term["field_key"]]["family"] for term in meta["terms"]}
        require(meta["world_id"] in worlds, "query has no base atomic world source inventory")
        world = worlds[meta["world_id"]]
        if meta["query_kind"] == "absent":
            specs, absent_provenance = build._query_specs(world, properties, by_family)
            absent = next(spec for spec in specs if spec.query_kind == "absent")
            require(meta["query_id"] == absent.query_key
                    and meta["terms"] == build._term_metadata(absent.terms)
                    and row["question"] == absent.question,
                    "absent query differs from frozen generator form")
            require(authored_families == {absent_provenance["absent_family"]}
                    and absent_provenance["absent_property_id"] not in world_properties[meta["world_id"]]
                    and not any(field == absent_provenance["absent_property_id"]
                                for field in meta["page_fields"].values()),
                    "absent query source property has owned world evidence")
            require(row["gold"] == [UNKNOWN]
                    and all(meta["known"][cid] is False
                            and meta["teacher_scores"][cid] is None for cid in candidates),
                    "absent query parent/child truth is not UNKNOWN")
            expected_families = sorted(world.families)
            counts["generator_absent_source_family_checks"] += 1
        else:
            require(all(term["field_key"] in world_properties[meta["world_id"]]
                        for term in meta["terms"]),
                    "ordinary required property absent from world source inventory")
            expected_families = sorted(authored_families)
            counts["authored_required_source_family_checks"] += 1
        require(meta["family"] == expected_families,
                "required-source question family provenance mismatch")
        counts["required_source_family_checks"] += 1
        for cid in candidates:
            total, good = evaluate.Fraction(0), True
            for term in meta["terms"]:
                matching = [bid for bid, owner in meta["page_owners"].items()
                            if owner == cid and meta["page_fields"][bid] == term["field_key"]]
                counts["missing_required_children"] += not matching
                counts["contradictory_required_children"] += len(matching) > 1
                if len(matching) != 1:
                    good = False
                    continue
                grade = meta["page_grades"][matching[0]]
                directed = grade if term["orientation"] == 1 else 4 - grade
                total += evaluate.exact_weight(term) * evaluate.Fraction(directed, 4)
            known[cid] = good
            scores[cid] = total if good else None
            require(meta["known"][cid] is good, "authored candidate knownness mismatch")
            require(meta["teacher_scores"][cid] == (float(total) if good else None),
                    "authored exact teacher scalar mismatch")
        maxima = ([UNKNOWN] if not candidates or not all(known.values()) else
                  [cid for cid in candidates if scores[cid] == max(scores.values())])
        require(set(row["gold"]) == set(maxima), "authored exact maximal teacher set mismatch")
        truth_hash.update((fingerprint({"row_id": row["id"], "known": known,
                                       "scores": {c: str(v) if v is not None else None for c, v in scores.items()},
                                       "maxima": maxima}) + "\n").encode())
        counts["exact_teacher_rows"] += 1
    seen = set()
    child_knownness = {}
    for i, record in enumerate(data["records"]):
        if i % 1024 == 0:
            _resource_guard(torch, initial_swap)
        row = ir[record["row_id"]]
        meta, cid, ti = row["metadata"], record["candidate_id"], record["term_index"]
        identity = (row["id"], ti, cid)
        require(identity not in seen, "duplicate captured term/candidate")
        seen.add(identity)
        term = meta["terms"][ti]
        expected_pages = []
        for block in row["state_blocks"]:
            bid = block["id"]
            if not block["exact"] and meta["page_owners"].get(bid) == cid:
                expected_pages.append({"block_id": bid, "text": block["text"],
                                       "field_key": meta["page_fields"][bid],
                                       "grade_target": meta["page_grades"][bid] / 4})
        expected_pages.sort(key=lambda p: p["block_id"])
        require(record["pages"] == expected_pages, "captured page ownership/order/text mismatch")
        p = len(expected_pages)
        relevant = [page["field_key"] == term["field_key"] for page in expected_pages]
        count = sum(relevant)
        matching = [page["block_id"] for page, flag in zip(expected_pages, relevant) if flag]
        known = len(matching) == 1 and type(meta["page_grades"].get(matching[0])) is int
        grade_mask = [flag and known and grade is not None
                      for flag, grade in zip(relevant, (page["grade_target"] for page in expected_pages))]
        grade = [page["grade_target"] for page in expected_pages]
        orientation = [term["orientation"] if flag else None for flag in grade_mask]
        directed = [(g if term["orientation"] == 1 else 1 - g) if flag else None
                    for g, flag in zip(grade, grade_mask)]
        targets = {"relevance_target": [1 / count if flag else 0 for flag in relevant],
                   "grade_target": grade, "directed_grade_target": directed,
                   "grade_mask": grade_mask, "orientation_target": orientation,
                   "orientation_mask": grade_mask}
        for key, target in targets.items():
            require(record[key] == target, "captured private target mismatch: " + key)
            numerical = np.asarray([np.nan if v is None else v for v in target],
                                   dtype=data[key].dtype)
            require(np.array_equal(data[key][i, :p], numerical, equal_nan=True),
                    "packed target mismatch: " + key)
        require(bool(data["known_target"][i]) is known and record["known_target"] is known,
                "packed candidate knownness mismatch")
        require(np.array_equal(data["page_mask"][i], np.arange(data["page_mask"].shape[1]) < p),
                "packed masked width mismatch")
        require(record["question"] == (row["question"] if len(meta["terms"]) == 1 else term["question"])
                and record["split"] == phase and record["world_id"] == meta["world_id"]
                and record["teacher_score"] == meta["teacher_scores"][cid], "captured record provenance mismatch")
        parent_key = (row["id"], cid)
        child_knownness[parent_key] = child_knownness.get(parent_key, True) and known
        counts["child_local_records"] += 1
        if known and not meta["known"][cid]:
            counts["supported_sibling_in_unknown_parent"] += 1
        if not known and meta["known"][cid]:
            counts["unsupported_sibling_in_known_parent"] += 1
    conjunction_holds = True
    for row in ir.values():
        for cid in [c["id"] for c in row["candidates"] if c["id"] != UNKNOWN]:
            conjunction = child_knownness[(row["id"], cid)]
            conjunction_holds = conjunction_holds and conjunction is bool(row["metadata"]["known"][cid])
    require(conjunction_holds, "parent knownness is not the conjunction of required child truth")
    data["_leaf_truth_verified"] = True
    expected = sum(len(row["metadata"]["terms"]) * (len(row["candidates"]) - 1) for row in ir.values())
    require(len(seen) == expected, "required child/candidate inventory incomplete")
    return {"phase": phase, "checks": dict(counts), "captured_records": len(seen),
            "exact_truth_execution_sha256": truth_hash.hexdigest(), "independent_authored_recheck": True,
            "supervision": "child_local",
            "provenance_rule_scope": {
                "rule": "all authored required properties independent of evidence availability; frozen absent-query form and source-world family fallback",
                "excluded_rows": 0, "excluded_row_ids": [], "excluded_detail": []},
            "parent_known_equals_child_conjunction": True,
            "supported_sibling_in_unknown_parent": counts["supported_sibling_in_unknown_parent"],
            "unsupported_sibling_in_known_parent": counts["unsupported_sibling_in_known_parent"],
            "opaque_review_is_provenance_not_native_task_evidence": True}


def allocate_fields(directory, data):
    n, p = data["page_mask"].shape
    fields = {}
    for name in (*FIELDS, "known_probability"):
        shape = (n,) if name in {"score", "known_logits", "known_probability"} else (n, p)
        dtype = np.float64 if name == "known_probability" else np.float32
        fields[name] = np.lib.format.open_memmap(directory / (name + ".npy"), mode="w+",
                                                dtype=dtype, shape=shape)
    return fields


def field_artifacts(directory, fields):
    result = {}
    for name, values in fields.items():
        values.flush()
        result[name] = {**evaluate.artifact(directory / (name + ".npy")),
                        "dtype": str(values.dtype), "shape": list(values.shape)}
    return result


def baseline_fields(directory, control, phase, run_root, data, ir, cal, checkpoint, chunk_size, initial_swap,
                    protocol_hash=PARENT_PROTOCOL_SHA256):
    manifest_path = run_root / "evaluation" / phase / "evaluation.json"
    manifest = read_json(manifest_path)
    require(manifest["phase"] == phase and manifest["protocol_sha256"] == protocol_hash,
            "baseline evaluation lineage mismatch")
    require(manifest["checkpoint_files"][control]["sha256"] == checkpoint["sha256"],
            "baseline raw predictions are bound to another checkpoint")
    entries = manifest["artifacts"][control]
    retained = {}
    for key in ("term_candidate", "decisions"):
        entry = entries[key]
        source = Path(entry["path"])
        require(source.parent.resolve() == (run_root / "evaluation" / phase).resolve(),
                "baseline artifact escaped allowed phase")
        require(sha256_file(source) == entry["sha256"], "baseline prediction artifact changed")
        destination = directory / ("baseline_" + key + ".jsonl")
        shutil.copyfile(source, destination)
        retained[key] = evaluate.artifact(destination)
        require(retained[key]["sha256"] == entry["sha256"], "baseline was not retained verbatim")
    fields = allocate_fields(directory, data)
    for name, values in fields.items():
        values.fill(-np.inf if name in {"known_logits", "relevance_logits"} else
                    np.nan if name == "score" else 0)
    count = 0
    for i, raw in enumerate(json_rows(directory / "baseline_term_candidate.jsonl")):
        require(i < len(data["records"]) and raw["record_index"] == i, "baseline record order mismatch")
        record = data["records"][i]
        require(all(raw[key] == value for key, value in record.items()), "baseline/corrected capture mismatch")
        p = len(record["pages"])
        for name in (*FIELDS, "known_probability"):
            value = raw["prediction"][name]
            if fields[name].ndim == 1:
                fields[name][i] = (np.nan if name == "score" else -np.inf) if value is None else value
            else:
                require(len(value) == p, "baseline per-page width changed")
                fields[name][i, :p] = [(-np.inf if name == "relevance_logits" else np.nan)
                                      if v is None else v for v in value]
        count += 1
    require(count == len(data["records"]), "baseline prediction coverage incomplete")
    state = torch.load(run_root / (control + ".pt"), map_location="cpu", weights_only=False)
    require(state["protocol_sha256"] == protocol_hash, "baseline checkpoint protocol changed")
    parameters = [state["state_dict"][key].detach().to(dtype=torch.float32, device="cpu")
                  for key in ("bk", "uk", "vk")]
    with torch.inference_mode():
        for start in range(0, count, chunk_size):
            _resource_guard(torch, initial_swap)
            sl = slice(start, min(start + chunk_size, count))
            mask = torch.from_numpy(np.array(data["page_mask"][sl], copy=True))
            inputs = [torch.from_numpy(np.array(fields[name][sl], copy=True))
                      for name in ("relevance_logits", "direction", "raw_value")]
            rebuilt = _output(*inputs, mask, *parameters, False, direct_grade=control == "cross")
            for name, tensor in zip(FIELDS, rebuilt):
                require(np.array_equal(fields[name][sl], tensor.numpy(), equal_nan=True),
                        "FP32 identity reconstruction failed before substitution: " + control + "/" + name)
            probability = np.exp(-np.logaddexp(0, -fields["known_logits"][sl].astype(np.float64)))
            require(np.array_equal(probability, fields["known_probability"][sl]),
                    "baseline knownness probability reconstruction mismatch")
    decisions = evaluate.aggregate(fields, data, ir, cal)
    saved = list(json_rows(directory / "baseline_decisions.jsonl"))
    require(evaluate.plain(decisions) == saved, "baseline typed aggregation differs from retained predictions")
    reject_approximate_ties(decisions)
    return fields, decisions, parameters, state["normalizer"], {
        "baseline_evaluation": evaluate.artifact(manifest_path), "retained_verbatim": retained,
        "raw_arrays": field_artifacts(directory, fields), "FP32_identity_reconstruction": True,
        "reconstructed_fields": list(FIELDS), "reader_forwards": 0,
        "knownness_parameters": {key: float(value) for key, value in zip(("bk", "uk", "vk"), parameters)}}


def reject_approximate_ties(decisions):
    for row in decisions:
        if row["chosen"] == UNKNOWN:
            continue
        values = row["candidate_scores"]
        best = max(values.values())
        exact = {cid for cid, value in values.items() if value == best}
        require(exact == set(row["maximal_predictions"]),
                "frozen tie tolerance widened a distinct parent score; audit must not award exact-set credit")


def combination_fields(directory, control, bits, baseline, data, ir, parameters, chunk_size, initial_swap):
    result = allocate_fields(directory, data)
    with torch.inference_mode():
        for start in range(0, len(data["records"]), chunk_size):
            _resource_guard(torch, initial_swap)
            stop = min(start + chunk_size, len(data["records"]))
            sl = slice(start, stop)
            mask = torch.from_numpy(np.array(data["page_mask"][sl], copy=True))
            relevance, direction, value = [torch.from_numpy(np.array(baseline[name][sl], copy=True))
                                          for name in ("relevance_logits", "direction", "raw_value")]
            if bits["relevance"]:
                target = torch.from_numpy(np.array(data["relevance_target"][sl], copy=True))
                matching = target.sum(dim=-1, keepdim=True) > 0
                # log normalized matching-page targets, with uniform existing
                # pages when there is no match; no free oracle-logit amplitude.
                relevance = torch.where(matching, torch.log(target), torch.zeros_like(target))
            if bits.get("raw_extent") or bits.get("direct_grade") or bits.get("orientation"):
                for j, record in enumerate(data["records"][start:stop]):
                    term = ir[record["row_id"]]["metadata"]["terms"][record["term_index"]]
                    sign = term["orientation"]
                    p = len(record["pages"])
                    if bits.get("raw_extent") or bits.get("direct_grade"):
                        grades = torch.tensor([page["grade_target"] for page in record["pages"]], dtype=torch.float32)
                        value[j, :p] = grades if not bits.get("direct_grade") or sign == 1 else 1 - grades
                    if bits.get("orientation"):
                        direction[j, :p] = sign
            output = _output(relevance, direction, value, mask, *parameters, False,
                             direct_grade=control == "cross")
            for name, tensor in zip(FIELDS, output):
                result[name][sl] = tensor.numpy()
            result["known_probability"][sl] = np.exp(-np.logaddexp(0, -result["known_logits"][sl].astype(np.float64)))
            if bits["knownness"]:
                # Force candidate labels, not a recalibrated threshold. Other
                # combinations retain the original threshold and expression.
                labels = np.asarray(data["known_target"][sl], dtype=bool)
                result["known_probability"][sl] = labels.astype(np.float64)
                result["known_logits"][sl] = np.where(labels, np.inf, -np.inf)
            if bits["relevance"]:
                for j, record in enumerate(data["records"][start:stop]):
                    p = len(record["pages"])
                    target = np.asarray(record["relevance_target"], dtype=np.float32)
                    if not target.sum() and p:
                        target = np.full(p, np.float32(1 / p), dtype=np.float32)
                    require(np.array_equal(result["attention"][start + j, :p], target),
                            "oracle attention differs from normalized authored target")
    return result


def add_exact_metrics(ledger, decisions):
    for row in decisions:
        ledger.add("all_rows_exact_set", row["world_id"], row["exact_winner_set_correct"])
        ledger.add("all_rows_concrete_choice", row["world_id"], row["correct"])
        if row["variant"] == "base" and row["supported"] and row["term_count"] > 1:
            ledger.add("composition_exact_set", row["world_id"], row["exact_winner_set_correct"])
            ledger.add("composition_concrete_choice", row["world_id"], row["correct"])


def paired_descriptive(ledgers, weights):
    baseline = ledgers["learned"]
    result = {}
    for name, ledger in ledgers.items():
        if name == "learned":
            continue
        changes = {}
        for metric in sorted(set(baseline.values) & set(ledger.values)):
            require(set(baseline.values[metric]) == set(ledger.values[metric]),
                    "paired descriptive family population changed")
            # UNKNOWN precision legitimately changes its denominator; the same
            # world draws remain paired, not independently resampled estimates.
            samples = ledger.estimate(metric, weights) - baseline.estimate(metric, weights)
            finite = samples[np.isfinite(samples)]
            changes[metric] = {"difference": float(ledger.estimate(metric) - baseline.estimate(metric)),
                               "CI95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                               "valid_bootstrap_draws": len(finite), "p_value": "not_computed"}
        result[name] = changes
    return result


def run_factorial(directory, run_root, catalogue, calibrations, checkpoints, chunk_size, initial_swap,
                  experiment=capture.DEFAULT_EXPERIMENT):
    directory.mkdir()
    protocol_hash = experiment.protocol()[1]
    results, normalizer = {}, None
    cfg = evaluate.protocol(experiment)
    for phase in ("calibration", "development"):
        phase_root = directory / phase
        phase_root.mkdir()
        data, ir, evidence = load_phase(phase, experiment)
        truth = verify_truth(phase, data, ir, catalogue, initial_swap)
        worlds = sorted({record["world_id"] for record in data["records"]})
        require(len(worlds) == cfg["corpus"]["worlds"][phase], "factorial world population changed")
        expected_families = set(cfg["corpus"]["families"][:12 if phase == "calibration" else 16])
        observed_families = {family for row in ir.values()
                             if row["metadata"]["variant"] == "base"
                             and row["metadata"]["query_kind"] == "atomic" and UNKNOWN not in row["gold"]
                             for family in row["metadata"]["family"]}
        require(observed_families == expected_families, "fixed factorial family population changed")
        weights, bootstrap = evaluate.bootstrap_indices(phase_root, worlds)
        controls = {}
        for control in ("pages", "cross"):
            require(0 < calibrations[control]["knownness_threshold"] <= 1,
                    "forced knownness requires a positive probability threshold")
            control_root = phase_root / control
            control_root.mkdir()
            base_root = control_root / "learned"
            base_root.mkdir()
            baseline, baseline_decisions, parameters, stats, baseline_receipt = baseline_fields(
                base_root, control, phase, run_root, data, ir, calibrations[control],
                checkpoints[control], chunk_size, initial_swap, protocol_hash)
            if control == "pages":
                if normalizer is not None:
                    require(fingerprint(stats) == fingerprint(normalizer), "baseline normalizer changed across phases")
                normalizer = stats
            names = (("relevance", "raw_extent", "orientation", "knownness") if control == "pages" else
                     ("relevance", "direct_grade", "knownness"))
            summaries, artifacts, ledgers = {}, {}, {}
            for values in itertools.product((False, True), repeat=len(names)):
                bits = dict(zip(names, values))
                name = "learned" if not any(values) else "oracle_" + "_".join(k for k, v in bits.items() if v)
                combo_root = base_root if name == "learned" else control_root / name
                if name != "learned":
                    combo_root.mkdir()
                fields = baseline if name == "learned" else combination_fields(
                    combo_root, control, bits, baseline, data, ir, parameters, chunk_size, initial_swap)
                decisions = baseline_decisions if name == "learned" else evaluate.aggregate(fields, data, ir, calibrations[control])
                reject_approximate_ties(decisions)
                if all(values):
                    require(all(row["exact_winner_set_correct"] and row["correct"] for row in decisions),
                            "all-privileged truth failed exact teacher outputs; no learned interpretation authorized")
                    for row in decisions:
                        teacher = ir[row["row_id"]]["metadata"]["teacher_scores"]
                        require(all(score == teacher[cid] for cid, score in row["candidate_scores"].items()),
                                "all-privileged scalar composition differs from authored teacher")
                ledger, pairs = evaluate.metric_ledger(control, fields, data, ir, decisions,
                    calibrations[control]["ordinal_sigma"], cfg["corpus"]["families"], worlds)
                add_exact_metrics(ledger, decisions)
                ledgers[name] = ledger
                summaries[name] = ledger.summaries(weights)
                descriptor = {"control": control, "oracle_components": bits,
                              "privileged": any(values), "deployable": not any(values),
                              "opened_development_diagnostic": phase == "development",
                              "promotion": False, "certificate": "unavailable",
                              "source_capture_manifest": evidence["manifest"],
                              "raw_prediction_row_order": "captured records, same full masked page width",
                              "relevance_oracle_logits": "log(normalized matching-page target); no-match zero logits over existing pages",
                              "knownness": "forced authored candidate label" if bits["knownness"] else
                                           "unchanged baseline bk/softplus(uk)/softplus(vk), original calibrated threshold",
                              "cross_direction_causal_score_credit": False,
                              "typed_arithmetic": "frozen Fraction weight times persisted scalar; required-child/candidate UNKNOWN; exact score ties only"}
                if name != "learned":
                    decision_file = save_rows(combo_root / "decisions.jsonl", decisions)
                    raw_arrays = field_artifacts(combo_root, fields)
                else:
                    decision_file = baseline_receipt["retained_verbatim"]["decisions"]
                    raw_arrays = baseline_receipt["raw_arrays"]
                artifacts[name] = {"raw_predictions": raw_arrays, "decisions": decision_file,
                    "paired_interventions": save_rows(combo_root / "paired_interventions.jsonl", pairs),
                    "world_metric_ledger": evaluate.write_json(combo_root / "world_metrics.json", ledger.ledger()),
                    "aggregation_provenance": evaluate.write_json(combo_root / "provenance.json", descriptor)}
                if name != "learned":
                    del fields, decisions
            controls[control] = {"combinations": 2 ** len(names), "metrics": summaries,
                                 "artifacts": artifacts, "baseline": baseline_receipt,
                                 "paired_descriptive_differences": paired_descriptive(ledgers, weights),
                                 "all_privileged_exact_teacher_control": True}
            del baseline, baseline_decisions, ledgers
            gc.collect()
        results[phase] = {"truth_receipt": truth, "capture": evidence,
                          "bootstrap": bootstrap, "controls": controls,
                          "scope": "descriptive; fixed authored inventory; opened development, no inference/promotion",
                          "derived_swaps_are_independent_experiments": False}
        del data, ir, weights
        gc.collect()
    return results, normalizer


def supervised_unique(data, indices, properties, initial_swap):
    require(data.get("_leaf_truth_verified") is True,
            "supervised fitting requires an independent child-local truth check; parent-wide masks are invalid")
    unique = {}
    occurrences = 0
    for offset, i in enumerate(indices):
        if offset % 1024 == 0:
            _resource_guard(torch, initial_swap)
        record = data["records"][int(i)]
        for j in np.flatnonzero(data["grade_mask"][i]):
            page = record["pages"][int(j)]
            text_hash = hashlib.sha256(page["text"].encode()).hexdigest()
            feature_hash = hashlib.sha256(memoryview(np.asarray(data["pages"][i, j])).cast("B")).hexdigest()
            target = float(data["grade_target"][i, j])
            require(target == page["grade_target"] and math.isfinite(target), "supervised raw grade mismatch")
            item = {"text_sha256": text_hash, "field_key": page["field_key"],
                    "family": properties[page["field_key"]]["family"], "target": target,
                    "record_index": int(i), "page_index": int(j), "feature_sha256": feature_hash,
                    "occurrences": 1}
            if text_hash in unique:
                prior = unique[text_hash]
                require(all(prior[key] == item[key] for key in
                            ("field_key", "family", "target", "feature_sha256")),
                        "duplicate supervised page text has inconsistent target or cached feature")
                prior["occurrences"] += 1
            else:
                unique[text_hash] = item
            occurrences += 1
    rows = [unique[key] for key in sorted(unique)]
    require(rows, "empty supervised grade design")
    return rows, occurrences


def normalized_design(data, rows, stats):
    # Keep the common train normalizer, but perform the diagnostic fit in FP64.
    design = np.empty((len(rows), 384), dtype=np.float64)
    mean, std = np.asarray(stats["mean"], dtype=np.float64), np.asarray(stats["std"], dtype=np.float64)
    for i, row in enumerate(rows):
        np.subtract(data["pages"][row["record_index"], row["page_index"]], mean, out=design[i])
        design[i] /= std
    return design


def run_linear(directory, run_root, catalogue, normalizer, chunk_size, initial_swap,
               experiment=capture.DEFAULT_EXPERIMENT):
    directory.mkdir()
    train_data, train_ir, train_evidence = load_phase("train", experiment)
    train_truth = verify_truth("train", train_data, train_ir, catalogue, initial_swap)
    selected = train.optimizer_indices(train_data, "train", experiment)
    computed = train.fit_normalizer(train_data, selected, chunk_size)
    if normalizer is None:
        checkpoint = torch.load(run_root / "pages.pt", map_location="cpu", weights_only=False)
        require(checkpoint["protocol_sha256"] == protocol_hash, "grade checkpoint protocol mismatch")
        normalizer = checkpoint["normalizer"]
    require(fingerprint(normalizer) == fingerprint(computed), "common train q/page normalizer identity failed")
    stats = normalizer["pages"]
    rows, occurrences = supervised_unique(train_data, selected, catalogue[2], initial_swap)
    design = normalized_design(train_data, rows, stats)
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    _resource_guard(torch, initial_swap)
    coefficients, residuals, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    error = fitted - target
    # Independently accumulate Gram/cross-products, instead of reusing the SVD
    # solver's residual output or treating a tiny training MSE as generalization.
    gram = np.zeros((384, 384), dtype=np.float64)
    cross = np.zeros(384, dtype=np.float64)
    for vector, value in zip(design, target):
        gram += np.outer(vector, vector)
        cross += vector * value
    normal_residual = gram @ coefficients - cross
    direct_gradient = design.T @ error
    scale = max(1.0, float(np.linalg.norm(gram, ord=np.inf) * np.linalg.norm(coefficients, ord=np.inf)
                           + np.linalg.norm(cross, ord=np.inf)))
    require(np.linalg.norm(normal_residual, ord=np.inf) / scale <= 1e-10
            and np.linalg.norm(direct_gradient, ord=np.inf) / scale <= 1e-10,
            "independent normal-equation check failed")
    second, _, second_rank, second_singular = linalg.lstsq(
        design, target, cond=np.finfo(np.float64).eps * max(design.shape), lapack_driver="gelss")
    second_fitted = design @ second
    coefficient_scale = max(1.0, float(np.linalg.norm(coefficients)))
    coefficient_difference = float(np.linalg.norm(second - coefficients) / coefficient_scale)
    fit_difference = float(np.max(np.abs(second_fitted - fitted)))
    second_mse = float(np.mean(np.square(second_fitted - target)))
    mse = float(np.mean(np.square(error)))
    require(second_rank == rank and coefficient_difference <= 1e-7
            and fit_difference <= 1e-8 and abs(second_mse - mse) <= 1e-12 * max(1, mse),
            "independent GELSS minimum-norm/fit residual check failed")
    singular_condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
    retained_condition = float(singular[0] / singular[rank - 1]) if rank else None
    files = {"coefficients": save_array(directory / "coefficients.npy", coefficients),
             "singular_values": save_array(directory / "singular_values.npy", singular),
             "second_solver_coefficients": save_array(directory / "second_solver_coefficients.npy", second),
             "second_solver_singular_values": save_array(directory / "second_solver_singular_values.npy", second_singular),
             "normal_equation_residual": save_array(directory / "normal_equation_residual.npy", normal_residual),
             "direct_gradient": save_array(directory / "direct_gradient.npy", direct_gradient),
             "normalizer": evaluate.write_json(directory / "normalizer.json", normalizer),
             "optimizer_record_indices": save_array(directory / "optimizer_record_indices.npy", selected),
             "train_design": save_array(directory / "train_design.npy", design),
             "train_targets": save_array(directory / "train_targets.npy", target),
             "train_predictions": save_rows(directory / "train_predictions.jsonl",
                 ({**row, "prediction_raw_unclipped": float(prediction)} for row, prediction in zip(rows, fitted)))}
    grouped = Counter(row["family"] for row in rows)
    result = {"schema": "vey.eca.finite-grade-linear-diagnostic.v1", "bias": False, "dtype": "float64",
              "solver": "numpy.linalg.lstsq(rcond=None)", "rank": int(rank),
              "design_shape": list(design.shape), "singular_condition": singular_condition,
              "retained_singular_condition": retained_condition,
              "full_column_rank": rank == design.shape[1],
              "unidentified_coefficient_directions": int(design.shape[1] - rank),
              "train_MSE": mse, "solver_reported_residuals": residuals.tolist(),
              "unique_supervised_train_page_texts": len(rows), "supervised_occurrences": occurrences,
              "unique_train_counts_by_family": dict(grouped),
              "unique_train_counts_by_property": dict(Counter(row["field_key"] for row in rows)),
              "unique_train_counts_by_grade": dict(Counter(str(row["target"]) for row in rows)),
              "train_capture": train_evidence, "train_truth_receipt": train_truth,
              "optimizer_row_identity_sha256": record_identity_hash(train_data["records"], selected),
              "normal_equation": {"independent_gram_residual_inf": float(np.max(np.abs(normal_residual))),
                                  "direct_gradient_inf": float(np.max(np.abs(direct_gradient))),
                                  "normalization_scale": scale, "relative_tolerance": 1e-10, "pass": True},
              "second_solver": {"driver": "scipy.linalg.lstsq GELSS", "rank": int(second_rank),
                                "train_MSE": second_mse, "relative_coefficient_difference": coefficient_difference,
                                "maximum_train_prediction_difference": fit_difference,
                                "coefficient_tolerance": 1e-7, "prediction_tolerance": 1e-8,
                                "MSE_absolute_scaled_tolerance": 1e-12 * max(1, mse), "pass": True},
              "artifacts": files, "held": {},
              "scope": "finite pinned bias-free normalized page-feature linear class; not absence of semantic information or a ranking bound",
              "normalizer_policy": "existing optimizer-only joint train q/page statistics; stored FP32 mean/std, FP64 transform for this FP64 fit",
              "neural_inputs": "cached 384-dimensional page feature only; no exact numeric facts, provenance, IDs, gold or family",
              "encoder_forwards": 0, "final_activity": False, "promotion": False,
              "B_STEF_allowed": False, "certificate": "unavailable"}
    train_hashes = {row["text_sha256"] for row in rows}
    del train_data, train_ir, design, gram
    gc.collect()
    for phase in ("validation", "development"):
        data, ir, evidence = load_phase(phase, experiment)
        truth = verify_truth(phase, data, ir, catalogue, initial_swap)
        indices = (train.optimizer_indices(data, phase, experiment) if phase == "validation" else
                   np.arange(len(data["records"]), dtype=np.int64))
        held_rows, held_occurrences = supervised_unique(data, indices, catalogue[2], initial_swap)
        require(not train_hashes.intersection(row["text_sha256"] for row in held_rows),
                "grade-linear held page texts overlap fitting texts")
        held_design = normalized_design(data, held_rows, stats)
        predictions = held_design @ coefficients
        errors = np.abs(predictions - np.asarray([row["target"] for row in held_rows]))
        weights = np.asarray([row["occurrences"] for row in held_rows])
        by_family = defaultdict(list)
        by_property = defaultdict(list)
        for row, error in zip(held_rows, errors):
            by_family[row["family"]].append(float(error))
            by_property[row["field_key"]].append(float(error))
        prediction_file = save_rows(directory / (phase + "_predictions.jsonl"),
            ({**row, "prediction_raw_unclipped": float(prediction), "absolute_error": float(error)}
             for row, prediction, error in zip(held_rows, predictions, errors)))
        result["held"][phase] = {"raw_unclipped_MAE_unique_page_text": float(errors.mean()),
            "raw_unclipped_MAE_supervised_occurrences": float(np.average(errors, weights=weights)),
            "family_macro_unique_text_MAE": float(np.mean([np.mean(v) for v in by_family.values()])),
            "unique_page_texts": len(held_rows), "supervised_occurrences": held_occurrences,
            "counts_MAE_by_family": {k: {"count": len(v), "MAE": float(np.mean(v))} for k, v in by_family.items()},
            "counts_MAE_by_property": {k: {"count": len(v), "MAE": float(np.mean(v))} for k, v in by_property.items()},
            "raw_prediction_range": [float(predictions.min()), float(predictions.max())],
            "predictions": prediction_file, "capture": evidence, "truth_receipt": truth,
            "selection": "existing optimizer rows" if phase == "validation" else "all diagnostic supervised records, deduplicated by text",
            "used_for_fitting_or_selection": False, "descriptive_opened_development": phase == "development"}
        del data, ir, held_design
        gc.collect()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic", action="store_true",
                        help="run the component audit under the ECA-2 child-local experiment")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--part", choices=("all", "factorial", "linear"), default="all")
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args(argv)
    require(args.chunk_size > 0, "chunk size must be positive")
    require(sha256_file(AUDIT_PROTOCOL) == AUDIT_SHA256, "predeclared audit protocol changed")
    audit = read_json(AUDIT_PROTOCOL)
    require(not audit["promotion"] and not audit["B_STEF_allowed"], "audit scope changed")
    experiment = capture.resolve_experiment(args.atomic)
    _, protocol_hash = experiment.protocol()
    require(audit["parent_protocol_sha256"] in {PARENT_PROTOCOL_SHA256, protocol_hash},
            "audit parent protocol is neither the frozen ECA-1 nor the active protocol")
    args.run_root = (args.run_root or (capture.ATOMIC_ROOT / "runs/seed7" if args.atomic else DEFAULT_RUN)).resolve()
    args.output_root = (args.output_root or
                        (capture.ATOMIC_ROOT / "runs/interface-audit-v1/components" if args.atomic else DEFAULT_OUTPUT)).resolve()
    require(args.output_root.is_relative_to(Path(capture.ATOMIC_ROOT if args.atomic else DATA_ROOT)
                                            / "runs/interface-audit-v1/components"),
            "component artifacts must stay under their exclusive diagnostic output root")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    priority = _apply_process_priority()
    initial_swap = _parse_meminfo()[1]
    _resource_guard(torch, initial_swap)
    catalogue = source_catalogue(experiment)
    calibration_path = args.run_root / "calibration.json"
    calibration = read_json(calibration_path)
    checkpoints = {control: evaluate.artifact(args.run_root / (control + ".pt")) for control in ("pages", "cross")}
    require(calibration["protocol_sha256"] == protocol_hash, "baseline calibration protocol changed")
    for control, entry in checkpoints.items():
        require(calibration["checkpoint_files"][control] == entry, "baseline calibration/checkpoint mismatch")
    args.output_root.mkdir(parents=True, exist_ok=False)
    result = {"schema": "vey.eca.privileged-component-audit.v1", "part": args.part,
              "audit_protocol": evaluate.artifact(AUDIT_PROTOCOL),
              "parent_protocol_sha256": PARENT_PROTOCOL_SHA256,
              "source_truth_controls": catalogue[3], "baseline_checkpoints": checkpoints,
              "baseline_calibration": evaluate.artifact(calibration_path),
              "implementation": {path.name: evaluate.artifact(path) for path in
                                 (Path(__file__), Path(capture.__file__), Path(train.__file__),
                                  Path(evaluate.__file__), HERE / "ephemeral_pages_model.py",
                                  HERE / "ephemeral_pages_features.py")},
              "environment": {"python": sys.version, "executable": sys.executable,
                              "platform": platform.platform(), "numpy": np.__version__,
                              "torch": torch.__version__, "scipy": __import__("scipy").__version__,
                              "priority": priority, "torch_threads": torch.get_num_threads(),
                              "BLAS_policy_threads": 1, "device": "cpu"},
              "reader_forwards": 0, "encoder_forwards": 0,
              "opened_phases": [], "final_activity": False,
              "promotion": False, "B_STEF_allowed": False, "certificate": "unavailable",
              "scope": "privileged descriptive interface audit; fixed inventory; opened-development only; no exploratory p-values or final/neutral/competitor credit"}
    normalizer = None
    if args.part in {"all", "factorial"}:
        factorial, normalizer = run_factorial(args.output_root / "factorial", args.run_root,
            catalogue, calibration["controls"], checkpoints, args.chunk_size, initial_swap, experiment)
        result["factorial"] = evaluate.write_json(args.output_root / "factorial_result.json", factorial)
        result["opened_phases"].extend(("calibration", "development"))
    if args.part in {"all", "linear"}:
        linear = run_linear(args.output_root / "linear", args.run_root, catalogue, normalizer,
                            args.chunk_size, initial_swap, experiment)
        result["linear"] = evaluate.write_json(args.output_root / "linear_result.json", linear)
        result["opened_phases"].extend(("train", "validation", "development"))
    result["opened_phases"] = sorted(set(result["opened_phases"]))
    for control, entry in checkpoints.items():
        require(sha256_file(args.run_root / (control + ".pt")) == entry["sha256"], "immutable baseline changed during audit")
    require(sha256_file(calibration_path) == result["baseline_calibration"]["sha256"], "baseline calibration changed during audit")
    for entry in result["implementation"].values():
        require(sha256_file(entry["path"]) == entry["sha256"], "audit implementation changed during execution")
    for entry in catalogue[3].values():
        require(sha256_file(entry["path"]) == entry["sha256"], "authored truth controls changed during execution")
    require(sha256_file(AUDIT_PROTOCOL) == AUDIT_SHA256, "audit protocol changed during execution")
    result["unchanged_baseline_checkpoints"] = True
    result["resource_end"] = _resource_guard(torch, initial_swap)
    evaluate.write_json(args.output_root / "result.json", result)
    print(json.dumps({"result": str(args.output_root / "result.json"), "encoder_forwards": 0,
                      "reader_forwards": 0, "final_activity": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
