#!/usr/bin/env python3
"""Independently reconstruct sealed ECA-1/ECA-2 predictions without a model.

Only persisted scalar/page predictions are consumed. No reader, trainer or
feature-capture entry point runs; the capture driver is imported for its frozen
Experiment contract and protocol-pinned hashes only, and this module re-derives
every decision, target and metric itself. Verification is not promotion.

Target reconstruction is child-local: a term/candidate is supervised only by
the graded active-property pages that candidate owns, never by parent-wide
``metadata.known``. The ECA-1 parent-wide capture is still reproduced bit for
bit when it is the object under audit, but it is reported as numerically
faithful, semantically invalid supervision and earns no interface, ceiling or
promotion reading.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
import ephemeral_pages_capture as capture
from ephemeral_pages_features import ATOMIC_PROTOCOL_SHA256

PROTOCOL_SHA256 = "4c0c1efe8ad8ffdde79004536a44fc6d034b7701cdedde8e2034dc3f4e52b489"
DEFAULT_RUN = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/runs/seed7-grade-corrected")
DEFAULT_ATOMIC_RUN = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-atomic-v2/runs/seed7")
UNKNOWN = "__unknown__"
CONTROLS = ("pages", "cross", "cosine", "lexical", "query_blind")
INTERVENTIONS = ("zero_question", "zero_pages", "uniform_attention")
ROBUST = {"candidate_permutation_1", "candidate_permutation_2", "candidate_rename", "page_reorder", "question_reorder"}
MISSING = {"base", "relevant_page_erasure", "relevant_page_contradiction"}
SIGMAS = (.025, .05, .075, .1, .15, .2, .3, .4)
HISTORICAL_SOURCE_REVISIONS = ("98672ea", "6b85ae5", "2818b8b")
PARENT_PROTOCOL_SHA256 = PROTOCOL_SHA256
LEGACY_SUPERVISION_KEY = "grade_correction"
ATOMIC_SUPERVISION_KEY = "atomic_supervision"
LEGACY_STATUS = "verified_legacy_parent_wide_supervision_numerics_only"
ATOMIC_STATUS = "verified_child_local_supervision"
RECORD_INPUT_KEYS = ("row_id", "world_id", "split", "term_index", "candidate_id", "question",
                     "term_weight", "pages", "teacher_score")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def json_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise RuntimeError(f"blank persisted row: {path}:{number}")
            yield json.loads(line)


class Audit:
    def __init__(self):
        self.hashes: dict[str, str] = {}
        self.counts = Counter()
        self.diffs: list[dict] = []

    def require(self, condition: bool, label: str) -> None:
        self.counts["assertions"] += 1
        if not condition:
            self.diffs.append({"path": label, "reason": "assertion failed"})
            raise RuntimeError(label)

    def hash(self, path: Path, expected: str | None = None) -> str:
        path = path.resolve()
        actual = self.hashes.get(str(path))
        if actual is None:
            actual = digest(path)
            self.hashes[str(path)] = actual
        if expected is not None:
            self.require(actual == expected, f"artifact SHA256 mismatch: {path}")
        return actual

    def artifact(self, entry: dict, base: Path) -> Path:
        self.require(isinstance(entry, dict) and isinstance(entry.get("path"), str)
                     and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64,
                     "artifact must have path and SHA256")
        path = Path(entry["path"])
        if not path.is_absolute():
            path = base / path
        self.hash(path, entry["sha256"])
        return path.resolve()

    def load(self, path: Path) -> dict:
        self.hash(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def source(self, path: Path, expected: str) -> None:
        if self.hash(path) == expected:
            return
        relative = str(path.resolve().relative_to(HERE.parents[1]))
        for revision in HISTORICAL_SOURCE_REVISIONS:
            result = subprocess.run(["git", "show", f"{revision}:{relative}"],
                                    cwd=HERE.parents[1], capture_output=True, check=False)
            if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected:
                self.hashes[f"git:{revision}:{relative}"] = expected
                self.counts["historical_source_hashes"] += 1
                return
        self.require(False, f"source hash has no current or approved historical bytes: {path}")

    def equal(self, actual: Any, expected: Any, label: str, tolerance: float = 1e-10) -> None:
        self.counts["compared_values"] += 1
        if isinstance(actual, dict) and isinstance(expected, dict):
            self.require(set(actual) == set(expected), label + ": object keys differ")
            for key in actual:
                self.equal(actual[key], expected[key], label + "." + str(key), tolerance)
            return
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            self.require(len(actual) == len(expected), label + ": lengths differ")
            for index, (left, right) in enumerate(zip(actual, expected)):
                self.equal(left, right, f"{label}[{index}]", tolerance)
            return
        if type(actual) is bool or type(expected) is bool:
            ok = type(actual) is type(expected) and actual == expected
        elif isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
            ok = (actual == expected if type(actual) is int and type(expected) is int else
                  math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance))
        else:
            ok = actual == expected
        if not ok:
            self.diffs.append({"path": label, "reconstructed": actual, "persisted": expected})
            raise RuntimeError(f"reconstruction differs: {label}")


def checked_receipt(audit: Audit, path: Path, run_root: Path, protocol_hash: str) -> tuple[dict, dict]:
    """Finish this gate before touching any final corpus/capture/prediction file."""
    receipt = audit.load(path)
    audit.require(receipt.get("schema") == "vey.eca.selection-calibration.v1", "wrong final receipt schema")
    audit.require(receipt.get("protocol_sha256") == protocol_hash, "wrong final receipt protocol")
    for key, value in (("eligible_arm", "pages"), ("eligible_arms", ["pages"]), ("final_outcomes_used", False)):
        audit.equal(receipt.get(key), value, "receipt." + key)
    audit.require(set(receipt["checkpoint_files"]) == set(CONTROLS), "receipt omits a fixed control")
    for name, entry in receipt["checkpoint_files"].items():
        actual = audit.artifact(entry, path.parent)
        target = run_root / ("lexical.pkl" if name == "lexical" else name + ".pt")
        audit.require(actual == target.resolve(), "receipt checkpoint belongs to another run")
    selection_path = audit.artifact(receipt["selection_file"], path.parent)
    calibration_path = audit.artifact(receipt["calibration_file"], path.parent)
    development_path = audit.artifact(receipt["development_evaluation"], path.parent)
    audit.require(selection_path == (run_root / "selection.json").resolve(), "selection belongs to another run")
    audit.require(calibration_path == (run_root / "calibration.json").resolve(), "calibration belongs to another run")
    selection, calibration = audit.load(selection_path), audit.load(calibration_path)
    for label, obj, schema in (("selection", selection, "vey.eca.selection.v1"),
                               ("calibration", calibration, "vey.eca.calibration.v1")):
        audit.equal(obj.get("schema"), schema, label + ".schema")
        audit.equal(obj.get("protocol_sha256"), protocol_hash, label + ".protocol")
        audit.equal(obj.get("final_outcomes_used"), False, label + ".no_final_selection")
    audit.equal(selection.get("eligible_arm"), "pages", "selection.eligible_arm")
    audit.equal(selection.get("eligible_arms"), ["pages"], "selection.eligible_arms")
    audit.equal(selection["development_evaluation"], receipt["development_evaluation"], "selection.development")
    audit.equal(calibration["checkpoint_files"], receipt["checkpoint_files"], "calibration.checkpoints")
    audit.equal(calibration.get("fit_split"), "calibration", "calibration.fit_split")
    audit.require(set(calibration["controls"]) == set(CONTROLS), "sealed calibration omits a control")
    development = audit.load(development_path)
    audit.equal(development.get("phase"), "development", "sealed development.phase")
    audit.equal(development.get("final_outcomes_used_for_selection"), False, "development.no_final_selection")
    audit.equal(selection["gate_screens"], development["gate_screens"], "selection.gate_screens")
    audit.equal(selection["comparisons"], development["comparisons"], "selection.comparisons")
    audit.artifact(calibration["calibration_evaluation"], calibration_path.parent)
    for name, settings in calibration["controls"].items():
        audit.equal(settings["fit_split"], "calibration", name + ".calibration_split")
        audit.equal(settings["final_outcomes_used"], False, name + ".no_final_calibration")
    return receipt, calibration["controls"]


def correction_capture(audit: Audit, manifest: dict, features_root: Path, corpus_root: Path, cfg: dict) -> int:
    phase, repair = manifest["phase"], manifest["lineage"]["grade_correction"]
    root = Path(cfg["output_root"])
    audit.require(phase != "final", "final cannot use prefinal correction repack")
    original_path = audit.artifact(repair["original_capture_manifest"], features_root)
    audit.require(original_path == (root / "features" / f"{phase}_manifest.json").resolve(), "correction references another original capture")
    original = audit.load(original_path)
    audit.equal({key: value for key, value in manifest["lineage"].items() if key != "grade_correction"},
                original["lineage"], "correction preserves original encoder/reader lineage")
    audit.equal(repair["existing_inputs_reencoded"], 0, "correction reencoded existing inputs")
    audit.equal(repair["optimizer_reruns"], 0, "correction reran optimization")
    audit.equal(repair["original_corpus_sha256"], original["corpus_sha256"], "correction original corpus hash")
    audit.equal(repair["corrected_corpus_sha256"], manifest["corpus_sha256"], "correction current corpus hash")
    for key in ("amendment_file", "implementation"):
        audit.artifact(repair[key], features_root)
    old_corpus = root / "corpus.superseded-grade-intervention" / f"{phase}.jsonl"
    audit.hash(old_corpus, original["corpus_sha256"])
    with old_corpus.open(encoding="utf-8") as stream:
        old_rows = {json.loads(line)["id"]: line for line in stream}
    invariant, grades, changed, seen = 0, 0, 0, set()
    with (corpus_root / f"{phase}.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            audit.require(row["id"] in old_rows and row["id"] not in seen, "correction changed row identities")
            seen.add(row["id"])
            if row["metadata"]["variant"] != "relevant_page_grade_change":
                audit.require(line == old_rows[row["id"]], "correction changed non-grade IR bytes")
                invariant += 1
            else:
                grades += 1
                parent = json.loads(old_rows[row["metadata"]["parent_decision_id"]])
                changed += set(row["gold"]) != set(parent["gold"])
                audit.equal(row["metadata"]["exact_facts"], parent["metadata"]["exact_facts"], "grade repair changed numeric exact facts")
    audit.require(seen == set(old_rows), "correction dropped original row identities")
    del old_rows
    for key, value in (("invariant_non_grade_rows", invariant), ("grade_rows", grades), ("teacher_changing_grade_rows", changed)):
        audit.equal(repair[key], value, "correction.independent." + key)
    cache_indices, cache_vectors = {}, {}
    for modality in ("query", "page", "cross"):
        audit.equal(manifest["token_receipts"][modality], original["token_receipts"][modality], "correction preserves original token cache")
        entry = original["token_receipts"][modality]["features"]
        path = audit.artifact(entry, original_path.parent)
        cache = audit.load(path.with_name(f"{modality}_{original['counters'][modality]['cache_key']}.json"))
        cache_indices[modality] = {key: index for index, key in enumerate(cache["item_hashes"])}
        cache_vectors[modality] = np.load(path, mmap_mode="r", allow_pickle=False)
    extension = None
    extension_pairs = {}
    if "cross_extension" in manifest["token_receipts"]:
        extension = audit.load(features_root / f"{phase}_cross_extension.json")
        audit.equal(extension["files"], manifest["token_receipts"]["cross_extension"], "extension token receipt files")
        paths = {key: audit.artifact(entry, features_root) for key, entry in extension["files"].items()}
        arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()}
        pairs = extension["pairs"]
        audit.require(arrays["features"].shape == (len(pairs), 384) and arrays["features"].dtype == np.dtype("float32"),
                      "cross extension feature shape/dtype")
        audit.require(arrays["input_ids"].shape == arrays["attention_mask"].shape == (len(pairs), 128), "cross extension token shape")
        audit.require(np.isfinite(arrays["features"]).all() and np.isin(arrays["attention_mask"], (0, 1)).all()
                      and (arrays["attention_mask"].sum(axis=1) > 0).all(), "invalid cross extension arrays")
        identity = extension["encoder_lineage"]
        for key in ("protocol_sha256", "canonical_root", "canonical_files_sha256", "encoder_repo", "encoder_revision",
                    "encoder_weight_sha256", "tokenizer_sha256",
                    "serialization", "pooling", "token_limit", "truncation", "encoder_parameters_sha256_before"):
            audit.equal(identity[key], original["lineage"][key], "cross extension frozen identity." + key)
        # Receipt-only feature code and the unused loss helper changed; encoder
        # weights, tokenizer, pooling and parameter bytes still must be identical.
        audit.source(HERE / "ephemeral_pages_features.py", identity["feature_code_sha256"])
        audit.source(HERE / "ephemeral_pages_model.py", identity["reader_code_sha256"])
        audit.equal(identity["encoder_parameters_sha256_before"], extension["encoder_parameters_sha256_after"], "cross extension parameters frozen")
        audit.require([pair[:2] for pair in pairs] == sorted(pair[:2] for pair in pairs), "cross extension pair order")
        for index, (qh, ph, question, page) in enumerate(pairs):
            audit.equal(hashlib.sha256(question.encode()).hexdigest(), qh, "extension question text hash")
            audit.equal(hashlib.sha256(page.encode()).hexdigest(), ph, "extension page text hash")
            pair_key = hashlib.sha256(json.dumps([qh, ph], separators=(",", ":")).encode()).hexdigest()
            audit.require(pair_key not in cache_indices["cross"] and (qh, ph) not in extension_pairs,
                          "extension reencodes an existing/duplicate cross input")
            extension_pairs[qh, ph] = arrays["features"][index]
    packed = {key: np.load(audit.artifact(manifest["files"][key], features_root), mmap_mode="r", allow_pickle=False)
              for key in ("q", "pages", "cross", "page_mask")}
    required_pairs, missing, count = set(), set(), 0
    records_path = audit.artifact(manifest["files"]["records"], features_root)
    for count, record in enumerate(json_rows(records_path), 1):
        index = count - 1
        qh = hashlib.sha256(record["question"].encode()).hexdigest()
        audit.require(qh in cache_indices["query"], "repair introduced an uncaptured question")
        audit.require(np.array_equal(packed["q"][index], cache_vectors["query"][cache_indices["query"][qh]]),
                      "repacked query differs from original cached bytes")
        p = len(record["pages"])
        audit.require(packed["page_mask"][index, :p].all() and not packed["page_mask"][index, p:].any(), "repacked real-page mask")
        for page_index, page in enumerate(record["pages"]):
            ph = hashlib.sha256(page["text"].encode()).hexdigest()
            audit.require(ph in cache_indices["page"], "repair introduced an uncaptured page")
            audit.require(np.array_equal(packed["pages"][index, page_index], cache_vectors["page"][cache_indices["page"][ph]]),
                          "repacked page differs from original cached bytes")
            pair_key = hashlib.sha256(json.dumps([qh, ph], separators=(",", ":")).encode()).hexdigest()
            required_pairs.add((qh, ph))
            if pair_key in cache_indices["cross"]:
                vector = cache_vectors["cross"][cache_indices["cross"][pair_key]]
            else:
                missing.add((qh, ph))
                audit.require((qh, ph) in extension_pairs, "required cross pair has no original or extension feature")
                vector = extension_pairs[qh, ph]
            audit.require(np.array_equal(packed["cross"][index, page_index], vector), "repacked cross differs from pinned source bytes")
        audit.require(not np.count_nonzero(packed["pages"][index, p:]) and not np.count_nonzero(packed["cross"][index, p:]),
                      "repacked padding has nonzero feature bytes")
    audit.equal(count, manifest["record_count"], "correction captured record count")
    audit.require(missing == set(extension_pairs), "extension is not exactly the missing input set")
    audit.equal(repair["missing_cross_pairs"], len(missing), "independent missing cross pair count")
    cross = manifest["counters"]["cross"]
    audit.equal(cross["cache_hits"], len(required_pairs) - len(missing), "corrected cross exact-reuse hits")
    audit.equal(cross["cache_misses"], len(missing), "corrected cross cache misses")
    audit.equal(cross["encoded_examples_this_capture"], len(missing), "corrected cross encoded examples")
    audit.equal(cross["forward_calls_this_capture"], (len(missing) + 31) // 32, "corrected cross actual forward count")
    for modality in ("page", "query"):
        audit.equal(manifest["counters"][modality]["encoded_examples_this_capture"], 0, "correction existing single inputs reencoded")
        audit.equal(manifest["counters"][modality]["forward_calls_this_capture"], 0, "correction existing single inputs forwarded")
    audit.counts[phase + "_correction_missing_cross_pairs"] = len(missing)
    audit.counts[phase + "_correction_repacked_records"] = count
    return len(required_pairs)


def atomic_correction(audit: Audit, manifest: dict, features_root: Path, corpus_root: Path,
                      cfg: dict, protocol_hash: str) -> dict | None:
    """Independently re-derive child-local targets, custody and byte preservation.

    Returns the independently derived target-change ledger, or None when the
    capture is not child-local. Every counter is recomputed from the parent
    records and DecisionIR; nothing declared by the capture is trusted.
    """
    phase, lineage = manifest["phase"], manifest["lineage"]
    repair = lineage.get("atomic_correction")
    if repair is None:
        return None
    audit.require(phase != "final", "final cannot use the prefinal child-local correction")
    audit.equal(lineage.get(ATOMIC_SUPERVISION_KEY), "child-local-v1", "atomic_supervision")
    audit.equal(protocol_hash, ATOMIC_PROTOCOL_SHA256, "atomic protocol identity")
    contract = audit.artifact(repair["correction_protocol"], features_root)
    audit.require(contract == (HERE / "ephemeral_pages_atomic_protocol.json").resolve(),
                  "correction pins another correction protocol")
    audit.equal(repair["existing_inputs_reencoded"], 0, "child-local correction reencoded inputs")
    audit.require("optimizer_reruns" not in repair,
                  "child-local correction changed optimizer labels; refits are separate runs")
    original_path = audit.artifact(repair["original_capture_manifest"], features_root)
    audit.require(original_path != features_root.resolve() / f"{phase}_manifest.json",
                  "child-local correction must reference the parent capture")
    original = audit.load(original_path)
    audit.equal(original["phase"], phase, "original capture phase")
    audit.equal(original["protocol_sha256"], PARENT_PROTOCOL_SHA256, "original capture protocol")
    audit.equal(repair["original_corpus_sha256"], original["corpus_sha256"], "original corpus hash")
    audit.equal(original["corpus_sha256"], audit.hash(corpus_root / f"{phase}.jsonl"),
                "child-local corpus is not byte-identical to the borrowed parent IR")
    audit.equal(lineage["protocol_sha256"], protocol_hash, "new atomic lineage protocol")
    audit.hash(contract, protocol_hash)
    for key in ("canonical_root", "canonical_files_sha256", "encoder_repo",
                "encoder_revision", "encoder_weight_sha256", "tokenizer_sha256", "serialization",
                "pooling", "token_limit", "truncation", "encoder_parameters_sha256_before",
                "encoder_parameters_sha256_after", "native_pooler_unused", "classifier_unused"):
        audit.equal(lineage[key], original["lineage"][key], "atomic lineage preserves frozen identity." + key)
    audit.hash(HERE / "ephemeral_pages_features.py", lineage["feature_code_sha256"])
    audit.source(HERE / "ephemeral_pages_model.py", lineage["reader_code_sha256"])
    audit.equal(repair["feature_implementation_sha256"], lineage["feature_code_sha256"],
                "correction declares the applied target implementation")
    ir = {}
    for row in json_rows(corpus_root / f"{phase}.jsonl"):
        audit.require(row["id"] not in ir and row["split"] == phase, "duplicate/wrong-split DecisionIR")
        verify_teacher(audit, row, amended=False)
        ir[row["id"]] = row
    ledger = target_ledger(audit, ir, phase)
    original_records = list(json_rows(audit.artifact(original["files"]["records"], original_path.parent)))
    records_path = audit.artifact(manifest["files"]["records"], features_root)
    packed = {key: np.load(audit.artifact(manifest["files"][key], features_root), mmap_mode="r", allow_pickle=False)
              for key in ("known_target", "relevance_target", "grade_target", "directed_grade_target",
                          "grade_mask", "orientation_target", "orientation_mask")}
    for key, array in packed.items():
        prior_array = np.load(audit.artifact(original["files"][key], original_path.parent),
                              mmap_mode="r", allow_pickle=False)
        audit.equal([str(array.dtype), list(array.shape)],
                    [str(prior_array.dtype), list(prior_array.shape)], "target layout preserved." + key)
        if array.ndim == 2:
            mask = np.load(audit.artifact(manifest["files"]["page_mask"], features_root),
                           mmap_mode="r", allow_pickle=False)
            audit.require(np.array_equal(array[~mask].view(np.uint8),
                                         prior_array[~mask].view(np.uint8)), "target padding preserved." + key)
    child_records = 0
    changed = Counter()
    for index, record in enumerate(json_rows(records_path)):
        prior = original_records[index]
        audit.require(prior["row_id"] == record["row_id"] and prior["term_index"] == record["term_index"]
                      and prior["candidate_id"] == record["candidate_id"],
                      "child-local record order or identity changed")
        audit.equal({key: record[key] for key in RECORD_INPUT_KEYS},
                    {key: prior[key] for key in RECORD_INPUT_KEYS},
                    "child-local record inputs are not byte-identical")
        row = ir[record["row_id"]]
        audit.equal(record, expected_record(row, record["term_index"], record["candidate_id"], "child_local"),
                    "child-local record projection")
        width = len(record["pages"])
        for key, array in packed.items():
            boolean = key in {"grade_mask", "orientation_mask"}
            column = np.asarray(record[key], dtype=np.bool_ if boolean else array.dtype)
            observed = array[index, :width] if array.ndim == 2 else array[index:index + 1]
            audit.equal(observed.tobytes(), column.tobytes(),
                        "child-local packed target differs from reconstructed child truth: " + key)
        for key in ("relevance_target", "grade_target", "directed_grade_target", "grade_mask",
                    "orientation_target", "orientation_mask", "known_target"):
            changed[key] += prior[key] != record[key]
        child_records += 1
    audit.equal(child_records, len(original_records), "child-local record count")
    kept = child_records - changed["known_target"]
    expected_changes = {"train": 2560, "validation": 1280, "calibration": 2560, "development": 2560}
    for key in ("known_target", "grade_mask", "orientation_mask", "orientation_target", "directed_grade_target"):
        audit.equal(changed[key], expected_changes[phase], "frozen actual changed targets." + key)
    for key, expected in (("child_local_records", child_records),
                          *((f"changed_{key}_records", changed[key]) for key in
                            ("known_target", "grade_mask", "orientation_mask", "orientation_target", "directed_grade_target"))):
        audit.equal(repair[key], expected, "child-local correction." + key)
    audit.equal(repair["parent_conjunction_verified"], True, "child-local correction conjunction receipt")
    audit.equal(ledger["parent_known_equals_child_conjunction"], True, "independent parent conjunction")
    audit.equal(ledger["child_records"], child_records, "independent child record count")
    for key in ("q", "pages", "cross", "page_mask", "grade_target", "relevance_target"):
        before = np.load(audit.artifact(original["files"][key], original_path.parent),
                         mmap_mode="r", allow_pickle=False)
        after = np.load(audit.artifact(manifest["files"][key], features_root),
                        mmap_mode="r", allow_pickle=False)
        audit.equal([str(before.dtype), list(before.shape)], [str(after.dtype), list(after.shape)],
                    "child-local packed input layout changed: " + key)
        audit.require(np.array_equal(before.view(np.uint8), after.view(np.uint8)),
                      "child-local correction changed model inputs: " + key)
        audit.equal(manifest["files"][key]["sha256"], original["files"][key]["sha256"],
                    "retained file bytes changed: " + key)
        audit.counts[phase + "_byte_identical_packed_" + key] = int(before.shape[0])
    proof_entry = repair["projection_verification"]
    proof_path = Path(proof_entry["path"])
    if not proof_path.is_absolute():
        proof_path = (Path(cfg["output_root"]) / proof_path).resolve()
    audit.hash(proof_path, proof_entry["sha256"])
    audit.source(HERE / "ephemeral_pages_atomic_repack.py",
                 repair["implementation"]["sha256"])
    proof = audit.load(proof_path)
    audit.equal(proof["protocol_sha256"], ATOMIC_PROTOCOL_SHA256, "projection proof protocol")
    audit.equal(proof["feature_implementation_sha256"], lineage["feature_code_sha256"],
                "projection proof implementation")
    for key in ("final_opened", "model_training", "encoder_forwards"):
        audit.equal(proof[key], False if key != "encoder_forwards" else 0, "projection proof." + key)
    audit.equal(proof["status"], "child_local_projection_and_parent_conjunction_verified",
                "projection proof status")
    audit.equal(set(proof["phase_counts"]),
                {"train", "validation", "calibration", "development"},
                "projection proof phase population")
    audit.equal(sum(counts["child_records"] for counts in proof["phase_counts"].values()),
                264320, "projection proof frozen child population")
    recorded = proof["phase_counts"].get(phase)
    audit.require(recorded is not None, "projection proof omits this phase")
    for key in ("IR_rows", "child_records", "missing_children", "conflicting_children",
                "supported_siblings_in_unknown_parent"):
        audit.equal(recorded[key], ledger[key], "projection proof " + phase + "." + key)
    audit.counts[phase + "_child_local_records"] = child_records
    audit.counts[phase + "_changed_known_target_records"] = changed["known_target"]
    return {"phase": phase, "child_local_records": child_records,
            "changed_known_target_records": changed["known_target"],
            "changed_grade_mask_records": changed["grade_mask"],
            "changed_orientation_mask_records": changed["orientation_mask"],
            "unchanged_known_records": kept,
            "parent_known_equals_child_conjunction": ledger["parent_known_equals_child_conjunction"],
            "unsupported_sibling_in_known_parent": ledger["unsupported_sibling_in_known_parent"],
            "missing_children": ledger["missing_children"],
            "conflicting_children": ledger["conflicting_children"],
            "supported_sibling_in_unknown_parent": ledger["supported_siblings_in_unknown_parent"],
            "supervision": "child_local", "invalid_atomic_supervision": False,
            "interpretation": "eligible for interface, ceiling and promotion reading"}


def target_ledger(audit: Audit, ir: dict, phase: str) -> dict:
    """Count child truth from owned pages and check the parent conjunction."""
    totals = Counter()
    conjunction_holds, sibling_leaks = True, 0
    for row in ir.values():
        meta, ids = row["metadata"], candidates(row)
        parent = {cid: bool(meta["known"][cid]) for cid in ids}
        conjunction_holds = conjunction_holds and parent_knownness(row) == parent
        for index in range(len(meta["terms"])):
            for cid in ids:
                child, _ = child_knownness(row, index, cid)
                totals["child_records"] += 1
                if not child:
                    owned_active = [bid for bid, owner in meta["page_owners"].items()
                                    if owner == cid and meta["page_fields"].get(bid)
                                    == meta["terms"][index]["field_key"]]
                    totals["missing_children"] += not owned_active
                    totals["conflicting_children"] += len(owned_active) > 1
                if child and not parent[cid]:
                    totals["supported_siblings_in_unknown_parent"] += 1
                if not child and parent[cid]:
                    sibling_leaks += 1
    audit.require(conjunction_holds, "parent knownness is not the conjunction of required child truth")
    return {"phase": phase, "IR_rows": len(ir), "child_records": totals["child_records"],
            "missing_children": totals["missing_children"],
            "conflicting_children": totals["conflicting_children"],
            "supported_siblings_in_unknown_parent": totals["supported_siblings_in_unknown_parent"],
            "parent_known_equals_child_conjunction": conjunction_holds,
            "unsupported_sibling_in_known_parent": sibling_leaks}


def capture_manifest(audit: Audit, features_root: Path, corpus_root: Path, phase: str, cfg: dict,
                     receipt: Path | None, protocol_hash: str) -> dict:
    manifest = audit.load(features_root / f"{phase}_manifest.json")
    audit.equal(manifest["phase"], phase, "capture.phase")
    audit.equal(manifest["protocol_sha256"], protocol_hash, "capture.protocol")
    audit.hash(corpus_root / f"{phase}.jsonl", manifest["corpus_sha256"])
    for entry in manifest["files"].values():
        audit.artifact(entry, features_root)
    lineage = manifest["lineage"]
    for key, expected in (("protocol_sha256", protocol_hash), ("capture_phase", phase),
                          ("canonical_root", cfg["canonical_dependency"]["root"]),
                          ("canonical_files_sha256", cfg["canonical_dependency"]["files_sha256"]),
                          ("encoder_repo", cfg["encoder"]["repo"]), ("encoder_revision", cfg["encoder"]["revision"]),
                          ("encoder_weight_sha256", cfg["encoder"]["weight_sha256"]),
                          ("encoder_parameters_unchanged", True), ("truncation", False),
                          ("native_pooler_unused", True), ("classifier_unused", True), ("token_limit", 128)):
        audit.equal(lineage.get(key), expected, "capture.lineage." + key)
    audit.equal(lineage["encoder_parameters_sha256_before"], lineage["encoder_parameters_sha256_after"], "frozen encoder parameters")
    audit.source(HERE / "ephemeral_pages_features.py", lineage["feature_code_sha256"])
    audit.source(HERE / "ephemeral_pages_model.py", lineage["reader_code_sha256"])
    counters = manifest["counters"]
    audit.equal(counters["phase"], phase, "capture.counters.phase")
    if phase == "final":
        audit.require(receipt is not None, "final capture requires an already-checked receipt")
        audit.require(Path(manifest["selection_calibration_receipt"]).resolve() == receipt.resolve(), "final capture used another receipt")
        audit.equal(counters["selection_calibration_receipt_sha256"], audit.hash(receipt), "capture.receipt_hash")
    else:
        audit.equal(manifest["selection_calibration_receipt"], None, "prefinal capture.receipt")
    supervised = atomic_correction(audit, manifest, features_root, corpus_root, cfg, protocol_hash)
    if protocol_hash == ATOMIC_PROTOCOL_SHA256 and phase == "final":
        audit.require(supervised is None and LEGACY_SUPERVISION_KEY not in lineage,
                      "new final cannot borrow prefinal repairs")
        final_ir = {}
        for row in json_rows(corpus_root / "final.jsonl"):
            audit.require(row["id"] not in final_ir and row["split"] == "final", "final IR identity")
            verify_teacher(audit, row, amended=True)
            final_ir[row["id"]] = row
        targets = {key: np.load(audit.artifact(manifest["files"][key], features_root),
                                mmap_mode="r", allow_pickle=False)
                   for key in ("known_target", "relevance_target", "grade_target", "directed_grade_target",
                               "grade_mask", "orientation_target", "orientation_mask")}
        seen = set()
        for index, record in enumerate(json_rows(audit.artifact(manifest["files"]["records"], features_root))):
            identity = (record["row_id"], record["term_index"], record["candidate_id"])
            audit.require(identity not in seen, "duplicate final child record")
            seen.add(identity)
            row = final_ir[record["row_id"]]
            audit.equal(record, expected_record(row, record["term_index"], record["candidate_id"], "child_local"),
                        "new final independent child targets")
            for key, array in targets.items():
                expected = np.asarray(record[key], dtype=array.dtype)
                actual = array[index, :len(record["pages"])] if array.ndim == 2 else array[index:index + 1]
                audit.equal(actual.tobytes(), expected.tobytes(), "final packed child target." + key)
        audit.equal(len(seen), target_ledger(audit, final_ir, phase)["child_records"], "final child inventory")
        audit.equal(len(seen), manifest["record_count"], "final captured child population")
    cache_lineage = lineage
    cache_root = features_root
    if supervised is not None:
        parent_path = audit.artifact(lineage["atomic_correction"]["original_capture_manifest"], features_root)
        parent = audit.load(parent_path)
        cache_root = parent_path.parent
        parent_root = parent_path.parent.parent
        audit.equal(manifest["token_receipts"], parent["token_receipts"], "atomic inherits exact token receipts")
        correction_capture(audit, parent, parent_path.parent, parent_root / "corpus",
                           {**cfg, "output_root": str(parent_root)})
        cache_lineage = parent["lineage"]
        audit.equal(counters["total_encoder_forward_calls_this_capture"], 0, "atomic zero encoder calls")
        audit.equal(counters["total_encoded_examples_this_capture"], 0, "atomic zero encoded examples")
    corrected_cross_inputs = (correction_capture(audit, manifest, features_root, corpus_root, cfg)
                              if LEGACY_SUPERVISION_KEY in lineage else None)
    audit.require(not (supervised and corrected_cross_inputs),
                  "capture declares both legacy grade repair and child-local supervision")
    for modality in ("page", "query", "cross"):
        counter = counters[modality]
        audit.equal(manifest["token_receipts"][modality], counter["files"], "capture.token_receipts." + modality)
        paths = {key: audit.artifact(entry, cache_root) for key, entry in counter["files"].items()}
        cache_path = paths["features"].with_name(f"{modality}_{counter['cache_key']}.json")
        cache = audit.load(cache_path)
        audit.equal(cache["cache_key"], counter["cache_key"], "cache.key")
        audit.equal(cache["modality"], modality, "cache.modality")
        audit.equal(cache["lineage"], {key: value for key, value in cache_lineage.items()
                                     if key not in {"encoder_parameters_sha256_after", "encoder_parameters_unchanged",
                                                    "capture_phase", "selection_calibration_receipt",
                                                    LEGACY_SUPERVISION_KEY}}, "cache.lineage")
        n = counter["unique_inputs"]
        audit.equal(n, len(cache["item_hashes"]), "cache.unique_inputs")
        audit.require(len(set(cache["item_hashes"])) == n, "cache repeats an input")
        arrays = {key: np.load(target, mmap_mode="r", allow_pickle=False) for key, target in paths.items()}
        audit.require(arrays["input_ids"].shape == arrays["attention_mask"].shape == (n, 128), "token cache dimensions")
        audit.require(arrays["features"].shape == (n, 384), "feature cache dimensions")
        for key, dtype in (("input_ids", "int32"), ("attention_mask", "uint8"), ("features", "float32")):
            audit.equal(str(arrays[key].dtype), dtype, "token cache dtype." + key)
        audit.require(np.isfinite(arrays["features"]).all(), "nonfinite token-cache features")
        token_digest = hashlib.sha256()
        for start in range(0, n, 32):
            ids, mask = arrays["input_ids"][start:start + 32], arrays["attention_mask"][start:start + 32]
            audit.require(np.isin(mask, (0, 1)).all() and (mask.sum(axis=1) > 0).all(), "invalid token masks")
            token_digest.update(memoryview(np.ascontiguousarray(ids)).cast("B"))
            token_digest.update(memoryview(np.ascontiguousarray(mask)).cast("B"))
        audit.equal(token_digest.hexdigest(), cache["token_ids_and_masks_sha256"], "literal encoder input digest")
        if "token_ids_and_masks_sha256" in counter:
            audit.equal(token_digest.hexdigest(), counter["token_ids_and_masks_sha256"], "capture.token_digest")
        for key, target in paths.items():
            audit.equal(audit.hash(target), cache["files"][target.name]["sha256"], "cache.artifact_hash")
        total_inputs = corrected_cross_inputs if modality == "cross" and corrected_cross_inputs is not None else n
        audit.equal(counter["cache_hits"] + counter["cache_misses"], total_inputs, "capture.hit_miss_count")
        audit.equal(counter["original_encoded_examples"], n, "capture.original_examples")
        audit.equal(counter["original_forward_calls"], (n + 31) // 32, "capture.original_forwards")
    audit.equal(counters["total_encoder_forward_calls_this_capture"], sum(counters[k]["forward_calls_this_capture"] for k in ("page", "query", "cross")), "capture.total_forwards")
    audit.equal(counters["total_encoded_examples_this_capture"], sum(counters[k]["encoded_examples_this_capture"] for k in ("page", "query", "cross")), "capture.total_examples")
    return manifest, supervised, corrected_cross_inputs



def factor(value: Any) -> Fraction:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("typed semantic factor must be finite and numeric")
    return Fraction(str(value))


def candidates(row: dict) -> list[str]:
    return [candidate["id"] for candidate in row["candidates"] if candidate["id"] != UNKNOWN]


def expected_record(row: dict, term_index: int, cid: str, supervision: str) -> dict:
    """Project one term/candidate record straight from the parsed DecisionIR.

    ``supervision`` selects which knownness gate labels this child:
    ``"child_local"`` uses only the candidate's own graded active-property page,
    which is the ECA-2 contract; ``"parent_wide"`` additionally requires every
    sibling requirement of the parent program to hold, which is what the frozen
    ECA-1 capture contains. Parent truth is separately checked as the conjunction
    of child truth, so the two modes can be compared rather than conflated.
    """
    meta, term = row["metadata"], row["metadata"]["terms"][term_index]
    pages = []
    for block in row["state_blocks"]:
        if not block["exact"] and meta["page_owners"].get(block["id"]) == cid:
            grade = meta["page_grades"].get(block["id"])
            pages.append({"block_id": block["id"], "text": block["text"],
                          "field_key": meta["page_fields"].get(block["id"]),
                          "grade_target": None if grade is None else grade / 4})
    pages.sort(key=lambda page: page["block_id"])
    relevant = [page["field_key"] == term["field_key"] for page in pages]
    count = sum(relevant)
    child, _ = child_knownness(row, term_index, cid)
    known = child if supervision == "child_local" else bool(meta["known"][cid])
    orientation = term["orientation"]
    values = [page["grade_target"] for page in pages]
    grade_mask = [rel and known and value is not None for rel, value in zip(relevant, values)]
    orientation_targets = [orientation if rel and known else None for rel in relevant]
    directed = [(value if orientation == 1 else 1 - value) if use else None
                for value, use in zip(values, grade_mask)]
    return {"row_id": row["id"], "world_id": meta["world_id"], "split": row["split"],
            "term_index": term_index, "candidate_id": cid,
            "question": row["question"] if len(meta["terms"]) == 1 else term["question"],
            "term_weight": float(term["weight"]), "pages": pages,
            "relevance_target": [1 / count if rel else 0.0 for rel in relevant],
            "grade_target": values, "grade_mask": grade_mask, "directed_grade_target": directed,
            "orientation_target": orientation_targets,
            "orientation_mask": [value is not None for value in orientation_targets],
            "known_target": known, "teacher_score": meta["teacher_scores"].get(cid)}


def child_knownness(row: dict, term_index: int, cid: str) -> tuple[bool, bool]:
    """Derive child truth plus whether parent truth equals the child conjunction."""
    meta, term = row["metadata"], row["metadata"]["terms"][term_index]
    owned = [bid for bid, owner in meta["page_owners"].items()
             if owner == cid and not exact_block(row, bid)]
    active = [bid for bid in owned if meta["page_fields"].get(bid) == term["field_key"]]
    graded = [bid for bid in active if type(meta["page_grades"].get(bid)) is int]
    return len(active) == 1 and len(graded) == 1, bool(meta["known"][cid])


def exact_block(row: dict, block_id: str) -> bool:
    for block in row["state_blocks"]:
        if block["id"] == block_id:
            return bool(block["exact"])
    raise RuntimeError(f"authored page provenance names an absent block: {row['id']}/{block_id}")


def parent_knownness(row: dict) -> dict[str, bool]:
    """Parent truth must equal the conjunction of every required child truth."""
    meta, ids = row["metadata"], candidates(row)
    result = {}
    for cid in ids:
        conjunction = True
        for index in range(len(meta["terms"])):
            child, _ = child_knownness(row, index, cid)
            conjunction = conjunction and child
        result[cid] = conjunction
    return result


def verify_teacher(audit: Audit, row: dict, amended: bool) -> None:
    meta, scores = row["metadata"], {}
    blocks = {block["id"]: block for block in row["state_blocks"]}
    audit.require(len(blocks) == len(row["state_blocks"]), "duplicate evidence block identity")
    audit.require(set(meta["page_owners"]) <= set(blocks), "authored page provenance refers to an absent block")
    audit.require(all(not blocks[bid]["exact"] and owner in candidates(row)
                      for bid, owner in meta["page_owners"].items()), "exact/unowned material became semantic evidence")
    audit.equal(parent_knownness(row), {cid: meta["known"][cid] for cid in candidates(row)},
                "teacher.parent_known_equals_child_conjunction")
    for cid in candidates(row):
        total = Fraction(0)
        known = True
        for term in meta["terms"]:
            pages = [bid for bid, owner in meta["page_owners"].items()
                     if owner == cid and meta["page_fields"].get(bid) == term["field_key"]]
            if len(pages) != 1 or type(meta["page_grades"].get(pages[0])) is not int:
                known = False
                break
            grade = meta["page_grades"][pages[0]]
            audit.require(0 <= grade <= 4 and term["orientation"] in (-1, 1), "invalid authored grade/orientation")
            total += factor(term["weight"]) * Fraction(grade if term["orientation"] == 1 else 4 - grade, 4)
        scores[cid] = total if known else None
        audit.equal(meta["known"][cid], known, "teacher.known." + cid)
        audit.equal(meta["teacher_scores"][cid], float(total) if known else None, "teacher.score." + cid)
    gold = ([UNKNOWN] if not scores or any(value is None for value in scores.values()) else
            [cid for cid, value in scores.items() if value == max(scores.values())])
    audit.require(set(row["gold"]) == set(gold), "DecisionIR gold differs from independent authored-grade execution")
    audit.require(len(candidates(row)) == len(set(candidates(row))), "duplicate candidate identity")
    audit.require(set(candidates(row)) == set(meta["stable_ordinals"]), "stable ordinal coverage differs")
    audit.require(all(type(value) is int for value in meta["stable_ordinals"].values()), "stable ordinals are not integers")


def authored_winners(row: dict, replacement: tuple[str, int]) -> set[str]:
    meta, totals = row["metadata"], {}
    replaced_page, replaced_grade = replacement
    for cid in candidates(row):
        total = Fraction(0)
        for term in meta["terms"]:
            pages = [bid for bid, owner in meta["page_owners"].items()
                     if owner == cid and meta["page_fields"].get(bid) == term["field_key"]]
            if len(pages) != 1:
                return {UNKNOWN}
            grade = replaced_grade if pages[0] == replaced_page else meta["page_grades"].get(pages[0])
            if type(grade) is not int:
                return {UNKNOWN}
            total += factor(term["weight"]) * Fraction(grade if term["orientation"] == 1 else 4 - grade, 4)
        totals[cid] = total
    return {cid for cid, score in totals.items() if score == max(totals.values())}


def grade_diagnostics(audit: Audit, ir: dict, split: str, amended: bool) -> dict:
    rows = [row for row in ir.values() if row["metadata"]["variant"] == "relevant_page_grade_change"]
    changed = 0
    for row in rows:
        meta = row["metadata"]
        parent = ir[meta["parent_decision_id"]]
        old = parent["metadata"]
        changed += set(row["gold"]) != set(parent["gold"])
        if not amended:
            continue
        audit.require(set(row["gold"]) != set(parent["gold"]), "corrected grade variant preserves teacher maximal set")
        audit.equal(meta["terms"], old["terms"], "grade intervention changed typed criterion")
        audit.equal(meta["page_owners"], old["page_owners"], "grade intervention changed page ownership")
        audit.equal(meta["page_fields"], old["page_fields"], "grade intervention changed page field")
        changed_pages = [bid for bid in old["page_grades"] if old["page_grades"][bid] != meta["page_grades"].get(bid)]
        audit.equal(changed_pages, [meta["intervened_page_block_id"]], "grade intervention must replace one grade")
        field = meta["intervened_field_key"]
        options = []
        for cid in candidates(parent):
            pages = [bid for bid, owner in old["page_owners"].items()
                     if owner == cid and old["page_fields"].get(bid) == field]
            if len(pages) != 1:
                continue
            bid = pages[0]
            for grade in range(5):
                if grade == old["page_grades"][bid]:
                    continue
                winners = authored_winners(parent, (bid, grade))
                if winners == set(parent["gold"]):
                    continue
                payload = (json.dumps([parent["id"], field, cid, grade], ensure_ascii=False,
                                      sort_keys=True, separators=(",", ":")) + "\n").encode()
                options.append((bool(winners & set(parent["gold"])), hashlib.sha256(payload).hexdigest(), cid, bid, grade))
        audit.require(bool(options), "amended grade rule has no teacher-changing replacement")
        _, _, cid, bid, grade = min(options, key=lambda option: option[:2])
        audit.equal((meta["intervened_candidate_id"], meta["intervened_page_block_id"], meta["new_grade"]),
                    (cid, bid, grade), "independent deterministic amended grade selection")
        audit.equal(meta["old_grade"], old["page_grades"][bid], "grade intervention original grade")
    audit.counts[split + "_grade_variants"] = len(rows)
    audit.counts[split + "_grade_teacher_changing_variants"] = changed
    audit.require(bool(rows), "frozen phase omits grade intervention diagnostics")
    return {"variants": len(rows), "teacher_changing": changed, "amended_rule_checked": amended}


def masses(score: float, sigma: float) -> list[float]:
    exponents = [-((score - level / 4) / sigma) ** 2 / 2 for level in range(5)]
    maximum = max(exponents)
    values = [math.exp(value - maximum) for value in exponents]
    total = sum(values)
    return [value / total for value in values]


def sigmoid(logit: float | None) -> float:
    if logit is None:
        return 0.0  # JSON null is the persisted -infinity no-page logit.
    if not math.isfinite(logit):
        raise ValueError("nonfinite persisted knownness logit")
    return 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))


def reconstruct(row: dict, children: dict, threshold: float) -> dict:
    meta = row["metadata"]
    ids = candidates(row)
    expected = {(term, cid) for term in range(len(meta["terms"])) for cid in ids}
    if set(children) != expected:
        raise RuntimeError(f"missing/duplicate required term-candidate predictions: {row['id']}")
    scores, known, details, failures = {}, {}, {}, []
    for cid in ids:
        total, known[cid], details[cid] = Fraction(0), True, []
        for index, term in enumerate(meta["terms"]):
            score, probability = children[index, cid]
            weight = factor(term["weight"])
            child_known = probability >= threshold and score is not None and math.isfinite(score)
            details[cid].append({"term_index": index, "raw_score": score,
                                 "known_probability": probability, "known": child_known, "weight": str(weight)})
            if child_known:
                total += weight * Fraction(str(score))
            else:
                known[cid] = False
                failures.append({"candidate_id": cid, "term_index": index,
                                 "reason": "no_pages_or_nonfinite_score" if score is None or not math.isfinite(score)
                                 else "below_calibrated_knownness_threshold"})
        scores[cid] = float(total) if known[cid] else None
    if failures or not ids:
        chosen, maxima = UNKNOWN, [UNKNOWN]
    else:
        best = max(scores.values())
        maxima = [cid for cid in ids if scores[cid] == best]
        chosen = min(maxima, key=lambda cid: (meta["stable_ordinals"][cid], cid))
    remap = meta.get("identity_map", {})
    gold = row["gold"]
    return {"row_id": row["id"], "world_id": meta["world_id"], "split": row["split"],
            "variant": meta["variant"], "query_kind": meta["query_kind"], "query_id": meta["query_id"],
            "family": meta["family"], "candidate_count": len(ids), "term_count": len(meta["terms"]),
            "parent_row_id": meta.get("parent_decision_id"), "chosen": chosen, "source_chosen": remap.get(chosen, chosen),
            "maximal_predictions": maxima, "gold": gold, "source_gold": [remap.get(cid, cid) for cid in gold],
            "correct": chosen in gold, "supported": UNKNOWN not in gold,
            "exact_winner_set_correct": set(maxima) == set(gold), "candidate_scores": scores,
            "candidate_known": known, "required_children": details, "UNKNOWN_reasons": failures,
            "source": "semantic", "certificate": "unavailable", "fallback": chosen == UNKNOWN,
            "ordinal_probability": "descriptive_only; not a decision certificate"}


def primary(row: dict) -> bool:
    return row["variant"] == "base" and row["supported"] and row["candidate_count"] == 4 and row["term_count"] == 1


class WorldEvents:
    """Rebuilt event totals, never populated from a saved metric ledger."""
    def __init__(self, worlds: list[str]):
        self.worlds = worlds
        self.index = {world: index for index, world in enumerate(worlds)}
        self.totals = defaultdict(dict)

    def add(self, name: str, world: str, numerator: float, denominator: float = 1, family: str = "all"):
        if family not in self.totals[name]:
            self.totals[name][family] = np.zeros((len(self.worlds), 2), dtype=np.float64)
        self.totals[name][family][self.index[world]] += (numerator, denominator)

    def estimate(self, name: str, weights: np.ndarray | None = None):
        ratios = []
        for values in self.totals[name].values():
            sums = values.sum(axis=0) if weights is None else weights @ values
            numerator, denominator = sums[..., 0], sums[..., 1]
            ratios.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator != 0))
        return np.mean(ratios, axis=0)

    def summary(self, weights: np.ndarray) -> dict:
        result = {}
        for name, groups in sorted(self.totals.items()):
            draws = self.estimate(name, weights)
            valid = draws[np.isfinite(draws)]
            point = float(self.estimate(name))
            result[name] = {"point": point if math.isfinite(point) else None,
                            "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                            "valid_bootstrap_draws": len(valid), "families": sorted(groups),
                            "denominator": float(sum(values[:, 1].sum() for values in groups.values()))}
        for prefix in ("raw_distribution_", "calibrated_distribution_", "OOD_raw_distribution_", "OOD_calibrated_distribution_"):
            result[prefix + "ECE"] = self.ece(prefix, weights)
        return result

    def ece(self, prefix: str, weights: np.ndarray) -> dict:
        bins = [(self.totals[prefix + f"ECE_bin{b}_accuracy"]["all"], self.totals[prefix + f"ECE_bin{b}_confidence"]["all"])
                for b in range(10) if prefix + f"ECE_bin{b}_accuracy" in self.totals]
        if not bins:
            return {"point": None, "CI95": None, "denominator": 0}
        def compute(w):
            denominator, numerator = 0, 0
            for accuracy, confidence in bins:
                a = accuracy.sum(axis=0) if w is None else w @ accuracy
                c = confidence.sum(axis=0) if w is None else w @ confidence
                numerator = numerator + np.abs(a[..., 0] - c[..., 0])
                denominator = denominator + a[..., 1]
            return np.divide(numerator, denominator, out=np.full_like(np.asarray(numerator), np.nan, dtype=float), where=denominator > 0)
        samples = compute(weights)
        valid = samples[np.isfinite(samples)]
        return {"point": float(compute(None)), "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                "denominator": float(sum(a[:, 1].sum() for a, _ in bins)), "bins": 10}


def distribution_event(events: WorldEvents, prefix: str, world: str, score: float, truth: float, sigma: float):
    mass = masses(score, sigma)
    label = round(truth * 4)
    events.add(prefix + "NLL", world, -math.log(max(mass[label], np.finfo(float).tiny)))
    events.add(prefix + "Brier", world, sum((value - int(index == label)) ** 2 for index, value in enumerate(mass)))
    confidence = max(mass)
    bucket = min(9, int(confidence * 10))
    events.add(prefix + f"ECE_bin{bucket}_confidence", world, confidence)
    events.add(prefix + f"ECE_bin{bucket}_accuracy", world, mass.index(confidence) == label)


def page_events(events: WorldEvents, record: dict, prediction: dict, row: dict, cross: bool, sigma: float, families: list[str]):
    if not (row["metadata"]["variant"] == "base" and UNKNOWN not in row["gold"]
            and len(candidates(row)) == 4 and len(row["metadata"]["terms"]) == 1):
        return
    world = record["world_id"]
    mask = record["grade_mask"]
    if any(mask):
        target = record["directed_grade_target"] if cross else record["grade_target"]
        error = sum(abs(value - truth) for value, truth, use in zip(prediction["raw_value"], target, mask) if use)
        events.add("ordinal_MAE", world, error, sum(mask))
        directed = [value for value, use in zip(record["directed_grade_target"], mask) if use]
        if len(directed) != 1:
            raise RuntimeError("supported atomic directed truth must have exactly one page")
        truth = directed[0]
        events.add("candidate_directed_ordinal_MAE", world, abs(prediction["score"] - truth))
        for prefix, use_sigma in (("raw_distribution_", .1), ("calibrated_distribution_", sigma)):
            distribution_event(events, prefix, world, prediction["score"], truth, use_sigma)
            if row["metadata"]["family"][0] in families[12:]:
                distribution_event(events, "OOD_" + prefix, world, prediction["score"], truth, use_sigma)
    orientation_mask = record["orientation_mask"]
    if any(orientation_mask):
        correct = sum(int(np.sign(value)) == truth for value, truth, use in
                      zip(prediction["direction"], record["orientation_target"], orientation_mask) if use)
        events.add("orientation", world, correct, sum(orientation_mask))
    relevance, attention = record["relevance_target"], prediction["attention"]
    if attention and sum(relevance) > 0:
        best = max(attention)
        tied = [index for index, value in enumerate(attention) if abs(value - best) <= 1e-12]
        selected = min(tied, key=lambda index: record["pages"][index]["block_id"])
        events.add("page_attribution", world, relevance[selected] > 0)
        events.add("matched_page_attention_mass", world, sum(value for value, rel in zip(attention, relevance) if rel > 0))


def causal(events: WorldEvents, old: dict, new: dict, kind: str) -> dict:
    teacher_changed = set(old["source_gold"]) != set(new["source_gold"])
    student_changed = old["source_chosen"] != new["source_chosen"]
    correct_new = new["source_chosen"] in new["source_gold"]
    world = new["world_id"]
    if teacher_changed:
        events.add(kind + "_teacher_changing_correct_new", world, correct_new)
        events.add(kind + "_changed_to_new", world, student_changed and correct_new)
        if kind in {"criterion_swap", "grade_swap"} and new["supported"]:
            events.add("teacher_changing_correct_new", world, correct_new)
            events.add("teacher_changing_changed_to_new", world, student_changed and correct_new)
    else:
        events.add(kind + "_teacher_stable_unwanted_changes", world, student_changed)
        if kind in {"criterion_swap", "grade_swap"}:
            events.add("teacher_stable_unwanted_changes", world, student_changed)
    return {"kind": kind, "world_id": world, "old_row_id": old["row_id"], "new_row_id": new["row_id"],
            "old_gold": old["source_gold"], "new_gold": new["source_gold"],
            "old_chosen": old["source_chosen"], "new_chosen": new["source_chosen"],
            "teacher_changed": teacher_changed, "student_changed": student_changed,
            "correct_new": correct_new, "changed_to_new": student_changed and correct_new, "new_supported": new["supported"]}


def decision_events(audit: Audit, events: WorldEvents, rows: list[dict], ir: dict, families: list[str]) -> list[dict]:
    by_id = {row["row_id"]: row for row in rows}
    pairs, groups = [], defaultdict(list)
    for row in rows:
        world, variant = row["world_id"], row["variant"]
        if primary(row):
            audit.require(len(row["family"]) == 1, "primary criterion has multiple families")
            family = row["family"][0]
            groups[world].append(row)
            for name, value, use_family in (("atomic_choice_macro", row["exact_winner_set_correct"], family),
                                            ("atomic_choice_micro", row["exact_winner_set_correct"], "all"),
                                            ("concrete_winner_choice_macro", row["correct"], family)):
                events.add(name, world, value, family=use_family)
            if family in families[-4:]:
                events.add("held4_family_choice_macro", world, row["exact_winner_set_correct"], family=family)
            if family in families[12:16]:
                events.add("development4_family_choice_macro", world, row["exact_winner_set_correct"], family=family)
        if variant == "base" and row["supported"] and row["term_count"] > 1:
            events.add("composition", world, row["correct"])
        if variant in MISSING:
            unknown, gold_unknown = row["chosen"] == UNKNOWN, not row["supported"]
            events.add("UNKNOWN_precision", world, unknown and gold_unknown, unknown)
            events.add("UNKNOWN_recall", world, unknown and gold_unknown, gold_unknown)
            events.add("supported_coverage", world, row["supported"] and not unknown, row["supported"])
        if variant == "base" or variant.startswith("probe_k"):
            events.add(f"K{row['candidate_count']}_choice", world, row["correct"])
        if variant in ROBUST:
            parent = by_id[row["parent_row_id"]]
            audit.equal(parent["world_id"], world, "robust world identity")
            mapping = ir[row["row_id"]]["metadata"].get("identity_map", {})
            mapped = {mapping.get(cid, cid): score for cid, score in row["candidate_scores"].items()}
            source = parent["candidate_scores"]
            audit.require(set(mapped) == set(source), "robust source candidate coverage")
            invariant = row["source_chosen"] == parent["source_chosen"]
            events.add("permutation_rename_reorder", world, invariant)
            events.add(variant + "_invariance", world, invariant)
            comparable = [abs(mapped[cid] - source[cid]) for cid in mapped if mapped[cid] is not None and source[cid] is not None]
            pairs.append({"kind": variant, "world_id": world, "parent_row_id": parent["row_id"],
                          "new_row_id": row["row_id"], "mapping_invariant": invariant,
                          "mapped_candidate_scores": mapped, "source_candidate_scores": source,
                          "candidate_knownness_invariant": all((mapped[cid] is None) == (source[cid] is None) for cid in mapped),
                          "maximum_score_deviation": max(comparable, default=0.0)})
        if variant in {"relevant_page_grade_change", "relevant_page_erasure", "relevant_page_contradiction"}:
            parent = by_id[row["parent_row_id"]]
            audit.equal(parent["world_id"], world, "causal world identity")
            pairs.append(causal(events, parent, row, "grade_swap" if variant == "relevant_page_grade_change" else variant))
    for group in groups.values():
        for old, new in itertools.permutations(group, 2):
            pairs.append(causal(events, old, new, "criterion_swap"))
    return pairs


def reconstruct_calibration(audit: Audit, path: Path, ir: dict, supervision: str) -> dict:
    probabilities, labels, ordinal_scores, truths = defaultdict(list), {}, [], []
    for raw in json_rows(path):
        row, prediction = ir[raw["row_id"]], raw["prediction"]
        expected = expected_record(row, raw["term_index"], raw["candidate_id"], supervision)
        audit.equal({key: raw[key] for key in expected}, expected, "calibration.raw_record")
        probability = sigmoid(prediction["known_logits"])
        audit.equal(probability, prediction["known_probability"], "calibration.known_probability")
        key = raw["row_id"], raw["candidate_id"]
        probabilities[key].append((raw["term_index"], probability))
        # Candidate-level calibration stays whole-program truth: parent knownness
        # is the conjunction of this candidate's required child truth.
        labels[key] = parent_knownness(row)[raw["candidate_id"]]
        target = [value for value, use in zip(raw["directed_grade_target"], raw["grade_mask"]) if use]
        if len(target) == 1 and prediction["score"] is not None:
            ordinal_scores.append(prediction["score"])
            truths.append(round(target[0] * 4))
    keys = sorted(probabilities)
    audit.require(set(keys) == {(row_id, cid) for row_id, row in ir.items() for cid in candidates(row)},
                  "calibration candidate population omits authored required candidates")
    for row_id, cid in keys:
        audit.require(sorted(index for index, _ in probabilities[row_id, cid]) == list(range(len(ir[row_id]["metadata"]["terms"]))), "calibration required-child coverage")
    p = np.array([min(value for _, value in probabilities[key]) for key in keys])
    y = np.array([labels[key] for key in keys], dtype=bool)
    audit.require(y.any() and (~y).any(), "calibration needs known and unknown candidates")
    grid = [{"threshold": index / 20, "balanced_candidate_accuracy": float(((p[y] >= index / 20).mean() + (p[~y] < index / 20).mean()) / 2)} for index in range(1, 20)]
    best = max(grid, key=lambda entry: (entry["balanced_candidate_accuracy"], -entry["threshold"]))
    sigma_grid = []
    for sigma in SIGMAS:
        nll = np.mean([-math.log(max(masses(score, sigma)[label], np.finfo(float).tiny)) for score, label in zip(ordinal_scores, truths)])
        sigma_grid.append({"sigma": sigma, "NLL": float(nll)})
    winner = min(sigma_grid, key=lambda entry: entry["NLL"])
    audit.counts["calibration_candidates"] += len(keys)
    return {"knownness_threshold": best["threshold"], "knownness_grid": grid,
            "candidate_count": len(keys), "known_candidates": int(y.sum()),
            "candidate_rule": "min required-child knownness; every captured calibration decision/candidate",
            "ordinal_sigma": winner["sigma"], "ordinal_sigma_grid": sigma_grid, "ordinal_candidate_count": len(truths),
            "fit_split": "calibration", "certificate": "unavailable", "final_outcomes_used": False}


def rebuild_control(audit: Audit, name: str, entries: dict, ir: dict, records_path: Path, calibration: dict,
                    worlds: list[str], weights: np.ndarray, cfg: dict, base: Path,
                    supervision: str) -> tuple[WorldEvents, list[dict]]:
    paths = {key: audit.artifact(entry, base) for key, entry in entries.items()}
    events = WorldEvents(worlds)
    groups = defaultdict(dict)
    count = 0
    raw_fields = {"record_index", "prediction", "ordinal_mass_uncalibrated_sigma_0_1", "ordinal_mass_calibrated", "variant", "certificate"}
    for count, (raw, captured) in enumerate(itertools.zip_longest(json_rows(paths["term_candidate"]), json_rows(records_path)), 1):
        audit.require(raw is not None and captured is not None, name + ": raw/capture length mismatch")
        audit.equal(raw["record_index"], count - 1, name + ".record_index")
        audit.equal({key: value for key, value in raw.items() if key not in raw_fields}, captured, name + ".capture_record")
        row = ir[raw["row_id"]]
        audit.equal(captured, expected_record(row, raw["term_index"], raw["candidate_id"], supervision),
                    name + ".independent_IR_projection")
        audit.equal(raw["variant"], row["metadata"]["variant"], name + ".variant")
        audit.equal(raw["certificate"], "unavailable", name + ".certificate")
        prediction = raw["prediction"]
        score, probability = prediction["score"], sigmoid(prediction["known_logits"])
        audit.equal(probability, prediction["known_probability"], name + ".known_probability")
        p = len(raw["pages"])
        for key in ("attention", "direction", "raw_value", "relevance_logits"):
            audit.require(len(prediction[key]) == p, name + ": page prediction shape")
        audit.require(all(value is not None and math.isfinite(value) for key in ("attention", "direction", "raw_value") for value in prediction[key]), name + ": invalid page predictions")
        audit.require(score is None or math.isfinite(score), name + ": invalid scalar score")
        for key, sigma in (("ordinal_mass_uncalibrated_sigma_0_1", .1), ("ordinal_mass_calibrated", calibration["ordinal_sigma"])):
            audit.equal(masses(score, sigma) if score is not None else None, raw[key], name + "." + key)
        key = raw["term_index"], raw["candidate_id"]
        audit.require(key not in groups[raw["row_id"]], name + ": duplicate term candidate")
        groups[raw["row_id"]][key] = score, probability
        page_events(events, raw, prediction, row, name == "cross", calibration["ordinal_sigma"], cfg["corpus"]["families"])
    audit.require(set(groups) == set(ir), name + ": raw prediction DecisionIR coverage")
    audit.counts[name + "_term_candidate_rows"] = count
    decisions, seen = [], set()
    for saved in json_rows(paths["decisions"]):
        row_id = saved["row_id"]
        audit.require(row_id not in seen and row_id in ir, name + ": duplicate/unknown decision")
        seen.add(row_id)
        rebuilt = reconstruct(ir[row_id], groups.pop(row_id), calibration["knownness_threshold"])
        audit.equal(rebuilt, saved, name + ".decision." + row_id, tolerance=1e-12)
        del rebuilt["required_children"], rebuilt["UNKNOWN_reasons"]
        decisions.append(rebuilt)
    audit.require(seen == set(ir), name + ": persisted decisions omit IR rows")
    audit.counts[name + "_decisions"] = len(decisions)
    pairs = decision_events(audit, events, decisions, ir, cfg["corpus"]["families"])
    for index, (rebuilt, saved) in enumerate(itertools.zip_longest(pairs, json_rows(paths["paired_interventions"])), 1):
        audit.require(rebuilt is not None and saved is not None, name + ": causal pair count differs")
        audit.equal(rebuilt, saved, f"{name}.pair[{index}]")
    audit.counts[name + "_pairs"] = len(pairs)
    ledger = audit.load(paths["world_metric_ledger"])
    audit.equal(ledger["world_ids"], worlds, name + ".ledger.worlds")
    rebuilt_values = {key: {family: value.tolist() for family, value in groups.items()} for key, groups in events.totals.items()}
    audit.equal(rebuilt_values, ledger["metrics"], name + ".world_metric_events")
    return events, [row for row in decisions if primary(row)]


def paired_comparisons(events: dict, primary_rows: dict, weights: np.ndarray) -> dict:
    samples = {name: ledger.estimate("atomic_choice_macro", weights) for name, ledger in events.items()}
    points = {name: float(ledger.estimate("atomic_choice_macro")) for name, ledger in events.items()}
    result = {}
    for name in (*CONTROLS[1:], *INTERVENTIONS):
        difference = samples["pages"] - samples[name]
        valid = difference[np.isfinite(difference)]
        result["pages_minus_" + name] = {"point": points["pages"] - points[name],
                                        "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                                        "one_sided95_lower": float(np.quantile(valid, .05)) if len(valid) else None,
                                        "valid_bootstrap_draws": len(valid)}
    cross = {row["row_id"]: row for row in primary_rows["cross"]}
    worlds, discordant, harmed = set(), set(), set()
    for row in primary_rows["pages"]:
        other = cross.pop(row["row_id"])
        world = row["world_id"]
        worlds.add(world)
        delta = int(row["exact_winner_set_correct"]) - int(other["exact_winner_set_correct"])
        if delta:
            discordant.add(world)
        if delta < 0:
            harmed.add(world)
    if cross or not worlds:
        raise RuntimeError("paired NI primary population differs")
    n, h = len(worlds), len(harmed)
    upper = 1.0 if h == n else float(beta.ppf(.95, h + 1, n - h))
    lower = result["pages_minus_cross"]["one_sided95_lower"]
    sparse = len(discordant) < 20
    passed = lower is not None and lower > -.05 and (not sparse or upper < .05)
    result["internal_cross_NI"] = {"margin": .05, "one_sided95_lower": lower, "discordant_worlds": len(discordant),
                                   "harmed_worlds": h, "world_count": n, "CP_harmed_world_upper95": upper,
                                   "sparse_guard_required": sparse, "pass": passed,
                                   "status": "passes_internal_screen" if passed else "inconclusive_or_fails", "endgame_NI": False}
    benefits = {}
    for name in ("lexical", "zero_question", "zero_pages"):
        difference = samples["pages"] - samples[name]
        valid = difference[np.isfinite(difference)]
        lower = float(np.quantile(valid, .05 / 3)) if len(valid) else None
        benefits["pages_minus_" + name] = {"one_sided_Bonferroni_lower": lower, "alpha": .05 / 3,
                                          "pass": lower is not None and lower > 0}
    result["learned_benefits"] = {"differences": benefits, "pass": all(item["pass"] for item in benefits.values()),
                                  "familywise_alpha": .05, "claims": 3}
    return result


def exact_decision(row: dict, terms: list[dict]) -> dict:
    """Independent integer lane; no scalar conversion through binary floating point."""
    if not terms:
        raise ValueError("exact expression has no children")
    for term in terms:
        if not isinstance(term["fact_key"], str) or not term["fact_key"] or type(term["weight"]) is not int or type(term["sign"]) is not int or term["sign"] not in (-1, 1):
            raise ValueError("invalid typed exact child")
    ids, scores = candidates(row), {}
    for cid in ids:
        facts = row["metadata"].get("exact_facts", {}).get(cid, {})
        values = [facts.get(term["fact_key"]) for term in terms]
        scores[cid] = (sum(term["weight"] * term["sign"] * value for term, value in zip(terms, values))
                       if all(type(value) is int for value in values) else None)
    if not ids or any(value is None for value in scores.values()):
        chosen, maxima = UNKNOWN, [UNKNOWN]
    else:
        best = max(scores.values())
        maxima = [cid for cid in ids if scores[cid] == best]
        chosen = min(maxima, key=lambda cid: (row["metadata"]["stable_ordinals"][cid], cid))
    return {"chosen": chosen, "maximal_predictions": maxima, "candidate_scores": scores,
            "source": "exact", "probability": "not_applicable", "certificate": "not_applicable", "fallback": chosen == UNKNOWN}


def exact_replay(audit: Audit, ir: dict) -> dict:
    expressions = [([{ "fact_key": key, "weight": weight, "sign": sign}])
                   for key in ("setup_charge_cents", "handoff_minutes", "replacement_parts")
                   for weight in (1, 2) for sign in (-1, 1)]
    expressions += [[{"fact_key": "setup_charge_cents", "weight": 2, "sign": -1},
                     {"fact_key": "handoff_minutes", "weight": 1, "sign": 1}],
                    [{"fact_key": "replacement_parts", "weight": 2, "sign": 1},
                     {"fact_key": "handoff_minutes", "weight": 2, "sign": -1}]]
    cases, invariant_cases = 0, 0
    fingerprint = hashlib.sha256()
    parent_results = {}
    for row in ir.values():
        for index, terms in enumerate(expressions):
            result = exact_decision(row, terms)
            audit.require(not result["fallback"], "authored typed exact facts are missing/noninteger")
            audit.require(all(type(value) is int for value in result["candidate_scores"].values()), "exact lane rounded integers")
            fingerprint.update(json.dumps({"row_id": row["id"], "expression": index, **result}, sort_keys=True).encode())
            parent_results[row["id"], index] = result
            cases += 1
    for row in ir.values():
        if row["metadata"]["variant"] not in ROBUST:
            continue
        remap = row["metadata"].get("identity_map", {})
        for index in range(len(expressions)):
            result = parent_results[row["id"], index]
            parent = parent_results[row["metadata"]["parent_decision_id"], index]
            audit.equal(remap.get(result["chosen"], result["chosen"]), parent["chosen"], "exact stable identity invariance")
            audit.equal({remap.get(cid, cid): value for cid, value in result["candidate_scores"].items()}, parent["candidate_scores"], "exact score invariance")
            invariant_cases += 1
    fixture = copy.deepcopy(next(iter(ir.values())))
    ids = candidates(fixture)
    for ordinal, cid in enumerate(ids):
        fixture["metadata"]["exact_facts"][cid]["integer_probe"] = 2 ** 60 + ordinal
    term = [{"fact_key": "integer_probe", "weight": 2, "sign": 1}]
    probe = exact_decision(fixture, term)
    audit.equal(probe["chosen"], ids[-1], "large integer ordering")
    for cid in ids:
        fixture["metadata"]["exact_facts"][cid]["integer_probe"] = 2 ** 60
    tie = exact_decision(fixture, term)
    audit.equal(tie["chosen"], min(ids, key=lambda cid: (fixture["metadata"]["stable_ordinals"][cid], cid)), "integer tie stable ordinal")
    del fixture["metadata"]["exact_facts"][ids[0]]["integer_probe"]
    audit.equal(exact_decision(fixture, term)["chosen"], UNKNOWN, "unknown exact child propagates to whole choice")
    fixture["metadata"]["exact_facts"][ids[0]]["integer_probe"] = True
    audit.equal(exact_decision(fixture, term)["chosen"], UNKNOWN, "boolean is not integer exact fact")
    return {"typed_integer_cases": cases, "identity_invariant_cases": invariant_cases,
            "reconstruction_sha256": fingerprint.hexdigest(), "large_integer_and_missing_child_checks": 4,
            "evaluator_exact_literals_gate": "not established; parent-owned closed/runtime prerequisite"}


def gate_screens(summary: dict, cfg: dict) -> dict:
    rules = {"atomic_choice_macro": "atomic_choice_macro", "held4_family_choice_macro": "held4_family_choice_macro",
             "ordinal_MAE_max": "ordinal_MAE", "orientation": "orientation", "page_attribution": "page_attribution",
             "UNKNOWN_recall": "UNKNOWN_recall", "UNKNOWN_precision": "UNKNOWN_precision", "supported_coverage": "supported_coverage",
             "composition": "composition", "teacher_changing_correct_new": "teacher_changing_correct_new",
             "permutation_rename_reorder": "permutation_rename_reorder",
             "criterion_swap_teacher_changing_correct_new": "criterion_swap_teacher_changing_correct_new",
             "grade_swap_teacher_changing_correct_new": "grade_swap_teacher_changing_correct_new"}
    result = {}
    for gate, metric in rules.items():
        threshold = cfg["gates"]["teacher_changing_correct_new" if gate.endswith("_teacher_changing_correct_new") else gate]
        point = summary.get(metric, {}).get("point")
        available = point is not None and math.isfinite(point)
        passed = (point <= threshold if gate == "ordinal_MAE_max" else point >= threshold) if available else None
        result[gate] = {"threshold": threshold, "observed": point if available else None, "pass": passed,
                        "status": "evaluated_point_screen" if available else "unavailable_in_this_phase"}
    result["exact_literals"] = {"threshold": 1.0, "pass": None, "status": "closed_regression_prerequisite_not_supplied; parent_owned"}
    return result


def verify_run(audit: Audit, phase: str, run_root: Path, receipt_path: Path | None,
               features_root: Path | None = None, corpus_root: Path | None = None,
               allow_incomplete_grade_diagnostic: bool = False,
               experiment: capture.Experiment = capture.DEFAULT_EXPERIMENT) -> dict:
    cfg, protocol_hash = experiment.protocol()
    audit.hash(experiment.protocol_path, protocol_hash)
    atomic = protocol_hash == ATOMIC_PROTOCOL_SHA256
    # ECA-1 artefacts reproduce their own parent-wide masks exactly; ECA-2
    # artefacts must match the independent child-local projection instead.
    supervision = "child_local" if atomic else "parent_wide"
    audit.equal(cfg["immutable_reference"], "e6b046ffbd138cbdbfb2f89c6ae77525fe6b0b18", "frozen product reference")
    product = subprocess.run(["git", "diff", "--quiet", cfg["immutable_reference"], "--", "src", "tests", "examples"],
                             cwd=HERE.parents[1], capture_output=True, check=False)
    audit.require(product.returncode == 0, "frozen product source/tests/examples differ from immutable reference")
    root = Path(cfg["output_root"])
    audit.equal(root.resolve(), Path(experiment.corpus_root).parent.resolve(), "experiment root/protocol mismatch")
    audit.require(run_root.resolve().is_relative_to(root.resolve()), "run root must remain outside Git in frozen data root")
    # This is the only route to final artifacts, and the gate is deliberately first.
    if phase == "final" and receipt_path is None:
        raise RuntimeError("final requires --receipt before any final IR/features")
    if phase == "development" and receipt_path is not None:
        raise RuntimeError("development does not consume an explicit final receipt")
    receipt_path = receipt_path or run_root / "selection_calibration_receipt.json"
    if atomic and phase == "final":
        from ephemeral_pages_features import validate_final_receipt
        validate_final_receipt(receipt_path, experiment.protocol_path)
    receipt, calibrations = checked_receipt(audit, receipt_path, run_root, protocol_hash)
    corpus_root = corpus_root or experiment.corpus_root
    audit.require(corpus_root.resolve().is_relative_to(root.resolve()), "corpus root is outside frozen data root")
    if phase == "final":
        audit.require(not allow_incomplete_grade_diagnostic, "incomplete diagnostics cannot authorize final")
        audit.require(corpus_root.resolve() == experiment.corpus_root.resolve(), "final corpus must be current")
    if phase == "final" and not atomic:
        audit.require(not allow_incomplete_grade_diagnostic, "incomplete grade diagnostics cannot authorize final")
        audit.require(corpus_root.resolve() == (root / "corpus").resolve(), "archival corpus cannot authorize current final")
        for key in ("corpus_build_manifest", "amendment_file"):
            audit.require(key in receipt, "final receipt omits corrected corpus prerequisite: " + key)
        build_path = audit.artifact(receipt["corpus_build_manifest"], receipt_path.parent)
        amendment_path = audit.artifact(receipt["amendment_file"], receipt_path.parent)
        audit.require(build_path == (corpus_root / "build_manifest_v1.json").resolve(), "receipt pins another corpus")
        audit.require(amendment_path == (HERE / "ephemeral_pages_grade_amendment.json").resolve(), "receipt pins another amendment")
        audit.equal(audit.load(build_path)["grade_intervention_amendment"], receipt["amendment_file"], "sealed corpus amendment")
        amendment = audit.load(amendment_path)
        audit.require(amendment.get("parent_protocol_sha256") == PARENT_PROTOCOL_SHA256,
                      "grade amendment does not identify the frozen parent protocol")
    evaluation_path = run_root / "evaluation" / phase / "evaluation.json"
    evaluation = audit.load(evaluation_path)
    if features_root is None:
        features_root = experiment.cache_root
    audit.require(features_root.resolve().is_relative_to(root.resolve()), "features root is outside frozen data root")
    for relative, expected in cfg["canonical_dependency"]["files_sha256"].items():
        audit.hash(Path(cfg["canonical_dependency"]["root"]) / relative, expected)
    source_root = root
    if atomic:
        source_root, build = prefinal_provenance(audit, experiment)
    else:
        build = audit.load(corpus_root / "build_manifest_v1.json")
    audit.equal(build["protocol_sha256"], PARENT_PROTOCOL_SHA256, "builder.protocol")
    audit.source(HERE / "ephemeral_pages_build.py", build["builder_sha256"])
    amended = "grade_intervention_amendment" in build
    if amended:
        amendment_path = audit.artifact(build["grade_intervention_amendment"], source_root / "corpus")
        amendment = audit.load(amendment_path)
        audit.equal(amendment.get("parent_protocol_sha256"), PROTOCOL_SHA256, "grade amendment parent protocol")
    for relative, key in (("source/eca_authored_source_v1.json", "source_sha256"),
                          ("source/eca_source_inventory_v1.json", "source_inventory_sha256"),
                          ("audit/opaque_review_packet_v1.json", "opaque_packet_sha256"),
                          ("audit/sealed_target_key_v1.json", "sealed_key_sha256")):
        audit.hash(source_root / relative, build[key])
    audit.hash(source_root / "audit" / "accepted_receipt_v1.json", build["receipt"]["receipt_sha256"])
    audit.hash(source_root / "audit" / "raw_reviews_merged_v1.jsonl", build["receipt"]["raw_reviews_sha256"])
    if atomic and phase == "final":
        final_build_path = audit.artifact(receipt["corpus_build_manifest"], receipt_path.parent)
        audit.require(final_build_path == (corpus_root / "eca2_final_build_manifest.json").resolve(),
                      "atomic final pins another builder manifest")
        final_build = audit.load(final_build_path)
        audit.equal(final_build["protocol_sha256"], protocol_hash, "atomic final build protocol")
        audit.source(HERE / "ephemeral_pages_atomic_final.py", final_build["builder"]["sha256"])
        for relative, key in (("source/eca2_final_source.json", "source_sha256"),
                              ("source/eca2_final_inventory.json", "source_inventory_sha256"),
                              ("audit/eca2_opaque_review_packet.json", "opaque_packet_sha256"),
                              ("audit/eca2_sealed_target_key.json", "sealed_key_sha256")):
            audit.hash(root / relative, final_build[key])
        audit.artifact(final_build["prepare_manifest"], root)
        audit.hash(root / "audit/eca2_independent_audit_receipt.json", final_build["receipt"]["receipt_sha256"])
        audit.hash(root / final_build["receipt"]["raw_reviews_file"], final_build["receipt"]["raw_reviews_sha256"])
        for relative, entry in final_build["outputs"].items():
            audit.hash(root / relative, entry["sha256"])
        build = final_build
    captures = {split: capture_manifest(audit, features_root, corpus_root, split, cfg,
                                         receipt_path if split == "final" else None, protocol_hash)
                for split in dict.fromkeys(("train", "validation", "calibration", phase))}
    manifests = {split: entry[0] for split, entry in captures.items()}
    supervised = {split: captures[split][1] for split in captures if captures[split][1] is not None}
    audit.require(bool(supervised) == atomic,
                  "child-local supervision lineage and atomic protocol disagree")
    for name in CONTROLS:
        history = audit.load(run_root / f"{name}_history.json")
        audit.equal(history["protocol_sha256"], protocol_hash, name + ".history.protocol")
        for key, expected in (("control", name), ("seed", 7), ("smoke", False), ("epochs", 400), ("encoder_forwards", 0)):
            audit.equal(history.get(key), expected, name + ".history." + key)
        audit.equal(history["checkpoint_sha256"], receipt["checkpoint_files"][name]["sha256"], name + ".frozen_checkpoint")
        for filename, expected in history["source_sha256"].items():
            audit.source(HERE / filename, expected)
        for split in ("train", "validation"):
            training_lineage = manifests[split]["lineage"] if atomic else {
                key: value for key, value in manifests[split]["lineage"].items()
                if key not in {LEGACY_SUPERVISION_KEY, ATOMIC_SUPERVISION_KEY}}
            audit.equal(history[split + "_feature_lineage"], training_lineage, name + ".training_lineage")
            records_path = audit.artifact(manifests[split]["files"]["records"], features_root)
            selected = set(history["selected_indices"][split])
            selected_rows = set()
            selected_indices = []
            split_ir = {row["id"]: row for row in json_rows(corpus_root / f"{split}.jsonl")}
            if name == "pages":
                grade_diagnostics(audit, split_ir, split, amended)
            for index, record in enumerate(json_rows(records_path)):
                row = split_ir[record["row_id"]]
                eligible = (len(candidates(row)) == 4 and row["metadata"]["variant"] in MISSING)
                if eligible:
                    selected_indices.append(index)
                    selected_rows.add(row["id"])
                if index in selected:
                    audit.require(record["split"] == split and eligible, "optimizer/validation selected a held or diagnostic row")
            audit.equal(history["selected_indices"][split], selected_indices, name + ".frozen_optimizer_population")
            audit.equal(history["selected_row_ids"][split], sorted(selected_rows), name + ".selected_row_ids")
            if atomic:
                if name == "pages":
                    audit.equal(supervised[split]["changed_known_target_records"] > 0, True,
                                "child-local correction changed no optimizer labels on " + split)
            elif features_root.resolve() != (root / "features").resolve() and name == "pages":
                original_root = root / "features"
                original = audit.load(original_root / f"{split}_manifest.json")
                for key in ("q", "pages", "cross", "page_mask", "relevance_target", "grade_target",
                            "directed_grade_target", "grade_mask", "orientation_target", "orientation_mask", "known_target"):
                    before = np.load(audit.artifact(original["files"][key], original_root), mmap_mode="r", allow_pickle=False)
                    after = np.load(audit.artifact(manifests[split]["files"][key], features_root), mmap_mode="r", allow_pickle=False)
                    audit.require(before.dtype == after.dtype and before.shape == after.shape,
                                  "corrected packed feature/target layout changed: " + split + "." + key)
                    for start in range(0, len(selected_indices), 32):
                        chunk = selected_indices[start:start + 32]
                        audit.require(np.array_equal(before[chunk].view(np.uint8), after[chunk].view(np.uint8)),
                                      "grade correction changed fitted optimizer/validation bytes: " + split + "." + key)
                    audit.counts["unchanged_selected_feature_target_rows"] += len(selected_indices)
        if name == "lexical":
            audit.equal(history["regularization"], 1.0, "lexical.regularization")
        else:
            trajectory = history["history"]
            audit.equal([item["epoch"] for item in trajectory], list(range(1, 401)), name + ".epoch_sequence")
            winner = min(trajectory, key=lambda item: item["validation"]["total"])
            audit.equal(history["selected_epoch"], winner["epoch"], name + ".earliest_validation_minimum")
            audit.equal(history["selected_validation_objective"], winner["validation"]["total"], name + ".validation_minimum")
    calibration_ir = {row["id"]: row for row in json_rows(corpus_root / "calibration.jsonl")}
    grade_diagnostics(audit, calibration_ir, "calibration", amended)
    calibration_eval = audit.load(audit.artifact(audit.load(run_root / "calibration.json")["calibration_evaluation"], run_root))
    audit.equal(calibration_eval["phase"], "calibration", "calibration_evaluation.phase")
    audit.equal(calibration_eval["checkpoint_files"], receipt["checkpoint_files"], "calibration_evaluation.checkpoints")
    for name in CONTROLS:
        raw_path = audit.artifact(calibration_eval["artifacts"][name]["term_candidate"], run_root)
        rebuilt = reconstruct_calibration(audit, raw_path, calibration_ir, supervision)
        audit.equal(rebuilt, calibrations[name], name + ".independent_calibration")
    del calibration_ir
    audit.equal(evaluation["phase"], phase, "evaluation.phase")
    audit.equal(evaluation["protocol_sha256"], protocol_hash, "evaluation.protocol")
    audit.equal(evaluation["checkpoint_files"], receipt["checkpoint_files"], "evaluation.frozen_checkpoints")
    audit.source(HERE / "ephemeral_pages_evaluate.py", evaluation["implementation_sha256"])
    for key, expected in (("evaluation_encoder_forwards", 0), ("final_outcomes_used_for_selection", False),
                          ("certificate", "unavailable"), ("promotion", False), ("B_STEF_allowed", False), ("endgame_complete", False)):
        audit.equal(evaluation.get(key), expected, "evaluation." + key)
    for key, manifest_key in (("capture_lineage", "lineage"), ("capture_counters", "counters"), ("capture_token_receipts", "token_receipts")):
        audit.equal(evaluation[key], manifests[phase][manifest_key], "evaluation." + key)
    audit.hash(corpus_root / f"{phase}.jsonl", build["split_jsonl_sha256"][phase])
    ir = {}
    for row in json_rows(corpus_root / f"{phase}.jsonl"):
        audit.require(row["id"] not in ir and row["split"] == phase, "duplicate/wrong-split DecisionIR")
        verify_teacher(audit, row, amended)
        ir[row["id"]] = row
    audit.equal(len(ir), build["row_counts_by_split"][phase], "corpus.phase_row_count")
    if atomic and phase == "final":
        supervised["final"] = {**target_ledger(audit, ir, phase),
                               "supervision": "child_local", "capture": "genuine_new_final",
                               "invalid_atomic_supervision": False}
    grade_diagnostics(audit, ir, phase, amended)
    worlds = sorted({row["metadata"]["world_id"] for row in ir.values()})
    audit.equal(len(worlds), cfg["corpus"]["worlds"][phase], "frozen world count")
    expected_families = cfg["corpus"]["families"][:16 if phase == "development" else 20]
    observed_families = {family for row in ir.values() if row["metadata"]["variant"] == "base"
                         and len(row["metadata"]["terms"]) == 1 and UNKNOWN not in row["gold"] for family in row["metadata"]["family"]}
    audit.require(observed_families == set(expected_families), "fixed family population changed")
    exposures = {(row["metadata"]["world_id"], row["metadata"]["family"][0]) for row in ir.values()
                 if row["metadata"]["variant"] == "base" and len(row["metadata"]["terms"]) == 1 and UNKNOWN not in row["gold"]}
    audit.equal(dict(Counter(family for _, family in exposures)), build["family_world_counts_by_split"][phase],
                "independent family world exposure counts")
    bootstrap = evaluation["bootstrap"]
    audit.equal(bootstrap["world_ids"], worlds, "bootstrap.world_ids")
    audit.equal(bootstrap["draws"], 10000, "bootstrap.draws")
    audit.equal(bootstrap["seed"], 0, "bootstrap.seed")
    indices = np.load(audit.artifact(bootstrap, evaluation_path.parent), allow_pickle=False)
    generated = np.random.default_rng(0).integers(0, len(worlds), size=(10000, len(worlds)), dtype=np.int32)
    audit.require(indices.dtype == np.dtype("int32") and np.array_equal(indices, generated), "saved world bootstrap differs from frozen seed/draw order")
    weights = np.zeros((10000, len(worlds)), dtype=float)
    for index, draw in enumerate(indices):
        weights[index] = np.bincount(draw, minlength=len(worlds))
    expected_names = set(CONTROLS + INTERVENTIONS)
    audit.require(set(evaluation["artifacts"]) == set(evaluation["metrics"]) == expected_names, "evaluation omits fixed controls/interventions")
    ledgers, primary_rows, summaries, screens = {}, {}, {}, {}
    records_path = audit.artifact(manifests[phase]["files"]["records"], features_root)
    for name in (*CONTROLS, *INTERVENTIONS):
        settings = calibrations["pages" if name in INTERVENTIONS else name]
        ledger, rows = rebuild_control(audit, name, evaluation["artifacts"][name], ir, records_path,
                                       settings, worlds, weights, cfg, evaluation_path.parent, supervision)
        audit.equal(audit.counts[name + "_term_candidate_rows"], manifests[phase]["record_count"], name + ".capture_record_count")
        audit.require(set(ledger.totals["atomic_choice_macro"]) == set(expected_families), name + ": primary family omission")
        summary = ledger.summary(weights)
        audit.equal(summary, evaluation["metrics"][name], name + ".independent_summaries")
        ledgers[name], primary_rows[name], summaries[name] = ledger, rows, summary
        screens[name] = gate_screens(summary, cfg)
    comparisons = paired_comparisons(ledgers, primary_rows, weights)
    audit.equal(comparisons, evaluation["comparisons"], "independent.paired_comparisons")
    causal_pass = all(screens["pages"][kind + "_teacher_changing_correct_new"]["pass"] is True for kind in ("criterion_swap", "grade_swap"))
    screens["pages"]["mechanism"] = {"pass": comparisons["learned_benefits"]["pass"] and causal_pass,
                                       "corrected_learned_benefits": comparisons["learned_benefits"], "required_causal_quality_pass": causal_pass}
    screens["pages"]["internal_cross_NI_margin"] = comparisons["internal_cross_NI"]
    audit.equal(screens, evaluation["gate_screens"], "independent.gate_screens")
    populations = {}
    for kind in ("criterion_swap", "grade_swap"):
        metric = kind + "_teacher_changing_correct_new"
        populations[kind] = int(summaries["pages"].get(metric, {}).get("denominator", 0))
        audit.counts[kind + "_teacher_changing_pairs"] = populations[kind]
    incomplete = populations["grade_swap"] == 0
    audit.require(populations["criterion_swap"] > 0, "required criterion-swap teacher-changing denominator is absent")
    if incomplete and not allow_incomplete_grade_diagnostic:
        audit.require(False, "required grade-swap teacher-changing denominator is absent; use explicit retained-evidence diagnostic mode only")
    exact = exact_replay(audit, ir)
    audit.counts["DecisionIR_rows"] = len(ir)
    audit.counts["worlds"] = len(worlds)
    audit.counts["bootstrap_draws"] = len(indices)
    return {"evaluation_sha256": audit.hash(evaluation_path), "selection_receipt_sha256": audit.hash(receipt_path),
            "independent_metrics": summaries, "independent_comparisons": comparisons, "typed_exact_replay": exact,
            "teacher_changing_populations": populations, "causal_diagnostic_complete": not incomplete,
            "artifact_roots": {"corpus": str(corpus_root.resolve()), "features": str(features_root.resolve())},
            "experiment_context": experiment.context(),
            "supervision_verification": supervision_report(atomic, supervised, phase),
            "no_final_selection": True, "encoder_forwards": 0, "model_forwards": 0,
            "unresolved_prerequisites": ["parent-owned closed exact-literal regression", "parent-owned actual cached runtime verification"],
            "evaluator_exact_literals_gate_pass": None, "certificate": "unavailable", "promotion": False}


def supervision_report(atomic: bool, supervised: dict, phase: str) -> dict:
    """State plainly which supervision contract these numbers were produced under."""
    if atomic:
        return {"supervision": "child_local", "protocol_scope": "ECA-2",
                "invalid_atomic_supervision": False,
                "child_local_custody": supervised,
                "interpretation": ("child-local targets, caches, calibration and predictions are independently "
                                   "reconstructed; interface, ceiling and promotion reading is in scope")}
    return {"supervision": "legacy_parent_wide", "protocol_scope": "ECA-1",
            "invalid_atomic_supervision": True,
            "defect": "parent metadata.known labels a supported child UNKNOWN whenever a sibling requirement is missing",
            "numerical_reproduction": "exact: persisted predictions, decisions, ledgers and calibration rebuilt unchanged",
            "interpretation": ("numerical reproduction only; interface, ceiling and promotion readings are withheld "
                               "because supervision, not representation, was invalid"),
            "parent_protocol_sha256": PARENT_PROTOCOL_SHA256, "audited_phase": phase}


def prefinal_provenance(audit: Audit, experiment: capture.Experiment) -> tuple[Path, dict]:
    cfg, protocol_hash = experiment.protocol()
    path = experiment.corpus_root / "eca2_prefinal_provenance_manifest.json"
    provenance = audit.load(path)
    audit.equal(provenance["schema"], "vey.eca2.prefinal-provenance.v1", "prefinal provenance schema")
    audit.equal(provenance["protocol_sha256"], protocol_hash, "prefinal provenance protocol")
    audit.artifact(provenance["parent_protocol"], path.parent)
    audit.equal(provenance["parent_protocol"]["sha256"], PARENT_PROTOCOL_SHA256, "borrowed parent protocol")
    original_root = Path(provenance["original_root"]).resolve()
    build_path = audit.artifact(provenance["parent_build_manifest"], path.parent)
    audit.require(build_path == original_root / "corpus/build_manifest_v1.json", "borrowed build location")
    build = audit.load(build_path)
    audit.equal(build["protocol_sha256"], PARENT_PROTOCOL_SHA256, "borrowed build protocol")
    audit.source(HERE / "ephemeral_pages_build.py", build["builder_sha256"])
    for relative, key in (("source/eca_authored_source_v1.json", "source_sha256"),
                          ("source/eca_source_inventory_v1.json", "source_inventory_sha256"),
                          ("audit/opaque_review_packet_v1.json", "opaque_packet_sha256"),
                          ("audit/sealed_target_key_v1.json", "sealed_key_sha256")):
        audit.hash(original_root / relative, build[key])
    audit.hash(original_root / "audit/accepted_receipt_v1.json", build["receipt"]["receipt_sha256"])
    audit.hash(original_root / "audit/raw_reviews_merged_v1.jsonl", build["receipt"]["raw_reviews_sha256"])
    for phase in ("train", "validation", "calibration", "development"):
        entry = provenance["files"][phase]
        child = audit.artifact(entry, path.parent)
        audit.require(child == (experiment.corpus_root / f"{phase}.jsonl").resolve(), "borrowed phase location")
        source = Path(entry["original_path"]).resolve()
        audit.require(source == original_root / "corpus" / f"{phase}.jsonl", "original phase location")
        audit.hash(source, entry["original_sha256"])
        audit.equal(entry["sha256"], entry["original_sha256"], "borrowed IR byte identity")
        audit.equal(entry["sha256"], build["split_jsonl_sha256"][phase], "borrowed build phase hash")
    return original_root, build


def verify_prefit(audit: Audit, experiment: capture.Experiment,
                  features_root: Path | None = None, corpus_root: Path | None = None) -> dict:
    cfg, protocol_hash = experiment.protocol()
    audit.equal(protocol_hash, ATOMIC_PROTOCOL_SHA256, "prefit requires atomic protocol")
    audit.hash(experiment.protocol_path, protocol_hash)
    for relative, expected in cfg["canonical_dependency"]["files_sha256"].items():
        audit.hash(Path(cfg["canonical_dependency"]["root"]) / relative, expected)
    features_root = features_root or experiment.cache_root
    corpus_root = corpus_root or experiment.corpus_root
    root = Path(cfg["output_root"]).resolve()
    audit.require(features_root.resolve().is_relative_to(root)
                  and corpus_root.resolve() == experiment.corpus_root.resolve(), "prefit paths escape atomic custody")
    prefinal_provenance(audit, experiment)
    supervised = {}
    projection_proofs = {}
    for phase in ("train", "validation", "calibration", "development"):
        manifest, correction, _ = capture_manifest(
            audit, features_root, corpus_root, phase, cfg, None, protocol_hash)
        audit.require(correction is not None, "prefit phase lacks target-only correction")
        supervised[phase] = correction
        projection_proofs[phase] = manifest["lineage"]["atomic_correction"]["projection_verification"]
        audit.equal(manifest["record_count"], correction["child_local_records"], "prefit record population")
    return {"verified": True, "status": "verified_atomic_prefit", "child_local_custody": supervised,
            "protocol_sha256": protocol_hash, "experiment_context": experiment.context(),
            "projection_proofs": projection_proofs,
            "final_opened": False, "optimizer_opened": False, "calibration_opened": False,
            "encoder_forwards": 0, "model_forwards": 0, "promotion": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefit", action="store_true",
                        help="verify all four atomic prefinal captures without fitted artifacts")
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--atomic", action="store_true",
                        help="verify the ECA-2 child-local experiment under the atomic protocol and data root")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--features-root", type=Path, help="explicit capture root for retained archival evidence")
    parser.add_argument("--corpus-root", type=Path, help="explicit DecisionIR root for retained archival evidence")
    parser.add_argument("--allow-incomplete-grade-diagnostic", action="store_true",
                        help="verify retained reconstruction only; never authorize final or causal adequacy")
    parser.add_argument("--output", type=Path, help="exclusive verification receipt path outside Git")
    args = parser.parse_args(argv)
    if args.prefit and (not args.atomic or args.phase == "final" or args.receipt is not None):
        parser.error("--prefit requires --atomic and cannot consume final or a selection receipt")
    if args.phase == "final" and args.receipt is None:
        parser.error("final requires --receipt before any final IR/features")
    experiment = capture.resolve_experiment(args.atomic, args.features_root)
    run_root = (args.run_root or (DEFAULT_ATOMIC_RUN if args.atomic else DEFAULT_RUN)).resolve()
    cfg, protocol_hash = experiment.protocol()
    output_path = args.output or run_root / ("prefit_verification.json" if args.prefit else
                                           f"evaluation/{args.phase}/independent_verification.json")
    if (not output_path.resolve().is_relative_to(run_root)
            or not output_path.resolve().is_relative_to(Path(cfg["output_root"])) or output_path.exists()):
        parser.error("output must be a new path within the outside-Git data/run root")
    audit = Audit()
    report = {"schema": "vey.eca.independent-verification.v1", "phase": args.phase,
              "protocol_sha256": protocol_hash, "experiment_context": experiment.context(),
              "verifier_sha256": audit.hash(Path(__file__)),
              "status": "failed", "promotion": False, "encoder_forwards": 0, "model_forwards": 0}
    failure = None
    try:
        if args.prefit:
            report.update(verify_prefit(audit, experiment, args.features_root, args.corpus_root))
        else:
            report.update(verify_run(audit, args.phase, run_root, args.receipt,
                                    args.features_root, args.corpus_root,
                                    args.allow_incomplete_grade_diagnostic, experiment))
            complete = "reconstruction_verified_incomplete_causal_diagnostic" if not report["causal_diagnostic_complete"] \
                else (ATOMIC_STATUS if args.atomic else LEGACY_STATUS)
            report["status"] = complete
    except Exception as error:
        failure = error
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report.update(counts=dict(audit.counts), artifact_hashes=audit.hashes, diffs=audit.diffs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if failure is not None:
        raise RuntimeError(f"independent verification failed; receipt: {output_path}") from failure
    print(json.dumps({"status": report["status"], "receipt": str(output_path.resolve()), "sha256": digest(output_path),
                      "counts": dict(audit.counts), "promotion": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
