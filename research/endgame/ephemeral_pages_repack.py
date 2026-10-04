#!/usr/bin/env python3
"""Repack prefinal grade-corrected rows without re-encoding existing inputs."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ephemeral_pages_capture as capture
from ephemeral_pages_features import (FeatureCorpus, FeatureEncoder, BATCH_SIZE,
    _canonical_json, _text_hash, rows_to_examples, sha256_bytes, sha256_file)

ROOT = capture.CORPUS.parent
ORIGINAL_FEATURES = ROOT / "features"
ORIGINAL_CORPUS = ROOT / "corpus.superseded-grade-intervention"


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")


def raw_cache(manifest, modality):
    receipt = manifest["counters"][modality]
    entries = manifest["token_receipts"][modality]
    metadata = Path(entries["features"]["path"]).parent / f"{modality}_{receipt['cache_key']}.json"
    value = json.loads(metadata.read_text())
    for entry in entries.values():
        if sha256_file(entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"original feature cache changed: {entry['path']}")
    features = np.load(entries["features"]["path"], mmap_mode="r", allow_pickle=False)
    return features, {key: i for i, key in enumerate(value["item_hashes"])}


def repack(phase, device):
    if phase not in ("train", "validation", "calibration", "development"):
        raise ValueError("final must use the sealed capture driver, not repair")
    old_path = ORIGINAL_FEATURES / f"{phase}_manifest.json"
    old = json.loads(old_path.read_text())
    if sha256_file(ORIGINAL_CORPUS / f"{phase}.jsonl") != old["corpus_sha256"]:
        raise RuntimeError("original corpus no longer matches retained capture")
    original_rows = {}
    with (ORIGINAL_CORPUS / f"{phase}.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            original_rows[row["id"]] = line
    changed, invariant, causal = 0, 0, 0
    by_id = {}
    with (capture.CORPUS / f"{phase}.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            by_id[row["id"]] = row
            if row["metadata"]["variant"] == "relevant_page_grade_change":
                changed += 1
            else:
                if original_rows[row["id"]] != line:
                    raise RuntimeError("a non-grade diagnostic/optimizer row changed")
                invariant += 1
    if set(original_rows) != set(by_id):
        raise RuntimeError("repair changed row identities")
    for row in by_id.values():
        if row["metadata"]["variant"] == "relevant_page_grade_change":
            parent = by_id[row["metadata"]["parent_decision_id"]]
            if set(row["gold"]) == set(parent["gold"]):
                raise RuntimeError("corrected grade variant still has no teacher change")
            if row["metadata"]["exact_facts"] != parent["metadata"]["exact_facts"]:
                raise RuntimeError("repair changed exact state")
            causal += 1
    records = rows_to_examples(by_id.values())
    del by_id, original_rows
    q, qi = raw_cache(old, "query")
    pages, pi = raw_cache(old, "page")
    cross, old_pairs = raw_cache(old, "cross")
    pairs, missing = {}, {}
    for record in records:
        qh = _text_hash(record["question"])
        if qh not in qi:
            raise RuntimeError("repair introduced an unaudited/new question")
        for page in record["pages"]:
            ph = _text_hash(page["text"])
            if ph not in pi:
                raise RuntimeError("repair introduced a page absent from original capture")
            key = (qh, ph)
            digest = sha256_bytes(_canonical_json(list(key)))
            if digest in old_pairs:
                pairs[key] = old_pairs[digest]
            else:
                missing[key] = (record["question"], page["text"])
    capture.CACHE.mkdir(parents=True, exist_ok=True)
    token_receipts = copy.deepcopy(old["token_receipts"])
    counters = copy.deepcopy(old["counters"])
    calls = 0
    extension = None
    if missing:
        items = sorted(missing.items())
        vectors, ids_rows, mask_rows = [], [], []
        with FeatureEncoder(device=device) as encoder:
            identity = ("encoder_repo", "encoder_revision", "encoder_weight_sha256", "tokenizer_sha256")
            if any(encoder.lineage[key] != old["lineage"][key] for key in identity):
                raise RuntimeError("missing-pair encoder does not match original identity")
            before = encoder.parameter_hash()
            for start in range(0, len(items), BATCH_SIZE):
                chunk = items[start:start + BATCH_SIZE]
                ids, masks = encoder.tokenize([pair[0] for _, pair in chunk], [pair[1] for _, pair in chunk])
                vectors.append(encoder.encode_batch(ids, masks, "cross"))
                ids_rows.append(ids)
                mask_rows.append(masks)
                calls += 1
            if encoder.parameter_hash() != before:
                raise RuntimeError("frozen encoder changed during pair extension")
            extension = {"encoder_lineage": encoder.lineage, "encoder_parameters_sha256_after": before}
        extra = np.concatenate(vectors)
        for offset, (key, _) in enumerate(items):
            pairs[key] = len(cross) + offset
        for name, value in (("features", extra), ("input_ids", np.concatenate(ids_rows)), ("attention_mask", np.concatenate(mask_rows))):
            path = capture.CACHE / f"{phase}_cross_extension_{name}.npy"
            with path.open("xb") as stream:
                np.save(stream, value, allow_pickle=False)
            token_receipts.setdefault("cross_extension", {})[name] = artifact(path)
        extension["pairs"] = [[key[0], key[1], *text_pair] for key, text_pair in items]
        extension["files"] = token_receipts["cross_extension"]
        write_json(capture.CACHE / f"{phase}_cross_extension.json", extension)
        cross = np.concatenate((cross, extra))
    for modality in ("query", "page", "cross"):
        counters[modality]["forward_calls_this_capture"] = calls if modality == "cross" else 0
        counters[modality]["encoded_examples_this_capture"] = len(missing) if modality == "cross" else 0
        counters[modality]["cache_misses"] = len(missing) if modality == "cross" else 0
        counters[modality]["cache_hits"] = len(pairs) - len(missing) if modality == "cross" else len(qi if modality == "query" else pi)
    counters["total_encoder_forward_calls_this_capture"] = calls
    counters["total_encoded_examples_this_capture"] = len(missing)
    lineage = copy.deepcopy(old["lineage"])
    lineage["grade_correction"] = {
        "original_capture_manifest": artifact(old_path),
        "original_corpus_sha256": old["corpus_sha256"],
        "corrected_corpus_sha256": sha256_file(capture.CORPUS / f"{phase}.jsonl"),
        "amendment_file": artifact(Path(__file__).with_name("ephemeral_pages_grade_amendment.json")),
        "implementation": artifact(Path(__file__)),
        "invariant_non_grade_rows": invariant, "grade_rows": changed,
        "teacher_changing_grade_rows": causal, "existing_inputs_reencoded": 0,
        "missing_cross_pairs": len(missing), "optimizer_reruns": 0,
    }
    corpus = FeatureCorpus(records, q, pages, cross, qi, pi, pairs, capture.CACHE,
                           lineage, counters, token_receipts)
    path = capture.persist_phase(corpus, phase, None)
    print(json.dumps({"phase": phase, "manifest": artifact(path),
                      "repair": lineage["grade_correction"], "encoder_forwards": calls}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("train", "validation", "calibration", "development"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repack(args.phase, args.device)


if __name__ == "__main__":
    main()
