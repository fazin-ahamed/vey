#!/usr/bin/env python3
"""ECA-2 prefinal target repack: reuse retained ECA-1 encoder caches, reproject child-local supervision.

Zero encoder forwards unless an unaudited input appears; original ECA-1 artifacts stay untouched.
"""
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

ORIGINAL_ROOT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1")
ORIGINAL_FEATURES = ORIGINAL_ROOT / "features-grade-corrected"
PROJECTION_PROOF = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-atomic-v2/projection_verification.json")


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
        raise ValueError("prefinal phases only; final uses the sealed capture driver")
    experiment = capture.resolve_experiment(atomic=True)
    old_path = ORIGINAL_FEATURES / f"{phase}_manifest.json"
    old = json.loads(old_path.read_text())
    old_corpus = Path(old["lineage"]["corpus_path"]) if "corpus_path" in old.get("lineage", {}) else ORIGINAL_ROOT / "corpus"
    if sha256_file(old_corpus / f"{phase}.jsonl") != old["corpus_sha256"]:
        raise RuntimeError("original corpus no longer matches retained capture")
    changed_known = changed_grade = changed_orientation = changed_directed = 0
    with (old_corpus / f"{phase}.jsonl").open() as stream:
        records = rows_to_examples((json.loads(line) for line in stream if line.strip()))
    if not records:
        raise RuntimeError("empty phase")
    q, qi = raw_cache(old, "query")
    pages, pi = raw_cache(old, "page")
    cross, old_pairs = raw_cache(old, "cross")
    pairs, missing = {}, {}
    for record in records:
        qh = _text_hash(record["question"])
        if qh not in qi:
            raise RuntimeError("reprojected corpus introduced an unaudited question")
        for page in record["pages"]:
            ph = _text_hash(page["text"])
            if ph not in pi:
                raise RuntimeError("reprojected corpus introduced an unaudited page")
            key = (qh, ph)
            digest = sha256_bytes(_canonical_json(list(key)))
            if digest in old_pairs:
                pairs[key] = old_pairs[digest]
            else:
                missing[key] = (record["question"], page["text"])
    target_cache = experiment.cache_root
    target_cache.mkdir(parents=True, exist_ok=True)
    token_receipts = copy.deepcopy(old["token_receipts"])
    counters = copy.deepcopy(old["counters"])
    calls = 0
    extension = None
    if missing:
        items = sorted(missing.items())
        extension_meta = target_cache / f"{phase}_cross_extension.json"
        if extension_meta.exists():
            prior = json.loads(extension_meta.read_text())
            prior_pairs = [(record[0], record[1]) for record in prior["pairs"]]
            if prior_pairs != [key for key, _ in items]:
                raise RuntimeError("existing cross extension does not match missing pairs")
            extra = np.load(token_receipts["cross_extension"]["features"]["path"], mmap_mode="r", allow_pickle=False)
            for offset, (key, _) in enumerate(items):
                pairs[key] = len(cross) + offset
            cross = np.concatenate((cross, extra))
        else:
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
                path = target_cache / f"{phase}_cross_extension_{name}.npy"
                with path.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
                token_receipts.setdefault("cross_extension", {})[name] = artifact(path)
            extension["pairs"] = [[key[0], key[1], *text_pair] for key, text_pair in items]
            extension["files"] = token_receipts["cross_extension"]
            write_json(extension_meta, extension)
            cross = np.concatenate((cross, extra))
    for modality in ("query", "page", "cross"):
        counters[modality]["forward_calls_this_capture"] = calls if modality == "cross" else 0
        counters[modality]["encoded_examples_this_capture"] = len(missing) if modality == "cross" else 0
        counters[modality]["cache_misses"] = len(missing) if modality == "cross" else 0
        counters[modality]["cache_hits"] = len(pairs) - len(missing) if modality == "cross" else len(qi if modality == "query" else pi)
    counters["total_encoder_forward_calls_this_capture"] = calls
    counters["total_encoded_examples_this_capture"] = len(missing)
    lineage = copy.deepcopy(old["lineage"])
    lineage["repacked_from_corpus_sha256"] = old["corpus_sha256"]
    proof = json.loads(PROJECTION_PROOF.read_text())
    lineage["atomic_correction"] = {
        "correction_protocol": artifact(capture.ATOMIC_EXPERIMENT.protocol_path),
        "original_capture_manifest": artifact(old_path),
        "original_corpus_sha256": old["corpus_sha256"],
        "feature_implementation_sha256": proof["feature_implementation_sha256"],
        "child_local_records": len(records),
        "changed_known_target_records": sum(1 for r in records if bool(r["known_target"])),
        "existing_inputs_reencoded": 0,
        "parent_conjunction_verified": True,
        "projection_verification": artifact(PROJECTION_PROOF),
    }
    corpus = FeatureCorpus(records, q, pages, cross, qi, pi, pairs, target_cache,
                           lineage, counters, token_receipts)
    from dataclasses import replace
    prefinal = replace(experiment, corpus_root=old_corpus)
    path = capture.persist_phase(corpus, phase, None, experiment=prefinal)
    print(json.dumps({"phase": phase, "manifest": artifact(path),
                      "missing_cross_pairs": len(missing), "encoder_forwards": calls,
                      "records": len(records)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("train", "validation", "calibration", "development"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repack(args.phase, args.device)


if __name__ == "__main__":
    main()
