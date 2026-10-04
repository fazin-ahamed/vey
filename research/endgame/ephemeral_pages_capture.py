#!/usr/bin/env python3
"""Phase-scoped ECA-1 feature capture driver.

Streams one split's DecisionIR JSONL through the pinned frozen encoder with
content-keyed mmap caches. Final remains locked behind the selection/
calibration receipt. No training, scoring or model outcome occurs here.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ephemeral_pages_features import (
    CAPTURE_PHASES, PROTOCOL_SHA256, capture_features, sha256_file,
    validate_final_receipt,
)

CORPUS = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/corpus")
CACHE = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/features-grade-corrected")


def rows_for(split: str):
    path = CORPUS / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)



def load_phase(phase: str) -> dict:
    """Load and hash-check a captured phase without constructing an encoder."""
    if phase not in CAPTURE_PHASES:
        raise ValueError(f"invalid phase: {phase}")
    path = CACHE / f"{phase}_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["phase"] != phase or manifest["protocol_sha256"] != PROTOCOL_SHA256:
        raise RuntimeError("phase manifest does not match frozen protocol")
    if phase == "final":
        validate_final_receipt(manifest["selection_calibration_receipt"])
    if sha256_file(CORPUS / f"{phase}.jsonl") != manifest["corpus_sha256"]:
        raise RuntimeError("captured IR changed")
    arrays = {}
    for name, entry in manifest["files"].items():
        target = CACHE / entry["path"]
        if sha256_file(target) != entry["sha256"]:
            raise RuntimeError(f"captured artifact changed: {target}")
        if name == "records":
            with target.open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
        else:
            arrays[name] = np.load(target, mmap_mode="r", allow_pickle=False)
    if len(records) != manifest["record_count"]:
        raise RuntimeError("captured record count changed")
    return {**arrays, "raw_q": arrays["q"], "raw_pages": arrays["pages"],
            "records": records, "lineage": manifest["lineage"],
            "counters": manifest["counters"], "token_receipts": manifest["token_receipts"]}


def persist_phase(corpus, phase: str, receipt: Path | None) -> Path:
    arrays = corpus.arrays()
    records_path = CACHE / f"{phase}_records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in corpus.records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
    files = {"records": {"path": records_path.name, "sha256": sha256_file(records_path)}}
    for name, value in arrays.items():
        if isinstance(value, np.memmap) and name not in {"raw_q", "raw_pages"}:
            target = Path(value.filename)
            files[name] = {"path": str(target.relative_to(CACHE)), "sha256": sha256_file(target)}
    manifest = {
        "phase": phase, "protocol_sha256": PROTOCOL_SHA256,
        "corpus_sha256": sha256_file(CORPUS / f"{phase}.jsonl"),
        "record_count": len(corpus.records), "files": files,
        "lineage": corpus.lineage, "counters": corpus.counters,
        "token_receipts": corpus.token_receipts,
        "selection_calibration_receipt": str(receipt.resolve()) if receipt else None,
    }
    path = CACHE / f"{phase}_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(CAPTURE_PHASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "final":
        if args.receipt is None:
            parser.error("final requires --receipt")
        validate_final_receipt(args.receipt)
    elif args.receipt is not None:
        parser.error("--receipt is only valid for final")
    manifest_path = CACHE / f"{args.phase}_manifest.json"
    if manifest_path.exists():
        arrays = load_phase(args.phase)
        print(json.dumps({"status": "captured_phase_verified", "phase": args.phase,
                          "records": len(arrays["records"]), "encoder_forwards": 0}))
        return 0
    corpus = capture_features(rows_for(args.phase), CACHE, phase=args.phase, device=args.device,
                              selection_calibration_receipt=args.receipt)
    path = persist_phase(corpus, args.phase, args.receipt)
    print(json.dumps({"status": "features_captured", "phase": args.phase,
                      "records": len(corpus.records), "counters": corpus.counters,
                      "manifest": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
