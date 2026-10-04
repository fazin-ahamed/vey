#!/usr/bin/env python3
"""Phase-scoped ECA-1 feature capture driver.

Streams one split's DecisionIR JSONL through the pinned frozen encoder with
content-keyed mmap caches. Final remains locked behind the selection/
calibration receipt. No training, scoring or model outcome occurs here.

Every path and protocol digest is carried by an explicit Experiment; the
module constants below are the frozen ECA-1 historical custody context, not a
mutable default.
"""
import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ephemeral_pages_features import (
    CAPTURE_PHASES, PROTOCOL_PATH, ATOMIC_PROTOCOL_PATH, _protocol,
    capture_features, sha256_file, validate_final_receipt,
)

CORPUS = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/corpus")
CACHE = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/features-grade-corrected")


@dataclass(frozen=True)
class Experiment:
    protocol_path: Path
    corpus_root: Path
    cache_root: Path

    def protocol(self):
        return _protocol(self.protocol_path)

    def context(self):
        _, digest = self.protocol()
        return {"protocol_sha256": digest, "protocol_path": str(self.protocol_path.resolve()),
                "corpus_root": str(self.corpus_root.resolve()),
                "cache_root": str(self.cache_root.resolve())}


DEFAULT_EXPERIMENT = Experiment(PROTOCOL_PATH, CORPUS, CACHE)
ATOMIC_ROOT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-atomic-v2")
ATOMIC_EXPERIMENT = Experiment(ATOMIC_PROTOCOL_PATH, ATOMIC_ROOT / "corpus", ATOMIC_ROOT / "features")


def resolve_experiment(atomic=False, features_root=None):
    experiment = ATOMIC_EXPERIMENT if atomic else DEFAULT_EXPERIMENT
    return replace(experiment, cache_root=Path(features_root)) if features_root is not None else experiment


def rows_for(split: str, experiment=DEFAULT_EXPERIMENT):
    path = experiment.corpus_root / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)



def load_phase(phase: str, experiment=DEFAULT_EXPERIMENT) -> dict:
    """Load and hash-check a captured phase without constructing an encoder."""
    if phase not in CAPTURE_PHASES:
        raise ValueError(f"invalid phase: {phase}")
    _, protocol_hash = experiment.protocol()
    path = experiment.cache_root / f"{phase}_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["phase"] != phase or manifest["protocol_sha256"] != protocol_hash:
        raise RuntimeError("phase manifest does not match frozen protocol")
    if phase == "final":
        validate_final_receipt(manifest["selection_calibration_receipt"], experiment.protocol_path)
    if sha256_file(experiment.corpus_root / f"{phase}.jsonl") != manifest["corpus_sha256"]:
        raise RuntimeError("captured IR changed")
    arrays = {}
    for name, entry in manifest["files"].items():
        target = experiment.cache_root / entry["path"]
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
            "counters": manifest["counters"], "token_receipts": manifest["token_receipts"],
            "experiment_context": experiment.context()}


def persist_phase(corpus, phase: str, receipt: Path | None, experiment=DEFAULT_EXPERIMENT) -> Path:
    _, protocol_hash = experiment.protocol()
    if phase == "final":
        validate_final_receipt(receipt, experiment.protocol_path)
    arrays = corpus.arrays()
    records_path = experiment.cache_root / f"{phase}_records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for record in corpus.records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False) + "\n")
    files = {"records": {"path": records_path.name, "sha256": sha256_file(records_path)}}
    for name, value in arrays.items():
        if isinstance(value, np.memmap) and name not in {"raw_q", "raw_pages"}:
            target = Path(value.filename)
            files[name] = {"path": str(target.relative_to(experiment.cache_root)), "sha256": sha256_file(target)}
    manifest = {
        "phase": phase, "protocol_sha256": protocol_hash,
        "corpus_sha256": sha256_file(experiment.corpus_root / f"{phase}.jsonl"),
        "record_count": len(corpus.records), "files": files,
        "lineage": corpus.lineage, "counters": corpus.counters,
        "token_receipts": corpus.token_receipts,
        "experiment_context": experiment.context(),
        "selection_calibration_receipt": str(receipt.resolve()) if receipt else None,
    }
    path = experiment.cache_root / f"{phase}_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=sorted(CAPTURE_PHASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--atomic", action="store_true")
    parser.add_argument("--features-root", type=Path)
    args = parser.parse_args()
    experiment = resolve_experiment(args.atomic, args.features_root)
    if args.phase == "final":
        if args.receipt is None:
            parser.error("final requires --receipt")
        validate_final_receipt(args.receipt, experiment.protocol_path)
    elif args.receipt is not None:
        parser.error("--receipt is only valid for final")
    manifest_path = experiment.cache_root / f"{args.phase}_manifest.json"
    if manifest_path.exists():
        arrays = load_phase(args.phase, experiment)
        print(json.dumps({"status": "captured_phase_verified", "phase": args.phase,
                          "records": len(arrays["records"]), "encoder_forwards": 0}))
        return 0
    corpus = capture_features(rows_for(args.phase, experiment), experiment.cache_root,
                              phase=args.phase, device=args.device, protocol_path=experiment.protocol_path,
                              selection_calibration_receipt=args.receipt)
    path = persist_phase(corpus, args.phase, args.receipt, experiment)
    print(json.dumps({"status": "features_captured", "phase": args.phase,
                      "records": len(corpus.records), "counters": corpus.counters,
                      "manifest": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
