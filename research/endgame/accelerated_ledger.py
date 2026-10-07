#!/usr/bin/env python3
"""Pretokenized input ledger for the accelerated local 3060 research lane.

Builds a deterministic cache of tokenizer outputs for registered QNATIVE-2
phases so successive-elimination arms never re-tokenize during training.
CPU-only, no model, no GPU. Reads through the same custody-bound loaders as
the live study; refuses overwrite; all pins are written into the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from . import neutral_qasper_cross_data as data
except ImportError:
    import neutral_qasper_cross_data as data

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "accelerated_local_protocol.json"
LEDGER_SCHEMA = "vey.accel.ledger.v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build_row(row, windows):
    return {
        "id": row["id"], "endpoint": row["endpoint"], "component_id": row["component_id"],
        "phase": row["phase"], "paper_id": row["paper_id"], "group_id": row["group_id"],
        "question_ordinal": row["question_ordinal"], "input_sha256": row["input_sha256"],
        "question": row["serving"]["question"],
        "windows": windows,
        "candidates": row["serving"]["candidates"],
        "target_available": row["target"].get("target_available"),
    }


def window_payloads(tokenizer, row):
    if row["endpoint"] in data.INTENTS:
        return []
    windows, _census = data.source_windows(tokenizer, row["serving"]["question"], row["blocks"])
    return [{"block_id": w["block_id"], "window_index": w["window_index"],
             "input_ids": w["input_ids"], "attention_mask": w["attention_mask"],
             "source_positions": w["source_positions"], "offsets": w["offsets"],
             "question_positions": w["question_positions"]} for w in windows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phases", default="fit,selection")
    args = parser.parse_args()

    protocol_pin = data.pin(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "vey.accelerated-endgame.protocol.v1", "Unexpected accelerated protocol")
    citation = (HERE / "neutral_qasper_cross_transport_input_amendment.json")
    dataset_root = Path(json.loads(citation.read_text())["derived_root"])
    dataset_manifest = data.pin(dataset_root / "dataset_manifest.json")
    dataset_receipt = data.pin(dataset_root / "verification_receipt.json")

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers unavailable for ledger build") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-xsmall", revision="eb2d654bf0a5b628c8be6c4be7d29118fbef95b8")
    tokenizer_sha = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    require(phases, "Phases list is empty")
    rows = {}
    for phase in phases:
        for row in data.load_phase(phase, data.AMENDMENT):
            require(row["id"] not in rows, "Duplicate ledger id " + row["id"])
            rows[row["id"]] = build_row(row, window_payloads(tokenizer, row))

    out = args.root / "ledger.jsonl"
    manifest_out = args.root / "ledger_manifest.json"
    for target in (out, manifest_out):
        require(not target.exists(), "Refusing to overwrite ledger artifacts")
    with out.open("x", encoding="utf-8") as stream:
        for rid in sorted(rows):
            stream.write(data.canonical(rows[rid]) + "\n")
        stream.flush()
    payload_pin = data.pin(out)
    manifest = {"schema": LEDGER_SCHEMA, "status": "MATERIALIZED",
                "accelerated_protocol": protocol_pin, "dataset_manifest": dataset_manifest,
                "dataset_verification_receipt": dataset_receipt,
                "tokenizer_sha256": tokenizer_sha, "payload": payload_pin,
                "phases": {phase: {"rows": sum(1 for r in rows.values() if r["phase"] == phase)} for phase in phases},
                "row_count": len(rows), "model_loads": 0, "model_forwards": 0}
    with manifest_out.open("x", encoding="utf-8") as stream:
        stream.write(data.canonical(manifest) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
