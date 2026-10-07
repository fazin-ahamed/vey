#!/usr/bin/env python3
"""Teacher Replay Warehouse for the accelerated lane.

Runs the completed NATIVE-2 cross checkpoint once over the registered screen
slice and persists per-row candidate logits and targets. Downstream arms
consume this artifact; they do not call the teacher. Local 3060 only, as
permitted by accelerated_local_protocol.json. The legacy QNATIVE custody and
its Modal-only gates remain in force for any future rerun of that study; this
file deliberately does not call build_model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "accelerated_local_protocol.json"
SCIENTIFIC = HERE / "neutral_qasper_cross_study_amendment.json"


def pin(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def load_teacher(cfg):
    from research.endgame.native_arch_model import CrossArchitecture, load_arch_encoder, canonical_modules
    from research.endgame.native_field_model import fingerprint, load_stock
    from research.endgame.native_field_data import protocol as native_protocol
    canonical_modules()
    native_cfg = native_protocol()
    tokenizer, encoder, custody = load_arch_encoder(native_cfg, {}, "cuda")
    model = CrossArchitecture(encoder).to(device="cuda", dtype=torch.float32)
    entry = cfg["current_authority"]["selected_initial_cross_checkpoint"]
    require(pin(entry["path"])["sha256"] == entry["sha256"], "Teacher checkpoint bytes differ")
    from safetensors.torch import load_file
    state = load_file(entry["path"], device="cpu")
    model.load_state_dict(state, strict=True)
    meta_entry = cfg["current_authority"]["selected_initial_cross_metadata"]
    meta = json.loads(Path(meta_entry["path"]).read_text())
    require(meta["checkpoint"]["state_dict_fingerprint"] == fingerprint(model.state_dict())
            and meta["checkpoint"]["selected_epoch"] == 10, "Teacher identity mismatch")
    del state
    model.eval()
    return tokenizer, model, custody


def pair_ids(row, catalogues):
    out = {}
    for candidate in row["candidates"]:
        out[candidate["id"]] = catalogues[row["endpoint"]][candidate["id"]]
    return [(row["state"] + "\n" + out[cid]) for cid in [c["id"] for c in row["candidates"]]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=0)
    args = parser.parse_args(argv)

    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "vey.accelerated-endgame.protocol.v1", "Unexpected protocol")
    cfg = json.loads(SCIENTIFIC.read_text())
    require(cfg["schema"] == "vey.neutral.qasper.cross-study-amendment.v1", "Unexpected amendment")

    tokenizer, model, custody = load_teacher(cfg)
    require(torch.cuda.is_available(), "CUDA required; local CPU replay is not registered")

    citation = json.loads((HERE / "neutral_qasper_cross_transport_input_amendment.json").read_text())
    dataset_root = Path(citation["derived_root"])
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    require(manifest["schema"] == "vey.qnative.cross.dataset.v1", "Dataset manifest schema differs")
    receipt = json.loads((dataset_root / "verification_receipt.json").read_text())
    require(receipt.get("schema") == "vey.qnative.cross.dataset-verification.v1"
            and receipt.get("status") == "PASS"
            and receipt.get("manifest", {}).get("sha256") == pin(dataset_root / "dataset_manifest.json")["sha256"],
            "Dataset receipt must bind the current manifest")
    targets = {}
    import gzip
    for phase, entry in manifest["phases"].items():
        if phase not in ("fit", "selection"):
            continue
        path = dataset_root / Path(entry["path"]).name
        require(pin(path)["sha256"] == entry["sha256"], "Phase bytes differ: " + phase)
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                targets[row["id"]] = {"target": row["target"], "input_sha256_ref": row["input_sha256"]}
    require(targets, "No targets recovered")

    rows = []
    with (Path(args.slice_root) / "screen_slice.jsonl").open() as stream:
        for line in stream:
            rows.append(json.loads(line))
    if args.rows:
        rows = rows[:args.rows]
    intent_rows = [r for r in rows if r["endpoint"] in ("banking77.intent", "massive.intent")]
    require(intent_rows, "No intent rows in slice")
    catalogues = {}
    for row in intent_rows:
        id_by_natural = {c["id"]: c["text"] for c in row["candidates"]}
        if id_by_natural:
            catalogues.setdefault(row["endpoint"], id_by_natural)

    args.out_root.mkdir(parents=True, exist_ok=True)
    out = args.out_root / "teacher_logits.jsonl"
    receipt_path = args.out_root / "teacher_warehouse_manifest.json"
    for target in (out, receipt_path):
        require(not target.exists(), "Refusing to overwrite: " + str(target))

    batch_size = 32
    encoded = 0
    forward_seconds = 0.0
    start_wall = time.monotonic()
    with out.open("x", encoding="utf-8") as stream:
        for row in intent_rows:
            pairs = pair_ids(row, catalogues)
            if not pairs:
                continue
            row_logits = []
            torch.cuda.synchronize()
            takeoff = time.monotonic()
            for index in range(0, len(pairs), batch_size):
                chunk = pairs[index:index + batch_size]
                batch = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                                  return_tensors="pt").to("cuda")
                with torch.no_grad():
                    ids = batch["input_ids"]
                    mask = batch["attention_mask"]
                    hidden = model.scorer.pool.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
                    pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
                    scores = model.scorer.head(pooled).squeeze(-1)
                encoded += len(chunk)
                row_logits.extend(float(v) for v in scores.float().cpu())
                del ids, mask, hidden, pooled, scores, batch
            torch.cuda.synchronize()
            forward_seconds += time.monotonic() - takeoff
            record = {
                "id": row["id"], "endpoint": row["endpoint"], "component_id": row["component_id"],
                "phase": row["phase"], "input_sha256": row["input_sha256"],
                "candidate_ids": [c["id"] for c in row["candidates"]], "teacher_logits": row_logits,
                "window_count": len(row.get("windows", [])),
                "target": targets[row["id"]]["target"],
            }
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "vey.accel.teacher-warehouse.v1", "status": "MATERIALIZED",
        "protocol_pin": pin(PROTOCOL), "teacher_checkpoint": cfg["current_authority"]["selected_initial_cross_checkpoint"],
        "slice_manifest": pin(Path(args.slice_root) / "screen_slice_manifest.json"),
        "out_rows": len(intent_rows), "encoded_sequences": encoded,
        "forward_gpu_seconds": round(forward_seconds, 3),
        "wall_seconds": round(time.monotonic() - start_wall, 1),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "payload": pin(out), "quality_credit": False, "performance_credit": False,
    }
    with receipt_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
