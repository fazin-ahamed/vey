#!/usr/bin/env python3
"""ACCEL-1 QASPER window arms: full-catalog relevance training vs FWS.

Trains the shared cross scorer over pretokenized window tokens against
native retrieval targets. The sparse arm selects gold blocks plus the top
teacher windows by warehouse v2 logits; the vanilla arm keeps full blocks.
Both write the same receipt schema so rounds are comparable CPU-seconds or
GPU-seconds per decision.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "accelerated_local_protocol.json"


def pin(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def load_dataset_rows(root):
    manifest = json.loads((Path(root) / "dataset_manifest.json").read_text())
    require(manifest["schema"] == "vey.qnative.cross.dataset.v1", "Unexpected dataset manifest")
    targets = {}
    rows_by_id = {}
    for phase, entry in manifest["phases"].items():
        if phase not in ("fit", "selection"):
            continue
        name = Path(entry["path"]).name
        require(pin(Path(root) / name)["sha256"] == entry["sha256"], "Phase bytes differ: " + phase)
        with gzip.open(Path(root) / name, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                targets[row["id"]] = row["target"]
                rows_by_id[row["id"]] = row
    return targets, rows_by_id


def block_positive_window_indices(row, target, window_block_ids):
    """Per-window indicator from native matched block ids."""
    hits = []
    for source, matches in enumerate(target.get("ratings", [])):
        ids = []
        for item in matches.get("matches", []):
            ids.extend(item.get("block_ids", []))
        for bid in ids:
            for i, wid in enumerate(window_block_ids):
                if wid == bid and i not in hits:
                    hits.append(i)
    return hits


def subset_windows(row, capacity, teacher_logits=None):
    windows = row["windows"]
    if not capacity or len(windows) <= capacity:
        return windows, list(range(len(windows))), False
    ranked = sorted(range(len(windows)), key=lambda i: (-teacher_logits[i], windows[i]["block_id"], windows[i]["window_index"]))
    chosen = sorted(ranked[:capacity])
    return [windows[i] for i in chosen], chosen, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Refusing to overwrite: " + str(args.out))

    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "vey.accelerated-endgame.protocol.v1", "Unexpected protocol")
    slice_manifest = pin(Path(args.slice_root) / "screen_slice_manifest.json")

    rows = []
    with (Path(args.slice_root) / "screen_slice.jsonl").open() as stream:
        for line in stream:
            rows.append(json.loads(line))
    rows = [r for r in rows if r["endpoint"] == "qasper.evidence_retrieval"]
    require(rows, "No retrieval rows in slice")

    warehouse = {}
    with (args.warehouse / "teacher_logits.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if "window_logits" in r:
                warehouse[r["id"]] = r
    require(warehouse, "Warehouse v2 lacks window logits")

    targets, source_rows = load_dataset_rows(Path(json.loads((HERE / "neutral_qasper_cross_transport_input_amendment.json").read_text())["derived_root"]))
    device = "cuda"
    require(torch.cuda.is_available(), "CUDA required for device-forward runs")
    from transformers import AutoTokenizer
    from research.endgame.native_arch_model import canonical_modules
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall",
                                              revision="eb2d654bf0a5b628c8be6c4be7d29118fbef95b8")
    canonical_modules()
    from research.endgame.native_arch_model import load_arch_encoder
    from research.endgame.native_field_data import protocol as native_protocol
    tokenizer, encoder, custody = load_arch_encoder(native_protocol(), {}, device)
    seed = 7
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = random.Random(seed)

    from research.endgame.native_arch_model import CrossArchitecture
    model = CrossArchitecture(encoder).to(device=device, dtype=torch.float32)
    head_params = list(model.scorer.head.parameters())
    pool_params = list(model.scorer.pool.parameters())
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": args.lr},
                                   {"params": pool_params, "lr": 0.00002}], weight_decay=0.01)
    model.train()
    torch.cuda.reset_peak_memory_stats()

    forward_seconds = 0.0
    backward_seconds = 0.0
    losses = 0
    covered_positive = 0
    covered_rows = 0
    start = time.monotonic()
    for step in range(args.steps):
        row = rows[step % len(rows)]
        target = targets[row["id"]]
        if not target.get("target_available") or not target.get("matched_block_ids"):
            continue
        gold = target["matched_block_ids"]
        windows, keep, reduced = subset_windows(row, args.cap, warehouse[row["id"]]["window_logits"]) if args.cap else (row["windows"], None, False)
        scored_proba = []
        active_positive = 0
        block_indices = []
        window_blocks = [w["block_id"] for w in windows]
        block_positions = {}
        for i, bid in enumerate(window_blocks):
            block_positions.setdefault(bid, []).append(i)
        for bid in {w["block_id"] for w in windows}:
            block_positions = [i for i, bi in enumerate(window_blocks) if bi == bid]
            if bid in gold:
                active_positive += 1
                block_indices.append((bid, block_positions))
        if not block_indices:
            continue
        covered_rows += 1
        covered_positive += active_positive
        logits_row = []
        for bid, positions in block_indices:
            chunk = [windows[p] for p in positions]
            batch = tokenizer.pad([{k: w[k] for k in ("input_ids", "attention_mask")} for w in chunk],
                                  padding=True, return_tensors="pt")
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            takeoff = time.monotonic()
            hidden = model.scorer.pool.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            scores = model.scorer.head(pooled).squeeze(-1)
            scored = scores.max()
            torch.cuda.synchronize()
            forward_seconds += time.monotonic() - takeoff
            logits_row.append(scored)
        if not logits_row:
            continue
        target = torch.ones(len(logits_row), device=device)
        logits = torch.stack(logits_row)
        loss = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
        losses += 1
        takeoff = time.monotonic()
        loss.backward()
        torch.cuda.synchronize()
        backward_seconds += time.monotonic() - takeoff
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    receipt = {
        "schema": "vey.accel.arm-receipt.v1", "arm": args.arm,
        "config": {"cap": args.cap, "batch_size": args.batch_size, "steps": args.steps, "lr": args.lr},
        "protocol": pin(PROTOCOL), "slice_manifest": slice_manifest,
        "warehouse_pin": pin(args.warehouse / "teacher_warehouse_manifest.json"),
        "rows_covered": covered_rows, "positive_windows_covered": covered_positive,
        "forward_gpu_seconds": round(forward_seconds, 3),
        "backward_gpu_seconds": round(backward_seconds, 3),
        "loss_batches": losses,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "wall_seconds": round(time.monotonic() - start, 1),
        "status": "COMPLETE", "quality_credit": False, "performance_credit": False,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
