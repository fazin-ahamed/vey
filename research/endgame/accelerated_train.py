#!/usr/bin/env python3
"""ACCEL-1 screen arm runner for intent rows.

Trains the canonical CrossArchitecture on the frozen 5% screen slice and
periodically scores against the slice selection rows, writing per-arm
receipts with measured forward/backward GPU-seconds, peak VRAM, and quality
proxy on the slice. Local 3060 only under the accelerated protocol; this
runner is CPU-local and does NOT construct the Modal-custodied QNATIVE
artifacts. It loads the stock pretrained encoder with a fresh scorer head -
the screen compares mechanisms against the warehouse'd teacher, not jump to
the legacy checkpoint.
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


def pin(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.file_digest(p.open("rb"), "sha256").hexdigest()}


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def load_slice(root):
    rows = []
    with (Path(root) / "screen_slice.jsonl").open() as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows


def teacher_subset_ids(row, teacher_logits, cap):
    """Gold plus top-(cap-1) by teacher scores; original order preserved."""
    ids = [c["id"] for c in row["candidates"]]
    if cap <= 0 or len(ids) <= cap:
        return ids, list(range(len(ids)))
    distribution = row["target"]["distribution"]
    gold_index = distribution.index(1.0) if 1.0 in distribution else None
    require(gold_index is not None, "Screen slice rows used for arms must have singleton gold")
    ranked = sorted(range(len(ids)), key=lambda i: (-teacher_logits[i], ids[i]))
    keep = sorted(dict.fromkeys([gold_index] + ranked[:cap - 1]))
    return [ids[i] for i in keep], keep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--candidate-cap", type=int, default=0,
                        help="Gold plus top-(cap-1) teacher candidates when positive")
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Refusing to overwrite: " + str(args.out))

    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "vey.accelerated-endgame.protocol.v1", "Unexpected protocol")
    rows = load_slice(args.slice_root)
    intent_rows = [r for r in rows if r["endpoint"] in ("banking77.intent", "massive.intent")]
    intent_rows = sorted(intent_rows, key=lambda r: r["id"])
    require(intent_rows, "No intent rows in slice")
    dataset_root = Path(json.loads((HERE / "neutral_qasper_cross_transport_input_amendment.json").read_text())["derived_root"])
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    import gzip
    targets = {}
    for phase, entry in manifest["phases"].items():
        if phase not in ("fit", "selection"):
            continue
        with gzip.open(dataset_root / Path(entry["path"]).name, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                targets[row["id"]] = row["target"]
    teacher = {}
    with (args.warehouse / "teacher_logits.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if "teacher_logits" not in r:
                continue
            teacher[r["id"]] = r["teacher_logits"]
    require(teacher, "Warehouse empty")
    intent_rows = [r for r in intent_rows if r["id"] in teacher]
    require(intent_rows, "No slice rows covered by warehouse")

    import random
    rng = random.Random(7)
    rng.shuffle(intent_rows)
    for row in intent_rows:
        row["target"] = targets[row["id"]]

    from research.endgame.native_arch_model import CrossArchitecture, load_arch_encoder
    from research.endgame.native_field_data import protocol as native_protocol
    import torch.nn.functional as F

    device = "cuda"
    require(torch.cuda.is_available(), "CUDA required; no CPU arm fallback")
    native_cfg = native_protocol()
    tokenizer, encoder, custody = load_arch_encoder(native_cfg, {}, device)
    model = CrossArchitecture(encoder).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW([
        {"params": model.scorer.head.parameters(), "lr": args.lr},
        {"params": model.scorer.pool.parameters(), "lr": 0.00002},
    ], weight_decay=0.01)
    model.train()
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.cuda.reset_peak_memory_stats()

    start = time.monotonic()
    forward_gpu = 0.0
    backward_gpu = 0.0
    optimizer_micro_steps = 0
    kl_forward = 0.0
    kl_batches = 0
    for step in range(args.steps):
        row = intent_rows[step % len(intent_rows)]
        teacher_logits = teacher[row["id"]]
        if args.candidate_cap:
            candidate_ids, keep = teacher_subset_ids(row, teacher_logits, args.candidate_cap)
            pairs = [row["state"] + "\n" + row["candidates"][i]["text"] for i in keep]
            target = torch.tensor([row["target"]["distribution"][i] for i in keep], device=device, dtype=torch.float32)
        else:
            candidate_ids = [c["id"] for c in row["candidates"]]
            pairs = [row["state"] + "\n" + c["text"] for c in row["candidates"]]
            target = torch.tensor(row["target"]["distribution"], device=device, dtype=torch.float32)
        batch = tokenizer(pairs, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)
        takeoff = time.monotonic()
        ids = batch["input_ids"].unsqueeze(0)
        mask = batch["attention_mask"].unsqueeze(0)
        values = model.scorer(ids, mask).squeeze(0)
        loss = -(target * F.log_softmax(values, 0)).sum()
        torch.cuda.synchronize()
        forward_gpu += time.monotonic() - takeoff
        takeoff = time.monotonic()
        loss.backward()
        torch.cuda.synchronize()
        backward_gpu += time.monotonic() - takeoff
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_micro_steps += 1
        values = values.detach()
        dist = torch.log_softmax(values, 0)
        with torch.no_grad():
            reference = teacher[row["id"]]
            if args.candidate_cap:
                reference = [reference[i] for i in keep]
            teacher_logits = torch.tensor(reference, device=device, dtype=torch.float32)
            teacher_probs = torch.softmax(teacher_logits, 0)
            kl_forward += float((teacher_probs * (torch.log_softmax(teacher_logits, 0) - dist)).sum().item())
            kl_batches += 1

    model.eval()
    eval_correct = 0
    eval_total = 0
    teacher_agree = 0
    eval_seconds = 0.0
    for row in intent_rows:
        distribution = row["target"]["distribution"]
        if 1.0 not in distribution:
            continue
        gold_index = distribution.index(1.0)
        pairs = [row["state"] + "\n" + c["text"] for c in row["candidates"]]
        logits = []
        start_eval = time.monotonic()
        for index in range(0, len(pairs), args.batch_size):
            chunk = pairs[index:index + args.batch_size]
            batch = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                              return_tensors="pt").to(device)
            with torch.no_grad():
                scores = model.scorer(batch["input_ids"].unsqueeze(0),
                                      batch["attention_mask"].unsqueeze(0)).squeeze(0)
            logits.extend(float(v) for v in scores.cpu().tolist())
        eval_seconds += time.monotonic() - start_eval
        eval_total += 1
        winner = max(range(len(logits)), key=lambda i: logits[i])
        if winner == gold_index:
            eval_correct += 1
        teacher_winner = max(range(len(teacher[row["id"]])), key=lambda i: teacher[row["id"]][i])
        if winner == teacher_winner:
            teacher_agree += 1

    receipt = {
        "schema": "vey.accel.arm-receipt.v1",
        "arm": args.arm,
        "config": {"batch_size": args.batch_size, "steps": args.steps, "lr": args.lr},
        "protocol": pin(PROTOCOL),
        "slice_root_pin": pin(args.slice_root / "screen_slice_manifest.json"),
        "warehouse_pin": pin(args.warehouse / "teacher_warehouse_manifest.json"),
        "rows_covered": len(intent_rows),
        "steps": optimizer_micro_steps,
        "forward_gpu_seconds": round(forward_gpu, 3),
        "backward_gpu_seconds": round(backward_gpu, 3),
        "gper_second_per_decision": round((forward_gpu + backward_gpu) / max(optimizer_micro_steps, 1), 6),
        "teacher_kl_mean": kl_forward / max(kl_batches, 1),
        "eval_rows": eval_total, "eval_top1": eval_correct / max(eval_total, 1),
        "teacher_agreement": teacher_agree / max(eval_total, 1),
        "eval_gpu_seconds": round(eval_seconds, 3),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": round(time.monotonic() - start, 1),
        "status": "COMPLETE", "quality_credit": False, "performance_credit": False,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
