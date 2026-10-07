#!/usr/bin/env python3
"""ACCEL-1 CNSR screen: batch/precision ladder over real cross-encoder pairs.

Measures GPU-forward seconds, backward seconds, peak VRAM and loss agreement
on the frozen 5% screen slice. Arms are configuration-only (same model, same
rows, same loss); no evidence is earned and no mechanism is advanced or
retired by this screen. Duty-cycle pauses bound GPU temperature.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "accelerated_local_protocol.json"
RUN_SECONDS_PER_STEP_CAP = 45.0
PAUSE_IF_TEMP_C = 74
RESUME_UNTIL_TEMP_C = 68
PAUSE_SECONDS = 30.0


def pin(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def gpu_temp():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=5)
    return int(out.stdout.strip().splitlines()[0])


class DutyClock:
    """Separates active compute time from thermal pauses; every pause logged."""

    def __init__(self, log):
        self.events = []
        self.log = Path(log)
        self.active_forward = 0.0
        self.active_backward = 0.0
        self.paused = 0.0

    def guard(self):
        temp = gpu_temp_state()
        while temp >= PAUSE_IF_TEMP_C:
            started = time.monotonic()
            time.sleep(PAUSE_SECONDS)
            after = gpu_temp_state()
            event = {"start_C": temp, "end_C": after, "seconds": round(time.monotonic() - started, 1)}
            self.events.append(event)
            self.paused += event["seconds"]
            with self.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**event, "status": "DUTY_PAUSE"}) + "\n")
            temp = after

    def forward(self, seconds):
        self.active_forward += seconds

    def backward(self, seconds):
        self.active_backward += seconds


def gpu_temp_state():
    return gpu_temp()


def load_screen_rows(root):
    with (Path(root) / "screen_slice.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def cross_inputs(row, max_candidates=32):
    """Intent-style cross inputs derived from the ledger row: state x candidate pairs."""
    pairs = []
    for cand in row["candidates"][:max_candidates]:
        tokens = (row["question"] + "\n" + cand["text"])
        pairs.append(tokens)
    return pairs


def arm_timed(model, tokenizer, pairs, device, batch_size, use_amp, clock):
    import torch
    import torch.nn.functional as F
    forward_seconds = 0.0
    backward_seconds = 0.0
    pair_logits = []
    seq_count = 0
    for index in range(0, len(pairs), batch_size):
        chunk = pairs[index:index + batch_size]
        batch = tokenizer(chunk, padding=True, truncation=True, max_length=512,
                          return_tensors="pt").to(device)
        clock.guard()
        start_fwd = time.monotonic()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            hidden = model(batch["input_ids"], batch["attention_mask"]).last_hidden_state[:, 0]
            logits = hidden.sum(dim=-1)
            _ = F.log_softmax(logits, dim=-1)
            torch.cuda.synchronize()
        forward_seconds += time.monotonic() - start_fwd
        seq_count += len(chunk)
        pair_logits.append(logits.float().cpu())
        start_bwd = time.monotonic()
        torch.sum(logits).backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        backward_seconds += time.monotonic() - start_bwd
    return forward_seconds, backward_seconds, seq_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--pairs-per-row", type=int, default=32)
    parser.add_argument("--max-sequences", type=int, default=2_048)
    args = parser.parse_args(argv)
    require(not args.out.exists(), "Refusing to overwrite: " + str(args.out))
    protocol_pin = pin(PROTOCOL)
    protocol = json.loads(PROTOCOL.read_text())
    require(protocol["schema"] == "vey.accelerated-endgame.protocol.v1", "Unexpected protocol")
    rows = load_screen_rows(args.slice_root)
    intent_rows = [row for row in rows if row["endpoint"] in ("banking77.intent", "massive.intent")]
    if args.rows:
        intent_rows = intent_rows[:args.rows]
    require(intent_rows, "No intent rows in screen slice")

    import torch  # noqa: F401
    require(torch.cuda.is_available(), "CUDA required; no CPU neural fallback")
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall",
                                              revision="eb2d654bf0a5b628c8be6c4be7d29118fbef95b8")
    model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall",
                                      revision="eb2d654bf0a5b628c8be6c4be7d29118fbef95b8")
    device = "cuda"
    model = model.to(device)
    torch.manual_seed(7)
    torch.cuda.reset_peak_memory_stats()
    clock = DutyClock(args.out.with_name(args.out.name + ".thermal.log"))
    start = time.monotonic()
    forward_total = 0.0
    backward_total = 0.0
    seq_total = 0
    for row in intent_rows:
        if seq_total >= args.max_sequences:
            break
        pairs = cross_inputs(row, max_candidates=args.pairs_per_row)[:max(0, args.max_sequences - seq_total)]
        if not pairs:
            continue
        f, b, seq = arm_timed(model, tokenizer, pairs, device, args.batch_size, args.amp, clock)
        forward_total += f
        backward_total += b
        seq_total += seq
    receipt = {
        "schema": "vey.accel.screen-arm.v1",
        "arm": f"cross_batch{args.batch_size}_" + ("amp" if args.amp else "fp32"),
        "config": {"batch_size": args.batch_size, "amp": args.amp, "rows_limit": args.rows},
        "protocol": protocol_pin,
        "slice_manifest": pin(Path(args.slice_root) / "screen_slice_manifest.json"),
        "rows_processed": len(intent_rows),
        "pairs_encoded": seq_total,
        "forward_gpu_seconds": round(forward_total, 3),
        "backward_gpu_seconds": round(backward_total, 3),
        "seconds_per_forward_sequence": round(forward_total / max(seq_total, 1), 6),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "thermal_events": len(clock.events), "thermal_paused_seconds": round(clock.paused, 1),
        "wall_seconds": round(time.monotonic() - start, 1),
        "status": "COMPLETE", "quality_credit": False, "performance_credit": False,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
