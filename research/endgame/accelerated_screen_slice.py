#!/usr/bin/env python3
"""Derive the frozen 5% screen slice for ACCEL-1 inside the pretokenized ledger.

Deterministic whole-component bucket by SHA256(component_id) with seed 7;
per-endpoint family counts must stay nonzero. CPU-only, no model, no GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pin(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def slice_id(row, seed=7):
    return int(hashlib.sha256(f"accel-screen|{seed}|{row['component_id']}".encode()).hexdigest(), 16) % 20 == 0


def materialize(ledger_root: Path, out_root: Path):
    a = ledger_root / "ledger_manifest.json"
    require(a.exists(), "Ledger manifest missing")
    manifest = json.loads(a.read_text())
    require(manifest["schema"] == "vey.accel.ledger.v1" and manifest["status"] == "MATERIALIZED",
            "Ledger manifest not settled")
    rows = []
    with (ledger_root / "ledger.jsonl").open() as stream:
        for line in stream:
            rows.append(json.loads(line))
    keep = {r["id"]: r for r in rows if slice_id(r)}
    require(keep, "Screen slice empty")
    counts = {}
    for rid, r in keep.items():
        counts[r["endpoint"]] = counts.get(r["endpoint"], 0) + 1
    out = out_root / "screen_slice.jsonl"
    for target in (out, out_root / "screen_slice_manifest.json"):
        require(not target.exists(), "Refusing to overwrite: " + str(target))
    with out.open("x", encoding="utf-8") as stream:
        for rid in sorted(keep):
            stream.write(json.dumps(keep[rid], sort_keys=True, ensure_ascii=False) + "\n")
    summary = {"schema": "vey.accel.screen-slice.v1", "status": "MATERIALIZED",
                "ledger_manifest": pin(a), "rows": len(keep), "endpoint_counts": counts,
                "seed": 7, "rule": "sha256(accel-screen|7|component_id) % 20 == 0",
                "payload": pin(out), "model_loads": 0, "model_forwards": 0}
    with (out_root / "screen_slice_manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = materialize(args.ledger_root, args.out_root)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
