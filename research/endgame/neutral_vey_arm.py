#!/usr/bin/env python3
"""Vey arm for the neutral development comparison.

Runs the FROZEN product (vey-2-final, e6b046f) exactly as shipped: no source
edit, no substitute scorer, no proxy head. Every row persists the compiled
decision program, lane, and full candidate distribution, because the arm's
scope limit (a fixed-label classification instruction compiles to an empty
program whose compose fallback is alphabetical) is only visible from those
records.

Usage:
  python neutral_vey_arm.py --limit N --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import neutral_comparison_rows as R  # noqa: E402


def input_digest(row) -> str:
    payload = {
        "id": row["id"],
        "question": row["question"],
        "state": row["state"],
        "candidate_ids": row["candidate_ids"],
        "candidates": row["candidates"],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0 = all labeled dev rows")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    R.protocol()  # fails closed if the preregistration bytes changed after commit

    from vey.crux.compiler import compile_instruction
    from vey.decision import DecisionRequest
    from vey.runtime import Runtime

    runtime = Runtime(device="cpu", enable_crux=True)


    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    n = 0
    empty_program = 0
    modes = {}
    correct = 0
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": "vey.neutral.vey-arm-run.v1",
            "arm": "vey",
            "product_commit": "e6b046ffbd138cbdbfb2f89c6ae77525fe6b0b18",
            "comparison_protocol_sha256": R.COMPARISON_PROTOCOL_SHA256,
            "amendment_sha256": hashlib.sha256(
                (HERE / "neutral_development_comparison_amendment_v1.json").read_bytes()
            ).hexdigest(),
            "endpoint": R.CHOICE_ENDPOINT,
            "limit": args.limit,
            "started_unix": started,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        for row in R.choice_rows():
            if args.limit and n >= args.limit:
                break
            program = compile_instruction(row["question"])
            if not program:
                empty_program += 1

            request = DecisionRequest(
                question=row["question"],
                candidates=dict(row["candidates"]),
                state=row["state"],
                explain=True,
            )
            decided = runtime.decide(request)
            modes[decided.decision_mode] = modes.get(decided.decision_mode, 0) + 1
            hit = decided.answer == row["gold"]
            correct += int(hit)
            record = {
                "id": row["id"],
                "group_id": row["group_id"],
                "locale": row["locale"],
                "endpoint": row["endpoint"],
                "arm": "vey",
                "gold": row["gold"],
                "answer": decided.answer,
                "correct": hit,
                "decision_mode": decided.decision_mode,
                "program": [s.label() for s in program],
                "program_empty": not program,
                "trust": decided.trust,
                "probabilities": decided.probabilities,
                "certificate": decided.certificate.to_dict() if decided.certificate else None,
                "input_sha256": input_digest(row),
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
            if n % 500 == 0:
                print(f"vey {n} rows, acc {correct / n:.4f}, {time.time() - started:.0f}s", flush=True)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({
        "rows": n,
        "accuracy": correct / n if n else None,
        "empty_program_fraction": empty_program / n if n else None,
        "decision_modes": modes,
        "seconds": time.time() - started,
        "peak_rss_kb": peak_kb,
        "out": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    raise SystemExit(main())