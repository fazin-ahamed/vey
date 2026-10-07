#!/usr/bin/env python3
"""Hardware/precision ward for the accelerated lane: one bounded 10-minute
workload across T4, L4, A10 and A100-40GB, measuring wall-clock, peak VRAM
and estimated dollars so Modal hardware selection follows cost per training
progress instead of cheapest hourly rate. Registered via
`accelerated_local_protocol.json`; pre-results (no quality), no promotions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRICES_USD_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A100-40GB": 2.10}
CAP_USD_THIS_PERIOD = 1.50


def pin(path):
    import hashlib
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def boundary_guard():
    proto = json.loads(Path("/home/fazinahamed/Documents/vey-public/research/endgame/accelerated_local_protocol.json").read_text())
    require(proto["schema"] == "vey.accelerated-endgame.protocol.v1", "Protocol identity differs")
    budget = proto["authority"]["budget"]
    require(budget["modal_usd_this_period_cap"] == CAP_USD_THIS_PERIOD, "Spend cap differs")


def plan(spend_ledger_usd=0.0):
    estimated = {gpu: round((600 / 3600) * price + 0.03, 4) for gpu, price in PRICES_USD_HOUR.items()}
    estimated["total"] = round(sum(estimated[g] for g in PRICES_USD_HOUR) + spend_ledger_usd, 4)
    return {"schema": "vey.accel.microbenchmark-plan.v1", "status": "PROSPECTIVE",
            "purpose": "GPU-hour-to-dollar ratio under a bounded 10-minute workload; no quality metric",
            "cost_usd_estimated": estimated, "cap_usd": CAP_USD_THIS_PERIOD}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    boundary_guard()
    result = plan()
    require(not args.out.exists(), "Refusing to overwrite: " + str(args.out))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
