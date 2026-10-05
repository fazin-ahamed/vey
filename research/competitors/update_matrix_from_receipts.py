#!/usr/bin/env python3
"""Update the 21-cell competitor matrix from verified comparison receipts.

Only receipts produced by neutral_comparison_verify.py are accepted. A cell is
filled from a receipt, never from an arm's own printed summary, and a cell that
has no receipt stays UNMEASURED. Vey's architectural cells keep their frozen
Vey 2 status and are never overwritten by a dev-partition measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MATRIX = HERE.parent / "competitors" / "COMPETITOR_MATRIX.json"

# Which verified receipt populates which matrix dimension, and with what.
# Choice accuracy and multilingual rows come from the route receipt.
ROUTE_TO_DIMENSION = {
    "massive.intent": "Choice accuracy",
}
SCORE_TO_DIMENSION = {
    "massive.grammar_score": "Score/ordinal quality",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes-receipt", required=True)
    parser.add_argument("--choice-receipt")
    parser.add_argument("--score-receipts", nargs="*", default=[])
    parser.add_argument("--note", required=True)
    args = parser.parse_args(argv)

    matrix = load(MATRIX)
    cells = {c["dimension"]: c for c in matrix["critical_cells"]}
    provenance = matrix.setdefault("verified_development_measurements", {})

    routes = load(args.routes_receipt)
    if routes.get("status") != "PASS":
        raise SystemExit("route receipt is not PASS; refusing to fill cells")
    choice = load(args.choice_receipt) if args.choice_receipt else None
    if choice and choice.get("status") != "PASS":
        raise SystemExit("choice receipt is not PASS; refusing to fill cells")

    def fill(dimension, values):
        cell = cells[dimension]
        for key, value in values.items():
            cell[key] = value
        cell["status"] = "MEASURED_DEVELOPMENT_PARTITION"
        # A development measurement on a legacy public benchmark never turns a
        # critical cell green: the endgame requires sealed final data and a
        # direct Jev contrast.
        cell["green"] = False
        cell["green_reason"] = ("Development partition on an exposed public benchmark. "
                                "Sealed final data and direct Jev contrast remain required.")

    dim = ROUTE_TO_DIMENSION["massive.intent"]
    if choice:
        fill(dim, {
            "Vey": choice["vey"]["top1_accuracy"],
            "Laya": choice["laya"]["top1_accuracy"],
            "paired_delta": choice["paired_group_bootstrap"]["observed_delta"],
            "CI95": choice["paired_group_bootstrap"]["ci95_bootstrap"],
            "sample_count": choice["discordance"]["shared_rows"],
        })
        provenance["choice_intent_dev"] = {
            "receipt": args.choice_receipt,
            "receipt_sha256": digest(args.choice_receipt),
            "endpoint": "massive.intent",
            "rows": choice["laya"]["rows"],
            "vey": choice["vey"],
            "laya_english": choice["laya"],
            "note": ("Vey arm is Vey-not-applicable by mechanism: the frozen decide() surface "
                     "compiles this question to an empty program and answers alphabetically. "
                     "The recorded accuracy is what the frozen reference actually produced."),
        }

    alt = routes.get("laya_multilingual")
    if alt:
        provenance["choice_intent_dev"]["laya_multilingual"] = alt
        provenance["route_comparison"] = routes["route_comparison"]
        provenance["vendor_claim_context"] = routes.get("vendor_claim_context")

    for path in args.score_receipts:
        receipt = load(path)
        if receipt.get("status") != "PASS":
            raise SystemExit(f"score receipt {path} is not PASS; refusing to fill cells")
        endpoint = receipt["endpoint"]
        fill(SCORE_TO_DIMENSION[endpoint], {
            "Laya": receipt["ordinal"]["normalized_mean_abs_error"],
            "sample_count": receipt["ordinal"]["rows"],
        })
        provenance[f"score_{endpoint.split('.')[-1]}_dev"] = {
            "receipt": path,
            "receipt_sha256": digest(path),
            "endpoint": endpoint,
            "ordinal": receipt["ordinal"],
            "vey": receipt["vey_arm"],
        }

    provenance["note"] = args.note
    provenance["evidence_class"] = ("MEASURED on the verified source-native dev projection. "
                                    "Legacy replication of an exposed public benchmark, not fresh "
                                    "transfer, and not a final or sealed-pool result.")
    provenance["direct_Jev_status"] = "PROVISIONAL - DIRECT JEV EVALUATION UNAVAILABLE"

    # Preserve the file's existing key order. Rewriting the whole document with
    # sort_keys buries a two-field change inside a full-file reformat, which
    # makes the diff unauditable in review.
    original_order = list(json.loads(MATRIX.read_text(encoding="utf-8")))
    ordered = {key: matrix[key] for key in original_order if key in matrix}
    ordered.update({key: value for key, value in matrix.items() if key not in original_order})
    # ensure_ascii matches the file's original escaping, so untouched strings
    # such as direct_Jev keep their exact bytes instead of being rewritten.
    MATRIX.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "matrix": str(MATRIX),
        "filled": sorted(provenance),
        "green_cells": sum(1 for c in ordered["critical_cells"] if c["green"]),
        "total_cells": len(ordered["critical_cells"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())