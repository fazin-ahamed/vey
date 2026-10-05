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
    "massive.spelling_score": "Score/ordinal quality",
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
    if (routes.get("status") != "PASS" or routes.get("schema") !=
            "vey.neutral.development-comparison-verification.v2"):
        raise SystemExit("route receipt schema/status invalid; refusing to fill cells")
    choice = load(args.choice_receipt) if args.choice_receipt else None
    if choice and (choice.get("status") != "PASS" or choice.get("schema") !=
                   "vey.neutral.development-comparison-verification.v2"):
        raise SystemExit("choice receipt schema/status invalid; refusing to fill cells")

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
            "route": "english",
            "metric": "top1_accuracy",
            "vey_scope": "not_applicable; numerical fallback output retained",
        })
        provenance["choice_intent_dev"] = {
            "receipt": args.choice_receipt,
            "receipt_sha256": digest(args.choice_receipt),
            "endpoint": "massive.intent",
            "rows": choice["laya"]["rows"],
            "vey": {k: v for k, v in choice["vey"].items() if k != "per_locale"},
            "laya_english": {k: v for k, v in choice["laya"].items() if k != "per_locale"},
            "note": ("Vey arm is Vey-not-applicable by mechanism: the frozen decide() surface "
                     "compiles this question to an empty program and answers alphabetically. "
                     "The recorded accuracy is what the frozen reference actually produced."),
        }
        fill("probability calibration", {
            "Vey": choice["vey"]["ece15"], "Laya": choice["laya"]["ece15"],
            "metric": "ece15_predicted_label_confidence", "lower_is_better": True,
            "sample_count": choice["laya"]["rows"], "route": "english",
            "endpoint": "massive.intent", "paired_delta": None, "CI95": None,
            "vey_scope": "not_applicable; numerical fallback confidence retained",
            "measurement_limit": "Historical raw-bundle temperatures, not current served calibration; no trust claim.",
        })

    alt = routes.get("laya_multilingual")
    if alt:
        provenance.setdefault("choice_intent_dev", {})["laya_multilingual"] = {
            k: v for k, v in alt.items() if k != "per_locale"}
        provenance["route_comparison"] = routes["route_comparison"]
        provenance["vendor_claim_context"] = routes.get("vendor_claim_context")
        provenance["route_receipt"] = {
            "path": args.routes_receipt, "sha256": digest(args.routes_receipt)}
        fill("multilingual", {
            "Vey": choice["vey"]["macro_locale_accuracy"] if choice else None,
            "Laya": alt["macro_locale_accuracy"], "metric": "macro_locale_top1_accuracy",
            "sample_count": alt["rows"], "route": "multilingual",
            "endpoint": "massive.intent", "paired_delta": None, "CI95": None,
            "vey_scope": "not_applicable; numerical fallback output retained",
            "measurement_limit": "51 exposed locales with equal counts; not fresh multilingual transfer.",
        })

    score_routes = provenance.setdefault("score_routes", {})
    for path in args.score_receipts:
        receipt = load(path)
        if (receipt.get("status") != "PASS" or receipt.get("schema") !=
                "vey.neutral.ordinal-score-verification.v2" or
                not receipt.get("independent_of_arm_aggregation") or
                receipt.get("verified_rows") != receipt.get("projected_rows")):
            raise SystemExit(f"score receipt {path} is not independently verified")
        source = Path(receipt["source_file"])
        if digest(source) != receipt["source_file_sha256"]:
            raise SystemExit(f"score source for {path} changed")
        endpoint = receipt["endpoint"]
        route = receipt["arm_header"]["route"]
        score_routes.setdefault(endpoint, {})[route] = {
            "receipt": path, "receipt_sha256": digest(path),
            "source_file_sha256": receipt["source_file_sha256"],
            "ordinal": receipt["ordinal"], "vey": receipt["vey_arm"],
            "registration_limit": receipt["registration_limit"],
        }
    # The scalar cell is the registered English grammar endpoint, not whichever
    # receipt happens to appear last. Every other route/rubric stays explicit.
    grammar = score_routes.get("massive.grammar_score", {}).get("english")
    if grammar:
        fill(SCORE_TO_DIMENSION["massive.grammar_score"], {
            "Laya": grammar["ordinal"]["normalized_mean_abs_error"],
            "sample_count": grammar["ordinal"]["rows"],
            "metric": "normalized_mean_abs_error", "lower_is_better": True,
            "route": "english", "endpoint": "massive.grammar_score",
            "vey_scope": "not_applicable on measured frozen decide() surface",
            "paired_delta": None, "CI95": None,
        })
    else:
        cells[SCORE_TO_DIMENSION["massive.grammar_score"]].update({
            "Laya": None, "sample_count": None, "paired_delta": None, "CI95": None,
            "status": "UNMEASURED", "green": False, "route": "english",
            "endpoint": "massive.grammar_score", "metric": "normalized_mean_abs_error",
            "lower_is_better": True,
            "green_reason": "No independently verified registered English grammar receipt.",
        })
    if "score_grammar_score_dev" in provenance:
        provenance["score_grammar_score_dev"]["superseded_by"] = "score_routes"
        provenance["score_grammar_score_dev"]["limitation"] = (
            "Historical v1 verifier trusted arm-derived errors and did not enforce "
            "complete unique membership; retained as superseded evidence.")

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