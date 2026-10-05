#!/usr/bin/env python3
"""Reconstruct completed ECA-2 adaptation outputs without encoder/model calls."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np

import ephemeral_pages_verify as reference

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = reference.capture.ATOMIC_ROOT / "runs/interface-audit-v1/adaptation-v1"
ARM = "pages_adapted_final_layer"
CLAIMS = ("pages_frozen_baseline", "adapter_blind", "zero_pages", "lexical")
RAW_FIELDS = {"record_index", "prediction", "ordinal_mass_uncalibrated_sigma_0_1",
              "ordinal_mass_calibrated", "variant", "certificate", "source_metadata"}
ECE_NAMES = {prefix + "ECE" for prefix in ("raw_distribution_", "calibrated_distribution_",
                                          "OOD_raw_distribution_", "OOD_calibrated_distribution_")}


def rebuild(audit, name, entries, ir, records, calibration, worlds, metadata, source, cfg, root):
    paths = {key: audit.artifact(value, root) for key, value in entries.items()}
    events = reference.WorldEvents(worlds)
    groups = {}
    count = 0
    for count, (raw, captured) in enumerate(itertools.zip_longest(
            reference.json_rows(paths["term_candidate"]), reference.json_rows(records)), 1):
        audit.require(raw is not None and captured is not None, name + ": raw/capture length")
        audit.equal(raw["record_index"], count - 1, name + ": record index")
        audit.equal(raw["source_metadata"], metadata, name + ": raw source provenance")
        audit.equal({key: value for key, value in raw.items() if key not in RAW_FIELDS},
                    captured, name + ": original captured record")
        row = ir[raw["row_id"]]
        audit.equal(captured, reference.expected_record(row, raw["term_index"],
                    raw["candidate_id"], "child_local"), name + ": child-local IR projection")
        audit.equal(raw["variant"], row["metadata"]["variant"], name + ": variant")
        audit.equal(raw["certificate"], "unavailable", name + ": certificate")
        prediction = raw["prediction"]
        score = prediction["score"]
        probability = reference.sigmoid(prediction["known_logits"])
        audit.equal(probability, prediction["known_probability"], name + ": known probability")
        for key in ("attention", "direction", "raw_value", "relevance_logits"):
            audit.require(len(prediction[key]) == len(raw["pages"]), name + ": page shape")
        audit.require(all(value is not None and math.isfinite(value)
                          for key in ("attention", "direction", "raw_value")
                          for value in prediction[key]), name + ": finite page outputs")
        audit.require(all(0 <= value <= 1 for value in prediction["raw_value"]),
                      name + ": raw-value range")
        audit.require(all(-1 <= value <= 1 for value in prediction["direction"]),
                      name + ": direction range")
        if score is not None:
            mixture = sum(a * (0.5 + d * (v - 0.5)) for a, d, v in zip(
                prediction["attention"], prediction["direction"], prediction["raw_value"]))
            audit.equal(mixture, score, name + ": independent page mixture", tolerance=2e-6)
        for key, sigma in (("ordinal_mass_uncalibrated_sigma_0_1", .1),
                           ("ordinal_mass_calibrated", calibration["ordinal_sigma"])):
            audit.equal(reference.masses(score, sigma) if score is not None else None,
                        raw[key], name + ": ordinal distribution")
        children = groups.setdefault(raw["row_id"], {})
        key = raw["term_index"], raw["candidate_id"]
        audit.require(key not in children, name + ": duplicate child")
        children[key] = score, probability
        reference.page_events(events, raw, prediction, row, False,
                              calibration["ordinal_sigma"], cfg["corpus"]["families"])
    audit.require(set(groups) == set(ir), name + ": complete child population")
    decisions, seen = [], set()
    for saved in reference.json_rows(paths["decisions"]):
        row_id = saved["row_id"]
        audit.require(row_id in ir and row_id not in seen, name + ": unique decision")
        seen.add(row_id)
        audit.equal(saved["source_metadata"], metadata, name + ": decision provenance")
        audit.equal(saved["source"], source, name + ": source label")
        rebuilt = reference.reconstruct(ir[row_id], groups.pop(row_id),
                                        calibration["knownness_threshold"])
        compared = {key: value for key, value in saved.items() if key != "source_metadata"}
        # Provenance is checked above; the reference decoder names its source differently.
        compared["source"] = rebuilt["source"]
        audit.equal(rebuilt, compared, name + ": complete decision", tolerance=1e-12)
        del rebuilt["required_children"], rebuilt["UNKNOWN_reasons"]
        decisions.append(rebuilt)
    audit.require(seen == set(ir), name + ": complete decision population")
    pairs = reference.decision_events(audit, events, decisions, ir, cfg["corpus"]["families"])
    if "paired_interventions" in paths:
        for left, right in itertools.zip_longest(pairs, reference.json_rows(paths["paired_interventions"])):
            audit.require(left is not None and right is not None, name + ": causal pair count")
            audit.equal(left, right, name + ": causal pair")
    ledger = audit.load(paths["world_metric_ledger"])
    audit.equal(ledger["world_ids"], worlds, name + ": ledger world ordering")
    rebuilt_metrics = {metric: {family: value.tolist() for family, value in groups.items()}
                       for metric, groups in events.totals.items()}
    audit.equal(rebuilt_metrics, ledger["metrics"], name + ": metric events")
    audit.counts[name + "_term_candidate_rows"] = count
    audit.counts[name + "_decision_rows"] = len(decisions)
    return events, decisions


def phase_replay(audit, root, manifest, history, context, phase, cfg, calibration=None):
    report = audit.load(audit.artifact(manifest["reports"][phase], root))
    audit.equal(report["phase"], phase, "phase identity")
    audit.equal(report["source_metadata"]["source_sha256"], history["source_sha256"], "phase sources")
    for key in ("final_pool_access", "promotion", "B_STEF_allowed"):
        audit.equal(report[key], False, "phase scope: " + key)
    for key in ("controls_refitted", "controls_recalibrated", "controls_reselected"):
        audit.equal(report[key], 0, "frozen controls: " + key)
    features, corpus = Path(context["cache_root"]), Path(context["corpus_root"])
    capture, _, _ = reference.capture_manifest(audit, features, corpus, phase, cfg,
                                               None, history["protocol_sha256"])
    ir = {}
    for row in reference.json_rows(corpus / (phase + ".jsonl")):
        audit.require(row["id"] not in ir, "unique phase IR")
        audit.equal(row["split"], phase, "IR phase")
        reference.verify_teacher(audit, row, amended=True)
        ir[row["id"]] = row
    records = audit.artifact(capture["files"]["records"], features)
    worlds = report["bootstrap"]["world_ids"]
    audit.equal(worlds, sorted({row["metadata"]["world_id"] for row in ir.values()}), "bootstrap worlds")
    indices = np.load(audit.artifact(report["bootstrap"], root), mmap_mode="r", allow_pickle=False)
    expected = np.random.default_rng(0).integers(0, len(worlds), size=(10000, len(worlds)), dtype=np.int32)
    audit.require(np.array_equal(indices, expected), "registered bootstrap indices")
    weights = np.zeros((10000, len(worlds)), dtype=np.float64)
    np.add.at(weights, (np.arange(10000)[:, None], indices), 1)
    if phase == "calibration":
        calibration = reference.reconstruct_calibration(audit,
            audit.artifact(report["artifacts"]["term_candidate"], root), ir, "child_local")
    audit.equal(calibration, report["calibration"], "independent calibration/fit-only isolation")
    events, decisions = rebuild(audit, phase + "_" + ARM, report["artifacts"], ir, records,
        calibration, worlds, report["source_metadata"], "semantic_adapted_final_layer", cfg, root)
    summaries = events.summary(weights)
    audit.equal(summaries, report["metrics"], "independent phase metrics")
    audit.equal(reference.gate_screens(summaries, cfg), report["gate_screens"][ARM], "absolute gates")
    return {"report": report, "ir": ir, "records": records, "worlds": worlds,
            "weights": weights, "events": events, "decisions": decisions, "calibration": calibration}


def verify(root):
    root = root.resolve()
    audit = reference.Audit()
    experiment = reference.capture.ATOMIC_EXPERIMENT
    audit.require(root.is_relative_to(reference.capture.ATOMIC_ROOT.resolve()), "run outside frozen data root")
    cfg, protocol_hash = experiment.protocol()
    audit.hash(experiment.protocol_path, protocol_hash)
    history = audit.load(root / (ARM + "_history.json"))
    manifest = audit.load(root / "evaluation_manifest.json")
    launch = audit.load(audit.artifact(manifest["launch_receipt"], root))
    audit.equal(launch["experiment_context"], experiment.context(), "registered experiment context")
    audit.equal(history["protocol_sha256"], protocol_hash, "history protocol")
    audit.equal(manifest["source_sha256"], history["source_sha256"], "fit/evaluation source identity")
    for name, expected in history["source_sha256"].items():
        audit.hash(HERE / name, expected)
    audit.artifact(history["checkpoint"], root)
    audit.equal(list(reference.json_rows(audit.artifact(history["epoch_progress"], root))),
                history["history"], "durable epoch journal")
    audit.require(len(history["history"]) == 400 and len(history["immutability_checks"]) == 400,
                  "fixed400epochs and frozen-byte checks")
    audit.equal([row["epoch"] for row in history["history"]], list(range(1, 401)), "sequential fit epochs")
    checks = history["immutability_checks"]
    audit.equal([row["epoch"] for row in checks], list(range(1, 401)), "sequential freeze checks")
    audit.equal(history["frozen_parameter_sha256"], launch["frozen_parameter_sha256"], "initial frozen digest")
    for check in checks:
        audit.equal(check["frozen_parameter_sha256"], launch["frozen_parameter_sha256"], "per-epoch frozen digest")
        audit.equal(check["adapter_names"], checks[0]["adapter_names"], "stable adapter names")
        audit.equal(check["changed_adapter_names"], check["adapter_names"], "all six adapters changed")
        audit.equal(check["changed_final_layer_names"], checks[0]["changed_final_layer_names"], "stable final-layer names")
        audit.require(len(check["adapter_names"]) == 6 and len(check["changed_final_layer_names"]) == 16,
                      "registered adapter/final-layer tensor population")
        audit.require(all(name.startswith("deberta.encoder.layer.11.") for name in check["changed_final_layer_names"]),
                      "changes restricted to final layer")
    for entry in history["screen_receipts"].values():
        receipt = audit.load(audit.artifact(entry, root))
        audit.equal(entry["verdict"], "pass", "screen reference verdict")
        audit.equal(receipt["verdict"], "pass", "screen receipt verdict")
    selected = min(history["history"], key=lambda row: row["validation"]["total"])
    audit.equal(selected["epoch"], history["selected_epoch"], "earliest minimum validation epoch")
    audit.equal(selected["validation"]["total"], history["selected_validation_objective"], "selection objective")
    audit.equal(history["encoder_counters"]["encoder_backward_calls"], 400, "one encoder backward per epoch")
    frozen = subprocess.run(["git", "diff", "--quiet", cfg["immutable_reference"],
                             "--", "src", "tests", "examples"], cwd=HERE.parents[1], check=False)
    audit.require(frozen.returncode == 0, "immutable product source/tests/examples changed")
    calibration = phase_replay(audit, root, manifest, history, launch["experiment_context"], "calibration", cfg)
    development = phase_replay(audit, root, manifest, history, launch["experiment_context"],
                               "development", cfg, calibration["calibration"])
    report = development["report"]
    ablations = {}
    for name, entry in report["ablations"].items():
        receipt = {key: value for key, value in entry.items() if key not in ("artifacts", "metrics", "recording")}
        metadata = {**report["source_metadata"], "ablation": name, **receipt}
        events, _ = rebuild(audit, name, entry["artifacts"], development["ir"], development["records"],
            development["calibration"], development["worlds"], metadata,
            "semantic_adapted_final_layer_ablation", cfg, root)
        audit.equal({key: value for key, value in events.summary(development["weights"]).items()
                     if key not in ECE_NAMES}, entry["metrics"], "independent ablation metrics: " + name)
        ablations[name] = events
    audit.equal(sorted(ablations), ["adapter_blind", "layer11_frozen", "uniform_attention", "zero_pages", "zero_question"],
                "complete frozen ablation set")
    baseline_path = audit.artifact(report["baseline_development_manifest"], root)
    baseline = audit.load(baseline_path)
    baseline_calibration = audit.load(audit.artifact(report["baseline_calibration"], root))
    controls = {}
    for name in ("pages", "lexical"):
        controls[name], _ = reference.rebuild_control(audit, name, baseline["artifacts"][name],
            development["ir"], development["records"], baseline_calibration["controls"][name],
            development["worlds"], development["weights"], cfg, baseline_path.parent, "child_local")
        audit.equal(controls[name].summary(development["weights"]), baseline["metrics"][name],
                    "independent frozen baseline metrics: " + name)
    targets = {"pages_frozen_baseline": controls["pages"], "lexical": controls["lexical"],
               "adapter_blind": ablations["adapter_blind"], "zero_pages": ablations["zero_pages"]}
    claims = {}
    for name in CLAIMS:
        other = targets[name]
        audit.equal(sorted(development["events"].totals["atomic_choice_macro"]),
                    sorted(other.totals["atomic_choice_macro"]), "fixed contrast families: " + name)
        samples = development["events"].estimate("atomic_choice_macro", development["weights"]) - other.estimate(
            "atomic_choice_macro", development["weights"])
        valid = samples[np.isfinite(samples)]
        lower = float(np.quantile(valid, .05 / 4))
        claims[name] = {"metric": "atomic_choice_macro",
            "point": float(development["events"].estimate("atomic_choice_macro") - other.estimate("atomic_choice_macro")),
            "CI95": np.quantile(valid, [.025, .975]).tolist(), "one_sided_Bonferroni_lower": lower,
            "alpha": .05 / 4, "valid_bootstrap_draws": len(valid),
            "families": sorted(development["events"].totals["atomic_choice_macro"]), "pass": lower > 0}
        audit.equal(claims[name], report["mechanism_claims"][name], "independent binding claim: " + name)
    audit.equal(all(claims[name]["pass"] for name in CLAIMS), report["mechanism_claims"]["pass"], "claim conjunction")
    return {"schema": "vey.eca2.adaptation-independent-reconstruction.v2", "status": "PASS",
        "evidence_class": "MEASURED reconstruction of saved authored development evidence",
        "run_root": str(root), "script": {"path": str(Path(__file__).resolve()), "sha256": reference.digest(Path(__file__))},
        "checkpoint": history["checkpoint"], "selected_epoch": history["selected_epoch"],
        "epochs": 400, "counts": dict(audit.counts), "verified_hashes": audit.hashes,
        "development_absolute_gates": report["gate_screens"][ARM], "binding_claims": claims,
        "binding_claim_set_pass": all(claims[name]["pass"] for name in CLAIMS),
        "encoder_forwards": 0, "model_forwards": 0, "optimizer_updates": 0,
        "readout_refits": 0, "recalibration": False, "calibration_numeric_reconstruction_only": True,
        "final_pool_access": False, "promotion": False, "B_STEF_allowed": False, "endgame_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    result = verify(args.run_root)
    output = args.run_root / "fullfit_independent_reconstruction_v2.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
    print(json.dumps({"receipt": str(output), "sha256": reference.digest(output),
                      "status": result["status"], "binding_claim_set_pass": result["binding_claim_set_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
