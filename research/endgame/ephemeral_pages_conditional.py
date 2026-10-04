#!/usr/bin/env python3
"""ECA frozen-interface conditional-reader diagnostic; never opens ECA final."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ephemeral_pages_capture as capture
import ephemeral_pages_evaluate as evaluator
import ephemeral_pages_model as readers
import ephemeral_pages_train as trainer

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1")
RUN_ROOT = DATA_ROOT / "runs/interface-audit-v1/conditional"
ATOMIC_RUN_ROOT = capture.ATOMIC_ROOT / "runs/interface-audit-v1/conditional"
BASELINE_ROOT = DATA_ROOT / "runs/seed7-grade-corrected"
AUDIT_PATH = HERE / "ephemeral_pages_audit_protocol.json"
AUDIT_SHA256 = "7ff11075c9e8298100a29756354951017a71c85dff34193b63a811d7ac61c567"
PARENT_PROTOCOL_SHA256 = "4c0c1efe8ad8ffdde79004536a44fc6d034b7701cdedde8e2034dc3f4e52b489"
ALLOWED_PHASES = frozenset({"train", "validation", "calibration", "development"})
BASELINES = ("pages", "cross", "lexical")


def baseline_root(experiment):
    """Baseline checkpoints live beside the run they were fitted for."""
    return capture.ATOMIC_ROOT / "runs/seed7" if experiment is capture.ATOMIC_EXPERIMENT else BASELINE_ROOT

SCOPE = {
    "diagnostic": True, "eligible_ECA_arm": False, "promotion": False,
    "B_STEF_allowed": False, "endgame_complete": False,
    "encoder_forwards": 0, "final_outcomes_used": False,
    "certificate": "unavailable", "competitor_credit": False,
    "neutral_task_credit": False,
}


class ConditionalReader(nn.Module):
    """Independent q/page heads; direction is supervised but never multiplies grade."""

    def __init__(self):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(2 * readers.WIDTH, readers.CROSS_HIDDEN),
                                 nn.GELU(), nn.Linear(readers.CROSS_HIDDEN, 1))
        self.relevance_head = head()
        self.grade_head = head()
        self.direction_head = head()
        self.bk = nn.Parameter(torch.zeros(()))
        self.uk = nn.Parameter(torch.zeros(()))
        self.vk = nn.Parameter(torch.zeros(()))

    def forward(self, q, pages, page_mask, raw_q=None, raw_pages=None, *,
                uniform_attention=False):
        readers._check_pages(pages, page_mask, "pages")
        if q.ndim != 2 or q.shape != (pages.shape[0], readers.WIDTH):
            raise ValueError(f"q must have shape [N,{readers.WIDTH}]")
        if not q.is_floating_point():
            raise TypeError("q must be floating point")
        joint = torch.cat((q.unsqueeze(1).expand(-1, pages.shape[1], -1), pages), dim=-1)
        relevance = self.relevance_head(joint).squeeze(-1)
        grade = torch.sigmoid(self.grade_head(joint).squeeze(-1))
        direction = torch.tanh(self.direction_head(joint).squeeze(-1))
        return readers._output(relevance, direction, grade, page_mask,
                               self.bk, self.uk, self.vk, uniform_attention,
                               direct_grade=True)


def _hash_json(value):
    return hashlib.sha256(json.dumps(evaluator.plain(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _authorize(experiment=capture.DEFAULT_EXPERIMENT):
    if trainer._hash_file(AUDIT_PATH) != AUDIT_SHA256:
        raise RuntimeError("frozen interface-audit protocol changed")
    cfg, protocol_hash = experiment.protocol()
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit["parent_protocol_sha256"] not in {PARENT_PROTOCOL_SHA256, protocol_hash}:
        raise RuntimeError("audit parent protocol is neither the frozen ECA-1 nor the active protocol")
    return cfg


def _source_hashes():
    return {name: trainer._hash_file(HERE / name) for name in (
        Path(__file__).name, "ephemeral_pages_train.py", "ephemeral_pages_model.py",
        "ephemeral_pages_evaluate.py", "ephemeral_pages_capture.py",
        "ephemeral_pages_features.py", "ephemeral_pages_protocol.json",
        "ephemeral_pages_audit_protocol.json", "ephemeral_pages_grade_amendment.json")}


def _environment():
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "variables": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}


def _baseline_checkpoints(experiment=capture.DEFAULT_EXPERIMENT):
    protocol_hash = experiment.protocol()[1]
    root = baseline_root(experiment)
    calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
    if calibration["protocol_sha256"] != protocol_hash:
        raise RuntimeError("baseline calibration protocol mismatch")
    result = {}
    for name in trainer.CONTROLS:
        path = root / ("lexical.pkl" if name == "lexical" else f"{name}.pt")
        entry = evaluator.artifact(path)
        if entry != calibration["checkpoint_files"][name]:
            raise RuntimeError(f"immutable baseline checkpoint changed: {name}")
        result[name] = entry
    return result


def _safe_root(run_root, smoke=False, experiment=capture.DEFAULT_EXPERIMENT):
    root = Path(run_root).resolve()
    if smoke:
        root = root / "smoke"
    protected = (baseline_root(experiment), experiment.corpus_root, experiment.cache_root)
    if any(root == path.resolve() or path.resolve() in root.parents for path in protected):
        raise ValueError("audit output cannot modify baseline or captured data")
    return root


def load_phase(phase, experiment=capture.DEFAULT_EXPERIMENT):
    if phase not in ALLOWED_PHASES:
        raise ValueError("conditional audit permits only train/validation/calibration/development")
    data = capture.load_phase(phase, experiment)
    if any(record["split"] != phase for record in data["records"]):
        raise RuntimeError("captured record split mismatch")
    return data


def leaf_truth_guard(data, phase, experiment=capture.DEFAULT_EXPERIMENT):
    """Reject parent-wide valid-leaf masks before any optimizer sees the labels.

    Child-local supervision is re-derived here from the authored DecisionIR, not
    trusted from the capture. A disagreement blocks the fit outright; it is
    never silently repaired.
    """
    import ephemeral_pages_components as components
    from ephemeral_pages_features import _parse_meminfo
    ir = {}
    for row in capture.rows_for(phase, experiment):
        if row["split"] != phase or row["id"] in ir:
            raise RuntimeError("DecisionIR phase/identity mismatch")
        ir[row["id"]] = row
    if {record["row_id"] for record in data["records"]} != set(ir):
        raise RuntimeError("capture/IR coverage mismatch")
    receipt = components.verify_truth(phase, data, ir, components.source_catalogue(experiment),
                                      _parse_meminfo()[1])
    data["_leaf_truth_verified"] = True
    return receipt


def _phase_evidence(phase, data, indices=None, experiment=capture.DEFAULT_EXPERIMENT):
    if phase not in ALLOWED_PHASES:
        raise ValueError("forbidden audit phase")
    ix = np.arange(len(data["records"])) if indices is None else np.asarray(indices)
    records = [data["records"][int(i)] for i in ix]
    return {"phase": phase,
            "feature_manifest": evaluator.artifact(experiment.cache_root / f"{phase}_manifest.json"),
            "corpus": evaluator.artifact(experiment.corpus_root / f"{phase}.jsonl"),
            "feature_lineage": data["lineage"], "capture_counters": data["counters"],
            "capture_token_receipts": data["token_receipts"],
            "selected_indices": ix.tolist(),
            "selected_row_ids": sorted({record["row_id"] for record in records}),
            "selected_records_sha256": _hash_json(records),
            "selected_indices_sha256": _hash_json(ix.tolist())}




def objective(model, normalizer, arrays, indices, denominators, chunk_size=32,
              baseline=None, backward=False):
    """Accumulate exact full-batch component means before a single optimizer step."""
    if baseline is None:
        baseline = trainer._parse_meminfo()[1]
    totals = dict.fromkeys(trainer.COMPONENTS, 0.0)
    device = next(model.parameters()).device
    for ix in trainer._chunks(indices, chunk_size):
        trainer._resource_guard(torch, baseline)
        targets = trainer._batch(arrays, ix, device)
        output = trainer._forward("conditional", model, normalizer, arrays, ix)
        losses = readers.eca_loss(output, targets, directed_grade=True)
        counts = trainer._counts(targets, "cross")
        weighted = []
        for key in trainer.COMPONENTS:
            if counts[key] and denominators[key]:
                term = losses[key] * (counts[key] / denominators[key])
                if not torch.isfinite(term):
                    raise FloatingPointError(f"nonfinite conditional {key} objective")
                totals[key] += float(term.detach())
                weighted.append(term)
        if backward and weighted:
            sum(weighted).backward()
    totals["total"] = sum(totals.values())
    return totals


def train_conditional(run_root=RUN_ROOT, device="cuda", chunk_size=32, smoke=False,
                      smoke_records=64, experiment=capture.DEFAULT_EXPERIMENT):
    protocol_hash = experiment.protocol()[1]
    _authorize(experiment)
    if chunk_size < 1 or smoke_records < 1:
        raise ValueError("chunk and smoke record counts must be positive")
    root = _safe_root(run_root, smoke, experiment)
    root.mkdir(parents=True, exist_ok=False)
    baseline_files = _baseline_checkpoints(experiment)
    train, validation = load_phase("train", experiment), load_phase("validation", experiment)
    # The optimizer must never see parent-wide valid-leaf masks: every supervised
    # label is re-derived from the owned active-property pages first.
    truth = leaf_truth_guard(train, "train", experiment)
    val_truth = leaf_truth_guard(validation, "validation", experiment)
    all_train_ix = trainer.optimizer_indices(train, "train", experiment)
    train_ix = all_train_ix[:smoke_records] if smoke else all_train_ix
    val_ix = trainer.optimizer_indices(validation, "validation", experiment)
    if smoke:
        val_ix = val_ix[:smoke_records]
    with trainer._resources(device) as (baseline, priority):
        trainer._seed()
        # Even a reduced smoke uses the immutable common full-fitting normalizer.
        normalizer = trainer.fit_normalizer(train, all_train_ix, chunk_size)
        baseline_checkpoint = torch.load(baseline_root(experiment) / "pages.pt", map_location="cpu", weights_only=False)
        for key in ("q", "pages", "cross"):
            expected = baseline_checkpoint["normalizer"][key]
            if (normalizer[key]["count"] != expected["count"] or
                    not np.array_equal(normalizer[key]["mean"], expected["mean"]) or
                    not np.array_equal(normalizer[key]["std"], expected["std"])):
                raise RuntimeError("conditional normalizer differs from common baseline train normalizer")
        del baseline_checkpoint
        model = ConditionalReader().to(device)
        initial_hash = trainer._parameter_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.0001)
        train_den = trainer._denominators(train, train_ix, "cross", chunk_size)
        val_den = trainer._denominators(validation, val_ix, "cross", chunk_size)
        environment = _environment()
        sources = _source_hashes()
        evidence = {"train": _phase_evidence("train", train, train_ix, experiment),
                    "validation": _phase_evidence("validation", validation, val_ix, experiment),
                    "leaf_truth_guard": {"train": truth, "validation": val_truth},
                    "normalizer_train_indices_sha256": _hash_json(all_train_ix.tolist())}
        evaluator.write_json(root / "launch_receipt.json", {
            "schema": "vey.eca.conditional-launch.v1", **SCOPE,
            "source_sha256": sources, "environment": environment,
            "row_evidence": evidence, "baseline_checkpoint_files": baseline_files,
            "normalizer_sha256": _hash_json(normalizer), "seed": 7,
            "epochs": 2 if smoke else 400, "smoke": smoke,
            "optimizer": "AdamW", "lr": .01, "weight_decay": .0001,
            "steps_per_epoch": 1, "chunk_size": chunk_size,
            "component_denominators": {"train": train_den, "validation": val_den},
            "experiment_context": experiment.context()})
        history = []
        best = float("inf")
        selected = None
        epochs = 2 if smoke else 400
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = objective(model, normalizer, train, train_ix, train_den,
                                   chunk_size, baseline, backward=True)
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                val_loss = objective(model, normalizer, validation, val_ix, val_den,
                                     chunk_size, baseline)
            history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
            if val_loss["total"] < best:
                best = val_loss["total"]
                selected = {"state_dict": {key: value.detach().cpu().clone()
                                            for key, value in model.state_dict().items()},
                            "epoch": epoch, "parameter_sha256": trainer._parameter_hash(model)}
        if selected is None:
            raise RuntimeError("no finite conditional checkpoint selected")
        checkpoint = {**selected, "control": "conditional", "normalizer": normalizer,
                      "protocol_sha256": protocol_hash,
                      "audit_protocol_sha256": AUDIT_SHA256, "smoke": smoke,
                      "source_sha256": sources, "environment": environment,
                      "row_evidence": evidence, "baseline_checkpoint_files": baseline_files,
                      "scope": SCOPE}
        with (root / "conditional.pt").open("xb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        record = {"schema": "vey.eca.conditional-training.v1", **SCOPE,
                  "control": "conditional", "protocol_sha256": protocol_hash,
                  "audit_protocol_sha256": AUDIT_SHA256, "smoke": smoke,
                  "seed": 7, "epochs": epochs, "optimizer": "AdamW", "lr": .01,
                  "weight_decay": .0001, "steps_per_epoch": 1, "chunk_size": chunk_size,
                  "component_denominators": {"train": train_den, "validation": val_den},
                  "selection": "earliest minimum held-world validation objective",
                  "selected_epoch": selected["epoch"], "selected_validation_objective": best,
                  "initial_parameter_sha256": initial_hash,
                  "selected_parameter_sha256": selected["parameter_sha256"],
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "source_sha256": sources, "environment": environment,
                  "environment_sha256": _hash_json(environment), "resources": priority,
                  "row_evidence": evidence, "baseline_checkpoint_files": baseline_files,
                  "normalizer_sha256": _hash_json(normalizer), "history": history,
                  "checkpoint": evaluator.artifact(root / "conditional.pt")}
        evaluator.write_json(root / "conditional_history.json", record)
        if _baseline_checkpoints(experiment) != baseline_files:
            raise RuntimeError("baseline changed during conditional fitting")
    return root


def load_conditional(run_root=RUN_ROOT, device="cpu", allow_smoke=False,
                     experiment=capture.DEFAULT_EXPERIMENT):
    _authorize(experiment)
    root = _safe_root(run_root, experiment=experiment)
    history = json.loads((root / "conditional_history.json").read_text(encoding="utf-8"))
    if evaluator.artifact(root / "conditional.pt") != history["checkpoint"]:
        raise RuntimeError("conditional checkpoint changed")
    checkpoint = torch.load(root / "conditional.pt", map_location="cpu", weights_only=False)
    if (checkpoint["control"] != "conditional" or checkpoint["protocol_sha256"] != experiment.protocol()[1] or
            checkpoint["audit_protocol_sha256"] != AUDIT_SHA256):
        raise RuntimeError("conditional checkpoint protocol mismatch")
    if checkpoint["smoke"] and not allow_smoke:
        raise RuntimeError("smoke checkpoint cannot enter calibration or development")
    if checkpoint["source_sha256"] != _source_hashes():
        raise RuntimeError("conditional implementation changed after fitting")
    if checkpoint["baseline_checkpoint_files"] != _baseline_checkpoints(experiment):
        raise RuntimeError("baseline checkpoints changed after conditional fitting")
    model = ConditionalReader()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if trainer._parameter_hash(model) != checkpoint["parameter_sha256"]:
        raise RuntimeError("conditional parameter hash mismatch")
    return model.to(device).eval(), checkpoint["normalizer"]


def predict_conditional(model, normalizer, data, indices=None, chunk_size=32):
    if any(record["split"] not in ALLOWED_PHASES for record in data["records"]):
        raise ValueError("forbidden conditional prediction split")
    return trainer.predict_control("conditional", model, normalizer, data,
                                   indices=indices, chunk_size=chunk_size)


def _verified_baseline_artifact(entry, name, experiment=capture.DEFAULT_EXPERIMENT):
    path = baseline_root(experiment) / "evaluation/development" / name
    actual = evaluator.artifact(path)
    if actual != entry:
        raise RuntimeError(f"immutable baseline artifact changed: {name}")
    return path


def _saved_baseline(name, data, ir, worlds, families, baseline_manifest, calibration,
                    experiment=capture.DEFAULT_EXPERIMENT):
    """Reconstruct only persisted development outputs, never run a baseline model."""
    entries = baseline_manifest["artifacts"][name]
    raw_path = _verified_baseline_artifact(entries["term_candidate"], f"{name}_term_candidate.jsonl", experiment)
    decisions_path = _verified_baseline_artifact(entries["decisions"], f"{name}_decisions.jsonl", experiment)
    ledger_path = _verified_baseline_artifact(entries["world_metric_ledger"], f"{name}_world_metrics.json", experiment)
    n, pmax = data["page_mask"].shape
    fields = {key: np.full((n,) if key in {"score", "known_logits"} else (n, pmax),
                           np.nan if key == "score" else -np.inf if key in {"known_logits", "relevance_logits"} else 0,
                           dtype=np.float32) for key in readers.ReaderOutput._fields}
    count = 0
    with raw_path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            if i >= n:
                raise RuntimeError("baseline raw prediction has extra records")
            row = json.loads(line)
            record = data["records"][i]
            if row["record_index"] != i or any(row[key] != value for key, value in record.items()):
                raise RuntimeError("baseline raw record identity or supervision differs")
            width = len(record["pages"])
            for key in readers.ReaderOutput._fields:
                value = row["prediction"][key]
                if key in {"score", "known_logits"}:
                    fields[key][i] = (np.nan if key == "score" else -np.inf) if value is None else value
                else:
                    if len(value) != width:
                        raise RuntimeError("baseline raw prediction page count differs")
                    fields[key][i, :width] = [(-np.inf if key == "relevance_logits" else np.nan)
                                              if x is None else x for x in value]
            count += 1
    if count != n:
        raise RuntimeError("baseline raw prediction is incomplete")
    fields = evaluator.prediction_fields(readers.ReaderOutput(*(fields[key] for key in readers.ReaderOutput._fields)), data)
    with decisions_path.open(encoding="utf-8") as stream:
        saved_decisions = [json.loads(line) for line in stream]
    reconstructed = evaluator.aggregate(fields, data, ir, calibration)
    if evaluator.plain(reconstructed) != saved_decisions:
        raise RuntimeError("saved baseline typed decisions do not reconstruct exactly")
    metrics, _ = evaluator.metric_ledger(name, fields, data, ir, reconstructed,
                                        calibration["ordinal_sigma"], families, worlds)
    saved_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if metrics.ledger() != saved_ledger:
        raise RuntimeError("saved baseline world ledger does not reconstruct exactly")
    return metrics, {"raw_predictions": evaluator.artifact(raw_path),
                     "decisions": evaluator.artifact(decisions_path),
                     "world_metric_ledger": evaluator.artifact(ledger_path),
                     "identity_reconstruction": True, "baseline_model_runs": 0}


def paired_deltas(conditional, baselines, weights):
    result = {}
    for name, baseline in baselines.items():
        metrics = {}
        for metric in sorted(set(conditional.values) & set(baseline.values)):
            left, right = conditional.values[metric], baseline.values[metric]
            if set(left) != set(right):
                raise RuntimeError(f"paired metric family population differs: {name}/{metric}")
            if metric in {"atomic_choice_macro", "concrete_winner_choice_macro"} and any(
                    not np.array_equal(left[family][:, 1], right[family][:, 1]) for family in left):
                raise RuntimeError(f"paired choice denominator differs: {name}/{metric}")
            samples = conditional.estimate(metric, weights) - baseline.estimate(metric, weights)
            valid = samples[np.isfinite(samples)]
            metrics[metric] = {"point": float(conditional.estimate(metric) - baseline.estimate(metric)),
                               "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                               "valid_bootstrap_draws": len(valid),
                               "families": sorted(left)}
        for required in ("atomic_choice_macro", "concrete_winner_choice_macro"):
            if required not in metrics:
                raise RuntimeError(f"missing paired choice metric: {required}")
        result["conditional_minus_" + name] = metrics
    return result


def evaluate_conditional(run_root=RUN_ROOT, device="cuda", chunk_size=32,
                        experiment=capture.DEFAULT_EXPERIMENT):
    protocol_hash = experiment.protocol()[1]
    cfg = _authorize(experiment)
    root = _safe_root(run_root, experiment=experiment)
    if any((root / path).exists() for path in ("calibration.json", "evaluation")):
        raise FileExistsError("refusing to replace conditional evaluation artifacts")
    baseline_files = _baseline_checkpoints(experiment)
    baseline_calibration_path = baseline_root(experiment) / "calibration.json"
    baseline_calibration = json.loads(baseline_calibration_path.read_text(encoding="utf-8"))
    baseline_manifest_path = baseline_root(experiment) / "evaluation/development/evaluation.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if (baseline_manifest["phase"] != "development" or
            baseline_manifest["protocol_sha256"] != protocol_hash or
            baseline_manifest["checkpoint_files"] != baseline_files):
        raise RuntimeError("baseline development manifest mismatch")
    with trainer._resources(device) as (baseline, priority):
        model, normalizer = load_conditional(root, device, experiment=experiment)
        calibration = None
        reports = {}
        for phase in ("calibration", "development"):
            trainer._resource_guard(torch, baseline)
            data = load_phase(phase, experiment)
            ir = evaluator.load_ir(phase, experiment=experiment)
            if {record["row_id"] for record in data["records"]} != set(ir):
                raise RuntimeError("conditional capture and DecisionIR populations differ")
            worlds = sorted({record["world_id"] for record in data["records"]})
            families = cfg["corpus"]["families"]
            expected_families = families[:12 if phase == "calibration" else 16]
            observed = {family for row in ir.values() if row["metadata"]["variant"] == "base"
                        and row["metadata"]["query_kind"] == "atomic" and evaluator.UNKNOWN not in row["gold"]
                        for family in row["metadata"]["family"]}
            if len(worlds) != cfg["corpus"]["worlds"][phase] or observed != set(expected_families):
                raise RuntimeError("conditional evaluation changed frozen world/family populations")
            directory = root / "evaluation" / phase
            directory.mkdir(parents=True, exist_ok=False)
            output = predict_conditional(model, normalizer, data, chunk_size=chunk_size)
            fields = evaluator.prediction_fields(output, data)
            if phase == "calibration":
                calibration = evaluator.calibrate(fields, data, ir)
            decisions = evaluator.aggregate(fields, data, ir, calibration)
            source = {"control": "conditional", **SCOPE,
                      "checkpoint": evaluator.artifact(root / "conditional.pt"),
                      "protocol_sha256": protocol_hash,
                      "audit_protocol_sha256": AUDIT_SHA256,
                      "source_sha256": _source_hashes(), "phase": phase,
                      "grade_semantics": "directed; direction diagnostic only"}
            for row in decisions:
                row["source"] = "conditional_reader_diagnostic"
                row["source_metadata"] = source
            annotated = {**data, "records": [{**record, "source_metadata": source} for record in data["records"]]}
            saved = evaluator.save_predictions(directory, "conditional", fields, annotated, ir,
                                               decisions, calibration["ordinal_sigma"])
            # Preserve every padded ReaderOutput value byte-for-byte as well as the JSON projection.
            with (directory / "conditional_reader_output.npz").open("xb") as stream:
                np.savez(stream, **{key: np.asarray(getattr(output, key)) for key in readers.ReaderOutput._fields})
                stream.flush()
                os.fsync(stream.fileno())
            saved["reader_output"] = evaluator.artifact(directory / "conditional_reader_output.npz")
            metrics, pairs = evaluator.metric_ledger("cross", fields, data, ir, decisions,
                                                    calibration["ordinal_sigma"], families, worlds)
            weights, bootstrap = evaluator.bootstrap_indices(directory, worlds)
            summaries = metrics.summaries(weights)
            for prefix in ("raw_distribution_", "calibrated_distribution_", "OOD_raw_distribution_", "OOD_calibrated_distribution_"):
                summaries[prefix + "ECE"] = evaluator.ece_summary(metrics, weights, prefix)
            saved["world_metric_ledger"] = evaluator.write_json(directory / "conditional_world_metrics.json", metrics.ledger())
            with (directory / "conditional_paired_interventions.jsonl").open("x", encoding="utf-8") as stream:
                for pair in pairs:
                    stream.write(json.dumps(evaluator.plain(pair), sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            saved["paired_interventions"] = evaluator.artifact(directory / "conditional_paired_interventions.jsonl")
            comparisons, baseline_evidence = None, None
            if phase == "development":
                baseline_ledgers, baseline_evidence = {}, {}
                for name in BASELINES:
                    baseline_ledgers[name], baseline_evidence[name] = _saved_baseline(
                        name, data, ir, worlds, families, baseline_manifest,
                        baseline_calibration["controls"][name], experiment)
                comparisons = paired_deltas(metrics, baseline_ledgers, weights)
            report = {"schema": "vey.eca.conditional-evaluation.v1", **SCOPE,
                      "phase": phase, "source_metadata": source, "calibration": calibration,
                      "metrics": summaries, "paired_deltas": comparisons,
                      "artifacts": saved, "bootstrap": bootstrap,
                      "baseline_saved_output_evidence": baseline_evidence,
                      "baseline_checkpoint_files": baseline_files,
                      "baseline_development_manifest": evaluator.artifact(baseline_manifest_path),
                      "baseline_calibration": evaluator.artifact(baseline_calibration_path),
                      "phase_evidence": _phase_evidence(phase, data, experiment=experiment),
                      "environment": _environment(), "resources": priority,
                      "statistical_scope": "descriptive world-cluster paired differences, conditioned on fixed authored inventory; no promotion inference",
                      "primary": "exact maximal-set equality; concrete chosen-in-gold separate",
                      "direct_grade_metric_mode": "cross",
                      "strict_certificate_credit": False}
            reports[phase] = evaluator.write_json(directory / "evaluation.json", report)
            if phase == "calibration":
                evaluator.write_json(root / "calibration.json", {
                    "schema": "vey.eca.conditional-calibration.v1", **SCOPE,
                    "protocol_sha256": protocol_hash,
                    "audit_protocol_sha256": AUDIT_SHA256,
                    "control": "conditional", "calibration": calibration,
                    "checkpoint": evaluator.artifact(root / "conditional.pt"),
                    "evaluation": reports[phase]})
            del output, fields, data, annotated
        if _baseline_checkpoints(experiment) != baseline_files:
            raise RuntimeError("baseline changed during conditional evaluation")
        return evaluator.write_json(root / "evaluation_manifest.json", {
            "schema": "vey.eca.conditional-audit.v1", **SCOPE,
            "protocol_sha256": protocol_hash,
            "audit_protocol_sha256": AUDIT_SHA256, "reports": reports,
            "source_sha256": _source_hashes(),
            "checkpoint": evaluator.artifact(root / "conditional.pt"),
            "training_history": evaluator.artifact(root / "conditional_history.json")})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atomic", action="store_true",
                        help="use the ECA-2 child-local experiment, its data root and its baseline run")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-records", type=int, default=64)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size < 1 or args.smoke_records < 1:
        parser.error("chunk size and smoke record count must be positive")
    if args.smoke and args.evaluate_only:
        parser.error("smoke checkpoints are not eligible for evaluation")
    experiment = capture.resolve_experiment(args.atomic)
    run_root = (args.run_root or (ATOMIC_RUN_ROOT if args.atomic else RUN_ROOT)).resolve()
    root = _safe_root(run_root, args.smoke, experiment)
    if not args.evaluate_only:
        root = train_conditional(run_root, args.device, args.chunk_size,
                                 args.smoke, args.smoke_records, experiment)
    if not args.smoke:
        evaluate_conditional(root, args.device, args.chunk_size, experiment)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
