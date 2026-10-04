#!/usr/bin/env python3
"""Frozen ECA-1 cached-feature training and record-ordered inference."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ephemeral_pages_capture import load_phase, rows_for
from ephemeral_pages_features import (GPU_LOCK, PROTOCOL_PATH, PROTOCOL_SHA256,
    _apply_process_priority, _parse_meminfo, _resource_guard)
from ephemeral_pages_model import (PageReader, CosineReader, CrossReader,
    QueryBlindReader, LexicalReader, ReaderOutput, eca_loss, intervene_features)

RUN_ROOT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1/runs/seed7")
CONTROLS = ("pages", "cross", "cosine", "lexical", "query_blind")
COMPONENTS = ("relevance", "grade", "orientation", "known")
TARGETS = ("page_mask", "relevance_target", "grade_target", "directed_grade_target",
           "grade_mask", "orientation_target", "orientation_mask", "known_target")
FACTORIES = {"pages": PageReader, "cross": CrossReader, "cosine": CosineReader,
             "query_blind": QueryBlindReader}


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def _resources(device):
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    with GPU_LOCK.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            baseline = _parse_meminfo()[1]
            priority = _apply_process_priority()
            _resource_guard(torch, baseline)
            yield baseline, priority
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def optimizer_indices(arrays, split):
    """Select via same-split original IR, never via diagnostic record duplication."""
    if split not in {"train", "validation"}:
        raise ValueError("optimizer selection only permits train/validation")
    allowed = set()
    for row in rows_for(split):
        if row["split"] != split:
            raise ValueError("IR split mismatch")
        meta = row["metadata"]
        if (meta.get("candidate_count") == 4 and
                meta.get("variant") in {"base", "relevant_page_erasure", "relevant_page_contradiction"}):
            allowed.add(row["id"])
    indices = np.asarray([i for i, r in enumerate(arrays["records"])
                          if r["split"] == split and r["row_id"] in allowed], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"no eligible {split} optimizer records")
    return indices


def _chunks(indices, size):
    if size < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(indices), size):
        yield indices[start:start + size]


def fit_normalizer(arrays, indices, chunk_size=32):
    # Stream the existing FeatureNormalizer policy: joint q/page statistics,
    # and independent question/page-pair statistics for the cross control.
    result = {}
    for mode, keys in (("pages", ("q", "pages")), ("cross", ("cross",))):
        total = np.zeros(384, np.float64)
        squares = np.zeros(384, np.float64)
        count = 0
        for ix in _chunks(indices, chunk_size):
            mask = np.asarray(arrays["page_mask"][ix], dtype=bool)
            for key in keys:
                values = np.asarray(arrays[key][ix], dtype=np.float64)
                if key != "q":
                    values = values[mask]
                total += values.sum(axis=0)
                squares += np.square(values).sum(axis=0)
                count += len(values)
        if count == 0:
            raise ValueError(f"no training features for {mode}")
        mean = total / count
        std = np.sqrt(np.maximum(squares / count - mean * mean, 0)).clip(min=.01)
        result[mode] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "count": count}
    result["q"] = result["pages"]
    return result


def _batch(arrays, indices, device):
    return {key: torch.as_tensor(np.array(arrays[key][indices], copy=True), device=device)
            for key in TARGETS}


def _forward(control, model, normalizer, arrays, ix, intervention=None):
    device = next(model.parameters()).device
    mask = torch.as_tensor(np.array(arrays["page_mask"][ix], copy=True), device=device)
    def feature(key):
        raw = torch.as_tensor(np.array(arrays[key][ix], copy=True), device=device)
        stats = normalizer[key]
        mean = torch.as_tensor(stats["mean"], device=device)
        std = torch.as_tensor(stats["std"], device=device)
        return (raw - mean) / std, raw
    if intervention not in {None, "zero_question", "zero_pages", "uniform_attention"}:
        raise ValueError(f"unknown intervention {intervention!r}")
    uniform = intervention == "uniform_attention"
    if control == "cross":
        if intervention in {"zero_question", "zero_pages"}:
            raise ValueError("joint cross features cannot isolate question/page interventions")
        pair, _ = feature("cross")
        return model(pair, mask, uniform_attention=uniform)
    pages, raw_pages = feature("pages")
    if control == "query_blind":
        if intervention == "zero_pages":
            pages = torch.zeros_like(pages)
        return model(pages, mask)
    q, raw_q = feature("q")
    q, pages, mask, raw_q, raw_pages, uniform = intervene_features(
        q, pages, mask, raw_q, raw_pages, zero_question=intervention == "zero_question",
        zero_pages=intervention == "zero_pages", uniform_attention=uniform)
    return model(q, pages, mask, raw_q, raw_pages, uniform_attention=uniform)


def _counts(targets, control):
    mask = targets["page_mask"].bool()
    rel = targets["relevance_target"]
    mass = torch.where(mask & torch.isfinite(rel) & (rel >= 0), rel, 0).sum(-1)
    grade = targets["directed_grade_target" if control == "cross" else "grade_target"]
    return {
        "relevance": int(((mass > 0) & mask.any(-1)).sum()),
        "grade": int((mask & targets["grade_mask"].bool() & torch.isfinite(grade)).sum()),
        "orientation": int((mask & targets["orientation_mask"].bool() &
                            torch.isfinite(targets["orientation_target"])).sum()),
        "known": int(torch.isfinite(targets["known_target"]).sum()),
    }


def _denominators(arrays, indices, control, chunk_size):
    result = dict.fromkeys(COMPONENTS, 0)
    for ix in _chunks(indices, chunk_size):
        counts = _counts(_batch(arrays, ix, "cpu"), control)
        for key in COMPONENTS:
            result[key] += counts[key]
    return result


def _objective(control, model, normalizer, arrays, indices, denominators,
               chunk_size, baseline, backward=False):
    totals = dict.fromkeys(COMPONENTS, 0.0)
    device = next(model.parameters()).device
    for ix in _chunks(indices, chunk_size):
        _resource_guard(torch, baseline)
        targets = _batch(arrays, ix, device)
        output = _forward(control, model, normalizer, arrays, ix)
        losses = eca_loss(output, targets, directed_grade=control == "cross")
        counts = _counts(targets, control)
        weighted = []
        for key in COMPONENTS:
            if counts[key] and denominators[key]:
                term = losses[key] * (counts[key] / denominators[key])
                if not torch.isfinite(term):
                    raise FloatingPointError(f"nonfinite {control} {key} objective")
                totals[key] += float(term.detach())
                weighted.append(term)
        if backward and weighted:
            sum(weighted).backward()
    totals["total"] = sum(totals.values())
    return totals


def _parameter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_control(control, run_root, device="cpu"):
    root = Path(run_root)
    if control == "lexical":
        return LexicalReader.load(str(root / "lexical.pkl")), None
    if control not in FACTORIES:
        raise ValueError(f"unknown control {control!r}")
    checkpoint = torch.load(root / f"{control}.pt", map_location="cpu", weights_only=False)
    if checkpoint["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("checkpoint protocol mismatch")
    model = FACTORIES[control]()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint["normalizer"]


def predict_control(control, model, normalizer, arrays, indices=None,
                    intervention=None, chunk_size=32):
    ix = np.arange(len(arrays["records"]), dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    pmax = arrays["page_mask"].shape[1]
    fields = [[] for _ in ReaderOutput._fields]
    if control == "lexical" and intervention is not None:
        raise ValueError("feature interventions apply to neural controls")
    with torch.inference_mode():
        for chunk in _chunks(ix, chunk_size):
            output = (model.predict_records([arrays["records"][int(i)] for i in chunk])
                      if control == "lexical" else
                      _forward(control, model, normalizer, arrays, chunk, intervention))
            for position, value in enumerate(output):
                data = value.detach().cpu().numpy()
                if data.ndim == 2 and data.shape[1] < pmax:
                    fill = -np.inf if position == 2 else 0
                    data = np.pad(data, ((0, 0), (0, pmax - data.shape[1])), constant_values=fill)
                fields[position].append(data)
    return ReaderOutput(*(np.concatenate(values, axis=0) if values else
                          np.empty((0,) if i < 2 else (0, pmax), dtype=np.float32)
                          for i, values in enumerate(fields)))


def train_controls(run_root=RUN_ROOT, device="cuda", controls=CONTROLS,
                   chunk_size=32, smoke=False, smoke_epochs=2, smoke_records=64):
    if _hash_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol changed")
    if any(control not in CONTROLS for control in controls):
        raise ValueError("unknown control")
    root = Path(run_root)
    if smoke:
        root = root / "smoke"
    root.mkdir(parents=True, exist_ok=True)
    if any((root / ("lexical.pkl" if c == "lexical" else f"{c}.pt")).exists() or
           (root / f"{c}_history.json").exists() for c in controls):
        raise FileExistsError("refusing to replace experiment artifacts")
    train = load_phase("train")
    validation = load_phase("validation")
    train_ix = optimizer_indices(train, "train")
    val_ix = optimizer_indices(validation, "validation")
    if smoke:
        train_ix = train_ix[:smoke_records]
        val_ix = val_ix[:smoke_records]
    epochs = smoke_epochs if smoke else 400
    if epochs < 1 or smoke_records < 1:
        raise ValueError("smoke dimensions must be positive")
    with _resources(device) as (baseline, priority):
        normalizer = fit_normalizer(train, train_ix, chunk_size)
        lineage = {
            "protocol_sha256": PROTOCOL_SHA256, "smoke": smoke, "seed": 7,
            "epochs": epochs, "chunk_size": chunk_size, "optimizer": "AdamW",
            "lr": .01, "weight_decay": .0001, "encoder_forwards": 0,
            "train_feature_lineage": train["lineage"],
            "validation_feature_lineage": validation["lineage"],
            "capture_counters": {"train": train["counters"], "validation": validation["counters"]},
            "selected_indices": {"train": train_ix.tolist(), "validation": val_ix.tolist()},
            "selected_row_ids": {"train": sorted({train["records"][int(i)]["row_id"] for i in train_ix}),
                                 "validation": sorted({validation["records"][int(i)]["row_id"] for i in val_ix})},
            "source_sha256": {Path(__file__).name: _hash_file(__file__),
                              "ephemeral_pages_model.py": _hash_file(Path(__file__).with_name("ephemeral_pages_model.py"))},
            "resources": priority,
        }
        for control in controls:
            _seed()
            history = {**lineage, "control": control, "history": []}
            if control == "lexical":
                model = LexicalReader().fit([train["records"][int(i)] for i in train_ix])
                model.save(str(root / "lexical.pkl"))
                history.update(regularization=1.0, checkpoint_sha256=_hash_file(root / "lexical.pkl"),
                               vocabulary_sizes={name: len(vec.vocabulary_) for name, vec in model.vectorizer.transformer_list})
                # Lexical fits have no epoch/checkpoint search, but retain identical
                # component-wise train/validation objective reporting.
                for split, arrays, indices in (("train", train, train_ix), ("validation", validation, val_ix)):
                    denominators = _denominators(arrays, indices, control, chunk_size)
                    totals = dict.fromkeys(COMPONENTS, 0.0)
                    for ix in _chunks(indices, chunk_size):
                        _resource_guard(torch, baseline)
                        targets = _batch(arrays, ix, "cpu")
                        out = model.predict_records([arrays["records"][int(i)] for i in ix])
                        width = targets["page_mask"].shape[1]
                        out = ReaderOutput(*(F.pad(v, (0, width-v.shape[1]), value=-float("inf") if j == 2 else 0)
                                             if v.ndim == 2 else v for j, v in enumerate(out)))
                        loss = eca_loss(out, targets)
                        counts = _counts(targets, control)
                        for key in COMPONENTS:
                            if denominators[key] and counts[key]:
                                totals[key] += float(loss[key]) * counts[key] / denominators[key]
                    totals["total"] = sum(totals.values())
                    history[split] = totals
                del model
            else:
                model = FACTORIES[control]().to(device)
                history["initial_parameter_sha256"] = _parameter_hash(model)
                history["parameter_count"] = sum(p.numel() for p in model.parameters())
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.0001)
                train_den = _denominators(train, train_ix, control, chunk_size)
                val_den = _denominators(validation, val_ix, control, chunk_size)
                history["component_denominators"] = {"train": train_den, "validation": val_den}
                best = float("inf")
                for epoch in range(1, epochs + 1):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    train_loss = _objective(control, model, normalizer, train, train_ix, train_den, chunk_size, baseline, backward=True)
                    optimizer.step()
                    model.eval()
                    with torch.inference_mode():
                        val_loss = _objective(control, model, normalizer, validation, val_ix, val_den, chunk_size, baseline)
                    history["history"].append({"epoch": epoch, "train": train_loss, "validation": val_loss})
                    if val_loss["total"] < best:
                        best = val_loss["total"]
                        history["selected_epoch"] = epoch
                        history["selected_validation_objective"] = best
                        history["selected_parameter_sha256"] = _parameter_hash(model)
                        torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                                    "normalizer": normalizer, "epoch": epoch,
                                    "protocol_sha256": PROTOCOL_SHA256, "smoke": smoke,
                                    "control": control}, root / f"{control}.pt")
                history["checkpoint_sha256"] = _hash_file(root / f"{control}.pt")
                del optimizer, model
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            with (root / f"{control}_history.json").open("x", encoding="utf-8") as stream:
                json.dump(history, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--controls", nargs="+", choices=CONTROLS, default=list(CONTROLS))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true", help="Reduced diagnostic only; writes run-root/smoke, never actual artifacts")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-records", type=int, default=64)
    args = parser.parse_args()
    root = train_controls(**vars(args))
    print(json.dumps({"run_root": str(root), "smoke": args.smoke, "controls": args.controls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
