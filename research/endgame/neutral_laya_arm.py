#!/usr/bin/env python3
"""Laya arm for the neutral development comparison.

Loads the PINNED English bundle through the bundle's own runtime code
(rl_common.build_sequence / DecisionModel) and its own safetensors weights, at
the revision recorded in the preregistration. No substitution: if this bundle
cannot load, the run fails and the failure receipt is retained.

Both question types are produced by the same pinned runtime:
  choice -> option markers, softmax over K=60 options
  score  -> level markers, distribution over the source rubric levels

Usage:
  python neutral_laya_arm.py --limit N --out FILE [--qtype choice|score]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import neutral_comparison_rows as R  # noqa: E402

BUNDLE = Path(
    "/home/fazinahamed/Documents/vey-data/decisionmix/hf-home/hub/"
    "models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
)
BUNDLE_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
QTYPES = {"choice": 0, "score": 1, "noul": 2}


def load_runtime():
    spec = importlib.util.spec_from_file_location("laya_rl_common", BUNDLE / "rl_common.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["laya_rl_common"] = module
    spec.loader.exec_module(module)
    return module


def build(laya, route: str, device: str):
    """Build one pinned route. route in {english, multilingual, typed-decisions}."""
    import torch
    from safetensors.torch import load_file

    sub = BUNDLE if route == "english" else BUNDLE / route
    cfg = laya.load_cfg(str(BUNDLE / "rl_agent_config.json"))
    if route != "english":
        cfg = laya.load_cfg(str(sub / "rl_agent_config.json"))
    enc_dir = sub / "encoder"
    model = laya.build_model(cfg, encoder_dir=str(enc_dir))
    state = load_file(str(sub / "model.safetensors") if route != "english" else str(BUNDLE / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"pinned weights missing tensors: {sorted(missing)[:5]}")
    tok_src = str(sub / "tokenizer") if route != "english" else str(BUNDLE / "tokenizer")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_src)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model, cfg, {"missing": list(missing), "unexpected": list(unexpected)}


def temperature_for(laya, cfg, qtype: int, n_options: int):
    """Delegate to the bundle's own temp_bucket lookup.

    Re-deriving the band table here is a bug source: the pinned table contains
    open-ended "11+" keys that a naive split misreads as the closed range [11,11].
    The bundle already owns that mapping, so this arm does not reimplement it.
    """
    key = laya.temp_bucket(qtype, n_options)
    table = cfg.get("temperature_by_options") or {}
    if key in table:
        return float(table[key]), key
    raw = cfg.get("temperature") or []
    if len(raw) > qtype:
        return float(raw[qtype]), f"per_qtype_index_{qtype}"
    raise RuntimeError(f"pinned config carries no temperature for {key}")


def option_token_lengths(ids, markers):
    """Per-option token span from the pinned output itself.

    build_sequence emits each option as a contiguous [MASK] + text block, so the
    gap to the next marker is that option's exact surviving length. No duplicate
    of the bundle's truncation arithmetic is needed, and when the bundle shrinks
    options evenly under its marker budget this reports the real damage.
    """
    spans = []
    for i, start in enumerate(markers):
        end = markers[i + 1] if i + 1 < len(markers) else start + 1
        spans.append(end - start)
    return spans


def run(laya, tok, model, cfg, rows, qtype_name, device, out, limit, endpoint):
    import numpy as np
    import torch

    qtype = QTYPES[qtype_name]
    started = time.time()
    n = 0
    hits = 0
    abs_err = 0.0
    temps = {}
    min_option_tokens = None
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": "vey.neutral.laya-arm-run.v1",
            "arm": "laya",
            "route": "english",
            "bundle_revision": BUNDLE_REVISION,
            "weights_sha256": hashlib.sha256((BUNDLE / "model.safetensors").read_bytes()).hexdigest(),
            "comparison_protocol_sha256": R.COMPARISON_PROTOCOL_SHA256,
            "endpoint": endpoint,
            "qtype": qtype_name,
            "limit": limit,
            "started_unix": started,
        }, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            if limit and n >= limit:
                break
            if qtype_name == "choice":
                q = {"t": "choice", "ins": row["question"], "crit": dict(row["candidates"])}
            else:
                q = {"t": "score", "ins": row["question"],
                     "crit": [row["candidates"][cid] for cid in row["candidate_ids"]]}
            n_opt = len(q["crit"])
            temp, key = temperature_for(laya, cfg, qtype, n_opt)
            temps[key] = temps.get(key, 0) + 1
            ids, markers = laya.build_sequence(tok, row["state"], q, cfg["max_len"], cfg["head_max_len"])
            spans = option_token_lengths(ids, markers)
            min_option_tokens = min(spans) if min_option_tokens is None else min(min_option_tokens, min(spans))
            # Identical to the pinned API path: the bundle's own collate builds
            # the attention mask from tok.pad_token_id, not from a nonzero test,
            # and the bundle raises rather than silently dropping options.
            item = {"ids": ids, "markers": markers, "qtype": qtype, "target": [0.0] * len(markers),
                    "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "neutral"}
            if len(markers) != n_opt:
                raise RuntimeError(f"row {row['id']}: {len(markers)} markers for {n_opt} options; "
                                   "options do not fit the pinned head_max_len")
            batch = laya.collate_items([[item]], tok.pad_token_id)
            use_amp = device != "cpu"
            with torch.autocast(device_type="cuda", dtype=laya.amp_dtype(cfg.get("amp_dtype", "bf16")),
                                enabled=use_amp):
                logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                  batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                                  batch["qtype"].to(device))
            logits = logits.float()[0, : len(markers)]
            valid = batch["marker_mask"].to(device)[0, : len(markers)]
            probs = torch.softmax((logits / temp).masked_fill(~valid, -1e4), -1).cpu().numpy()
            record = {
                "id": row["id"], "group_id": row["group_id"], "locale": row["locale"],
                "endpoint": row["endpoint"], "arm": "laya", "route": "english",
                "question": row["question"], "state": row["state"],
                "candidate_ids": row["candidate_ids"],
                "probs": [round(float(p), 8) for p in probs],
                "temperature": temp, "temperature_key": key,
                "markers": len(markers), "options": n_opt, "seq_len": len(ids),
                "min_option_tokens": min(spans), "option_tokens_head": spans[:8],
            }
            if qtype_name == "choice":
                pred = row["candidate_ids"][int(np.argmax(probs[: len(markers)]))]
                record["gold"] = row["gold"]
                record["answer"] = pred
                record["correct"] = pred == row["gold"]
                hits += int(record["correct"])
            else:
                levels = np.arange(len(probs[: len(markers)]))
                expected = float((probs[: len(markers)] * levels).sum())
                target = row["mean_level"]
                record["expected_level"] = round(expected, 8)
                record["mean_level"] = target
                record["native_level_max"] = row["level_max"]
                record["observed_raters"] = row["observed_raters"]
                record["abs_error"] = abs(expected - target)
                record["normalized_abs_error"] = record["abs_error"] / row["level_max"]
                record["probs_full"] = [round(float(p), 8) for p in probs]
                record["target_distribution"] = row["distribution"]
                record["target_counts"] = row["counts"]
                abs_err += record["abs_error"]
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
            if n % 1000 == 0:
                msg = f"laya {qtype_name} {n} rows {time.time() - started:.0f}s"
                print(msg + (f" acc {hits / n:.4f}" if qtype_name == "choice" else f" mae {abs_err / n:.4f}"), flush=True)
    return {
        "rows": n,
        "accuracy": hits / n if qtype_name == "choice" and n else None,
        "mean_abs_error": abs_err / n if qtype_name == "score" and n else None,
        "temperature_keys": temps,
        "min_option_tokens_observed": min_option_tokens,
        "seconds": time.time() - started,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "out": str(out),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtype", choices=["choice", "score"], default="choice")
    parser.add_argument("--endpoint", default=R.CHOICE_ENDPOINT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    R.protocol()
    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    laya = load_runtime()
    tok, model, cfg, loadinfo = build(laya, "english", device)
    print("loaded pinned english route", json.dumps(loadinfo), "device", device, flush=True)

    if args.qtype == "choice":
        rows = R.choice_rows()
    else:
        rows = (r for r in R.score_rows() if r["endpoint"] == args.endpoint)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = run(laya, tok, model, cfg, rows, args.qtype, device, out, args.limit, args.endpoint)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    raise SystemExit(main())