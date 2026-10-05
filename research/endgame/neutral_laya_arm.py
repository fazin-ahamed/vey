#!/usr/bin/env python3
"""Laya arm for the neutral development comparison.

Loads the selected PINNED English or multilingual bundle through its runtime code
(rl_common.build_sequence / DecisionModel) and its own safetensors weights, at
the revision recorded in the preregistration. No substitution: if this bundle
cannot load, the run fails and the failure receipt is retained.

Both question types are produced by the same pinned runtime:
  choice -> option markers, softmax over K=60 options
  score  -> level markers, distribution over the source rubric levels

Usage:
  python neutral_laya_arm.py --out FILE --qtype score --endpoint massive.grammar_score --route english
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
SCORE_AMENDMENT_SHA256 = "247cf2e3f6c0c488803d91deffca3c842d8ffe7372d09f323e22036c98e6f67c"
MULTILINGUAL_CHOICE_SHA256 = "73f823b543581ebd5ba4ff46dd38985df3e6fd9c5b40acde5c6fd736f3183b52"


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
    if missing or unexpected:
        raise RuntimeError(f"pinned weights differ: missing={sorted(missing)[:5]}, "
                           f"unexpected={sorted(unexpected)[:5]}")
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


def option_token_lengths(ids, markers, sep_id):
    """Per-option token span measured from the pinned output itself.

    build_sequence emits each option as a contiguous [MASK] + text block, so the
    span runs from one marker to the next. The LAST option runs to the closing
    [SEP], not to markers[-1]+1; without that the final level always reports a
    spurious length of 1 and understates real truncation.
    """
    spans = []
    for i, start in enumerate(markers):
        if i + 1 < len(markers):
            spans.append(markers[i + 1] - start)
        else:
            tail = ids[start + 1:]
            stop = next((j for j, t in enumerate(tail) if t == sep_id), len(tail))
            spans.append(max(1, stop))
    return spans


def run(laya, tok, model, cfg, rows, qtype_name, device, out, limit, endpoint, route):
    import numpy as np
    import torch

    qtype = QTYPES[qtype_name]
    started = time.time()
    n = 0
    hits = 0
    abs_err = 0.0
    temps = {}
    min_option_tokens = None
    sub = BUNDLE if route == "english" else BUNDLE / route
    with (sub / "model.safetensors").open("rb") as weights:
        weights_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": "vey.neutral.laya-arm-run.v1",
            "arm": "laya",
            "route": route,
            "bundle_revision": BUNDLE_REVISION,
            "weights_sha256": weights_sha256,
            "comparison_protocol_sha256": R.COMPARISON_PROTOCOL_SHA256,
            "score_amendment_sha256": SCORE_AMENDMENT_SHA256 if qtype_name == "score" else None,
            "protocol_sha256": MULTILINGUAL_CHOICE_SHA256
                if qtype_name == "choice" and route == "multilingual" else None,
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
            spans = option_token_lengths(ids, markers, tok.sep_token_id)
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
                "endpoint": row["endpoint"], "arm": "laya", "route": route,
                "question": row["question"], "state": row["state"],
                "candidate_ids": row["candidate_ids"],
                "candidates": row["candidates"],
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
    parser.add_argument("--route", choices=["english", "multilingual"], default="english")
    args = parser.parse_args(argv)

    R.protocol()
    if args.qtype == "score":
        amendment = HERE / "neutral_score_route_amendment_v2.json"
        if hashlib.sha256(amendment.read_bytes()).hexdigest() != SCORE_AMENDMENT_SHA256:
            raise RuntimeError("Score route amendment changed")
        if args.endpoint not in R.SCORE_ENDPOINTS:
            parser.error("--endpoint must name a registered Score endpoint")
    else:
        if args.endpoint != R.CHOICE_ENDPOINT:
            parser.error("--endpoint must name the registered Choice endpoint")
        if args.route == "multilingual":
            protocol = HERE / "neutral_multilingual_route_protocol.json"
            if hashlib.sha256(protocol.read_bytes()).hexdigest() != MULTILINGUAL_CHOICE_SHA256:
                raise RuntimeError("Multilingual Choice protocol changed")
    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    laya = load_runtime()
    tok, model, cfg, loadinfo = build(laya, args.route, device)
    print("loaded pinned", args.route, "route", json.dumps(loadinfo), "device", device, flush=True)

    if args.qtype == "choice":
        rows = R.choice_rows()
    else:
        rows = (r for r in R.score_rows() if r["endpoint"] == args.endpoint)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = run(laya, tok, model, cfg, rows, args.qtype, device, out, args.limit, args.endpoint, args.route)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    raise SystemExit(main())