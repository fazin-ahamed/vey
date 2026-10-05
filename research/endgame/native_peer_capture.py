#!/usr/bin/env python3
"""Capture actual published Laya 0.3.27 English choice on the native English DEV rows.

Runs the unmodified complete published release source through its own typed
``laya.Agent(...).system_one`` API, one choice question per native DEV row, on
CPU eager. Reads no sealed phase, fits nothing, and claims no quality or
performance credit. Fail-closed: any custody, load or numerical problem raises
and the partial per-row capture is retained.

Usage:
  python native_peer_capture.py --out FILE
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "native_peer_protocol.json"
PROTOCOL_SHA256 = "8ca37c2a64f22af5846c698a16ae1269e76f988770226aff4446d7abac3b31ca"
CAPTURE_SCHEMA = "vey.native-peer.laya-capture.v1"
ROUTE = "english"
DEV_ENDPOINTS = ("banking77.intent", "massive.intent")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked(entry):
    path = Path(entry["path"])
    if "bytes" in entry:
        require(path.stat().st_size == entry["bytes"], "Pinned artifact size differs: " + str(path))
    require(sha(path) == entry["sha256"], "Pinned artifact identity differs: " + str(path))
    return path


def protocol():
    require(sha(PROTOCOL) == PROTOCOL_SHA256, "Peer protocol identity differs")
    cfg = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    require(cfg["schema"] == "vey.native-field.current-peer-protocol.v1", "Unregistered peer schema")
    require(cfg["laya"]["route"] == ROUTE, "Registered peer route differs")
    return cfg


def custody(cfg):
    manifest = json.loads(checked(cfg["laya"]["custody_manifest"]).read_text(encoding="utf-8"))
    compat = json.loads(checked(cfg["laya"]["compatibility_protocol"]).read_text(encoding="utf-8"))
    require(manifest["package_version"] == cfg["laya"]["published_version"], "Published version differs")
    require(manifest["release_commit"] == cfg["laya"]["release_commit"] == compat["release_commit"],
            "Published release commit differs")
    require(compat["complete_release_source_sha256"] == cfg["laya"]["complete_release_source"]["sha256"],
            "Compatibility complete-source digest differs")
    release = json.loads(checked(cfg["laya"]["complete_release_source"]).read_text(encoding="utf-8"))
    source_root = Path(release["source_root"])
    actual = {str(path.relative_to(source_root)) for path in (source_root / "laya").rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    require(actual == set(release["files"]), "Complete release source member set differs")
    for name, info in release["files"].items():
        require(sha(source_root / name) == info["sha256"], "Published release source changed: " + name)
    route = manifest["routes"][ROUTE]
    for relative, info in route["files"].items():
        source = Path(info["resolved_path"])
        require(source.stat().st_size == info["bytes"] and sha(source) == info["sha256"],
                "English route payload changed: " + relative)
    return manifest, compat, release, source_root, route


def materialize(root, route):
    directory = root / "peer-capture-artifacts-v1" / ROUTE
    directory.mkdir(parents=True, exist_ok=False)
    expected = {}
    for relative, info in route["files"].items():
        source = Path(info["resolved_path"])
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative == "model.safetensors":
            target.symlink_to(source)
        else:
            shutil.copyfile(source, target)
        expected[relative] = info["sha256"]
    return directory, expected


def offline(*_args, **_kwargs):
    raise RuntimeError("Network denied by offline peer capture")


def load_agent(source_root, directory, expected):
    import torch

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(),
            "CPU-only capture required")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source_root))
    import laya

    require(laya.__version__ == "0.3.27", "Wrong runtime version")
    require(Path(laya.__file__).resolve() == (source_root / "laya/__init__.py").resolve(), "Wrong runtime source")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    return laya, torch


def dev_rows(cfg):
    path = checked(cfg["data"]["dev"])
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            decision = row["decisions"][0]
            require(decision["endpoint"] in DEV_ENDPOINTS, "Unregistered DEV endpoint")
            require(decision["task"] == "choice", "Native DEV decision is not Choice")
            require(type(decision["id"]) is str and decision["id"], "Invalid native decision identity")
            yield {
                "id": decision["id"], "record_id": row["id"],
                "group_id": row["group_id"], "component_id": row["component_id"],
                "locale": row["locale"], "state": row["state"], "endpoint": decision["endpoint"],
                "question": decision["question"], "candidate_ids": list(decision["candidate_ids"]),
                "candidates": dict(decision["candidates"]), "gold": decision["gold"],
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = protocol()
    for variable, value in cfg["laya"]["environment_variables"].items():
        require(os.environ.get(variable) == value, "Environment guard: " + variable)
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "LAYA_SHA256_DIGESTS"):
        require(not os.environ.get(variable), "Forbidden credential/override: " + variable)
    require(sys.executable == cfg["laya"]["executable"], "Unexpected interpreter")
    for package, version in cfg["laya"]["packages"].items():
        require(importlib.metadata.version(package) == version, "Dependency pin changed: " + package)

    manifest, compat, release, source_root, route = custody(cfg)
    root = Path(cfg["data"]["root"])
    directory, expected = materialize(root, route)
    socket.socket.connect = offline
    socket.socket.connect_ex = offline
    socket.create_connection = offline
    laya, torch = load_agent(source_root, directory, expected)
    # One model resident for the whole capture, matching the registered resource
    # contract; the exact published constructor and typed call are unchanged.
    agent = laya.Agent(str(directory), device="cpu", backend="eager", expected_sha256=expected)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    hits = 0
    header = {
        "schema": CAPTURE_SCHEMA, "arm": "laya", "route": ROUTE,
        "published_version": cfg["laya"]["published_version"],
        "release_commit": cfg["laya"]["release_commit"],
        "complete_release_source_sha256": cfg["laya"]["complete_release_source"]["sha256"],
        "peer_protocol_sha256": PROTOCOL_SHA256,
        "compatibility_protocol_sha256": cfg["laya"]["compatibility_protocol"]["sha256"],
        "custody_manifest_sha256": cfg["laya"]["custody_manifest"]["sha256"],
        "state_convention": "system_one(state={'text': exact rendered state})",
        "criteria_convention": "choice criteria {candidate_id: exact rendered candidate text}",
        "endpoints": list(DEV_ENDPOINTS), "limit": args.limit, "started_unix": time.time(),
        "final_benchmark_access": False, "quality_claim": False, "performance_claim": False,
    }
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        try:
            for row in dev_rows(cfg):
                if args.limit and rows >= args.limit:
                    break
                questions = {"q": {"type": "choice", "instructions": row["question"],
                                   "criteria": {cid: row["candidates"][cid] for cid in row["candidate_ids"]}}}
                prediction = agent.system_one({"text": row["state"]}, questions)
                answer = prediction["answers"]["q"]
                probabilities = answer["probabilities"]
                require(set(probabilities) == set(row["candidate_ids"]), "Probability keys differ")
                values = [float(probabilities[cid]) for cid in row["candidate_ids"]]
                require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid probability")
                require(abs(sum(values) - 1) <= len(values) * 0.00005 + 1e-6, "Serialization bound exceeded")
                chosen = answer["choice"]
                require(chosen in row["candidate_ids"], "Illegal returned choice")
                record = {
                    "id": row["id"], "group_id": row["group_id"], "component_id": row["component_id"],
                    "locale": row["locale"], "endpoint": row["endpoint"], "arm": "laya", "route": ROUTE,
                    "question": row["question"], "state": row["state"],
                    "candidate_ids": row["candidate_ids"], "candidates": row["candidates"],
                    "gold": row["gold"], "answer": chosen, "correct": chosen == row["gold"],
                    "probabilities": {cid: probabilities[cid] for cid in row["candidate_ids"]},
                    "raw_probabilities": {cid: probabilities[cid] for cid in row["candidate_ids"]},
                    "probability_decimals": 4,
                }
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                rows += 1
                hits += int(record["correct"])
                if rows % 250 == 0:
                    print(json.dumps({"rows": rows, "top1": hits / rows}), flush=True)
        except BaseException:
            stream.write(json.dumps({"schema": "vey.native-peer.laya-capture-failure.v1",
                                     "error": traceback.format_exc(), "rows_before_failure": rows},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            raise
    print(json.dumps({"rows": rows, "top1": hits / rows if rows else None, "out": str(out)}), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
