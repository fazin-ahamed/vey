#!/usr/bin/env python3
"""Resource-governed remaining-ID continuation of the published Laya DEV capture.

Implements the committed native_peer_resource_recovery_protocol.json: the 924
retained valid/raw rows of the original stop are immutable and never re-run;
only the 1,355 missing native DEV rows are called through the unchanged
published typed CPU API, under per-call RAM/swap governors that stop gracefully
before the next call instead of exhausting the machine. A complete continuation
merges retained plus new rows into the full 2,279-row original native order.

Usage:
  python native_peer_capture_recovery.py --attempt-out FILE [--dry-run]
  python native_peer_capture_recovery.py --merge-out FILE --raw-merge-out FILE \
      --attempt-out CONTINUATION.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

try:
    from . import native_peer_capture as capture
except ImportError:
    import native_peer_capture as capture

HERE = Path(__file__).resolve().parent
RECOVERY = HERE / "native_peer_resource_recovery_protocol.json"
RECOVERY_SHA256 = "3a09cf97a099335f44b28e7f463abaa5fcb0d6628d472d87f8953d0fec3901a6"
STOP_MANIFEST = HERE / "native_peer_resource_stop_result_manifest.json"
PARTIAL_DIR = Path(capture.protocol()["data"]["root"]) / "peer-full-dev-v1"
GIB = 1024 ** 3
MIB = 1024 ** 2
CAPTURE_SCHEMA = capture.CAPTURE_SCHEMA


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return capture.sha(Path(path))


def read_meminfo(path="/proc/meminfo") -> Tuple[int, int]:
    """Return (MemAvailable bytes, swap-used bytes) parsed from a meminfo file."""
    fields = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            parts = value.split()
            if parts and parts[0].isdigit():
                fields[name] = int(parts[0]) * 1024
    require("MemAvailable" in fields and "SwapTotal" in fields and "SwapFree" in fields,
            "meminfo lacks MemAvailable/Swap fields: " + str(path))
    return fields["MemAvailable"], fields["SwapTotal"] - fields["SwapFree"]


class ResourceBreach(RuntimeError):
    """Raised when the governor refuses to make the next model call."""

    def __init__(self, message, snapshot):
        super().__init__(message)
        self.snapshot = snapshot


class Governor:
    """Per-call RAM/swap guard; never retries, never suppresses."""

    def __init__(self, reader: Callable[[], Tuple[int, int]], load_floor=GIB * 10,
                 call_floor=GIB * 4, swap_growth_limit=MIB * 256):
        self.reader = reader
        self.load_floor = load_floor
        self.call_floor = call_floor
        self.swap_growth_limit = swap_growth_limit
        available, self.start_swap = reader()
        require(available >= load_floor,
                "MemAvailable %.2f GiB below the %.0f GiB load floor"
                % (available / GIB, load_floor / GIB))
        self.start_available = available

    def snapshot(self):
        available, swap = self.reader()
        return {"MemAvailable_bytes": available, "swap_used_bytes": swap,
                "attempt_start_swap_bytes": self.start_swap,
                "swap_growth_bytes": swap - self.start_swap}

    def check(self, stage):
        snapshot = self.snapshot()
        if snapshot["MemAvailable_bytes"] < self.call_floor:
            raise ResourceBreach(stage + ": MemAvailable %.2f GiB below the %.0f GiB per-call floor"
                                 % (snapshot["MemAvailable_bytes"] / GIB, self.call_floor / GIB),
                                 snapshot)
        if snapshot["swap_growth_bytes"] > self.swap_growth_limit:
            raise ResourceBreach(stage + ": swap growth %.1f MiB exceeds the %d MiB limit"
                                 % (snapshot["swap_growth_bytes"] / MIB, self.swap_growth_limit / MIB),
                                 snapshot)
        return snapshot


def load_jsonl(path) -> Tuple[dict, List[dict]]:
    """Return (header, rows) of a capture file; the first line must be the header."""
    rows = []
    header = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if header is None:
                header = value
                continue
            if value.get("schema") == "vey.native-peer.laya-capture-failure.v1":
                raise RuntimeError("Retained capture contains a failure row: " + str(path))
            rows.append(value)
    require(header is not None, "Capture file lacks a header: " + str(path))
    return header, rows


def retained_universe(validated, raw) -> Dict[str, Tuple[dict, dict]]:
    """Pair validated rows with their raw payloads by identical unique id order."""
    require(len(validated) == len(raw), "Retained validated/raw row counts differ")
    paired = {}
    for valid, payload in zip(validated, raw):
        require(valid["id"] == payload["id"],
                "Retained raw/validated id order differs at " + valid["id"])
        require(valid["id"] not in paired, "Duplicate retained id " + valid["id"])
        paired[valid["id"]] = (valid, payload)
    return paired


def remaining_rows(dev_rows: List[dict], retained: Dict[str, Tuple[dict, dict]]) -> List[dict]:
    """Rows of the original native DEV order absent from the retained universe."""
    require(len({row["id"] for row in dev_rows}) == len(dev_rows), "Duplicate DEV row id")
    missing = [row for row in dev_rows if row["id"] not in retained]
    for row in dev_rows:
        if row["id"] in retained:
            valid, _ = retained[row["id"]]
            require(valid["state"] == row["state"] and valid["question"] == row["question"]
                    and valid["candidate_ids"] == row["candidate_ids"]
                    and valid["candidates"] == row["candidates"]
                    and valid["gold"] == row["gold"],
                    "Retained row inputs differ from the verified DEV projection: " + row["id"])
    return missing


def validate_prediction(row, prediction) -> dict:
    """Published-payload checks identical to the original capture; no relaxation."""
    require(set(prediction["answers"]) == {"q"}, "Question keys differ")
    answer = prediction["answers"]["q"]
    probabilities = answer["probabilities"]
    require(set(probabilities) == set(row["candidate_ids"]), "Probability keys differ")
    values = [float(probabilities[cid]) for cid in row["candidate_ids"]]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), "Invalid probability")
    require(abs(sum(values) - 1) <= len(values) * 0.00005 + 1e-6, "Serialization bound exceeded")
    chosen = answer["choice"]
    require(chosen in row["candidate_ids"], "Illegal returned choice")
    return {
        "id": row["id"], "group_id": row["group_id"], "component_id": row["component_id"],
        "locale": row["locale"], "endpoint": row["endpoint"], "arm": "laya",
        "route": capture.ROUTE, "question": row["question"], "state": row["state"],
        "candidate_ids": row["candidate_ids"], "candidates": row["candidates"],
        "gold": row["gold"], "answer": chosen, "correct": chosen == row["gold"],
        "probabilities": {cid: probabilities[cid] for cid in row["candidate_ids"]},
        "raw_probabilities": {cid: probabilities[cid] for cid in row["candidate_ids"]},
        "probability_decimals": 4,
    }


def merge_full(header, dev_rows, retained, continuation_validated, continuation_raw):
    """Interleave retained and continuation rows in original native DEV order."""
    merged_validated, merged_raw = {}, {}
    for record in continuation_validated:
        require(record["id"] not in merged_validated, "Duplicate continuation id " + record["id"])
        merged_validated[record["id"]] = record
    for payload in continuation_raw:
        require(payload["id"] not in merged_raw, "Duplicate continuation raw id " + payload["id"])
        merged_raw[payload["id"]] = payload
    for rid, (valid, payload) in retained.items():
        require(rid not in merged_validated and rid not in merged_raw,
                "Continuation re-ran a retained id: " + rid)
        merged_validated[rid], merged_raw[rid] = valid, payload
    ordered_validated = [merged_validated[row["id"]] for row in dev_rows if row["id"] in merged_validated]
    ordered_raw = [merged_raw[row["id"]] for row in dev_rows if row["id"] in merged_raw]
    require(len(ordered_validated) == len(ordered_raw) == len(dev_rows),
            "Merged universe is not the complete DEV population")
    return ordered_validated, ordered_raw


def custody():
    require(digest(RECOVERY) == RECOVERY_SHA256, "Recovery protocol identity differs")
    cfg = capture.protocol()
    stop = json.loads(STOP_MANIFEST.read_text(encoding="utf-8"))
    require(stop["status"] == "INCOMPLETE_OWNED_RESOURCE_STOP", "Unexpected stop manifest status")
    for key in ("capture", "raw_sidecar"):
        entry = stop[key]
        path = Path(entry["path"])
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                "Retained partial evidence differs: " + key)
    capture.custody(cfg)
    return cfg, stop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-out", type=Path)
    parser.add_argument("--merge-out", type=Path)
    parser.add_argument("--raw-merge-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    merge_mode = bool(args.merge_out or args.raw_merge_out)
    require(bool(args.attempt_out), "--attempt-out is required")
    require(not merge_mode or (args.merge_out and args.raw_merge_out),
            "Merge mode requires both --merge-out and --raw-merge-out")
    for target in (args.attempt_out, args.merge_out, args.raw_merge_out):
        if target:
            require(not target.exists(), "Refusing to overwrite: " + str(target))

    cfg, stop = custody()
    header, validated = load_jsonl(Path(stop["capture"]["path"]))
    _, raw = load_jsonl(Path(stop["raw_sidecar"]["path"]))
    require(header.get("schema") == CAPTURE_SCHEMA, "Retained capture schema differs")
    retained = retained_universe(validated, raw)
    require(len(retained) == stop["captured_validated_rows"], "Retained row count differs from the stop manifest")
    dev_rows = list(capture.dev_rows(cfg))
    require(len(dev_rows) == stop["registered_rows"], "DEV universe differs from the registered count")
    missing = remaining_rows(dev_rows, retained)
    require(len(missing) == stop["remaining_rows"], "Remaining row count differs from the stop manifest")
    for row in dev_rows[:len(validated)]:
        require(row["id"] in retained, "Retained rows are not a prefix of the original DEV order")

    if merge_mode:
        cont_header, cont_validated = load_jsonl(args.attempt_out)
        _, cont_raw = load_jsonl(args.attempt_out.with_name(args.attempt_out.name + ".raw.jsonl"))
        require(cont_header.get("schema") == CAPTURE_SCHEMA and cont_header.get("started_unix") >= header["started_unix"],
                "Continuation header differs")
        ordered_validated, ordered_raw = merge_full(header, dev_rows, retained, cont_validated, cont_raw)
        with args.merge_out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
            for record in ordered_validated:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        with args.raw_merge_out.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({**header, "schema": "vey.native-peer.laya-raw-capture.v1"},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            for payload in ordered_raw:
                stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"merged_rows": len(ordered_validated), "retained": len(retained),
                          "continued": len(cont_validated), "out": str(args.merge_out)}))
        return 0

    governor = Governor(lambda: read_meminfo())
    state = {"schema": "vey.native-peer.recovery-attempt.v1", "status": "STARTED",
             "remaining_rows": len(missing), "retained_rows": len(retained),
             "dry_run": args.dry_run, "events": [], "processed_rows": 0}
    if args.dry_run:
        state.update(status="DRY_RUN_COMPLETE", model_loaded=False,
                     resource=governor.snapshot())
        args.attempt_out.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(state, sort_keys=True))
        return 0

    raw_out = args.attempt_out.with_name(args.attempt_out.name + ".raw.jsonl")
    require(not raw_out.exists(), "Refusing to overwrite: " + str(raw_out))
    source_root, directory, expected = reusable_materialization(cfg)
    laya, torch = capture.load_agent(source_root, directory, expected)
    agent = laya.Agent(str(directory), device="cpu", backend="eager",
                       expected_sha256=expected)
    try:
        with args.attempt_out.open("x", encoding="utf-8") as stream, \
                raw_out.open("x", encoding="utf-8") as raw_stream:
            header_row = {**header, "recovery": True, "remaining_ids": [r["id"] for r in missing]}
            stream.write(json.dumps(header_row, ensure_ascii=False, sort_keys=True) + "\n")
            raw_stream.write(json.dumps({**header_row, "schema": "vey.native-peer.laya-raw-capture.v1"},
                                         ensure_ascii=False, sort_keys=True) + "\n")
            hits = 0
            for index, row in enumerate(missing):
                snapshot = governor.check("before call " + str(index))
                questions = {"q": {"type": "choice", "instructions": row["question"],
                                    "criteria": {cid: row["candidates"][cid] for cid in row["candidate_ids"]}}}
                prediction = agent.system_one({"text": row["state"]}, questions)
                raw_stream.write(json.dumps({"id": row["id"], "endpoint": row["endpoint"],
                                              "prediction": prediction},
                                             ensure_ascii=False, sort_keys=True) + "\n")
                raw_stream.flush()
                record = validate_prediction(row, prediction)
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                state["processed_rows"] = index + 1
                hits += int(record["correct"])
                if state["processed_rows"] % 250 == 0:
                    event = {"rows": state["processed_rows"], "top1": hits / state["processed_rows"],
                             "resource": snapshot}
                    state["events"].append(event)
                    print(json.dumps(event, sort_keys=True), flush=True)
        state.update(status="COMPLETE", model_loaded=True, processed_rows=len(missing))
    except ResourceBreach as breach:
        state.update(status="RESOURCE_STOP", error=str(breach),
                     error_type="ResourceBreach", resource=breach.snapshot,
                     processed_ids=[r["id"] for r in missing[:state["processed_rows"]]][:0] or None)
        state["error_snapshot"] = breach.snapshot
    except BaseException as error:
        state.update(status="FAILED", error=repr(error), error_type=type(error).__name__,
                     processed_rows=state["processed_rows"])
    finally:
        receipt = args.attempt_out.with_name(args.attempt_out.name + ".receipt.json")
        require(not receipt.exists(), "Refusing to overwrite: " + str(receipt))
        receipt.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": state["status"], "processed_rows": state["processed_rows"]}))
    return 0 if state["status"] == "COMPLETE" else 2


def reusable_materialization(cfg):
    """Verify and reuse the already-materialized published route directory."""
    manifest, compat, release, source_root, route = capture.custody(cfg)
    root = Path(cfg["data"]["root"])
    directory = root / "peer-capture-artifacts-v1" / capture.ROUTE
    require(directory.is_dir(), "Published route materialization missing")
    expected = {}
    for relative, info in route["files"].items():
        target = directory / relative
        require(target.exists() and capture.sha(target) == info["sha256"],
                "Materialized route payload changed: " + relative)
        expected[relative] = info["sha256"]
    return source_root, directory, expected


if __name__ == "__main__":
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
