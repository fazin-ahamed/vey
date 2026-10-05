#!/usr/bin/env python3
"""Persist QNATIVE-2 membership under its already registered paper-level split."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

try:
    from . import native_field_data, neutral_qasper_native_compile as projection
except ImportError:
    import native_field_data
    import neutral_qasper_native_compile as projection

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "neutral_qasper_question_study_protocol.json"
PROTOCOL_SHA256 = "40d66173e91a827725dc107553a644894f39212f4667d4d47c33e45491f73f1f"
OUT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/neutral-source-v2/qasper/native-question-split-v1")
PHASES = ("fit", "selection", "calibration", "dev")


def pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run():
    require(pin(PROTOCOL)["sha256"] == PROTOCOL_SHA256, "Question study protocol differs")
    cfg = json.loads(PROTOCOL.read_text())
    for source in (PROTOCOL, Path(__file__), Path(native_field_data.__file__)):
        relative = source.relative_to(REPO).as_posix()
        committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=REPO)
        require(committed == source.read_bytes(), "Uncommitted split authority: " + relative)
    for entry in cfg["authority"].values():
        require(pin(entry["path"]) == entry, "Question study authority differs: " + entry["path"])
    require(not OUT.exists(), "Refusing to overwrite split evidence")
    endpoints = tuple(cfg["data"]["endpoints"])
    rows = {phase: [] for phase in PHASES}
    ids, component_phase = set(), {}
    for outer in ("train", "dev"):
        counts = Counter()
        for provenance in projection.open_phase(outer, "provenance"):
            endpoint, identity, component = (provenance[key] for key in ("endpoint", "id", "component_id"))
            require(endpoint in endpoints and identity not in ids, "Unexpected endpoint or duplicate decision")
            phase = "dev" if outer == "dev" else native_field_data.split_component(component)
            require(component_phase.setdefault(component, phase) == phase, "Paper crosses phases")
            ids.add(identity)
            counts[endpoint] += 1
            rows[phase].append({key: provenance[key] for key in
                                ("id", "endpoint", "component_id", "group_id", "paper_id", "question_ordinal")})
        require(dict(counts) == {endpoint: cfg["data"]["endpoints"][endpoint]["rows"][outer]
                                 for endpoint in endpoints}, "Endpoint census differs from preregistration")
    summaries = {}
    for phase, records in rows.items():
        counts = Counter(row["endpoint"] for row in records)
        require(all(counts[endpoint] > 0 for endpoint in endpoints), "Empty endpoint in " + phase)
        records.sort(key=lambda row: row["id"])
        summaries[phase] = {"rows": len(records), "components": len({row["component_id"] for row in records}),
                            "endpoint_counts": dict(counts)}
    OUT.mkdir(parents=True, exist_ok=False)
    for phase, records in rows.items():
        path = OUT / (phase + ".jsonl")
        with path.open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        summaries[phase]["file"] = pin(path)
    manifest = {"schema": "vey.neutral.qasper.question-split-census.v1", "status": "MATERIALIZED",
                "protocol": pin(PROTOCOL), "implementation": pin(Path(__file__)),
                "split_implementation": pin(Path(native_field_data.__file__)), "authority": cfg["authority"],
                "split_rule": "SHA256(native-field-v1|7| + component_id) modulo100; <70 fit, <85 selection, else calibration",
                "phases": summaries, "membership_fields_only": True,
                "provenance_annotation_fields": "Deserialized by the existing guarded provenance reader, unused and not persisted",
                "source_phases_read": ["train", "dev"], "sealed_phases_accessed": False,
                "model_loads": 0, "model_forwards": 0, "architecture_selected": False, "quality_credit": False}
    with (OUT / "manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(OUT), "phases": summaries}, sort_keys=True))
    return manifest


if __name__ == "__main__":
    run()
