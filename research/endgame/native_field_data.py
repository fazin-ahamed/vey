#!/usr/bin/env python3
"""Guarded English native records for the registered state-field study."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

try:
    from . import neutral_native_compile as compiler
except ImportError:
    import neutral_native_compile as compiler

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "native_field_protocol.json"
PHASES = ("fit", "selection", "calibration", "dev")
ENDPOINTS = ("banking77.intent", "massive.intent", "massive.grammar_score", "massive.spelling_score")
IMPLEMENTATION = ("native_field_data.py", "native_field_model.py", "native_field_train.py", "native_field_verify.py")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def protocol():
    cfg = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if cfg["native_compiler_protocol_sha256"] != compiler.PROTOCOL_SHA256:
        raise RuntimeError("Native compiler authority differs")
    return cfg


def committed():
    for path in (PROTOCOL, *(HERE / name for name in IMPLEMENTATION)):
        relative = path.relative_to(REPO).as_posix()
        committed_bytes = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=REPO)
        if committed_bytes != path.read_bytes():
            raise RuntimeError("Uncommitted native-field implementation: " + relative)


def split_component(component):
    value = hashlib.sha256(("native-field-v1|7|" + component).encode()).hexdigest()
    bucket = int(value, 16) % 100
    return "fit" if bucket < 70 else "selection" if bucket < 85 else "calibration"


def selected(provenance):
    return provenance["source_id"] == compiler.BANKING or (
        provenance["source_id"] == compiler.MASSIVE and provenance["locale"] == "en-US")


def phase_records(phase, catalogues):
    if phase not in ("train", "dev"):
        raise RuntimeError("Sealed/unknown source phase denied")
    provenance = {}
    for row in compiler.open_phase(phase, "provenance"):
        if selected(row):
            if row["id"] in provenance:
                raise RuntimeError("Duplicate selected native provenance")
            provenance[row["id"]] = row
    serving, targets, decisions = {}, {}, {}
    for stream, retained in (("serving", serving), ("targets", targets), ("decisions", decisions)):
        for row in compiler.open_phase(phase, stream):
            if row["id"] in provenance:
                if row["id"] in retained:
                    raise RuntimeError("Duplicate selected native " + stream)
                retained[row["id"]] = row
    if not (provenance.keys() == serving.keys() == targets.keys() == decisions.keys()):
        raise RuntimeError("Selected native stream membership differs")
    grouped = {}
    for rid in sorted(provenance):
        prov, view, target, decision = provenance[rid], serving[rid], targets[rid], decisions[rid]
        endpoint = prov["endpoint"]
        if endpoint not in ENDPOINTS or prov["final_split"] != phase or decision["split"] != phase:
            raise RuntimeError("Native endpoint/phase differs")
        if decision["license"] != "CC-BY-4.0" or prov["license"] != "CC-BY-4.0":
            raise RuntimeError("Native source license differs")
        if view["locale"] != prov["locale"] or view["task"] != decision["task"]:
            raise RuntimeError("Native serving metadata differs")
        candidate_ids = [item["id"] for item in view["candidates"]]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise RuntimeError("Duplicate native candidate")
        if [item["id"] for item in decision["candidates"]] != candidate_ids:
            raise RuntimeError("Native IR/serving catalogue differs")
        if endpoint in catalogues and catalogues[endpoint] != candidate_ids:
            raise RuntimeError("Endpoint catalogue changed")
        catalogues[endpoint] = candidate_ids
        if not target["target_available"]:
            if view["task"] != "score" or target["observed_raters"] != 0:
                raise RuntimeError("Invalid absent native target")
            continue
        projected = {"id": rid, "endpoint": endpoint, "task": view["task"],
                     "candidate_ids": candidate_ids, "candidates": {c["id"]: c["text"] for c in view["candidates"]},
                     "question": view["question"]}
        if view["task"] == "choice":
            gold = target["label_id"]
            if decision["gold"] != [gold] or gold not in candidate_ids:
                raise RuntimeError("Native categorical target differs")
            projected.update(gold=gold, target_distribution=[float(label == gold) for label in candidate_ids])
        elif view["task"] == "score":
            maximum, counts, raters = target["native_level_max"], target["counts"], target["observed_raters"]
            if candidate_ids != [str(i) for i in range(maximum + 1)] or len(counts) != maximum + 1:
                raise RuntimeError("Native ordinal coordinates differ")
            if any(type(count) is not int or count < 0 for count in counts) or sum(counts) != raters or raters < 1:
                raise RuntimeError("Invalid retained-rater counts")
            distribution = [count / raters for count in counts]
            mean = sum(i * count for i, count in enumerate(counts)) / raters
            if distribution != target["distribution"] or mean != target["mean_level"]:
                raise RuntimeError("Native ordinal target reconstruction differs")
            projected.update(target_distribution=distribution, counts=counts, observed_raters=raters,
                             level_max=maximum, mean_level=mean)
        else:
            raise RuntimeError("Unsupported native task")
        key = prov["row_key"]
        state = {"id": key, "component_id": prov["component_id"], "group_id": prov["group_id"],
                 "locale": prov["locale"], "source_id": prov["source_id"], "state": view["state"]}
        if key not in grouped:
            grouped[key] = {**state, "decisions": []}
        elif any(grouped[key][name] != value for name, value in state.items()):
            raise RuntimeError("Co-occurring native state metadata differs")
        grouped[key]["decisions"].append(projected)
    records = sorted(grouped.values(), key=lambda row: row["id"])
    for row in records:
        row["decisions"].sort(key=lambda decision: decision["endpoint"])
    return records


def source_exports(cfg):
    root = Path(cfg["native_root"])
    result = {}
    for phase in ("train", "dev"):
        path = root / phase / "manifest.json"
        receipt = root / phase / "verification_receipt.json"
        result[phase] = {"manifest_path": str(path), "manifest_sha256": digest(path),
                         "verification_path": str(receipt), "verification_sha256": digest(receipt)}
    return result


def materialize():
    cfg = protocol()
    committed()
    root = Path(cfg["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    try:
        return _materialize(cfg, root)
    except BaseException as error:
        with (root / "failure.json").open("x", encoding="utf-8") as stream:
            json.dump({"schema": "vey.native-field.preparation-failure.v1",
                       "error_type": type(error).__name__, "message": str(error),
                       "model_executed": False, "sealed_phases_accessed": False,
                       "retry_requires_prospective_execution_correction": True}, stream, indent=2)
            stream.write("\n")
        raise


def _materialize(cfg, root):
    catalogues = {}
    train = phase_records("train", catalogues)
    dev = phase_records("dev", catalogues)
    train_components = {row["component_id"] for row in train}
    if train_components & {row["component_id"] for row in dev}:
        raise RuntimeError("Native train/dev component leakage")
    phases = {name: [] for name in PHASES}
    for row in train:
        phases[split_component(row["component_id"])].append(row)
    phases["dev"] = dev
    counts = {phase: Counter(d["endpoint"] for row in rows for d in row["decisions"])
              for phase, rows in phases.items()}
    for phase, expected in (("train", {"banking77.intent": 4105, "massive.intent": 3971}),
                            ("dev", {"banking77.intent": 1027, "massive.intent": 1252})):
        actual = Counter(d["endpoint"] for row in (train if phase == "train" else dev) for d in row["decisions"])
        if any(actual[endpoint] != count for endpoint, count in expected.items()):
            raise RuntimeError("Registered English intent census differs")
    eligible = [endpoint for endpoint in ENDPOINTS if endpoint.endswith(".intent") or
                (counts["fit"][endpoint] >= 250 and counts["dev"][endpoint] >= 100)]
    for endpoint in eligible:
        if any(counts[phase][endpoint] == 0 for phase in PHASES):
            raise RuntimeError("Eligible field has an empty fitting/selection/calibration/dev phase")
    files = {}
    for phase, records in phases.items():
        path = root / (phase + ".jsonl")
        with path.open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        files[phase] = {"path": str(path), "sha256": digest(path), "states": len(records),
                        "decisions": sum(len(row["decisions"]) for row in records),
                        "components": len({row["component_id"] for row in records}),
                        "endpoint_counts": dict(counts[phase])}
    manifest = {"schema": "vey.native-field.dataset.v1", "protocol_sha256": digest(PROTOCOL),
                "catalogues": catalogues, "eligible_endpoints": eligible,
                "ineligible_endpoints": [endpoint for endpoint in ENDPOINTS if endpoint not in eligible],
                "phases": files, "source_exports": source_exports(cfg),
                "source_files": {str(HERE / name): digest(HERE / name) for name in IMPLEMENTATION},
                "semantic_model_files": {str(Path("/home/fazinahamed/Documents/vey/vey_u/semantic/models.py")):
                                         digest("/home/fazinahamed/Documents/vey/vey_u/semantic/models.py")},
                "train_payloads_decoded": True, "sealed_phases_accessed": False,
                "shipping_training_sources": [compiler.BANKING, compiler.MASSIVE]}
    with (root / "dataset_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def execution_correction(manifest):
    path = HERE / "native_field_execution_correction_v1.json"
    if not path.exists():
        return None
    raw = path.read_bytes()
    relative = path.relative_to(REPO).as_posix()
    if subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=REPO) != raw:
        raise RuntimeError("Uncommitted native execution correction")
    correction = json.loads(raw)
    root = Path(protocol()["output_root"])
    if (correction["schema"] != "vey.native-field.execution-correction.v1" or
            correction["dataset_manifest_sha256"] != digest(root / "dataset_manifest.json") or
            correction["protocol_sha256"] != manifest["protocol_sha256"] or
            correction["original_source_files"] != manifest["source_files"] or
            set(correction["source_files"]) != set(manifest["source_files"])):
        raise RuntimeError("Unregistered native execution transition")
    for entry in correction["retained_artifacts"]:
        if digest(entry["path"]) != entry["sha256"]:
            raise RuntimeError("Retained pre-correction evidence changed")
    return correction


def execution_sources(manifest):
    correction = execution_correction(manifest)
    return correction["source_files"] if correction else manifest["source_files"]


def verify_dataset(root):
    cfg = protocol()
    if Path(root).resolve() != Path(cfg["output_root"]).resolve():
        raise RuntimeError("Unregistered native dataset root")
    manifest = json.loads((Path(root) / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "vey.native-field.dataset.v1" or manifest["protocol_sha256"] != digest(PROTOCOL):
        raise RuntimeError("Native dataset/protocol identity differs")
    for mapping in (execution_sources(manifest), manifest["semantic_model_files"]):
        for path, expected in mapping.items():
            if digest(path) != expected:
                raise RuntimeError("Native implementation changed after materialization")
    if manifest["source_exports"] != source_exports(cfg):
        raise RuntimeError("Native source projection custody changed")
    if set(manifest["phases"]) != set(PHASES):
        raise RuntimeError("Native inner phase membership differs")
    for phase, entry in manifest["phases"].items():
        path = Path(root) / (phase + ".jsonl")
        if entry["path"] != str(path) or digest(path) != entry["sha256"]:
            raise RuntimeError("Native selected-record hash differs")
    return manifest


def load_dataset(root):
    manifest = verify_dataset(root)
    records = {}
    components = {}
    decision_ids = set()
    for phase in PHASES:
        with Path(manifest["phases"][phase]["path"]).open(encoding="utf-8") as stream:
            records[phase] = [json.loads(line) for line in stream]
        entry = manifest["phases"][phase]
        if len(records[phase]) != entry["states"] or sum(len(r["decisions"]) for r in records[phase]) != entry["decisions"]:
            raise RuntimeError("Selected-record census differs")
        for row in records[phase]:
            component = row["component_id"]
            if component in components and components[component] != phase:
                raise RuntimeError("Native inner component leakage")
            components[component] = phase
            if phase != "dev" and split_component(component) != phase:
                raise RuntimeError("Native hash split differs")
            for decision in row["decisions"]:
                if decision["id"] in decision_ids:
                    raise RuntimeError("Duplicate native decision ID")
                decision_ids.add(decision["id"])
                if decision["candidate_ids"] != manifest["catalogues"][decision["endpoint"]]:
                    raise RuntimeError("Selected-record catalogue differs")
    return manifest, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialize", action="store_true", required=True)
    parser.parse_args()
    result = materialize()
    print(json.dumps({"phases": result["phases"], "eligible_endpoints": result["eligible_endpoints"],
                      "ineligible_endpoints": result["ineligible_endpoints"]}, sort_keys=True))


if __name__ == "__main__":
    main()
