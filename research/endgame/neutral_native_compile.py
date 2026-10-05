#!/usr/bin/env python3
"""Custodial train/dev projection of pinned Banking77 and MASSIVE source rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "neutral_native_compiler_protocol.json"
PROTOCOL_SHA256 = "f1da1e227e6195c2230f4f1a10c016e0c683a036a192030704ed9c67da259642"
PHASES = ("train", "dev")
STREAMS = ("serving", "decisions", "targets", "provenance")
BANKING = "PolyAI/banking77"
MASSIVE = "AmazonScience/massive"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def value_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def checked(entry):
    path = Path(entry["path"])
    require(digest(path) == entry["sha256"], "Pinned artifact hash differs")
    return path


def metadata():
    require(digest(PROTOCOL) == PROTOCOL_SHA256, "Compiler protocol changed")
    cfg = json.loads(PROTOCOL.read_text())
    for entry in cfg["authority"].values():
        if isinstance(entry, dict) and "sha256" in entry:
            checked(entry)
    catalogue_manifest = json.loads(checked(cfg["authority"]["catalogue_manifest"]).read_text())
    catalogues = {}
    for source, entry in catalogue_manifest["sources"].items():
        checked(entry["original_metadata"])
        catalogue = json.loads(checked(entry["catalogue"]).read_text())
        candidates = catalogue["candidates"]
        require(len(candidates) == entry["candidate_count"], "Catalogue count differs")
        require(len({item["id"] for item in candidates}) == len(candidates), "Duplicate catalogue ID")
        require(all(item["description"] == item["id"].replace("_", " ") for item in candidates),
                "Unregistered catalogue description")
        catalogues[source] = candidates
    massive = catalogue_manifest["sources"][MASSIVE]
    checked(massive["rubric_document"])
    rubrics = json.loads(checked(massive["rubrics"]).read_text())
    return cfg, catalogues, rubrics


def ir_modules(cfg):
    root = Path(cfg["authority"]["IR"]["path"]).parents[1]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    ir = importlib.import_module("vey_u.ir")
    renderer = importlib.import_module("vey_u.semantic.format")
    for module, key in ((ir, "IR"), (renderer, "renderer")):
        require(Path(module.__file__).resolve() == checked(cfg["authority"][key]).resolve(),
                "Unexpected canonical module import")
    return ir, renderer


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank source/output row")
            yield json.loads(line)


def _memory(cfg):
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    require(int(fields["MemAvailable"].split()[0]) * 1024 >= cfg["resource_budget"]["RAM_guard_available_bytes"],
            "Available RAM below compiler guard")


def _committed():
    repo = HERE.parents[1]
    for path in (Path(__file__).resolve(), PROTOCOL):
        relative = path.relative_to(repo)
        snapshot = subprocess.run(["git", "show", "HEAD:" + str(relative)], cwd=repo,
                                  capture_output=True, check=False)
        require(snapshot.returncode == 0 and hashlib.sha256(snapshot.stdout).hexdigest() == digest(path),
                "Compiler and protocol must be committed before source materialization")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.strip()


def _membership(cfg, phase):
    require(phase in PHASES, "Selectors/custodian may request train or dev only")
    expected = json.loads(checked(cfg["authority"]["source_custody_result"]).read_text())["banking77_and_massive"]["corrected_counts"]
    groups, ordered, keys = {}, defaultdict(list), {}
    for row in _rows(checked(cfg["data"]["groups"])):
        if row["final_split"] != phase:
            continue
        require(row["source_id"] in cfg["data"]["sources"], "Unregistered selected source")
        gid = row["group_id"]
        require(gid not in groups, "Duplicate selected group")
        groups[gid] = row
        ordered[row["source_id"]].append(gid)
        for key in row["ordered_source_row_keys"]:
            require(key not in keys, "Duplicate selected source key")
            keys[key] = gid
    summaries = {}
    for source in cfg["data"]["sources"]:
        gids = ordered[source]
        require(gids == sorted(gids, key=lambda gid: (groups[gid]["rank_sha256"], gid)), "Source group order differs")
        components = list(dict.fromkeys(groups[gid]["component_id"] for gid in gids))
        summary = {"source_lineage_group_count": len(gids), "connected_component_count": len(components),
                   "ordered_group_ids_sha256": value_digest(gids),
                   "ordered_component_ids_sha256": value_digest(components)}
        require(summary == expected["group_split_hashes"][source][phase], "Original group/component membership differs")
        count = sum(len(groups[gid]["ordered_source_row_keys"]) for gid in gids)
        require(count == expected["source_row_counts_by_final_split"][source][phase], "Original row membership count differs")
        summaries[source] = {**summary, "source_row_count": count}
    return groups, keys, summaries


def projection(envelope, cfg, catalogues, rubrics, ir, renderer):
    """Only a previously membership-checked train/dev envelope may enter here."""
    phase = envelope["final_split"]
    require(phase in PHASES, "Sealed source projection denied")
    source = envelope["source_id"]
    require(source in catalogues, "Unregistered projection source")
    original = envelope["source_record_with_original_labels_and_annotations"]
    require(isinstance(original, dict), "Selected source payload must be an object")
    field = "text" if source == BANKING else "utt"
    text = original[field]
    require(isinstance(text, str) and bool(text.strip()), "Invalid selected utterance")
    require(text == envelope["input_text"], "Original utterance bytes differ from captured input")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    require(hashlib.sha256(normalized.encode()).hexdigest() == envelope["normalized_input_sha256"],
            "Captured input-only fingerprint differs")
    native_label = original["category" if source == BANKING else "intent"]
    catalogue = catalogues[source]
    require(type(native_label) is str and native_label in {item["id"] for item in catalogue},
            "Invalid present source-native intent ID")
    base = "banking77" if source == BANKING else "massive"
    endpoints = [(base + ".intent", cfg["projection"]["choice"]["question"], catalogue,
                  (native_label,), {"kind": "source_native_categorical", "target_available": True,
                                    "label_id": native_label}, "choice")]
    if source == MASSIVE:
        presence = "missing" if "judgments" not in original else "null" if original["judgments"] is None else "present"
        judgments = [] if presence != "present" else original["judgments"]
        require(isinstance(judgments, list) and all(isinstance(j, dict) for j in judgments),
                "Invalid selected judgment collection")
        for name in ("grammar_score", "spelling_score"):
            rubric = rubrics[name]
            maximum = rubric["native_level_max"]
            ratings, counts = [], [0] * (maximum + 1)
            for index, judgment in enumerate(judgments):
                present = name in judgment
                value = judgment.get(name)
                observed = present and value is not None
                if observed:
                    require(type(value) is int and 0 <= value <= maximum, "Invalid present native rating type/range")
                    counts[value] += 1
                ratings.append({"rater_index": index, "present": present, "value": value, "observed": observed})
            n = sum(counts)
            target = {"kind": "source_retained_rater_ordinal", "target_available": n > 0,
                      "judgments_presence": presence, "ratings": ratings, "native_level_max": maximum,
                      "counts": counts, "observed_raters": n,
                      "distribution": [count / n for count in counts] if n else None,
                      "mean_level": sum(index * count for index, count in enumerate(counts)) / n if n else None,
                      "population_probability_claim": False}
            endpoints.append(("massive." + name, rubric["question"], rubric["candidates"], (), target, "score"))
    for endpoint, question, candidates, gold, target, task in endpoints:
        row_id = value_digest([envelope["row_key"], endpoint])
        row = ir.DecisionIR(id=row_id, source=source, split=phase, task=task,
            state_blocks=(ir.StateBlock(id="utterance", text=text),), question=question,
            candidates=tuple(ir.Candidate(**item) for item in candidates), gold=gold,
            license="CC-BY-4.0", metadata={"endpoint": endpoint, "target_available": target["target_available"]})
        serving = {"id": row.id, "task": task, "state": renderer.state_text(row),
                   "question": renderer.question_text(row), "locale": envelope["locale"],
                   "candidates": [{"id": candidate.id, "text": renderer.candidate_text(candidate)} for candidate in row.candidates]}
        provenance = {key: envelope[key] for key in ("row_key", "source_id", "official_partition", "locale",
            "original_utterance_id", "normalized_input_sha256", "source_file", "source_line_number", "group_id", "component_id")}
        provenance.update(id=row_id, endpoint=endpoint, final_split=phase, license="CC-BY-4.0",
            modification="Source-native DecisionIR projection; catalogue underscores rendered as spaces; native rater targets retained.")
        yield serving, asdict(row), {"id": row_id, **target}, provenance


def _writer(stack, path):
    raw = stack.enter_context(path.open("xb"))
    return stack.enter_context(gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0))


def compile_phase(phase):
    require(phase in PHASES, "Only train/dev materialization allowed")
    cfg, catalogues, rubrics = metadata()
    _memory(cfg)
    revision = _committed()
    groups, expected_keys, membership = _membership(cfg, phase)
    source_path = checked(cfg["data"]["source_rows"])
    checked(cfg["data"]["source_files_manifest"])
    ir, renderer = ir_modules(cfg)
    directory = Path(cfg["data"]["output_root"]) / phase
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    seen, source_counts, coverage = set(), Counter(), defaultdict(Counter)
    row_order_hash, decision_order_hash = hashlib.sha256(), hashlib.sha256()
    decisions = 0
    try:
        with ExitStack() as stack:
            outputs = {name: _writer(stack, directory / (name + ".jsonl.gz")) for name in STREAMS}
            for envelope in _rows(source_path):
                if envelope["final_split"] != phase:
                    continue
                key = envelope["row_key"]
                require(key in expected_keys and key not in seen, "Unexpected/duplicate selected source row")
                group = groups[expected_keys[key]]
                for field in ("group_id", "component_id", "source_id", "final_split", "official_partition"):
                    require(envelope[field] == group[field], "Selected source envelope membership differs")
                seen.add(key)
                source_counts[envelope["source_id"]] += 1
                row_order_hash.update((key + "\n").encode())
                if len(seen) % 1000 == 0:
                    _memory(cfg)
                for bundle in projection(envelope, cfg, catalogues, rubrics, ir, renderer):
                    serving, _, target, provenance = bundle
                    for name, value in zip(STREAMS, bundle):
                        outputs[name].write((canonical(value) + "\n").encode("utf-8"))
                    decision_order_hash.update((serving["id"] + "\n").encode())
                    decisions += 1
                    bucket = coverage[provenance["endpoint"] + "/" + str(provenance["locale"])]
                    bucket["decisions"] += 1
                    bucket["target_available"] += int(target["target_available"])
                    bucket["target_unavailable"] += int(not target["target_available"])
                    for rating in target.get("ratings", ()):
                        bucket["present_integer"] += int(rating["observed"])
                        bucket["missing_field"] += int(not rating["present"])
                        bucket["null_field"] += int(rating["present"] and not rating["observed"])
        require(seen == set(expected_keys), "Selected source population incomplete")
        for source, summary in membership.items():
            require(source_counts[source] == summary["source_row_count"], "Selected source denominator differs")
        artifacts = {name: {"path": name + ".jsonl.gz", "sha256": digest(directory / (name + ".jsonl.gz")),
                            "bytes": (directory / (name + ".jsonl.gz")).stat().st_size} for name in STREAMS}
        result = {"schema": "vey.neutral.native-projection.v1", "status": "projection_complete_selector_closed",
            "phase": phase, "protocol_sha256": PROTOCOL_SHA256, "compiler_sha256": digest(Path(__file__)),
            "git_commit": revision, "environment": {"python": platform.python_version(), "model_imports": False},
            "source_inputs": cfg["data"], "authority": cfg["authority"], "source_membership": membership,
            "source_rows": len(seen), "decision_rows": decisions, "source_row_counts": dict(source_counts),
            "coverage": {key: dict(value) for key, value in sorted(coverage.items())},
            "ordered_source_row_keys_newline_sha256": row_order_hash.hexdigest(),
            "ordered_decision_ids_newline_sha256": decision_order_hash.hexdigest(), "files": artifacts,
            "custodial_mixed_envelopes_parsed": True, "sealed_phase_payloads_projected": False,
            "selector_sealed_phase_access": False, "model_executed": False,
            "quality_credit": False, "promotion": False, "endgame_complete": False}
        with (directory / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(canonical(result) + "\n")
        return result
    except BaseException as error:
        with (directory / "failure.json").open("x", encoding="utf-8") as stream:
            stream.write(canonical({"phase": phase, "error_type": type(error).__name__,
                                    "message": str(error), "selector_open": False}) + "\n")
        raise


def open_phase(phase, stream="serving"):
    """Return a hashed completed train/dev stream only after independent verification."""
    require(phase in PHASES, "Sealed/unknown selector phase denied")
    require(stream in STREAMS, "Unregistered selector stream")
    cfg, _, _ = metadata()
    directory = Path(cfg["data"]["output_root"]) / phase
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads((directory / "verification_receipt.json").read_text())
    require(manifest["phase"] == phase and manifest["protocol_sha256"] == PROTOCOL_SHA256,
            "Selector manifest provenance differs")
    require(manifest["compiler_sha256"] == digest(Path(__file__)), "Selector compiler identity differs")
    require(verification["status"] == "PASS" and verification["manifest_sha256"] == digest(manifest_path),
            "Independent source projection verification required")
    entry = manifest["files"][stream]
    path = directory / (stream + ".jsonl.gz")
    require(entry["path"] == path.name and digest(path) == entry["sha256"], "Selector stream hash/path differs")
    return _rows(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    args = parser.parse_args(argv)
    result = compile_phase(args.phase)
    print(canonical({"phase": result["phase"], "source_rows": result["source_rows"],
                     "decision_rows": result["decision_rows"], "selector_open": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
