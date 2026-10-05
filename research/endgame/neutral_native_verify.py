#!/usr/bin/env python3
"""Independently reconstruct a sealed native train/dev projection without its compiler.

Source-native fields, membership, targets, renderer bytes and DecisionIR are
re-derived here from the pinned custody artifacts and the canonical DecisionIR
classes. The compiler's projection helpers are never imported; only its
CLI ``open_phase`` gate is exercised to prove the selector refuses sealed and
unverified phases. This is projection correctness, not model quality.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "neutral_native_compiler_protocol.json"
PROTOCOL_SHA256 = "f1da1e227e6195c2230f4f1a10c016e0c683a036a192030704ed9c67da259642"
PHASES = ("train", "dev")
STREAMS = ("serving", "decisions", "targets", "provenance")
BANKING = "PolyAI/banking77"
MASSIVE = "AmazonScience/massive"
RATINGS = (("grammar_score", 4), ("spelling_score", 2))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition, label):
    if not condition:
        raise RuntimeError(label)


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


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            require(bool(line.strip()), "Blank persisted row")
            yield json.loads(line)


def authoritative_catalogue(source_id, manifest):
    """Recover each candidate ID from the pinned original metadata bytes alone."""
    entry = manifest["sources"][source_id]
    body = checked(entry["original_metadata"]).read_bytes()
    if source_id == BANKING:
        require(len(body) == 2036 and hashlib.sha1(b"blob 2036\0" + body).hexdigest() == "cdd2a5c77a4079a455f8fb7e751d1ecee0e2a5a4",
                "Banking77 original catalogue bytes differ")
        return json.loads(body)
    require(len(body) == 30340 and hashlib.sha1(b"blob 30340\0" + body).hexdigest() == "4731be1adc5c42b075603588248d8ae99cb01896",
            "MASSIVE loader bytes differ")
    tree = ast.parse(body.decode("utf-8"))
    bindings = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, call in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "intent" and isinstance(call, ast.Call):
                    for keyword in call.keywords:
                        if keyword.arg == "names":
                            value = bindings[keyword.value.id] if isinstance(keyword.value, ast.Name) else keyword.value
                            found.append(ast.literal_eval(value))
    require(len(found) == 1, "MASSIVE loader intent catalogue is not uniquely declared")
    return found[0]


def independent_catalogue(source_id, manifest):
    ids = authoritative_catalogue(source_id, manifest)
    return [{"id": label, "description": label.replace("_", " ")} for label in ids]


def independent_rubrics(manifest):
    """Recover the exact native ordinal prompts and levels from the pinned rubric document."""
    entry = manifest["sources"][MASSIVE]
    document = checked(entry["rubric_document"]).read_text(encoding="utf-8")
    rubrics = {}
    for name, maximum in RATINGS:
        marker = name + ' : "'
        start = document.index(marker) + len(marker)
        end = document.index('"', start)
        question = document[start:end]
        levels, index = [], 0
        while index <= maximum:
            line_start = document.index("\n  " + str(index) + ": ", end) + 1
            line_end = document.index("\n", line_start)
            levels.append({"id": str(index), "description": document[line_start:line_end].split(": ", 1)[1]})
            index += 1
        require([level["id"] for level in levels] == [str(i) for i in range(maximum + 1)], "Native level set differs")
        rubrics[name] = {"question": question, "candidates": levels, "native_level_max": maximum}
    return rubrics


def expected_projection(envelope, cfg, catalogues, rubrics, ir, renderer):
    phase, source = envelope["final_split"], envelope["source_id"]
    require(phase in PHASES and source in catalogues, "Sealed or unregistered envelope")
    original = envelope["source_record_with_original_labels_and_annotations"]
    text = original["text" if source == BANKING else "utt"]
    require(isinstance(text, str) and text == envelope["input_text"], "Selected utterance bytes differ")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    require(hashlib.sha256(normalized.encode()).hexdigest() == envelope["normalized_input_sha256"],
            "Input-only fingerprint differs")
    native = original["category" if source == BANKING else "intent"]
    catalogue = catalogues[source]
    require(type(native) is str and native in {item["id"] for item in catalogue}, "Invalid native intent ID")
    base = "banking77" if source == BANKING else "massive"
    plans = [(base + ".intent", cfg["projection"]["choice"]["question"], catalogue, (native,),
              {"kind": "source_native_categorical", "target_available": True, "label_id": native}, "choice")]
    if source == MASSIVE:
        presence = "missing" if "judgments" not in original else "null" if original["judgments"] is None else "present"
        judgments = [] if presence != "present" else original["judgments"]
        require(isinstance(judgments, list) and all(isinstance(item, dict) for item in judgments), "Invalid judgments")
        for name, maximum in RATINGS:
            rubric = rubrics[name]
            ratings, counts = [], [0] * (maximum + 1)
            for index, judgment in enumerate(judgments):
                present = name in judgment
                value = judgment.get(name)
                observed = present and value is not None
                if observed:
                    require(type(value) is int and 0 <= value <= maximum, "Invalid present native rating")
                    counts[value] += 1
                ratings.append({"rater_index": index, "present": present, "value": value, "observed": observed})
            n = sum(counts)
            target = {"kind": "source_retained_rater_ordinal", "target_available": n > 0, "judgments_presence": presence,
                      "ratings": ratings, "native_level_max": maximum, "counts": counts, "observed_raters": n,
                      "distribution": [count / n for count in counts] if n else None,
                      "mean_level": sum(index * count for index, count in enumerate(counts)) / n if n else None,
                      "population_probability_claim": False}
            plans.append(("massive." + name, rubric["question"], rubric["candidates"], (), target, "score"))
    bundles = []
    for endpoint, question, candidates, gold, target, task in plans:
        row_id = value_digest([envelope["row_key"], endpoint])
        row = ir.DecisionIR(id=row_id, source=source, split=phase, task=task,
            state_blocks=(ir.StateBlock(id="utterance", text=text),), question=question,
            candidates=tuple(ir.Candidate(**item) for item in candidates), gold=gold, license="CC-BY-4.0",
            metadata={"endpoint": endpoint, "target_available": target["target_available"]})
        bundles.append(({"id": row.id, "task": task, "state": renderer.state_text(row), "question": renderer.question_text(row),
                          "locale": envelope["locale"],
                          "candidates": [{"id": c.id, "text": renderer.candidate_text(c)} for c in row.candidates]},
            asdict(row), {"id": row_id, **target},
            {"row_key": envelope["row_key"], "source_id": source, "official_partition": envelope["official_partition"],
             "locale": envelope["locale"], "original_utterance_id": envelope["original_utterance_id"],
             "normalized_input_sha256": envelope["normalized_input_sha256"], "source_file": envelope["source_file"],
             "source_line_number": envelope["source_line_number"], "group_id": envelope["group_id"],
             "component_id": envelope["component_id"], "id": row_id, "endpoint": endpoint, "final_split": phase,
             "license": "CC-BY-4.0",
             "modification": "Source-native DecisionIR projection; catalogue underscores rendered as spaces; native rater targets retained."}))
    return bundles


def ir_modules(cfg):
    root = Path(cfg["authority"]["IR"]["path"]).parents[1]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    ir = importlib.import_module("vey_u.ir")
    renderer = importlib.import_module("vey_u.semantic.format")
    for module, key in ((ir, "IR"), (renderer, "renderer")):
        require(Path(module.__file__).resolve() == checked(cfg["authority"][key]).resolve(), "Unexpected canonical module")
    return ir, renderer


def selector_denials(compiler, cfg):
    require(digest(PROTOCOL) == PROTOCOL_SHA256, "Verifier protocol identity differs")
    require(digest(compiler) == digest(HERE / "neutral_native_compile.py"), "Selector compiler identity differs")
    denials = {}
    for phase in ("calibration", "confirmation", "unused", "", "train "):
        try:
            compiler.open_phase(phase)
        except (RuntimeError, ValueError) as error:
            denials[phase] = type(error).__name__
        else:
            raise RuntimeError("Selector accepted a sealed/unknown phase: " + repr(phase))
    return denials


def verify(phase, compiler_path):
    require(phase in PHASES, "Only train/dev verification is registered")
    require(digest(PROTOCOL) == PROTOCOL_SHA256, "Compiler protocol changed")
    cfg = json.loads(PROTOCOL.read_text())
    for entry in cfg["authority"].values():
        if isinstance(entry, dict) and "sha256" in entry:
            checked(entry)
    custody = json.loads(checked(cfg["authority"]["source_custody_result"]).read_text())["banking77_and_massive"]["corrected_counts"]
    manifest = json.loads(checked(cfg["authority"]["catalogue_manifest"]).read_text())
    catalogues = {source: independent_catalogue(source, manifest) for source in (BANKING, MASSIVE)}
    for source, catalogue in catalogues.items():
        derived = json.loads(checked(manifest["sources"][source]["catalogue"]).read_text())["candidates"]
        require(catalogue == derived, "Registered catalogue differs from independently recovered source metadata: " + source)
    rubrics = independent_rubrics(manifest)
    require(rubrics == json.loads(checked(manifest["sources"][MASSIVE]["rubrics"]).read_text()),
            "Registered native rubrics differ from the pinned rubric document")
    directory = Path(cfg["data"]["output_root"]) / phase
    manifest_path = directory / "manifest.json"
    projection = json.loads(manifest_path.read_text())
    require(projection["phase"] == phase and projection["protocol_sha256"] == PROTOCOL_SHA256, "Projection provenance differs")
    require(projection["compiler_sha256"] == digest(Path(compiler_path)), "Projection compiler identity differs")
    for name, entry in projection["files"].items():
        path = directory / entry["path"]
        require(entry["path"] == name + ".jsonl.gz" and digest(path) == entry["sha256"] and path.stat().st_size == entry["bytes"],
                "Projection artifact hash differs")
    require(not (directory / "failure.json").exists(), "Failed projection retained a failure receipt")
    membership, expected_keys = {}, set()
    for row in rows(checked(cfg["data"]["groups"])):
        if row["final_split"] != phase:
            continue
        gid = row["group_id"]
        require(gid not in membership, "Duplicate selected group")
        membership[gid] = row
        expected_keys.update(row["ordered_source_row_keys"])
    per_source = defaultdict(list)
    for gid, row in membership.items():
        per_source[row["source_id"]].append((row["rank_sha256"], gid))
    for source, pairs in per_source.items():
        gids = [gid for _, gid in sorted(pairs)]
        components = list(dict.fromkeys(membership[gid]["component_id"] for gid in gids))
        expected = custody["group_split_hashes"][source][phase]
        require({"source_lineage_group_count": len(gids), "connected_component_count": len(components),
                 "ordered_group_ids_sha256": value_digest(gids), "ordered_component_ids_sha256": value_digest(components)} == expected,
                "Original source/component membership differs: " + source)
        require(sum(len(membership[gid]["ordered_source_row_keys"]) for gid in gids) == custody["source_row_counts_by_final_split"][source][phase],
                "Original source row membership differs: " + source)
    ir, renderer = ir_modules(cfg)
    streams = {name: rows(directory / (name + ".jsonl.gz")) for name in STREAMS}
    seen, decisions, coverage = set(), 0, defaultdict(Counter)
    order_hash, decision_hash = hashlib.sha256(), hashlib.sha256()
    selector_closure = None
    for envelope in rows(checked(cfg["data"]["source_rows"])):
        if envelope["final_split"] != phase:
            continue
        key = envelope["row_key"]
        require(key in expected_keys and key not in seen, "Unexpected/duplicate selected source row")
        seen.add(key)
        order_hash.update((key + "\n").encode())
        for bundle in expected_projection(envelope, cfg, catalogues, rubrics, ir, renderer):
            for name, value in zip(STREAMS, bundle):
                require(next(streams[name]) == value, "Independent projection differs from persisted stream: " + name)
            decision_hash.update((bundle[0]["id"] + "\n").encode())
            bucket = coverage[bundle[3]["endpoint"] + "/" + str(bundle[3]["locale"])]
            bucket["decisions"] += 1
            bucket["target_available"] += int(bundle[2]["target_available"])
            bucket["target_unavailable"] += int(not bundle[2]["target_available"])
            for rating in bundle[2].get("ratings", ()):
                bucket["present_integer"] += int(rating["observed"])
                bucket["missing_field"] += int(not rating["present"])
                bucket["null_field"] += int(rating["present"] and not rating["observed"])
            decisions += 1
    require(seen == expected_keys, "Selected source population incomplete")
    for stream in streams.values():
        require(next(stream, None) is None, "Persisted projection stream has extra rows")
    require(decisions == projection["decision_rows"] and len(seen) == projection["source_rows"], "Projection row totals differ")
    require(order_hash.hexdigest() == projection["ordered_source_row_keys_newline_sha256"], "Selected row order differs")
    require(decision_hash.hexdigest() == projection["ordered_decision_ids_newline_sha256"], "Decision order differs")
    require({key: dict(value) for key, value in sorted(coverage.items())} == projection["coverage"], "Coverage/type census differs")
    sys.path.insert(0, str(HERE))
    compiler = importlib.import_module("neutral_native_compile")
    denials = selector_denials(compiler, cfg)
    for name in STREAMS:
        require(list(compiler.open_phase(phase, name)) == list(rows(directory / (name + ".jsonl.gz"))), "Verified selector stream differs")
    selector_closure = {"sealed_phase_denials": denials, "verified_stream_openings": sorted(STREAMS)}
    receipt = {"schema": "vey.neutral.native-projection-verification.v1", "status": "PASS", "phase": phase,
        "protocol_sha256": PROTOCOL_SHA256, "projection_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
        "verifier": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "independent_catalogue_counts": {source: len(items) for source, items in catalogues.items()},
        "independent_rubric_levels": {name: maximum + 1 for name, maximum in RATINGS},
        "source_rows": len(seen), "decision_rows": decisions, "coverage": {key: dict(value) for key, value in sorted(coverage.items())},
        "access_control": selector_closure, "source_payloads_of_sealed_phases_read": False,
        "model_executed": False, "quality_credit": False, "promotion": False, "endgame_complete": False}
    with (directory / "verification_receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical(receipt) + "\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    args = parser.parse_args(argv)
    receipt = verify(args.phase, HERE / "neutral_native_compile.py")
    print(canonical({"phase": receipt["phase"], "status": receipt["status"], "decision_rows": receipt["decision_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())