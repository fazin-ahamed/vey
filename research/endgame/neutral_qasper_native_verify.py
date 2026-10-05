#!/usr/bin/env python3
"""Independently reconstruct QNATIVE-1 train/dev artifacts from guarded native fields."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "neutral_qasper_native_protocol.json"
PROTOCOL_SHA256 = "87b016779e700d63f4426f1626735ed8e5efbe90cbfe18ab9953f19c985b95b0"
COMPILER = HERE / "neutral_qasper_native_compile.py"
PHASES = ("train", "dev")
STREAMS = ("serving", "decisions", "targets", "provenance")
ENDPOINTS = ("qasper.yes_no", "qasper.answerability", "qasper.evidence_retrieval", "qasper.extractive_answer")
MODIFICATION = "QNATIVE-1: source-only DecisionIR projection; native annotations retained separately; answerability is the complement of native unanswerable."
READ_COLUMNS = {
    "source_groups": frozenset(("group_id", "source_id", "pinned_revision", "component_id", "final_split")),
    "papers": frozenset(("paper_pk", "paper_id", "source_partition", "title_json", "abstract_json", "full_text_json", "group_id", "question_count", "annotation_count")),
    "questions": frozenset(("question_pk", "paper_pk", "ordinal", "question_id_present", "question_id_json", "question_json", "annotation_count")),
    "annotations": frozenset(("paper_pk", "question_pk", "ordinal", "yes_no_present", "yes_no_json", "unanswerable_present", "unanswerable_json", "evidence_present", "evidence_json", "extractive_spans_present", "extractive_spans_json", "free_form_answer_present", "free_form_answer_json")),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_phase(phase):
    require(type(phase) is str and phase in PHASES, "Only train/dev verification is registered")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def value_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _nonfinite(_value):
    raise RuntimeError("Nonfinite source/artifact JSON is prohibited")


def decode(value):
    return json.loads(value, parse_constant=_nonfinite)


def checked(entry):
    path = Path(entry["path"])
    require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Pinned artifact identity differs")
    return path


def protocol():
    require(digest(PROTOCOL) == PROTOCOL_SHA256, "Verifier protocol identity differs")
    cfg = decode(PROTOCOL.read_text(encoding="utf-8"))
    require(cfg["schema"] == "vey.neutral.qasper.native-projection-protocol.v1", "Unregistered protocol schema")
    require(cfg["data"]["roles"] == list(PHASES), "Registered source phases differ")
    return cfg


def metadata():
    cfg = protocol()
    for entry in cfg["authority"].values():
        if isinstance(entry, dict) and "sha256" in entry:
            checked(entry)
    return cfg


def _committed():
    repo = HERE.parents[1]
    for path in (COMPILER, Path(__file__).resolve(), PROTOCOL):
        relative = path.relative_to(repo)
        snapshot = subprocess.run(["git", "show", "HEAD:" + str(relative)], cwd=repo, capture_output=True, check=False)
        require(snapshot.returncode == 0 and hashlib.sha256(snapshot.stdout).hexdigest() == digest(path),
                "Producer, independent verifier and protocol must be committed before source rows")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def ir_modules(cfg):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(cfg["authority"]["IR"]["path"]).parents[1]))
    ir = importlib.import_module("vey_u.ir")
    renderer = importlib.import_module("vey_u.semantic.format")
    for module, name in ((ir, "IR"), (renderer, "renderer")):
        require(Path(module.__file__).resolve() == checked(cfg["authority"][name]).resolve(), "Unexpected canonical module")
    return ir, renderer


def _read_authorizer(actual):
    def authorize(action, table, column, database, trigger):
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ and database == "main" and trigger is None and column in READ_COLUMNS.get(table, ()):
            actual.add(table + "." + column)
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return authorize


@contextmanager
def _database(phase, cfg, actual):
    check_phase(phase)
    connection = sqlite3.connect(Path(cfg["authority"]["database"]["path"]).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA cache_size=-16384")
        connection.execute("PRAGMA mmap_size=0")
        connection.row_factory = sqlite3.Row
        connection.set_authorizer(_read_authorizer(actual))
        yield connection
    finally:
        connection.close()


def _integer(value, label):
    require(type(value) is int and value >= 0, label)
    return value


def _field(row, field):
    present = row[field + "_present"]
    require(type(present) is int and present in (0, 1), "Invalid native field presence flag")
    raw = row[field + "_json"]
    require(present or raw is None, "Absent native field has a stored value")
    require(not present or type(raw) is str, "Present native field lacks its JSON value")
    return bool(present), decode(raw) if present else None


def _paper_fields(paper):
    fields, missing = {}, {}
    for field in ("title", "abstract", "full_text"):
        raw = paper[field + "_json"]
        fields[field] = decode(raw) if raw is not None else None
        missing[field] = {"present": raw is not None, "null": raw is not None and fields[field] is None}
    return fields, missing


def independent_state(paper, ir):
    fields, missing = _paper_fields(paper)
    blocks, sections = [], []

    def add(block_id, text):
        require(text is None or type(text) is str, "Invalid non-null source text")
        if text:
            blocks.append(ir.StateBlock(id=block_id, text=text))

    for field in ("title", "abstract"):
        add(field, fields[field])
    full = fields["full_text"]
    missing["full_text_shape"] = "missing" if not missing["full_text"]["present"] else "null" if full is None else "sections" if type(full) is list else "columnar"
    missing["column_fields"] = None
    if full is not None:
        if type(full) is list:
            sections = full
        else:
            require(type(full) is dict, "Invalid native full_text shape")
            columns = {}
            missing["column_fields"] = {}
            for field in ("section_name", "paragraphs"):
                value = full.get(field)
                require(value is None or type(value) is list, "Invalid native full_text column")
                columns[field] = value or []
                missing["column_fields"][field] = {"present": field in full, "null": field in full and value is None}
            for index in range(max(len(columns["section_name"]), len(columns["paragraphs"]))):
                sections.append({field: values[index] for field, values in columns.items() if index < len(values)})
    records = []
    for index, section in enumerate(sections):
        require(section is None or type(section) is dict, "Invalid non-null native section")
        record = {"section_ordinal": index, "section_null": section is None, "section_name_present": False,
                  "section_name_null": False, "paragraphs_present": False, "paragraphs_null": False, "null_paragraph_ordinals": []}
        if section is not None:
            for field in ("section_name", "paragraphs"):
                record[field + "_present"] = field in section
                record[field + "_null"] = field in section and section[field] is None
            add("section:%04d:heading" % index, section.get("section_name"))
            paragraphs = section.get("paragraphs")
            require(paragraphs is None or type(paragraphs) is list, "Invalid non-null native paragraphs")
            for position, text in enumerate(paragraphs or []):
                if text is None:
                    record["null_paragraph_ordinals"].append(position)
                add("section:%04d:paragraph:%04d" % (index, position), text)
        records.append(record)
    missing["section_fields"] = records
    missing["empty_document"] = not blocks
    if not blocks:
        blocks.append(ir.StateBlock(id="document", text=""))
    return tuple(blocks), missing


def independent_annotations(annotations):
    decoded = []
    for ordinal, annotation in enumerate(annotations):
        require(annotation["ordinal"] == ordinal and type(annotation["ordinal"]) is int, "Duplicate/noncontiguous native annotation ordinal")
        result = {"annotation_ordinal": ordinal}
        for field in ("yes_no", "unanswerable", "evidence", "extractive_spans", "free_form_answer"):
            present, value = _field(annotation, field)
            if value is not None:
                if field in ("yes_no", "unanswerable"):
                    require(type(value) is bool, "Invalid non-null native Boolean type")
                elif field in ("evidence", "extractive_spans"):
                    require(type(value) is list and all(type(item) is str for item in value), "Invalid non-null native string-array type")
                else:
                    require(type(value) in (str, bool, int, float), "Native free-form provenance must be primitive")
            result[field] = {"present": present, "native_value": value}
        decoded.append(result)
    return decoded


def independent_boolean(annotations, field):
    ratings, counts, conflicts = [], [0, 0], []
    for annotation in annotations:
        native = annotation[field]
        value = native["native_value"]
        observed = native["present"] and type(value) is bool
        require(value is None or type(value) is bool, "Invalid non-null native Boolean type")
        mapped = (not value if field == "unanswerable" else value) if observed else None
        if observed:
            counts[int(mapped)] += 1
        ordinal = annotation["annotation_ordinal"]
        ratings.append({"annotation_ordinal": ordinal, **native, "observed": observed, "mapped_value": mapped})
        if type(annotation["yes_no"]["native_value"]) is bool and annotation["unanswerable"]["native_value"] is True:
            conflicts.append(ordinal)
    n = sum(counts)
    return {"kind": "source_transformed_answerability" if field == "unanswerable" else "source_retained_boolean",
            "target_available": n > 0, "ratings": ratings, "counts": counts,
            "distribution": [count / n for count in counts] if n else None, "observed_raters": n,
            "conflicting_annotation_ordinals": sorted(conflicts)}


def independent_native_items(annotations, blocks, extraction):
    field = "extractive_spans" if extraction else "evidence"
    ratings, resolved, resolved_seen = [], [], set()
    matched_ids, native_count, unmatched_count = set(), 0, 0
    for annotation in annotations:
        native = annotation[field]
        value = native["native_value"]
        require(value is None or type(value) is list and all(type(item) is str for item in value), "Invalid native string-array type")
        matches, unmatched = [], []
        for index, text in enumerate(value or []):
            native_count += 1
            item = {"source_index": index, "text": text}
            if extraction:
                occurrences = []
                if text:
                    for block in blocks:
                        position = block.text.find(text)
                        while position >= 0:
                            occurrence = {"block_id": block.id, "start": position, "end": position + len(text)}
                            occurrences.append(occurrence)
                            key = (block.id, position, position + len(text))
                            if key not in resolved_seen:
                                resolved_seen.add(key)
                                resolved.append(occurrence)
                            position = block.text.find(text, position + 1)
                matches.append({**item, "occurrences": occurrences})
                found = bool(occurrences)
            else:
                block_ids = [block.id for block in blocks if block.text == text]
                matched_ids.update(block_ids)
                matches.append({**item, "block_ids": block_ids})
                found = bool(block_ids)
            if not found:
                unmatched.append(item)
                unmatched_count += 1
        ratings.append({"annotation_ordinal": annotation["annotation_ordinal"], **native,
                        "observed": native["present"] and value is not None, "matches": matches, "unmatched": unmatched})
    target = {"kind": "source_native_extractive" if extraction else "source_native_evidence",
              "target_available": native_count > 0, "ratings": ratings, "native_item_count": native_count, "unmatched_item_count": unmatched_count}
    if not extraction:
        target["matched_block_ids"] = [block.id for block in blocks if block.id in matched_ids]
    return target, resolved


def independent_question(paper, question, annotations, cfg, phase, blocks, missing, ir, renderer):
    check_phase(phase)
    require(type(question["ordinal"]) is int and question["ordinal"] >= 0, "Invalid native question ordinal")
    qa = decode(question["question_json"])
    require(type(qa) is dict and type(qa.get("question")) is str, "Native question must be an original string")
    question_text = qa["question"]
    qid_present, qid_value = _field(question, "question_id")
    source_annotations = independent_annotations(annotations)
    targets = {ENDPOINTS[0]: independent_boolean(source_annotations, "yes_no"),
               ENDPOINTS[1]: independent_boolean(source_annotations, "unanswerable")}
    targets[ENDPOINTS[2]], _ = independent_native_items(source_annotations, blocks, False)
    targets[ENDPOINTS[3]], occurrences = independent_native_items(source_annotations, blocks, True)
    identities = {key: paper[key] for key in ("paper_id", "source_partition", "group_id", "component_id")}
    identities.update(question_ordinal=question["ordinal"], phase=phase,
                      source_question_id_present=qid_present, source_question_id_value=qid_value)
    endpoint_ids = {}
    bundles = []
    boolean_candidates = tuple(ir.Candidate(id=item["id"], description=item["description"]) for item in cfg["projection"]["boolean"]["candidates"])
    require(cfg["projection"]["answerability"]["candidates"] == cfg["projection"]["boolean"]["candidates"], "Boolean catalogues differ")
    for index, endpoint in enumerate(ENDPOINTS):
        target = targets[endpoint]
        eligible = index != 0 or target["target_available"]
        row_id = value_digest([cfg["source"]["id"], cfg["source"]["revision"], paper["group_id"], paper["paper_id"], paper["source_partition"], question["ordinal"], endpoint])
        endpoint_ids[endpoint] = {"id": row_id if eligible else None, "eligible": eligible}
        if not eligible:
            continue
        if index < 2:
            candidates = boolean_candidates
            gold = tuple(label for label, count in zip(("false", "true"), target["counts"]) if count)
            evidence = ()
            prompt = question_text if index == 0 else cfg["projection"]["answerability"]["question_prefix"] + question_text
            task = "bool"
        elif index == 2:
            candidates = tuple(ir.Candidate(id=block.id, description=block.text) for block in blocks)
            gold = tuple(target["matched_block_ids"])
            evidence = tuple(ir.Evidence(block_id=block_id) for block_id in gold)
            prompt = cfg["projection"]["evidence"]["question_prefix"] + question_text
            task = "retrieval"
        else:
            candidates, gold = (), ()
            evidence = tuple(ir.Evidence(block_id=item["block_id"], span=(item["start"], item["end"])) for item in occurrences)
            prompt, task = question_text, "extract"
        decision = ir.DecisionIR(id=row_id, source=cfg["source"]["id"], split=phase, task=task, state_blocks=blocks,
                                 question=prompt, candidates=candidates, gold=gold, evidence=evidence, license="CC-BY-4.0",
                                 metadata={"endpoint": endpoint, "target_available": target["target_available"]})
        serving = {"id": row_id, "task": task, "locale": "en", "state": renderer.state_text(decision),
                   "question": renderer.question_text(decision),
                   "candidates": [{"id": item.id, "text": renderer.candidate_text(item)} for item in candidates]}
        provenance = {"id": row_id, "endpoint": endpoint, **identities, "modification": MODIFICATION, "license": "CC-BY-4.0",
                      "free_form_answers": [{"annotation_ordinal": annotation["annotation_ordinal"], **annotation["free_form_answer"]} for annotation in source_annotations],
                      "state_missingness": missing}
        bundles.append((serving, asdict(decision), {"id": row_id, "endpoint": endpoint, **target}, provenance))
    census = {**identities, "source_id": cfg["source"]["id"], "source_revision": cfg["source"]["revision"],
              "annotation_count": len(annotations), "endpoints": endpoint_ids}
    return census, bundles, targets


def _coverage():
    common = ("questions", "annotations", "eligible_questions", "ineligible_questions", "decisions", "target_available", "target_unavailable", "observed_raters", "missing_fields", "null_fields", "conflicting_questions", "conflicting_annotations")
    native = ("native_item_count", "unmatched_item_count", "matched_item_count", "full_support_questions", "mixed_support_questions", "no_support_questions", "no_native_items_questions")
    return {endpoint: dict.fromkeys(common + (native if index >= 2 else ()), 0) for index, endpoint in enumerate(ENDPOINTS)}


def _count_question(coverage, census, targets):
    for index, endpoint in enumerate(ENDPOINTS):
        bucket, target = coverage[endpoint], targets[endpoint]
        eligible = census["endpoints"][endpoint]["eligible"]
        bucket["questions"] += 1
        bucket["annotations"] += census["annotation_count"]
        bucket["eligible_questions"] += int(eligible)
        bucket["ineligible_questions"] += int(not eligible)
        bucket["decisions"] += int(eligible)
        bucket["target_available"] += int(eligible and target["target_available"])
        bucket["target_unavailable"] += int(eligible and not target["target_available"])
        for rating in target["ratings"]:
            bucket["observed_raters"] += int(rating["observed"])
            bucket["missing_fields"] += int(not rating["present"])
            bucket["null_fields"] += int(rating["present"] and rating["native_value"] is None)
        if index < 2:
            conflicts = len(target["conflicting_annotation_ordinals"])
            bucket["conflicting_annotations"] += conflicts
            bucket["conflicting_questions"] += int(conflicts > 0)
        else:
            n, unmatched = target["native_item_count"], target["unmatched_item_count"]
            bucket["native_item_count"] += n
            bucket["unmatched_item_count"] += unmatched
            bucket["matched_item_count"] += n - unmatched
            bucket["no_native_items_questions"] += int(n == 0)
            bucket["full_support_questions"] += int(n > 0 and unmatched == 0)
            bucket["mixed_support_questions"] += int(0 < unmatched < n)
            bucket["no_support_questions"] += int(n > 0 and unmatched == n)


def _source_groups(connection, phase, cfg):
    check_phase(phase)
    collision = connection.execute(
        "SELECT s.group_id FROM source_groups AS s JOIN source_groups AS other ON other.component_id=s.component_id "
        "WHERE s.final_split=? AND (other.final_split IS NULL OR other.final_split<>s.final_split) LIMIT 1", (phase,)).fetchone()
    require(collision is None, "Selected component crosses source roles")
    groups = {}
    for row in connection.execute(
            "SELECT group_id,source_id,pinned_revision,component_id,final_split FROM source_groups WHERE final_split=? ORDER BY group_id", (phase,)):
        require(row["source_id"] == cfg["source"]["id"] and row["pinned_revision"] == cfg["source"]["revision"], "Original source group identity differs")
        require(row["final_split"] == phase and type(row["component_id"]) is str and row["component_id"], "Invalid selected source component")
        require(type(row["group_id"]) is str and row["group_id"] and row["group_id"] not in groups, "Duplicate/invalid selected source group")
        groups[row["group_id"]] = row["component_id"]
    return groups


def _papers(connection, phase):
    check_phase(phase)
    return connection.execute(
        "SELECT p.paper_pk,p.paper_id,p.source_partition,p.title_json,p.abstract_json,p.full_text_json,p.group_id,"
        "p.question_count,p.annotation_count,s.component_id,s.final_split FROM papers AS p "
        "JOIN source_groups AS s ON s.group_id=p.group_id WHERE s.final_split=? "
        "ORDER BY p.group_id,p.paper_id,p.source_partition,p.paper_pk", (phase,))


def _questions(connection, phase, paper):
    check_phase(phase)
    return connection.execute(
        "SELECT q.question_pk,q.paper_pk,q.ordinal,q.question_id_present,q.question_id_json,q.question_json,q.annotation_count "
        "FROM questions AS q JOIN papers AS p ON p.paper_pk=q.paper_pk JOIN source_groups AS s ON s.group_id=p.group_id "
        "WHERE s.final_split=? AND p.paper_pk=? ORDER BY q.ordinal,q.question_pk", (phase, paper["paper_pk"]))


def _paper_annotation_membership(connection, phase, paper):
    check_phase(phase)
    orphan = connection.execute(
        "SELECT a.ordinal FROM annotations AS a JOIN papers AS p ON p.paper_pk=a.paper_pk "
        "JOIN source_groups AS s ON s.group_id=p.group_id LEFT JOIN questions AS q ON q.question_pk=a.question_pk "
        "WHERE s.final_split=? AND p.paper_pk=? AND (q.question_pk IS NULL OR q.paper_pk<>p.paper_pk) LIMIT 1",
        (phase, paper["paper_pk"])).fetchone()
    require(orphan is None, "Selected paper annotation has no matching native question")


def _annotations(connection, phase, paper, question):
    check_phase(phase)
    mismatch = connection.execute(
        "SELECT a.ordinal FROM annotations AS a JOIN questions AS q ON q.question_pk=a.question_pk "
        "JOIN papers AS p ON p.paper_pk=q.paper_pk JOIN source_groups AS s ON s.group_id=p.group_id "
        "WHERE s.final_split=? AND p.paper_pk=? AND q.question_pk=? AND a.paper_pk<>p.paper_pk LIMIT 1",
        (phase, paper["paper_pk"], question["question_pk"])).fetchone()
    require(mismatch is None, "Annotation and question paper identity differ")
    fields = "a.paper_pk,a.question_pk,a.ordinal,a.yes_no_present,a.yes_no_json,a.unanswerable_present,a.unanswerable_json,a.evidence_present,a.evidence_json,a.extractive_spans_present,a.extractive_spans_json,a.free_form_answer_present,a.free_form_answer_json"
    return list(connection.execute(
        "SELECT " + fields + " FROM annotations AS a JOIN questions AS q ON q.question_pk=a.question_pk "
        "JOIN papers AS p ON p.paper_pk=q.paper_pk JOIN source_groups AS s ON s.group_id=p.group_id "
        "WHERE s.final_split=? AND p.paper_pk=? AND q.question_pk=? AND a.paper_pk=p.paper_pk ORDER BY a.ordinal",
        (phase, paper["paper_pk"], question["question_pk"])))


def _artifact(directory, name, entry):
    require(set(entry) == {"path", "sha256", "bytes", "rows"} and entry["path"] == name + ".jsonl.gz", "Unrooted/invalid projection artifact")
    _integer(entry["rows"], "Invalid declared artifact row count")
    _integer(entry["bytes"], "Invalid declared artifact byte count")
    path = directory / entry["path"]
    require(path.resolve().parent == directory.resolve() and not path.is_symlink(), "Projection artifact escapes phase root")
    require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Projection artifact identity differs")
    with path.open("rb") as raw:
        header = raw.read(10)
    require(len(header) == 10 and header[:3] == b"\x1f\x8b\x08" and header[4:8] == b"\0\0\0\0", "Non-deterministic projection gzip header")
    return path


def _equal_line(stream, expected, name):
    require(stream.readline() == (canonical(expected) + "\n").encode("utf-8"), "Independent reconstruction differs: " + name)


def _archive_receipt(directory):
    path = directory / "verification_receipt.json"
    if not path.exists():
        return None
    version = 1
    while (directory / ("verification_receipt_superseded_v%d.json" % version)).exists():
        version += 1
    retained = directory / ("verification_receipt_superseded_v%d.json" % version)
    identity = {"path": retained.name, "sha256": digest(path)}
    path.rename(retained)
    return identity


def _write_receipt(directory, receipt):
    with (directory / "verification_receipt.json").open("x", encoding="utf-8") as stream:
        stream.write(canonical(receipt) + "\n")


def _selector_module():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    compiler = importlib.import_module("neutral_qasper_native_compile")
    require(Path(compiler.__file__).resolve() == COMPILER.resolve(), "Unexpected phase selector module")
    return compiler


def _denied(call, label):
    try:
        call()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        return type(error).__name__
    raise RuntimeError(label)


def _selector_denials(compiler):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Denied selector attempted filesystem/database access")

    denials = {"phases": {}, "streams": {}}
    with patch("builtins.open", forbidden), patch.object(Path, "open", forbidden), patch.object(Path, "read_text", forbidden), patch.object(Path, "read_bytes", forbidden), patch.object(Path, "stat", forbidden), patch.object(Path, "exists", forbidden), patch.object(sqlite3, "connect", forbidden), patch.object(gzip, "open", forbidden):
        for phase in ("calibration", "confirmation", "final", "unused", "", "train ", None):
            denials["phases"][str(phase)] = _denied(lambda phase=phase: compiler.open_phase(phase), "Selector accepted sealed/unknown phase")
        for stream in ("source_census", "payload", "", None):
            denials["streams"][str(stream)] = _denied(lambda stream=stream: compiler.open_phase("train", stream), "Selector accepted unregistered stream")
    return denials


def _resource_guard(cfg):
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    require(int(fields["MemAvailable"].split()[0]) * 1024 >= cfg["resources"]["minimum_available_RAM_GiB"] * 1024 ** 3,
            "Available RAM below registered verifier guard")


def _verify_population(phase, cfg, directory, manifest, receipt):
    _resource_guard(cfg)
    started = time.monotonic()
    ir, renderer = ir_modules(cfg)
    actual, seen_papers, seen_questions, seen_groups = set(), set(), set(), set()
    coverage, counts, decision_ids = _coverage(), Counter(papers=0, questions=0, annotations=0), []
    with ExitStack() as stack:
        streams = {name: stack.enter_context(gzip.open(_artifact(directory, name, manifest["files"][name]), "rb")) for name in STREAMS}
        census_stream = stack.enter_context(gzip.open(_artifact(directory, "source_census", manifest["source_census"]), "rb"))
        connection = stack.enter_context(_database(phase, cfg, actual))
        groups = _source_groups(connection, phase, cfg)
        for native_paper in _papers(connection, phase):
            require(time.monotonic() - started <= cfg["resources"]["wall_timeout_seconds"], "Registered verification wall deadline exceeded")
            paper = dict(native_paper)
            pk = _integer(paper["paper_pk"], "Invalid native paper primary key")
            require(pk not in seen_papers and paper["final_split"] == phase, "Duplicate/off-phase native paper")
            require(type(paper["paper_id"]) is str and type(paper["source_partition"]) is str, "Invalid native paper identity")
            require(paper["group_id"] in groups and groups[paper["group_id"]] == paper["component_id"], "Native paper membership differs")
            seen_papers.add(pk)
            seen_groups.add(paper["group_id"])
            counts["papers"] += 1
            blocks, missing = independent_state(paper, ir)
            _paper_annotation_membership(connection, phase, paper)
            nq, na = 0, 0
            for native_question in _questions(connection, phase, paper):
                require(time.monotonic() - started <= cfg["resources"]["wall_timeout_seconds"], "Registered verification wall deadline exceeded")
                question = dict(native_question)
                require(type(question["ordinal"]) is int and question["ordinal"] == nq, "Duplicate/noncontiguous native question ordinal")
                qpk = _integer(question["question_pk"], "Invalid native question primary key")
                require(question["paper_pk"] == pk and qpk not in seen_questions, "Duplicate/mismatched native question identity")
                seen_questions.add(qpk)
                annotations = _annotations(connection, phase, paper, question)
                require(len(annotations) == _integer(question["annotation_count"], "Invalid native annotation count"), "Native question annotation census differs")
                census, bundles, targets = independent_question(paper, question, annotations, cfg, phase, blocks, missing, ir, renderer)
                _equal_line(census_stream, census, "source_census")
                _count_question(coverage, census, targets)
                for bundle in bundles:
                    for name, row in zip(STREAMS, bundle):
                        _equal_line(streams[name], row, name)
                    decision_ids.append(bundle[0]["id"])
                nq += 1
                na += len(annotations)
            require(nq == _integer(paper["question_count"], "Invalid native question count") and na == _integer(paper["annotation_count"], "Invalid native paper annotation count"), "Native paper census differs")
            counts["questions"] += nq
            counts["annotations"] += na
        require(seen_groups == set(groups), "Selected source group census incomplete")
        for stream in (*streams.values(), census_stream):
            require(stream.readline() == b"", "Projection artifact contains extra population rows")
    counts["groups"], counts["components"] = len(groups), len(set(groups.values()))
    require(len(decision_ids) == len(set(decision_ids)), "Duplicate native decision identity")
    membership = {"ordered_group_ids_sha256": value_digest(sorted(groups)),
                  "ordered_component_ids_sha256": value_digest(sorted(set(groups.values()))),
                  "ordered_decision_ids_sha256": value_digest(decision_ids)}
    expected = cfg["data"]["expected"][phase]
    require(dict(counts) == {key: expected[key] for key in ("papers", "questions", "annotations", "groups", "components")}, "Registered source population differs")
    require(membership["ordered_group_ids_sha256"] == expected["ordered_group_ids_sha256"], "Registered source group membership differs")
    require(canonical(manifest["source_counts"]) == canonical(dict(counts)) and canonical(manifest["membership"]) == canonical(membership), "Manifest census/membership differs")
    require(canonical(manifest["endpoints"]) == canonical(coverage), "Endpoint observed/missing/null/support coverage differs")
    require(manifest["source_census"]["rows"] == counts["questions"] and all(manifest["files"][name]["rows"] == len(decision_ids) for name in STREAMS), "Artifact row counts differ")
    declared = manifest["actual_SQL_READ_columns"]
    allowed = sorted(table + "." + column for table, columns in READ_COLUMNS.items() for column in columns)
    require(declared == allowed and sorted(actual) == allowed, "Independent/producer SQL READ columns differ from exact allowlist")
    receipt.update(source_counts=dict(counts), membership=membership, endpoints=coverage, decision_rows=len(decision_ids),
                   source_census_rows=counts["questions"], independent_actual_SQL_READ_columns=sorted(actual))


def verify_phase(phase):
    check_phase(phase)
    cfg = protocol()
    directory = Path(cfg["data"]["output_root"]) / phase
    require(directory.is_dir() and not directory.is_symlink(), "Completed projection phase directory required")
    marker = directory / "verification_in_progress.json"
    require(not marker.exists(), "Another verification attempt is active or interrupted")
    superseded = None
    receipt = {"schema": "vey.neutral.qasper.native-projection-verification.v1", "status": "FAIL", "phase": phase,
               "protocol_sha256": PROTOCOL_SHA256, "verifier_sha256": digest(Path(__file__)), "compiler_sha256": digest(COMPILER),
               "manifest_sha256": None, "sealed_labels_accessed": False, "model_outputs": 0, "quality_credit": False,
               "promotion": False, "endgame_complete": False}
    with marker.open("x", encoding="utf-8") as stream:
        stream.write(canonical({"phase": phase, "verifier_sha256": receipt["verifier_sha256"]}) + "\n")
    try:
        superseded = _archive_receipt(directory)
        if superseded is not None:
            receipt["supersedes"] = superseded
        cfg = metadata()
        receipt["source_rows_execution_commit"] = _committed()
        manifest_path = directory / "manifest.json"
        require(not manifest_path.is_symlink() and not (directory / "failure.json").exists(), "Compiler failure or unrooted manifest")
        manifest_hash = digest(manifest_path)
        receipt["manifest_sha256"] = manifest_hash
        manifest = decode(manifest_path.read_text(encoding="utf-8"))
        require(manifest["schema"] == "vey.neutral.qasper.native-projection-manifest.v1" and manifest["phase"] == phase and manifest["complete"] is True, "Immutable completed phase manifest required")
        require(manifest["protocol_sha256"] == PROTOCOL_SHA256 and manifest["compiler_sha256"] == receipt["compiler_sha256"] and manifest["verifier_sha256"] == receipt["verifier_sha256"], "Projection implementation identity differs")
        require(canonical(manifest["authority"]) == canonical(cfg["authority"]) and set(manifest["files"]) == set(STREAMS), "Projection authority/stream set differs")
        require(manifest["source_database_unchanged"] is True and manifest["source_rows_committed_code_only"] is True and manifest["sealed_labels_accessed"] is False and type(manifest["model_outputs"]) is int and manifest["model_outputs"] == 0 and manifest["quality_credit"] is False, "Projection custody/claim flags differ")
        compiler = _selector_module()
        receipt["access_control"] = {"invalid_selector_denials": _selector_denials(compiler),
                                     "in_progress_phase_denied": _denied(lambda: compiler.open_phase(phase), "Selector opened phase during verification")}
        marker.unlink()
        try:
            receipt["access_control"]["unverified_phase_denied"] = _denied(lambda: compiler.open_phase(phase), "Selector opened phase without an independent receipt")
        finally:
            with marker.open("x", encoding="utf-8") as stream:
                stream.write(canonical({"phase": phase, "verifier_sha256": receipt["verifier_sha256"]}) + "\n")
        _verify_population(phase, cfg, directory, manifest, receipt)
        for name in STREAMS:
            _artifact(directory, name, manifest["files"][name])
        _artifact(directory, "source_census", manifest["source_census"])
        metadata()
        _committed()
        require(digest(manifest_path) == manifest_hash and digest(Path(__file__)) == receipt["verifier_sha256"] and digest(COMPILER) == receipt["compiler_sha256"], "Source/manifest/implementation changed during verification")
        receipt.update(status="PASS", source_database_unchanged=True, source_rows_committed_code_only=True,
                       verifier={"path": str(Path(__file__).resolve()), "sha256": receipt["verifier_sha256"]},
                       projection_manifest={"path": str(manifest_path), "sha256": manifest_hash})
        _write_receipt(directory, receipt)
        marker.unlink()
        for name in STREAMS:
            n = 0
            with gzip.open(directory / (name + ".jsonl.gz"), "rb") as original:
                for row in compiler.open_phase(phase, name):
                    _equal_line(original, row, "verified selector " + name)
                    n += 1
                require(original.readline() == b"" and n == receipt["decision_rows"], "Verified selector population differs")
        require(digest(manifest_path) == manifest_hash and digest(Path(__file__)) == receipt["verifier_sha256"], "Verified selector changed custody")
        return receipt
    except BaseException as error:
        if (directory / "verification_receipt.json").exists():
            _archive_receipt(directory)
        receipt["status"] = "FAIL"
        receipt["failure"] = {"error_type": type(error).__name__, "check": str(error) if type(error) is RuntimeError else None, "source_text_disclosed": False}
        receipt.pop("source_database_unchanged", None)
        _write_receipt(directory, receipt)
        raise
    finally:
        if marker.exists():
            marker.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    args = parser.parse_args(argv)
    try:
        receipt = verify_phase(args.phase)
    except Exception as error:
        print(canonical({"phase": args.phase, "status": "FAIL", "error_type": type(error).__name__}))
        return 1
    print(canonical({"phase": receipt["phase"], "status": receipt["status"], "decision_rows": receipt["decision_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
