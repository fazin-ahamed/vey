#!/usr/bin/env python3
"""Compile the pinned QASPER train/dev source into separated native-task streams."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, closing
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "neutral_qasper_native_protocol.json"
PROTOCOL_SHA256 = "87b016779e700d63f4426f1626735ed8e5efbe90cbfe18ab9953f19c985b95b0"
VERIFIER = HERE / "neutral_qasper_native_verify.py"
PHASES = ("train", "dev")
STREAMS = ("serving", "decisions", "targets", "provenance")
ENDPOINTS = ("qasper.yes_no", "qasper.answerability", "qasper.evidence_retrieval", "qasper.extractive_answer")
LICENSE = "CC-BY-4.0"
MODIFICATION = ("QNATIVE-1: source-only DecisionIR projection; native annotations retained separately; "
                "answerability is the complement of native unanswerable.")
READ_COLUMNS = {
    "papers": frozenset({"paper_pk", "source_partition", "paper_id", "title_json", "abstract_json",
                         "full_text_json", "group_id", "question_count", "annotation_count"}),
    "questions": frozenset({"question_pk", "paper_pk", "ordinal", "question_id_present", "question_id_json",
                            "question_json", "annotation_count"}),
    "annotations": frozenset({"paper_pk", "question_pk", "ordinal", "unanswerable_present", "unanswerable_json",
                              "yes_no_present", "yes_no_json", "evidence_present", "evidence_json",
                              "extractive_spans_present", "extractive_spans_json",
                              "free_form_answer_present", "free_form_answer_json"}),
    "source_groups": frozenset({"group_id", "source_id", "pinned_revision", "component_id", "final_split", "rank_sha256"}),
}
COVERAGE_KEYS = ("questions", "annotations", "eligible_questions", "ineligible_questions", "decisions",
                 "target_available", "target_unavailable", "observed_raters", "missing_fields", "null_fields",
                 "conflicting_questions", "conflicting_annotations")
SUPPORT_KEYS = ("native_item_count", "unmatched_item_count", "matched_item_count", "full_support_questions",
                "mixed_support_questions", "no_support_questions", "no_native_items_questions")


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


def _reject_constant(_value):
    raise RuntimeError("Nonfinite JSON value denied")


def _json(value):
    require(isinstance(value, str), "Stored JSON value is malformed")
    return json.loads(value, parse_constant=_reject_constant)


def checked(entry):
    path = Path(entry["path"])
    require(path.is_file() and path.stat().st_size == entry["bytes"], "Pinned artifact size differs")
    require(digest(path) == entry["sha256"], "Pinned artifact hash differs")
    return path


def protocol():
    require(digest(PROTOCOL) == PROTOCOL_SHA256, "QASPER compiler protocol changed")
    cfg = _json(PROTOCOL.read_text(encoding="utf-8"))
    require(cfg["schema"] == "vey.neutral.qasper.native-projection-protocol.v1", "Unregistered protocol schema")
    require(tuple(cfg["data"]["roles"]) == PHASES and tuple(cfg["projection"]["streams"]) == STREAMS,
            "Protocol phase/stream boundary differs")
    return cfg


def metadata():
    cfg = protocol()
    for entry in cfg["authority"].values():
        if isinstance(entry, dict):
            checked(entry)
    return cfg


def _committed():
    repo = HERE.parents[1]
    for path in (Path(__file__).resolve(), VERIFIER, PROTOCOL):
        relative = path.relative_to(repo)
        snapshot = subprocess.run(["git", "show", "HEAD:" + str(relative)], cwd=repo,
                                  capture_output=True, check=False)
        require(snapshot.returncode == 0 and hashlib.sha256(snapshot.stdout).hexdigest() == digest(path),
                "QASPER producer, independent verifier and protocol must be committed before source access")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                          text=True, check=True).stdout.strip()


def ir_modules(cfg):
    root = Path(cfg["authority"]["IR"]["path"]).parents[1]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    ir = importlib.import_module("vey_u.ir")
    renderer = importlib.import_module("vey_u.semantic.format")
    for module, key in ((ir, "IR"), (renderer, "renderer")):
        require(Path(module.__file__).resolve() == checked(cfg["authority"][key]).resolve(),
                "Unexpected canonical IR/renderer module")
    return ir, renderer


def _memory(cfg):
    fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(fields["MemAvailable"].split()[0]) * 1024
    require(available >= cfg["resources"]["minimum_available_RAM_GiB"] * 1024 ** 3,
            "Available RAM below QASPER compiler guard")


def _deadline(deadline):
    require(time.monotonic() <= deadline, "QASPER compiler wall deadline exceeded")


def _native_field(row, name):
    flag = row[name + "_present"]
    raw = row[name + "_json"]
    require(type(flag) is int and flag in (0, 1), "Malformed annotation presence flag")
    require(flag or raw is None, "Absent annotation field has a stored value")
    value = _json(raw) if flag else None
    if value is not None:
        if name in ("yes_no", "unanswerable"):
            require(type(value) is bool, "Present non-Boolean native Boolean denied")
        elif name in ("evidence", "extractive_spans"):
            require(type(value) is list and all(type(item) is str for item in value),
                    "Present malformed native string-array target denied")
        else:
            require(type(value) in (str, bool, int, float), "Nonprimitive native free-form provenance denied")
    canonical(value)
    return {"present": bool(flag), "native_value": value}


def _source_papers(phase, cfg, read_columns, deadline):
    require(phase in PHASES, "Sealed/unknown source phase denied")
    path = Path(cfg["authority"]["database"]["path"])
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA cache_size=-" + str(cfg["resources"]["SQLite_cache_MiB"] * 1024))
        connection.execute("PRAGMA mmap_size=0")
        connection.execute("PRAGMA temp_store=FILE")

        def authorize(action, table, column, database, _trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and database == "main" and column in READ_COLUMNS.get(table, ()):
                read_columns.add(table + "." + column)
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        connection.set_authorizer(authorize)
        _deadline(deadline)
        collision = connection.execute(
            "SELECT selected.component_id FROM source_groups AS selected "
            "JOIN source_groups AS other ON other.component_id=selected.component_id "
            "WHERE selected.final_split=? AND (other.final_split IS NULL OR other.final_split!=?) LIMIT 1",
            (phase, phase)).fetchone()
        require(collision is None, "Selected component crosses phase allocation")
        mismatch = connection.execute(
            "SELECT a.paper_pk,a.question_pk,q.paper_pk FROM annotations AS a "
            "JOIN questions AS q ON q.question_pk=a.question_pk "
            "JOIN papers AS p ON p.paper_pk=q.paper_pk JOIN source_groups AS g ON g.group_id=p.group_id "
            "WHERE g.final_split=? AND a.paper_pk!=q.paper_pk LIMIT 1", (phase,)).fetchone()
        require(mismatch is None, "Selected annotation/question paper membership differs")
        groups = {}
        for group in connection.execute(
                "SELECT group_id,source_id,pinned_revision,component_id,final_split,rank_sha256 FROM source_groups "
                "WHERE final_split=? ORDER BY rank_sha256,group_id", (phase,)):
            require(group["source_id"] == cfg["source"]["id"] and
                    group["pinned_revision"] == cfg["source"]["revision"] and group["final_split"] == phase,
                    "Selected source group provenance differs")
            require(type(group["group_id"]) is str and type(group["component_id"]) is str,
                    "Selected source membership identifier missing")
            require(group["group_id"] not in groups, "Duplicate source group")
            groups[group["group_id"]] = dict(group)
        expected = cfg["data"]["expected"][phase]
        require(len(groups) == expected["groups"], "Selected group count differs")
        require(len({group["component_id"] for group in groups.values()}) == expected["components"],
                "Selected component count differs")
        require(value_digest(list(groups)) == expected["ordered_group_ids_sha256"],
                "Selected source group membership differs")
        seen_groups = set()
        papers = connection.execute(
            "SELECT p.paper_pk,p.source_partition,p.paper_id,p.title_json,p.abstract_json,p.full_text_json,"
            "p.group_id,p.question_count,p.annotation_count,g.component_id,g.final_split "
            "FROM papers AS p JOIN source_groups AS g ON g.group_id=p.group_id WHERE g.final_split=? "
            "ORDER BY p.group_id,p.paper_id,p.source_partition,p.paper_pk", (phase,))
        for paper_row in papers:
            _deadline(deadline)
            paper = dict(paper_row)
            group = groups.get(paper["group_id"])
            require(group is not None and paper["component_id"] == group["component_id"] and
                    paper["final_split"] == phase, "Selected paper source membership differs")
            paper["rank_sha256"] = group["rank_sha256"]
            require(type(paper["paper_id"]) is str and type(paper["source_partition"]) is str,
                    "Malformed native paper identity")
            require(type(paper["question_count"]) is int and paper["question_count"] >= 0 and
                    type(paper["annotation_count"]) is int and paper["annotation_count"] >= 0,
                    "Malformed paper source counts")
            seen_groups.add(paper["group_id"])
            questions = []
            for question_row in connection.execute(
                    "SELECT q.question_pk,q.paper_pk,q.ordinal,q.question_id_present,q.question_id_json,"
                    "q.question_json,q.annotation_count FROM questions AS q "
                    "JOIN papers AS p ON p.paper_pk=q.paper_pk JOIN source_groups AS g ON g.group_id=p.group_id "
                    "WHERE g.final_split=? AND p.paper_pk=? ORDER BY q.ordinal", (phase, paper["paper_pk"])):
                _deadline(deadline)
                question = dict(question_row)
                require(question["paper_pk"] == paper["paper_pk"] and
                        type(question["ordinal"]) is int and question["ordinal"] == len(questions),
                        "Question source membership/ordinal differs")
                qa = _json(question.pop("question_json"))
                require(type(qa) is dict and type(qa.get("question")) is str,
                        "Missing or malformed original source question")
                question["text"] = qa["question"]
                flag = question["question_id_present"]
                require(type(flag) is int and flag in (0, 1) and bool(flag) == ("question_id" in qa),
                        "Source question identifier presence differs")
                raw_id = question.pop("question_id_json")
                require(flag or raw_id is None, "Absent source question ID has a stored value")
                question["question_id_value"] = _json(raw_id) if flag else None
                require(question["question_id_value"] == qa.get("question_id"), "Source question identifier differs")
                del qa
                annotations = []
                for annotation_row in connection.execute(
                        "SELECT a.paper_pk,a.question_pk,a.ordinal,a.unanswerable_present,a.unanswerable_json,"
                        "a.yes_no_present,a.yes_no_json,a.evidence_present,a.evidence_json,"
                        "a.extractive_spans_present,a.extractive_spans_json,"
                        "a.free_form_answer_present,a.free_form_answer_json FROM annotations AS a "
                        "JOIN questions AS q ON q.question_pk=a.question_pk "
                        "JOIN papers AS p ON p.paper_pk=q.paper_pk JOIN source_groups AS g ON g.group_id=p.group_id "
                        "WHERE g.final_split=? AND p.paper_pk=? AND q.question_pk=? ORDER BY a.ordinal",
                        (phase, paper["paper_pk"], question["question_pk"])):
                    require(annotation_row["paper_pk"] == paper["paper_pk"] and
                            annotation_row["question_pk"] == question["question_pk"] and
                            type(annotation_row["ordinal"]) is int and annotation_row["ordinal"] == len(annotations),
                            "Annotation source membership/ordinal differs")
                    annotation = {"annotation_ordinal": annotation_row["ordinal"]}
                    for name in ("yes_no", "unanswerable", "evidence", "extractive_spans", "free_form_answer"):
                        annotation[name] = _native_field(annotation_row, name)
                    annotations.append(annotation)
                require(type(question["annotation_count"]) is int and
                        len(annotations) == question["annotation_count"], "Question annotation denominator differs")
                question["annotations"] = annotations
                questions.append(question)
            require(len(questions) == paper["question_count"] and
                    sum(len(question["annotations"]) for question in questions) == paper["annotation_count"],
                    "Paper question/annotation denominator differs")
            yield paper, questions
        require(seen_groups == set(groups), "Selected source groups lack their complete paper population")


def _flags(present, value):
    return {"present": bool(present), "null": bool(present and value is None)}


def state_blocks(paper, ir):
    blocks = []
    missingness = {"section_fields": [], "column_fields": None}

    def add(block_id, value):
        require(value is None or type(value) is str, "Malformed non-null native document text")
        if value:
            blocks.append(ir.StateBlock(block_id, value))

    values = {}
    for field in ("title", "abstract", "full_text"):
        raw = paper[field + "_json"]
        value = _json(raw) if raw is not None else None
        values[field] = value
        missingness[field] = _flags(raw is not None, value)
    add("title", values["title"])
    add("abstract", values["abstract"])

    def section(index, section_null, name_present, name, paragraphs_present, paragraphs):
        require(paragraphs is None or type(paragraphs) is list, "Malformed native section paragraphs")
        null_paragraphs = []
        add("section:%04d:heading" % index, name)
        if paragraphs is not None:
            for paragraph_index, paragraph in enumerate(paragraphs):
                add("section:%04d:paragraph:%04d" % (index, paragraph_index), paragraph)
                if paragraph is None:
                    null_paragraphs.append(paragraph_index)
        missingness["section_fields"].append({
            "section_ordinal": index, "section_null": section_null,
            "section_name_present": name_present, "section_name_null": bool(name_present and name is None),
            "paragraphs_present": paragraphs_present, "paragraphs_null": bool(paragraphs_present and paragraphs is None),
            "null_paragraph_ordinals": null_paragraphs})

    full_text = values["full_text"]
    if full_text is None:
        missingness["full_text_shape"] = "null" if missingness["full_text"]["present"] else "missing"
    elif type(full_text) is list:
        missingness["full_text_shape"] = "sections"
        for index, value in enumerate(full_text):
            require(value is None or type(value) is dict, "Malformed native full-text section")
            if value is None:
                section(index, True, False, None, False, None)
            else:
                section(index, False, "section_name" in value, value.get("section_name"),
                        "paragraphs" in value, value.get("paragraphs"))
    elif type(full_text) is dict:
        missingness["full_text_shape"] = "columnar"
        names, paragraphs = full_text.get("section_name"), full_text.get("paragraphs")
        require(names is None or type(names) is list, "Malformed native heading column")
        require(paragraphs is None or type(paragraphs) is list, "Malformed native paragraph column")
        missingness["column_fields"] = {"section_name": _flags("section_name" in full_text, names),
                                         "paragraphs": _flags("paragraphs" in full_text, paragraphs)}
        names = names if names is not None else []
        paragraphs = paragraphs if paragraphs is not None else []
        for index in range(max(len(names), len(paragraphs))):
            section(index, False, index < len(names), names[index] if index < len(names) else None,
                    index < len(paragraphs), paragraphs[index] if index < len(paragraphs) else None)
    else:
        raise RuntimeError("Malformed non-null native full_text shape")
    missingness["empty_document"] = not blocks
    if not blocks:
        blocks.append(ir.StateBlock("document", ""))
    return tuple(blocks), missingness


def _boolean_target(annotations, answerability):
    ratings, counts, conflicts = [], [0, 0], []
    for annotation in annotations:
        field = annotation["unanswerable" if answerability else "yes_no"]
        native = field["native_value"]
        observed = field["present"] and type(native) is bool
        mapped = (not native if answerability else native) if observed else None
        if observed:
            counts[int(mapped)] += 1
        ratings.append({"annotation_ordinal": annotation["annotation_ordinal"], **field,
                        "observed": observed, "mapped_value": mapped})
        if type(annotation["yes_no"]["native_value"]) is bool and annotation["unanswerable"]["native_value"] is True:
            conflicts.append(annotation["annotation_ordinal"])
    observed_raters = sum(counts)
    return {"kind": "source_transformed_answerability" if answerability else "source_retained_boolean",
            "target_available": bool(observed_raters), "ratings": ratings, "counts": counts,
            "distribution": [count / observed_raters for count in counts] if observed_raters else None,
            "observed_raters": observed_raters, "conflicting_annotation_ordinals": sorted(conflicts)}


def _native_array_target(annotations, blocks, catalogue, extraction):
    ratings, native_count, unmatched_count = [], 0, 0
    matched_blocks, seen_occurrences, occurrences = set(), set(), []
    for annotation in annotations:
        field = annotation["extractive_spans" if extraction else "evidence"]
        native = field["native_value"]
        observed = field["present"] and native is not None
        matches, unmatched = [], []
        for source_index, text in enumerate(native if observed else []):
            native_count += 1
            if extraction:
                found = []
                if text:
                    for block in blocks:
                        start = block.text.find(text)
                        while start != -1:
                            occurrence = {"block_id": block.id, "start": start, "end": start + len(text)}
                            found.append(occurrence)
                            key = (block.id, start, start + len(text))
                            if key not in seen_occurrences:
                                seen_occurrences.add(key)
                                occurrences.append(occurrence)
                            start = block.text.find(text, start + 1)
                matches.append({"source_index": source_index, "text": text, "occurrences": found})
            else:
                found = catalogue.get(text, [])
                matches.append({"source_index": source_index, "text": text, "block_ids": list(found)})
                matched_blocks.update(found)
            if not found:
                unmatched_count += 1
                unmatched.append({"source_index": source_index, "text": text})
        ratings.append({"annotation_ordinal": annotation["annotation_ordinal"], **field,
                        "observed": observed, "matches": matches, "unmatched": unmatched})
    target = {"kind": "source_native_extractive" if extraction else "source_native_evidence",
              "target_available": bool(native_count), "ratings": ratings,
              "native_item_count": native_count, "unmatched_item_count": unmatched_count}
    if not extraction:
        target["matched_block_ids"] = [block.id for block in blocks if block.id in matched_blocks]
    return target, occurrences


def _row_id(cfg, paper, question, endpoint):
    return value_digest([cfg["source"]["id"], cfg["source"]["revision"], paper["group_id"], paper["paper_id"],
                         paper["source_partition"], question["ordinal"], endpoint])


def _question_projection(phase, cfg, paper, question, blocks, missingness, catalogue, ir, renderer):
    annotations = question["annotations"]
    evidence, _ = _native_array_target(annotations, blocks, catalogue, False)
    extraction, occurrences = _native_array_target(annotations, blocks, catalogue, True)
    targets = {ENDPOINTS[0]: _boolean_target(annotations, False), ENDPOINTS[1]: _boolean_target(annotations, True),
               ENDPOINTS[2]: evidence, ENDPOINTS[3]: extraction}
    endpoint_info, bundles = {}, []
    for endpoint in ENDPOINTS:
        target = targets[endpoint]
        eligible = endpoint != ENDPOINTS[0] or target["target_available"]
        row_id = _row_id(cfg, paper, question, endpoint) if eligible else None
        endpoint_info[endpoint] = {"id": row_id, "eligible": bool(eligible)}
        if not eligible:
            continue
        task, prompt, gold, support = "bool", question["text"], (), ()
        if endpoint in ENDPOINTS[:2]:
            projection_key = "answerability" if endpoint == ENDPOINTS[1] else "boolean"
            candidates = tuple(ir.Candidate(item["id"], item["description"])
                               for item in cfg["projection"][projection_key]["candidates"])
            gold = tuple(candidate_id for candidate_id, count in zip(("false", "true"), target["counts"]) if count)
            if endpoint == ENDPOINTS[1]:
                prompt = cfg["projection"]["answerability"]["question_prefix"] + prompt
        elif endpoint == ENDPOINTS[2]:
            task = "retrieval"
            prompt = cfg["projection"]["evidence"]["question_prefix"] + prompt
            candidates = tuple(ir.Candidate(block.id, block.text) for block in blocks)
            gold = tuple(target["matched_block_ids"])
            support = tuple(ir.Evidence(block_id) for block_id in gold)
        else:
            task, candidates = "extract", ()
            support = tuple(ir.Evidence(item["block_id"], (item["start"], item["end"])) for item in occurrences)
        row = ir.DecisionIR(id=row_id, source=cfg["source"]["id"], split=phase, task=task, state_blocks=blocks,
                            question=prompt, candidates=candidates, gold=gold, license=LICENSE, evidence=support,
                            metadata={"endpoint": endpoint, "target_available": target["target_available"]})
        serving = {"id": row.id, "task": row.task, "locale": "en", "state": renderer.state_text(row),
                   "question": renderer.question_text(row),
                   "candidates": [{"id": item.id, "text": renderer.candidate_text(item)} for item in row.candidates]}
        provenance = {"id": row_id, "endpoint": endpoint, "paper_id": paper["paper_id"],
                      "source_partition": paper["source_partition"], "group_id": paper["group_id"],
                      "component_id": paper["component_id"], "question_ordinal": question["ordinal"], "phase": phase,
                      "source_question_id_present": bool(question["question_id_present"]),
                      "source_question_id_value": question["question_id_value"],
                      "modification": MODIFICATION, "license": LICENSE,
                      "free_form_answers": [{"annotation_ordinal": item["annotation_ordinal"],
                                             **item["free_form_answer"]} for item in annotations],
                      "state_missingness": missingness}
        bundles.append((serving, asdict(row), {"id": row_id, "endpoint": endpoint, **target}, provenance))
    census = {"source_id": cfg["source"]["id"], "source_revision": cfg["source"]["revision"],
              "paper_id": paper["paper_id"], "source_partition": paper["source_partition"],
              "group_id": paper["group_id"], "component_id": paper["component_id"],
              "question_ordinal": question["ordinal"], "phase": phase,
              "source_question_id_present": bool(question["question_id_present"]),
              "source_question_id_value": question["question_id_value"],
              "annotation_count": len(annotations), "endpoints": endpoint_info}
    return bundles, census, targets


def _coverage(bucket, target, eligible, annotation_count, native_array):
    bucket["questions"] += 1
    bucket["annotations"] += annotation_count
    bucket["eligible_questions"] += int(eligible)
    bucket["ineligible_questions"] += int(not eligible)
    bucket["decisions"] += int(eligible)
    if eligible:
        bucket["target_available"] += int(target["target_available"])
        bucket["target_unavailable"] += int(not target["target_available"])
    for rating in target["ratings"]:
        bucket["observed_raters"] += int(rating["observed"])
        bucket["missing_fields"] += int(not rating["present"])
        bucket["null_fields"] += int(rating["present"] and rating["native_value"] is None)
    conflicts = target.get("conflicting_annotation_ordinals", [])
    bucket["conflicting_questions"] += int(bool(conflicts))
    bucket["conflicting_annotations"] += len(conflicts)
    if native_array:
        native, unmatched = target["native_item_count"], target["unmatched_item_count"]
        matched = native - unmatched
        bucket["native_item_count"] += native
        bucket["unmatched_item_count"] += unmatched
        bucket["matched_item_count"] += matched
        bucket["full_support_questions"] += int(native > 0 and unmatched == 0)
        bucket["mixed_support_questions"] += int(matched > 0 and unmatched > 0)
        bucket["no_support_questions"] += int(native > 0 and matched == 0)
        bucket["no_native_items_questions"] += int(native == 0)


def _writer(stack, path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    raw = stack.enter_context(os.fdopen(fd, "wb"))
    return stack.enter_context(gzip.GzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0))


def _immutable_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _entry(directory, name, rows):
    path = directory / name
    return {"path": name, "sha256": digest(path), "bytes": path.stat().st_size, "rows": rows}


def compile_phase(phase):
    require(phase in PHASES, "Only train/dev QASPER materialization allowed")
    cfg = protocol()
    root = Path(cfg["data"]["output_root"])
    require(root.is_absolute() and root.resolve() == root, "QASPER output root must be absolute and nonsymlinked")
    directory = root / phase
    require(not directory.exists() and not directory.is_symlink(), "QASPER phase output overwrite denied")
    revision = _committed()
    require(metadata() == cfg, "Protocol changed during input pin checks")
    _memory(cfg)
    desired_nice = cfg["resources"]["nice"]
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < desired_nice:
        os.nice(desired_nice - current_nice)
    deadline = time.monotonic() + cfg["resources"]["wall_timeout_seconds"]
    ir, renderer = ir_modules(cfg)
    compiler_sha, verifier_sha = digest(Path(__file__)), digest(VERIFIER)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.mkdir(exist_ok=False, mode=0o700)
    source_counts, groups, components, seen_ids = Counter(), {}, set(), set()
    coverage = {endpoint: Counter({key: 0 for key in COVERAGE_KEYS +
                                  (SUPPORT_KEYS if endpoint in ENDPOINTS[2:] else ())}) for endpoint in ENDPOINTS}
    read_columns, decision_count = set(), 0
    decision_ids_hash = hashlib.sha256(b"[")
    try:
        with ExitStack() as stack:
            outputs = {name: _writer(stack, directory / (name + ".jsonl.gz")) for name in STREAMS}
            census_output = _writer(stack, directory / "source_census.jsonl.gz")
            for paper, questions in _source_papers(phase, cfg, read_columns, deadline):
                _deadline(deadline)
                source_counts["papers"] += 1
                if source_counts["papers"] % 32 == 0:
                    _memory(cfg)
                groups[paper["group_id"]] = paper["rank_sha256"]
                components.add(paper["component_id"])
                blocks, missingness = state_blocks(paper, ir)
                catalogue = {}
                for block in blocks:
                    catalogue.setdefault(block.text, []).append(block.id)
                for question in questions:
                    source_counts["questions"] += 1
                    source_counts["annotations"] += len(question["annotations"])
                    bundles, census, targets = _question_projection(
                        phase, cfg, paper, question, blocks, missingness, catalogue, ir, renderer)
                    census_output.write((canonical(census) + "\n").encode("utf-8"))
                    for endpoint in ENDPOINTS:
                        _coverage(coverage[endpoint], targets[endpoint], census["endpoints"][endpoint]["eligible"],
                                  len(question["annotations"]), endpoint in ENDPOINTS[2:])
                    for bundle in bundles:
                        row_id = bundle[0]["id"]
                        require(row_id not in seen_ids, "Duplicate source-native decision identity")
                        seen_ids.add(row_id)
                        for name, value in zip(STREAMS, bundle):
                            outputs[name].write((canonical(value) + "\n").encode("utf-8"))
                        if decision_count:
                            decision_ids_hash.update(b",")
                        decision_ids_hash.update(canonical(row_id).encode("utf-8"))
                        decision_count += 1
        decision_ids_hash.update(b"]")
        source_counts["groups"], source_counts["components"] = len(groups), len(components)
        expected = cfg["data"]["expected"][phase]
        require(all(source_counts[key] == expected[key] for key in ("papers", "questions", "annotations", "groups", "components")),
                "Selected source population denominator differs")
        group_sha = value_digest(sorted(groups, key=lambda group_id: (groups[group_id], group_id)))
        require(group_sha == expected["ordered_group_ids_sha256"], "Selected source group hash differs")
        require(metadata() == cfg, "Original pinned inputs changed during materialization")
        require(digest(Path(__file__)) == compiler_sha and digest(VERIFIER) == verifier_sha,
                "Producer/independent verifier changed during materialization")
        _committed()
        _deadline(deadline)
        result = {"schema": "vey.neutral.qasper.native-projection-manifest.v1", "phase": phase, "complete": True,
                  "protocol_sha256": PROTOCOL_SHA256, "compiler_sha256": compiler_sha, "verifier_sha256": verifier_sha,
                  "authority": cfg["authority"], "git_commit": revision,
                  "files": {name: _entry(directory, name + ".jsonl.gz", decision_count) for name in STREAMS},
                  "source_census": _entry(directory, "source_census.jsonl.gz", source_counts["questions"]),
                  "source_counts": dict(source_counts),
                  "membership": {"ordered_group_ids_sha256": group_sha,
                                 "ordered_component_ids_sha256": value_digest(sorted(components)),
                                 "ordered_decision_ids_sha256": decision_ids_hash.hexdigest()},
                  "endpoints": {endpoint: dict(coverage[endpoint]) for endpoint in ENDPOINTS},
                  "actual_SQL_READ_columns": sorted(read_columns), "source_database_unchanged": True,
                  "source_rows_committed_code_only": True, "sealed_labels_accessed": False,
                  "model_outputs": 0, "quality_credit": False}
        _immutable_json(directory / "manifest.json", result)
        return result
    except BaseException as error:
        try:
            _immutable_json(directory / "failure.json", {"phase": phase, "complete": False, "selector_open": False,
                            "error_type": type(error).__name__, "message": str(error), "protocol_sha256": PROTOCOL_SHA256,
                            "compiler_sha256": compiler_sha, "verifier_sha256": verifier_sha})
        except OSError:
            pass
        raise


def _rooted_entry(directory, entry, filename):
    require(entry["path"] == filename, "Selector stream relative path differs")
    path = directory / filename
    require(not path.is_symlink() and path.resolve().parent == directory.resolve(), "Selector stream escapes phase root")
    require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Selector stream size/hash differs")
    require(type(entry["rows"]) is int and entry["rows"] >= 0, "Malformed selector stream row count")
    return path


def _verified_rows(path, entry):
    with path.open("rb") as raw:
        result = hashlib.sha256()
        size = 0
        for block in iter(lambda: raw.read(1024 * 1024), b""):
            result.update(block)
            size += len(block)
        require(result.hexdigest() == entry["sha256"] and size == entry["bytes"], "Selector stream changed before opening")
        raw.seek(0)
        rows = 0
        with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
            for line in stream:
                require(line.endswith(b"\n"), "Incomplete selector JSONL record")
                value = _json(line.decode("utf-8"))
                require(type(value) is dict and line == (canonical(value) + "\n").encode("utf-8"),
                        "Noncanonical selector JSONL record")
                rows += 1
                require(rows <= entry["rows"], "Selector stream has extra records")
                yield value
        require(rows == entry["rows"], "Selector stream row count differs")


def open_phase(phase, stream="serving"):
    """Return only a rooted, hashed stream bound to a current independent PASS receipt."""
    require(phase in PHASES, "Sealed/unknown selector phase denied")
    require(stream in STREAMS, "Unregistered selector stream")
    _committed()
    cfg = metadata()
    root = Path(cfg["data"]["output_root"])
    directory = root / phase
    require(root.resolve() == root and not directory.is_symlink(), "Selector phase root differs")
    require(not (directory / "failure.json").exists() and not (directory / "verification_in_progress.json").exists(),
            "Incomplete/failed projection or active independent verification denied")
    manifest_path, receipt_path = directory / "manifest.json", directory / "verification_receipt.json"
    require(not manifest_path.is_symlink() and not receipt_path.is_symlink(), "Selector manifest/receipt link denied")
    manifest = _json(manifest_path.read_text(encoding="utf-8"))
    receipt = _json(receipt_path.read_text(encoding="utf-8"))
    compiler_sha, verifier_sha = digest(Path(__file__)), digest(VERIFIER)
    require(manifest["schema"] == "vey.neutral.qasper.native-projection-manifest.v1" and
            manifest["phase"] == phase and manifest["complete"] is True, "Selector completed manifest required")
    require(manifest["protocol_sha256"] == PROTOCOL_SHA256 and manifest["compiler_sha256"] == compiler_sha and
            manifest["verifier_sha256"] == verifier_sha and manifest["authority"] == cfg["authority"],
            "Selector manifest current-source identities differ")
    require(manifest["source_database_unchanged"] is True and manifest["source_rows_committed_code_only"] is True and
            manifest["sealed_labels_accessed"] is False and manifest["model_outputs"] == 0 and
            manifest["quality_credit"] is False, "Selector custody/claim boundary differs")
    require(receipt["schema"] == "vey.neutral.qasper.native-projection-verification.v1" and receipt["status"] == "PASS" and
            receipt["phase"] == phase and receipt["manifest_sha256"] == digest(manifest_path) and
            receipt["protocol_sha256"] == PROTOCOL_SHA256 and receipt["verifier_sha256"] == verifier_sha and
            receipt["compiler_sha256"] == compiler_sha, "Current independent source reconstruction PASS receipt required")
    expected = cfg["data"]["expected"][phase]
    require(manifest["source_counts"] == {key: expected[key] for key in ("papers", "questions", "annotations", "groups", "components")},
            "Selector source population differs")
    require(manifest["membership"]["ordered_group_ids_sha256"] == expected["ordered_group_ids_sha256"],
            "Selector source group membership differs")
    require(set(manifest["files"]) == set(STREAMS), "Selector stream inventory differs")
    paths = {name: _rooted_entry(directory, manifest["files"][name], name + ".jsonl.gz") for name in STREAMS}
    _rooted_entry(directory, manifest["source_census"], "source_census.jsonl.gz")
    require(manifest["source_census"]["rows"] == expected["questions"] and
            len({manifest["files"][name]["rows"] for name in STREAMS}) == 1,
            "Selector stream/census denominators differ")
    return _verified_rows(paths[stream], manifest["files"][stream])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    args = parser.parse_args(argv)
    result = compile_phase(args.phase)
    print(canonical({"phase": result["phase"], "source_counts": result["source_counts"],
                     "decision_rows": result["files"]["serving"]["rows"], "selector_open": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
