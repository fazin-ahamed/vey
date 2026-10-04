#!/usr/bin/env python3
"""Independently verify the sealed QASPER paper-custody attempt.

This verifier intentionally imports no project acquisition, grouping, allocation,
source-loader, evaluator, or model code. It reads the private SQLite database in
read-only mode and writes metadata-only evidence beneath the selected attempt.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import resource
import sqlite3
import stat
import subprocess
import sys
import tarfile
import unicodedata
import urllib.parse
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence


ATTEMPT_ID = "20261004T074414.392704Z-593810-cb44d7ae"
EXPECTED_MANIFEST_SHA256 = "de5bc1986d7aa0d89833ddab45c48e4a44f440f80a376acf744a874df5ff5c72"
EXPECTED_ATTEMPT = Path(
    "/home/fazinahamed/Documents/vey-data/decisionmix/endgame/neutral-source-v2/qasper/attempts"
) / ATTEMPT_ID
EXPECTED_REUSE_RECEIPT_SHA256 = "10600183edb1c7fb8d7513001affec9801433675d40d08e6a011061925dd05ed"
EXPECTED_ACQUISITION_COMMIT = "696be2f3116905517806d90ca8291098e58a0c3e"
SOURCE_ID = "allenai/qasper"
PINNED_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
SEED = "vey-neutral-v2:2026-10-04:1729"
PURPOSE = "neutral-acquisition"
PARTITIONS = ("train", "validation", "test")
SPLITS = ("train", "dev", "calibration", "confirmation", "unused")
OUTCOME_FIELDS = ("unanswerable", "extractive_spans", "yes_no", "free_form_answer")
PAPER_FIELDS = ("title", "abstract", "full_text", "qas", "figures_and_tables")
QUESTION_FIELDS = (
    "question", "question_id", "nlp_background", "topic_background", "paper_read",
    "search_query", "question_writer", "answers",
)
ANNOTATION_FIELDS = ("answer", "annotation_id", "worker_id")
ANSWER_FIELDS = (
    "unanswerable", "extractive_spans", "yes_no", "free_form_answer", "evidence",
    "highlighted_evidence",
)
SECTION_FIELDS = ("section_name", "paragraphs")
FIGURE_FIELDS = ("caption", "file")
ARXIV_ID = re.compile(
    r"(?i)(?<![0-9A-Za-z])(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?(?!\d)"
)
CHUNK_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_TAR_MEMBERS = 10000
MAX_TAR_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_SELECTED_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_PAPER_CHARS = 128 * 1024 * 1024
MAX_DOCUMENT_CHARS = 8 * 1024 * 1024
MAX_NGRAM_POSTINGS = 50_000_000
MAX_PREFIX_PAIR_WORK = 100_000_000
GRAM_CACHE_BYTES = 128 * 1024 * 1024
MAX_SQLITE_BYTES = 8 * 1024 * 1024 * 1024
# Conservative retained-size allowance for a five-codepoint Unicode string and its set-table slot.
GRAM_CACHE_ENTRY_BYTES = 256
ADDRESS_SPACE_LIMIT = 4 * 1024 * 1024 * 1024


class VerificationError(RuntimeError):
    def __init__(self, check: str):
        super().__init__(check)
        self.check = check


CHECK_NAMES = (
    "attempt_and_manifest_pin",
    "frozen_historical_snapshots",
    "source_plan_and_protocol_pins",
    "declared_schema_expectation",
    "preparse_receipt_and_raw_reuse_origin",
    "raw_artifact_hashes_and_metadata_license",
    "archive_member_safety_and_hashes",
    "readonly_database_fingerprint",
    "strict_json_and_database_row_reconstruction",
    "native_schema_census_and_source_descendants",
    "paper_identity_and_document_fingerprints",
    "bounded_complete_character_5gram_join",
    "group_edges_and_edge_hashes",
    "components_and_component_hashes",
    "allocation_membership_reasons_caps_and_hashes",
    "database_unchanged_after_read",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: Path, max_bytes: int | None = None) -> tuple[str, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise VerificationError("raw_artifact_hashes_and_metadata_license")
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(fd, "rb", closefd=False) as stream:
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                size += len(block)
                if max_bytes is not None and size > max_bytes:
                    raise VerificationError("raw_artifact_hashes_and_metadata_license")
                digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def verify_hardlink_reuse(current_path: Path, reused_path: Path) -> None:
    try:
        current = current_path.stat(follow_symlinks=False)
        reused = reused_path.stat(follow_symlinks=False)
    except OSError as exc:
        raise VerificationError("raw_artifact_hashes_and_metadata_license") from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(reused.st_mode)
        or current.st_dev != reused.st_dev
        or current.st_ino != reused.st_ino
        or current.st_nlink < 2
    ):
        raise VerificationError("raw_artifact_hashes_and_metadata_license")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON constant")


def parse_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def strict_json_bytes(data: bytes, *, object_required: bool = True) -> Any:
    value = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_json_constant,
        parse_float=parse_float,
    )
    if object_required and not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def ensure_plain_path(path: Path, *, directory: bool = False) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise VerificationError("attempt_and_manifest_pin")
    info = path.lstat()
    wanted = stat.S_ISDIR if directory else stat.S_ISREG
    if not wanted(info.st_mode):
        raise VerificationError("attempt_and_manifest_pin")


def validate_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    allowed = host in {
        "huggingface.co", "qasper-dataset.s3.us-west-2.amazonaws.com",
        "creativecommons.org",
    }
    if parsed.scheme != "https" or not allowed or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise VerificationError("raw_artifact_hashes_and_metadata_license")
    return url


def set_resource_guards() -> dict[str, int | bool]:
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[key] = ""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise VerificationError("attempt_and_manifest_pin")
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise VerificationError("attempt_and_manifest_pin")
    selected = set(allowed[:4])
    os.sched_setaffinity(0, selected)
    if len(os.sched_getaffinity(0)) > 4:
        raise VerificationError("attempt_and_manifest_pin")
    try:
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if current_nice < 10:
            os.nice(10 - current_nice)
        nice_value = os.getpriority(os.PRIO_PROCESS, 0)
    except (AttributeError, OSError) as exc:
        raise VerificationError("attempt_and_manifest_pin") from exc
    if nice_value < 10:
        raise VerificationError("attempt_and_manifest_pin")
    try:
        result = subprocess.run(
            ["ionice", "-c", "2", "-n", "7", "-p", str(os.getpid())],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise VerificationError("attempt_and_manifest_pin") from exc
    if result.returncode != 0:
        raise VerificationError("attempt_and_manifest_pin")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = ADDRESS_SPACE_LIMIT if hard == resource.RLIM_INFINITY else min(ADDRESS_SPACE_LIMIT, hard)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if limit < 512 * 1024 * 1024:
        raise VerificationError("attempt_and_manifest_pin")
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return {"cpu_affinity_count": len(selected), "blas_threads": 1, "nice_value": nice_value,
            "ionice_class": 2, "ionice_priority": 7, "gpu_visible_devices_disabled": True,
            "address_space_limit_bytes": limit}


def json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "other"


class Census:
    def __init__(self) -> None:
        self.parents: Counter[tuple[str, str]] = Counter()
        self.fields: Counter[tuple[str, str, str, str]] = Counter()

    def observe(self, partition: str, scope: str, obj: Mapping[str, Any], fields: Sequence[str]) -> None:
        self.parents[(partition, scope)] += 1
        for field in fields:
            kind = json_type(obj[field]) if field in obj else "missing"
            self.fields[(partition, scope, field, kind)] += 1

    def observe_value(self, partition: str, scope: str, kind: str) -> None:
        self.fields[(partition, scope, "_value", kind)] += 1

    def serialized(self) -> dict[str, Any]:
        result: dict[str, Any] = {"parents": {}, "fields": {}}
        for (partition, scope), count in sorted(self.parents.items()):
            result["parents"].setdefault(partition, {})[scope] = count
        for (partition, scope, field, kind), count in sorted(self.fields.items()):
            result["fields"].setdefault(partition, {}).setdefault(scope, {}).setdefault(field, {})[kind] = count
        return result


def iter_text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_text_values(item)


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def document_text(paper: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for field in ("title", "abstract"):
        value = paper.get(field)
        if isinstance(value, str):
            pieces.append(value)
    full_text = paper.get("full_text")
    if isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, dict):
                pieces.extend(iter_text_values(section.get("section_name")))
                pieces.extend(iter_text_values(section.get("paragraphs")))
    elif isinstance(full_text, dict):
        names = full_text.get("section_name")
        paragraphs = full_text.get("paragraphs")
        if isinstance(names, list) or isinstance(paragraphs, list):
            names_list = names if isinstance(names, list) else []
            paragraphs_list = paragraphs if isinstance(paragraphs, list) else []
            for index in range(max(len(names_list), len(paragraphs_list))):
                if index < len(names_list):
                    pieces.extend(iter_text_values(names_list[index]))
                if index < len(paragraphs_list):
                    pieces.extend(iter_text_values(paragraphs_list[index]))
    figures = paper.get("figures_and_tables")
    if isinstance(figures, list):
        for figure in figures:
            if isinstance(figure, dict):
                pieces.extend(iter_text_values(figure.get("caption")))
    return "\n".join(pieces)


def canonical_arxiv_id(paper_id: str) -> str | None:
    match = ARXIV_ID.search(paper_id)
    if match is None:
        return None
    return re.sub(r"v\d+$", "", match.group(0).lower())


def rank_hash(group_id: str) -> str:
    return hashlib.sha256(f"{SEED}\n{PURPOSE}\n{SOURCE_ID}\n{group_id}".encode("utf-8")).hexdigest()


def stream_top_level_object(path: Path) -> Iterator[tuple[str, Any]]:
    decoder = json.JSONDecoder(
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_json_constant,
        parse_float=parse_float,
    )
    with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
        buffer = ""
        position = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, eof
            if eof:
                return False
            chunk = stream.read(CHUNK_BYTES)
            if chunk == "":
                eof = True
                return False
            buffer += chunk
            return True

        def ensure(index: int) -> bool:
            while index >= len(buffer) and not eof:
                fill()
            return index < len(buffer)

        def skip_space(index: int) -> int:
            while True:
                while index < len(buffer) and buffer[index] in " \t\r\n":
                    index += 1
                if index < len(buffer) or eof:
                    return index
                fill()

        def decode_at(index: int, max_chars: int) -> tuple[Any, int]:
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, index)
                    if end - index > max_chars:
                        raise VerificationError("strict_json_and_database_row_reconstruction")
                    return value, end
                except json.JSONDecodeError:
                    if eof or len(buffer) - index > max_chars:
                        raise VerificationError("strict_json_and_database_row_reconstruction") from None
                    fill()
                except ValueError:
                    raise VerificationError("strict_json_and_database_row_reconstruction") from None

        position = skip_space(position)
        if not ensure(position) or buffer[position] != "{":
            raise VerificationError("strict_json_and_database_row_reconstruction")
        position += 1
        position = skip_space(position)
        if ensure(position) and buffer[position] == "}":
            position += 1
        else:
            while True:
                position = skip_space(position)
                key, end = decode_at(position, 1024 * 1024)
                if not isinstance(key, str):
                    raise VerificationError("strict_json_and_database_row_reconstruction")
                position = skip_space(end)
                if not ensure(position) or buffer[position] != ":":
                    raise VerificationError("strict_json_and_database_row_reconstruction")
                position = skip_space(position + 1)
                value, end = decode_at(position, MAX_PAPER_CHARS)
                position = skip_space(end)
                if not ensure(position):
                    raise VerificationError("strict_json_and_database_row_reconstruction")
                delimiter = buffer[position]
                if delimiter == ",":
                    position += 1
                    finished = False
                elif delimiter == "}":
                    position += 1
                    finished = True
                else:
                    raise VerificationError("strict_json_and_database_row_reconstruction")
                if position > CHUNK_BYTES:
                    buffer = buffer[position:]
                    position = 0
                yield key, value
                if finished:
                    break
        position = skip_space(position)
        while not eof and position >= len(buffer):
            fill()
            position = skip_space(position)
        if position != len(buffer):
            raise VerificationError("strict_json_and_database_row_reconstruction")


def json_field(obj: Mapping[str, Any], field: str) -> tuple[int, str | None]:
    if field not in obj:
        return 0, None
    return 1, canonical_json(obj[field])


def add_census_for_paper(census: Census, partition: str, paper: Mapping[str, Any]) -> tuple[int, int, int]:
    census.observe(partition, "paper", paper, PAPER_FIELDS)
    raw_qas = paper.get("qas")
    qas = raw_qas if isinstance(raw_qas, list) else []
    annotation_count = 0
    nonnull_outcome_count = 0
    for question in qas:
        if isinstance(question, dict):
            census.observe(partition, "question", question, QUESTION_FIELDS)
            raw_answers = question.get("answers")
            answers = raw_answers if isinstance(raw_answers, list) else []
        else:
            census.observe_value(partition, "question_item", json_type(question))
            answers = []
        for annotation in answers:
            annotation_count += 1
            if isinstance(annotation, dict):
                census.observe(partition, "annotation", annotation, ANNOTATION_FIELDS)
                answer_present = "answer" in annotation
                answer = annotation.get("answer")
                if isinstance(answer, dict):
                    census.observe(partition, "answer", answer, ANSWER_FIELDS)
                    if any(field in answer and answer[field] is not None for field in OUTCOME_FIELDS):
                        nonnull_outcome_count += 1
                else:
                    census.observe_value(
                        partition, "answer_value", json_type(answer) if answer_present else "missing"
                    )
            else:
                census.observe_value(partition, "annotation_item", json_type(annotation))
    figures = paper.get("figures_and_tables")
    if isinstance(figures, list):
        for figure in figures:
            if isinstance(figure, dict):
                census.observe(partition, "figure_or_table", figure, FIGURE_FIELDS)
            else:
                census.observe_value(partition, "figure_or_table_item", json_type(figure))
    full_text = paper.get("full_text")
    if isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, dict):
                census.observe(partition, "section", section, SECTION_FIELDS)
            else:
                census.observe_value(partition, "section_item", json_type(section))
    elif isinstance(full_text, dict):
        census.observe(partition, "full_text_columnar", full_text, SECTION_FIELDS)
    return len(qas), annotation_count, nonnull_outcome_count


def assert_row(actual: Sequence[Any] | None, expected: Sequence[Any]) -> None:
    if actual is None or tuple(actual) != tuple(expected):
        raise VerificationError("strict_json_and_database_row_reconstruction")


def compare_paper_to_database(
    connection: sqlite3.Connection,
    partition: str,
    paper_id: str,
    paper: Mapping[str, Any],
    doc_hash: str,
    full_hash: str,
    title_norm: str | None,
    title_hash: str | None,
    group_id: str,
    question_count: int,
    annotation_count: int,
    labeled_count: int,
) -> tuple[int, list[list[Any]], list[list[Any]]]:
    rows = connection.execute(
        "SELECT paper_pk,paper_id,source_partition,canonical_arxiv_id,title_json,abstract_json,full_text_json,figures_json,model_state_json,payload_json,full_document_sha256,normalized_document_sha256,normalized_title,normalized_title_sha256,group_id,question_count,annotation_count,nonnull_outcome_annotation_count FROM papers WHERE source_partition=? AND paper_id=?",
        (partition, paper_id),
    ).fetchall()
    if len(rows) != 1:
        raise VerificationError("strict_json_and_database_row_reconstruction")
    row = rows[0]
    state = {field: paper[field] for field in ("title", "abstract", "full_text") if field in paper}
    expected_fields = [
        paper_id, partition, canonical_arxiv_id(paper_id), json_field(paper, "title")[1],
        json_field(paper, "abstract")[1], json_field(paper, "full_text")[1],
        json_field(paper, "figures_and_tables")[1], canonical_json(state), canonical_json(paper),
        full_hash, doc_hash, title_norm or None, title_hash, group_id, question_count,
        annotation_count, labeled_count,
    ]
    assert_row(row[1:], expected_fields)
    paper_pk = int(row[0])
    question_rows = connection.execute(
        "SELECT question_pk,ordinal,question_id_present,question_id_json,question_type,question_json,annotation_count FROM questions WHERE paper_pk=? ORDER BY ordinal",
        (paper_pk,),
    ).fetchall()
    qas_value = paper.get("qas")
    qas = qas_value if isinstance(qas_value, list) else []
    if len(question_rows) != len(qas):
        raise VerificationError("strict_json_and_database_row_reconstruction")
    question_descendants: list[list[Any]] = []
    annotation_descendants: list[list[Any]] = []
    observed_annotations = 0
    for ordinal, question in enumerate(qas):
        question_pk = int(question_rows[ordinal][0])
        if isinstance(question, dict):
            qid_present, qid_json = json_field(question, "question_id")
            raw_answers = question.get("answers")
            answers = raw_answers if isinstance(raw_answers, list) else []
        else:
            qid_present, qid_json = 0, None
            answers = []
        assert_row(
            question_rows[ordinal][1:],
            [ordinal, qid_present, qid_json, json_type(question), canonical_json(question), len(answers)],
        )
        question_descendants.append([paper_id, ordinal, qid_present, qid_json])
        answer_rows = connection.execute(
            "SELECT ordinal,annotation_type,annotation_id_json,worker_id_json,annotation_json,answer_present,answer_json,unanswerable_present,unanswerable_json,extractive_spans_present,extractive_spans_json,yes_no_present,yes_no_json,free_form_answer_present,free_form_answer_json,evidence_present,evidence_json,highlighted_evidence_present,highlighted_evidence_json,nonnull_outcome_annotation FROM annotations WHERE question_pk=? ORDER BY ordinal",
            (question_pk,),
        ).fetchall()
        if len(answer_rows) != len(answers):
            raise VerificationError("strict_json_and_database_row_reconstruction")
        for answer_ordinal, annotation in enumerate(answers):
            ann_obj = annotation if isinstance(annotation, dict) else {}
            answer_value = ann_obj.get("answer")
            answer_obj = answer_value if isinstance(answer_value, dict) else {}
            answer_present, answer_json = json_field(ann_obj, "answer")
            expected_annotation = [
                answer_ordinal, json_type(annotation),
                canonical_json(ann_obj["annotation_id"]) if isinstance(annotation, dict) and "annotation_id" in ann_obj else None,
                canonical_json(ann_obj["worker_id"]) if isinstance(annotation, dict) and "worker_id" in ann_obj else None,
                canonical_json(annotation), answer_present, answer_json,
            ]
            has_nonnull = 0
            for field in ANSWER_FIELDS:
                present, value_json = json_field(answer_obj, field)
                expected_annotation.extend((present, value_json))
            if isinstance(answer_value, dict):
                has_nonnull = int(any(field in answer_value and answer_value[field] is not None for field in OUTCOME_FIELDS))
            expected_annotation.append(has_nonnull)
            assert_row(answer_rows[answer_ordinal], expected_annotation)
            annotation_id_json = expected_annotation[2]
            worker_id_json = expected_annotation[3]
            annotation_descendants.append(
                [paper_id, ordinal, answer_ordinal, json_type(annotation), annotation_id_json, worker_id_json]
            )
            observed_annotations += 1
    if observed_annotations != annotation_count:
        raise VerificationError("strict_json_and_database_row_reconstruction")
    return paper_pk, question_descendants, annotation_descendants


def safe_member_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise VerificationError("archive_member_safety_and_hashes")
    pieces = name.split("/")
    if any(piece == ".." for piece in pieces) or (pieces and re.match(r"^[A-Za-z]:", pieces[0])):
        raise VerificationError("archive_member_safety_and_hashes")
    normalized = "/".join(piece for piece in pieces if piece not in ("", "."))
    if not normalized:
        raise VerificationError("archive_member_safety_and_hashes")
    return normalized


def inspect_archive(
    archive_path: Path,
    archive_record: Mapping[str, Any],
    expected_members: Mapping[str, str],
    selected_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    archive_hash, archive_size = hash_file(archive_path, MAX_ARCHIVE_BYTES)
    if archive_hash != archive_record.get("sha256") or archive_size != archive_record.get("bytes"):
        raise VerificationError("archive_member_safety_and_hashes")
    selected: dict[str, tarfile.TarInfo] = {}
    seen: set[str] = set()
    count = 0
    unpacked = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                count += 1
                if count > MAX_TAR_MEMBERS:
                    raise VerificationError("archive_member_safety_and_hashes")
                name = safe_member_name(member.name)
                if name in seen:
                    raise VerificationError("archive_member_safety_and_hashes")
                seen.add(name)
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise VerificationError("archive_member_safety_and_hashes")
                if not (member.isfile() or member.isdir()) or getattr(member, "issparse", lambda: False)():
                    raise VerificationError("archive_member_safety_and_hashes")
                if member.size < 0:
                    raise VerificationError("archive_member_safety_and_hashes")
                if member.isdir() and member.size != 0:
                    raise VerificationError("archive_member_safety_and_hashes")
                if member.isfile():
                    unpacked += member.size
                    if unpacked > MAX_TAR_UNCOMPRESSED_BYTES:
                        raise VerificationError("archive_member_safety_and_hashes")
                if name in expected_members:
                    if not member.isfile():
                        raise VerificationError("archive_member_safety_and_hashes")
                    selected[name] = member
            if set(selected) != set(expected_members):
                raise VerificationError("archive_member_safety_and_hashes")
            header_bytes = {name: int(info.size) for name, info in selected.items()}
            if header_bytes != archive_record.get("selected_member_header_bytes"):
                raise VerificationError("archive_member_safety_and_hashes")
            if header_bytes != archive_record.get("selected_members"):
                raise VerificationError("archive_member_safety_and_hashes")
            observed_selected: dict[str, dict[str, Any]] = {}
            for name, info in selected.items():
                partition = expected_members[name]
                record = selected_records.get(partition)
                if record is None or record.get("member_name") != name:
                    raise VerificationError("archive_member_safety_and_hashes")
                if record.get("tar_header_bytes") != info.size or record.get("actual_bytes") != info.size:
                    raise VerificationError("archive_member_safety_and_hashes")
                source = archive.extractfile(info)
                if source is None:
                    raise VerificationError("archive_member_safety_and_hashes")
                digest = hashlib.sha256()
                size = 0
                with source:
                    while True:
                        block = source.read(CHUNK_BYTES)
                        if not block:
                            break
                        size += len(block)
                        if size > info.size or size > MAX_SELECTED_TOTAL_BYTES:
                            raise VerificationError("archive_member_safety_and_hashes")
                        digest.update(block)
                if size != info.size or digest.hexdigest() != record.get("sha256"):
                    raise VerificationError("archive_member_safety_and_hashes")
                observed_selected[partition] = {"bytes": size, "sha256": digest.hexdigest()}
    except VerificationError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise VerificationError("archive_member_safety_and_hashes") from exc
    if count != archive_record.get("tar_member_count") or unpacked != archive_record.get("declared_uncompressed_file_bytes"):
        raise VerificationError("archive_member_safety_and_hashes")
    return {"archive_bytes": archive_size, "archive_sha256": archive_hash, "tar_member_count": count,
            "declared_uncompressed_file_bytes": unpacked, "selected_members": observed_selected}


def observe_source_record(
    connection: sqlite3.Connection,
    partition: str,
    paper_id: str,
    paper: Mapping[str, Any],
    census: Census,
    scratch: sqlite3.Connection,
    data: dict[str, Any],
) -> None:
    if not isinstance(paper, dict):
        raise VerificationError("strict_json_and_database_row_reconstruction")
    paper_counts = add_census_for_paper(census, partition, paper)
    question_count, annotation_count, labeled_count = paper_counts
    doc_raw = document_text(paper)
    normalized_document = normalize_text(doc_raw)
    if len(normalized_document) > MAX_DOCUMENT_CHARS:
        raise VerificationError("paper_identity_and_document_fingerprints")
    doc_hash = sha256_bytes(normalized_document.encode("utf-8"))
    raw_document_fields = {
        field: paper[field]
        for field in ("title", "abstract", "full_text", "figures_and_tables")
        if field in paper
    }
    full_hash = sha256_bytes(canonical_json(raw_document_fields).encode("utf-8"))
    title = paper.get("title")
    title_norm = normalize_text(title) if isinstance(title, str) else ""
    title_hash = sha256_bytes(title_norm.encode("utf-8")) if title_norm else None
    canonical_arxiv = canonical_arxiv_id(paper_id)
    lineage = [f"qasper-paper:{paper_id.casefold()}"]
    if canonical_arxiv is not None:
        lineage.append(f"arxiv:{canonical_arxiv}")
    lineage = sorted(set(lineage))
    input_hashes = [doc_hash]
    group_id = canonical_sha256([SOURCE_ID, PINNED_REVISION, lineage, input_hashes])
    rank = rank_hash(group_id)
    _, q_desc, a_desc = compare_paper_to_database(
        connection, partition, paper_id, paper, doc_hash, full_hash, title_norm or None,
        title_hash, group_id, question_count, annotation_count, labeled_count,
    )
    data["counts"][partition]["papers"] += 1
    data["counts"][partition]["questions"] += question_count
    data["counts"][partition]["annotations"] += annotation_count
    data["counts"][partition]["labeled_annotations"] += labeled_count
    data["descendants"][partition]["papers"].append(
        ((paper_id, full_hash, group_id), [paper_id, full_hash, group_id])
    )
    data["descendants"][partition]["questions"].extend(
        ((paper_id, full_hash, row[1]), row) for row in q_desc
    )
    data["descendants"][partition]["annotations"].extend(
        ((paper_id, full_hash, row[1], row[2]), row) for row in a_desc
    )
    group = data["groups"].get(group_id)
    if group is None:
        group = {
            "source_id": SOURCE_ID, "pinned_revision": PINNED_REVISION,
            "lineage": lineage, "input_hashes": input_hashes, "rank": rank,
            "official_partition": partition, "partitions": defaultdict(lambda: Counter()),
            "component_id": None, "final_split": None, "allocation_reason": None,
            "exclusion_reason": None, "anchor_hash": None, "anchor_group": None,
        }
        data["groups"][group_id] = group
    else:
        if group["lineage"] != lineage or group["input_hashes"] != input_hashes or group["rank"] != rank:
            raise VerificationError("paper_identity_and_document_fingerprints")
        priority = {"train": 0, "validation": 1, "test": 2}
        if priority[partition] > priority[group["official_partition"]]:
            group["official_partition"] = partition
    group["partitions"][partition].update({
        "paper_count": 1, "question_count": question_count,
        "annotation_count": annotation_count, "labeled_annotation_count": labeled_count,
    })
    docs = data["docs"]
    existing = docs.get(doc_hash)
    if existing is None:
        docs[doc_hash] = {"text": normalized_document, "groups": {group_id}}
        scratch.execute(
            "INSERT INTO docs(doc_hash,normalized_text,representative_group) VALUES(?,?,?)",
            (doc_hash, normalized_document, group_id),
        )
    else:
        if existing["text"] != normalized_document:
            raise VerificationError("paper_identity_and_document_fingerprints")
        existing["groups"].add(group_id)
        scratch.execute(
            "UPDATE docs SET representative_group=MIN(representative_group,?) WHERE doc_hash=?",
            (group_id, doc_hash),
        )
    for reason, key in (
        ("same_canonical_arxiv_version_family", canonical_arxiv),
        ("same_normalized_complete_title", title_hash),
        ("same_exact_full_document_sha256", full_hash),
    ):
        if key:
            data["keys"][reason][key].add(group_id)


def add_edge(
    edges: dict[str, tuple[Any, ...]], group_a: str, group_b: str, reason: str,
    hash_a: str, hash_b: str | None = None, intersection: int | None = None,
    union: int | None = None,
) -> None:
    if group_a == group_b:
        return
    if group_a <= group_b:
        first, second, first_hash, second_hash = group_a, group_b, hash_a, hash_b
    else:
        first, second, first_hash, second_hash = group_b, group_a, hash_b, hash_a
    edge_key = canonical_sha256([first, second, reason, first_hash, second_hash])
    edges[edge_key] = (
        edge_key, first, second, reason, first_hash, second_hash, intersection, union,
    )


def hash_rows(rows: Sequence[Sequence[Any]]) -> str:
    return canonical_sha256([list(row) for row in rows])


def make_scratch_indexes(scratch: sqlite3.Connection) -> dict[str, int]:
    scratch.execute("CREATE TABLE grams(doc_id INTEGER NOT NULL,gram TEXT NOT NULL,PRIMARY KEY(doc_id,gram)) WITHOUT ROWID")
    scratch.execute("CREATE INDEX grams_by_gram ON grams(gram,doc_id)")
    total_postings = 0
    short_docs = 0
    for doc_id, text in scratch.execute("SELECT doc_id,normalized_text FROM docs ORDER BY doc_id"):
        grams = {text[index:index + 5] for index in range(max(0, len(text) - 4))}
        if not grams:
            short_docs += 1
            continue
        total_postings += len(grams)
        if total_postings > MAX_NGRAM_POSTINGS:
            raise VerificationError("bounded_complete_character_5gram_join")
        scratch.executemany("INSERT INTO grams(doc_id,gram) VALUES(?,?)", ((doc_id, gram) for gram in grams))
        scratch.execute("UPDATE docs SET gram_count=? WHERE doc_id=?", (len(grams), doc_id))
    scratch.execute("CREATE TABLE frequencies(gram TEXT PRIMARY KEY,document_frequency INTEGER NOT NULL) WITHOUT ROWID")
    scratch.execute("INSERT INTO frequencies SELECT gram,COUNT(*) FROM grams GROUP BY gram")
    scratch.execute("CREATE TABLE prefixes(doc_id INTEGER NOT NULL,gram TEXT NOT NULL,PRIMARY KEY(doc_id,gram)) WITHOUT ROWID")
    scratch.execute("CREATE INDEX prefixes_by_gram ON prefixes(gram,doc_id)")
    cursor = scratch.execute(
        "SELECT g.doc_id,g.gram,f.document_frequency,d.gram_count FROM grams g "
        "JOIN frequencies f ON f.gram=g.gram JOIN docs d ON d.doc_id=g.doc_id "
        "ORDER BY g.doc_id,f.document_frequency,g.gram"
    )
    current_doc: int | None = None
    position = 0
    prefix_limit = 0
    batch: list[tuple[int, str]] = []
    prefix_count = 0
    for doc_id, gram, _, gram_count in cursor:
        if doc_id != current_doc:
            current_doc = doc_id
            position = 0
            ceil_threshold = (9 * gram_count + 9) // 10
            prefix_limit = gram_count - ceil_threshold + 1
        position += 1
        if position <= prefix_limit:
            batch.append((doc_id, gram))
            prefix_count += 1
        if len(batch) >= 10000:
            scratch.executemany("INSERT INTO prefixes(doc_id,gram) VALUES(?,?)", batch)
            batch.clear()
    if batch:
        scratch.executemany("INSERT INTO prefixes(doc_id,gram) VALUES(?,?)", batch)
    scratch.commit()
    prefix_pair_bound = int(scratch.execute(
        "SELECT COALESCE(SUM(n*(n-1)/2),0) FROM (SELECT COUNT(*) n FROM prefixes GROUP BY gram)"
    ).fetchone()[0])
    return {
        "document_count": int(scratch.execute("SELECT COUNT(*) FROM docs").fetchone()[0]),
        "short_document_count": short_docs,
        "gram_posting_count": total_postings,
        "prefix_token_count": prefix_count,
        "raw_prefix_pair_bound": prefix_pair_bound,
    }


def near_duplicate_edges(scratch: sqlite3.Connection, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]], list[tuple[Any, ...]]]:
    index_stats = make_scratch_indexes(scratch)
    if index_stats["raw_prefix_pair_bound"] > MAX_PREFIX_PAIR_WORK:
        raise VerificationError("bounded_complete_character_5gram_join")
    exclusions: list[tuple[Any, ...]] = []
    for doc_hash, text, representative, gram_count in scratch.execute(
        "SELECT doc_hash,normalized_text,representative_group,gram_count FROM docs ORDER BY doc_hash"
    ):
        if gram_count == 0:
            exclusions.append((doc_hash, representative, len(text), "empty_raw_character_5gram_set"))
    groups_by_hash = {doc_hash: set(info["groups"]) for doc_hash, info in data["docs"].items()}
    edges: dict[str, tuple[Any, ...]] = {}
    for reason, key_map in data["keys"].items():
        for key, groups in key_map.items():
            ordered = sorted(groups)
            if len(ordered) < 2:
                continue
            evidence = sha256_bytes(key.encode("utf-8"))
            for group_id in ordered[1:]:
                add_edge(edges, ordered[0], group_id, reason, evidence, evidence)
    for doc_hash, groups in groups_by_hash.items():
        ordered = sorted(groups)
        for group_id in ordered[1:]:
            add_edge(edges, ordered[0], group_id, "exact_normalized_full_document_text", doc_hash)
    candidate_query = (
        "SELECT p1.doc_id,p2.doc_id FROM prefixes p1 JOIN prefixes p2 "
        "ON p1.gram=p2.gram AND p1.doc_id<p2.doc_id "
        "JOIN docs a ON a.doc_id=p1.doc_id JOIN docs b ON b.doc_id=p2.doc_id "
        "WHERE a.representative_group<>b.representative_group "
        "AND 10*MIN(a.gram_count,b.gram_count)>=9*MAX(a.gram_count,b.gram_count) "
        "GROUP BY p1.doc_id,p2.doc_id ORDER BY p1.doc_id,p2.doc_id"
    )
    gram_cache: OrderedDict[int, tuple[set[str], int]] = OrderedDict()
    cache_weight = 0

    def load_grams(doc_id: int) -> set[str]:
        nonlocal cache_weight
        cached = gram_cache.get(doc_id)
        if cached is not None:
            gram_cache.move_to_end(doc_id)
            return cached[0]
        values = {row[0] for row in scratch.execute("SELECT gram FROM grams WHERE doc_id=?", (doc_id,))}
        weight = len(values) * GRAM_CACHE_ENTRY_BYTES
        if weight <= GRAM_CACHE_BYTES:
            while gram_cache and cache_weight + weight > GRAM_CACHE_BYTES:
                _, (_, old_weight) = gram_cache.popitem(last=False)
                cache_weight -= old_weight
            gram_cache[doc_id] = (values, weight)
            cache_weight += weight
        return values

    candidates = rejected = accepted = 0
    for doc_a, doc_b in scratch.execute(candidate_query):
        candidates += 1
        if candidates > MAX_PREFIX_PAIR_WORK:
            raise VerificationError("bounded_complete_character_5gram_join")
        grams_a = load_grams(doc_a)
        grams_b = load_grams(doc_b)
        intersection = len(grams_a & grams_b)
        union = len(grams_a) + len(grams_b) - intersection
        if not union or 10 * intersection < 9 * union:
            rejected += 1
            continue
        row_a = scratch.execute("SELECT doc_hash,representative_group FROM docs WHERE doc_id=?", (doc_a,)).fetchone()
        row_b = scratch.execute("SELECT doc_hash,representative_group FROM docs WHERE doc_id=?", (doc_b,)).fetchone()
        add_edge(
            edges, row_a[1], row_b[1], "verified_char5gram_jaccard_at_least_0.90",
            row_a[0], row_b[0], intersection, union,
        )
        accepted += 1
    near = {
        **index_stats,
        "gram_posting_cap": MAX_NGRAM_POSTINGS,
        "unique_candidate_pairs_verified": candidates,
        "threshold_rejections": rejected,
        "accepted_near_duplicate_edges": accepted,
        "candidate_generation": "complete Jaccard prefix filter with document-frequency token order, exact length bound, and exact set-Jaccard verification",
        "candidate_join_pair_work_cap": MAX_PREFIX_PAIR_WORK,
        "threshold_exact_integer_test": "10*intersection >= 9*union",
        "inputs_shorter_than_5_codepoints": "retained for exact-input and source-lineage grouping; they have no raw character-5gram set and are counted explicitly, not silently discarded",
    }
    return near, edges, exclusions


def connected_components(group_ids: Sequence[str], edges: Mapping[str, Sequence[Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    parent = {group_id: group_id for group_id in group_ids}
    size = {group_id: 1 for group_id in group_ids}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            nxt = parent[item]
            parent[item] = root
            item = nxt
        return root

    for edge in sorted(edges.values(), key=lambda row: row[0]):
        root_a, root_b = find(edge[1]), find(edge[2])
        if root_a == root_b:
            continue
        if size[root_a] < size[root_b] or (size[root_a] == size[root_b] and root_a > root_b):
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]
    members: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids:
        members[find(group_id)].append(group_id)
    group_to_component: dict[str, str] = {}
    component_groups: dict[str, list[str]] = {}
    for group_list in members.values():
        ordered = sorted(group_list)
        component_id = canonical_sha256(ordered)
        component_groups[component_id] = ordered
        for group_id in ordered:
            group_to_component[group_id] = component_id
    return group_to_component, component_groups


def allocate_groups(
    groups: dict[str, dict[str, Any]], component_id_by_group: Mapping[str, str],
    component_groups: Mapping[str, Sequence[str]], protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[list[Any]], list[list[Any]]]:
    caps = {key: int(value) for key, value in protocol["allocation_parameters"]["caps_independent_groups_per_source"].items()}
    component_info: dict[str, dict[str, Any]] = {}
    for component_id, group_list in component_groups.items():
        component_info[component_id] = {
            "groups": list(group_list), "partitions": defaultdict(lambda: Counter()),
            "final_split": None, "reason": None, "anchor": None,
        }
    for group_id, record in groups.items():
        component = component_info[component_id_by_group[group_id]]
        for partition, counts in record["partitions"].items():
            component["partitions"][partition].update(counts)
    has_labeled_test = any(
        component["partitions"].get("test", {}).get("labeled_annotation_count", 0) > 0
        for component in component_info.values()
    )
    assigned: dict[str, str] = {}
    reasons: dict[str, str] = {}
    anchors: dict[str, tuple[str, str]] = {}
    unused: dict[str, str] = {}
    used = {split: 0 for split in ("train", "dev", "calibration", "confirmation")}

    def ranked_group(component_id: str, partition: str) -> tuple[str, str] | None:
        candidates = [
            (groups[group_id]["rank"], group_id)
            for group_id in component_info[component_id]["groups"]
            if groups[group_id]["partitions"].get(partition, {}).get("labeled_annotation_count", 0) > 0
        ]
        return min(candidates) if candidates else None

    test_candidates: list[tuple[tuple[str, str], str]] = []
    validation_candidates: list[tuple[tuple[str, str], str]] = []
    train_candidates: list[tuple[tuple[str, str], str]] = []
    for component_id, component in component_info.items():
        parts = component["partitions"]
        if "test" in parts:
            anchor = ranked_group(component_id, "test")
            if anchor is not None:
                test_candidates.append((anchor, component_id))
            else:
                unused[component_id] = "official_test_component_without_nonnull_source_annotation"
        elif "validation" in parts:
            anchor = ranked_group(component_id, "validation")
            if anchor is None:
                unused[component_id] = "official_validation_component_without_nonnull_source_annotation"
            elif has_labeled_test:
                validation_candidates.append((anchor, component_id))
            else:
                test_candidates.append((anchor, component_id))
        elif "train" in parts:
            anchor = ranked_group(component_id, "train")
            if anchor is not None:
                train_candidates.append((anchor, component_id))
            else:
                unused[component_id] = "official_train_component_without_nonnull_source_annotation"
        else:
            unused[component_id] = "no_eligible_official_partition"

    def assign(component_id: str, split: str, reason: str, anchor: tuple[str, str] | None) -> None:
        assigned[component_id] = split
        reasons[component_id] = reason
        if anchor is not None:
            anchors[component_id] = anchor
        used[split] += 1

    for anchor, component_id in sorted(test_candidates):
        if used["confirmation"] < caps["confirmation"]:
            reason = (
                "official_labeled_test_confirmation_only"
                if has_labeled_test and "test" in component_info[component_id]["partitions"]
                else "official_labeled_validation_confirmation_no_labeled_test"
            )
            assign(component_id, "confirmation", reason, anchor)
        else:
            unused[component_id] = "confirmation_cap_reached"
    for ordinal, (anchor, component_id) in enumerate(sorted(validation_candidates)):
        split = "dev" if ordinal % 2 == 0 else "calibration"
        if used[split] < caps[split]:
            assign(component_id, split, "ranked_official_validation_alternating_dev_calibration", anchor)
        else:
            unused[component_id] = f"{split}_cap_reached"
    recommendations: list[tuple[tuple[str, str], str, str]] = []
    for ordinal, (anchor, component_id) in enumerate(sorted(train_candidates)):
        slot = ordinal % 10
        split = "dev" if slot in (0, 1) else "calibration" if slot in (2, 3) else "train"
        recommendations.append((anchor, component_id, split))
    for anchor, component_id, split in sorted(recommendations, key=lambda item: (item[0], item[1])):
        if used[split] < caps[split]:
            assign(component_id, split, "ranked_official_train_sha_rank_mod_10", anchor)
        else:
            unused[component_id] = f"{split}_cap_reached"
    for component_id in component_info:
        if component_id not in assigned and component_id not in unused:
            unused[component_id] = "no_eligible_source_partition"

    for component_id, component in component_info.items():
        split = assigned.get(component_id, "unused")
        reason = reasons[component_id] if component_id in assigned else unused[component_id]
        anchor = anchors.get(component_id)
        component["final_split"] = split
        component["reason"] = reason
        component["anchor"] = anchor
        for group_id in component["groups"]:
            groups[group_id]["component_id"] = component_id
            groups[group_id]["final_split"] = split
            groups[group_id]["allocation_reason"] = reason if split != "unused" else None
            groups[group_id]["exclusion_reason"] = reason if split == "unused" else None
            groups[group_id]["anchor_hash"] = anchor[0] if anchor else None
            groups[group_id]["anchor_group"] = anchor[1] if anchor else None
    counts_by_split: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_components = [item for item in component_info.values() if item["final_split"] == split]
        group_ids = [gid for item in split_components for gid in item["groups"]]
        counts_by_split[split] = {
            "connected_components": len(split_components),
            "source_groups": len(group_ids),
            "papers": sum(counts["paper_count"] for gid in group_ids for counts in groups[gid]["partitions"].values()),
            "questions": sum(counts["question_count"] for gid in group_ids for counts in groups[gid]["partitions"].values()),
            "answer_annotation_records": sum(counts["annotation_count"] for gid in group_ids for counts in groups[gid]["partitions"].values()),
        }
    group_split_hashes: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        ordered_group_ids = sorted(
            (group_id for group_id, record in groups.items() if record["final_split"] == split),
            key=lambda group_id: (groups[group_id]["rank"], group_id),
        )
        group_split_hashes[split] = {
            "source_group_count": len(ordered_group_ids),
            "ordered_group_ids_sha256": canonical_sha256(ordered_group_ids),
        }
    group_rows = [
        [group_id, record["component_id"], record["rank"], record["final_split"],
         record["allocation_reason"], record["exclusion_reason"]]
        for group_id, record in sorted(groups.items())
    ]
    component_rows = [
        [component_id, canonical_json(list(component["groups"])), component["final_split"],
         component["reason"], component["anchor"][0] if component["anchor"] else None,
         component["anchor"][1] if component["anchor"] else None]
        for component_id, component in sorted(component_info.items())
    ]
    summary = {
        "seed": SEED, "purpose": PURPOSE,
        "caps_independent_groups_per_source": caps,
        "official_partition_precedence": "labeled official test -> confirmation; otherwise labeled official validation -> confirmation; with labeled test, validation alternates dev/calibration; train-only labeled components use rank mod 10; unlabeled or overflow components remain unused",
        "allocation_uses": "source partition, source annotation-presence schema, and input-only group/rank hashes; no answer values, question text, worker IDs, or model outputs",
        "has_labeled_test_partition": has_labeled_test,
        "counts_by_final_split": counts_by_split,
        "group_split_hashes": group_split_hashes,
        "group_assignment_rows_sha256": hash_rows(group_rows),
        "component_assignment_rows_sha256": hash_rows(component_rows),
        "unused_component_count": len(unused),
    }
    return summary, component_info, group_rows, component_rows


def read_json_file(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], str, int]:
    ensure_plain_path(path)
    digest, size = hash_file(path, max_bytes)
    data = path.read_bytes()
    if len(data) != size or sha256_bytes(data) != digest:
        raise VerificationError("frozen_historical_snapshots")
    value = strict_json_bytes(data)
    return value, digest, size


class Verifier:
    def __init__(self, attempt: Path, report_dir: Path, resource_info: Mapping[str, Any]):
        self.attempt = attempt
        self.report_dir = report_dir
        self.checks: dict[str, dict[str, Any]] = {name: {"status": "not_run"} for name in CHECK_NAMES}
        self.inputs: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        self.resource_info = dict(resource_info)

    def check(self, name: str, ok: bool, details: Mapping[str, Any] | None = None) -> None:
        self.checks[name] = {"status": "pass" if ok else "fail"}
        if details:
            self.checks[name]["details"] = dict(details)
        if not ok:
            raise VerificationError(name)

    def run(self) -> None:
        if self.attempt.resolve(strict=True) != EXPECTED_ATTEMPT:
            raise VerificationError("attempt_and_manifest_pin")
        ensure_plain_path(self.attempt, directory=True)
        if stat.S_IMODE(self.attempt.stat().st_mode) & 0o077:
            raise VerificationError("attempt_and_manifest_pin")
        manifest_path = self.attempt / "manifest.json"
        ensure_plain_path(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest_hash = sha256_bytes(manifest_bytes)
        self.check("attempt_and_manifest_pin", manifest_hash == EXPECTED_MANIFEST_SHA256)
        manifest = strict_json_bytes(manifest_bytes)
        self.check(
            "attempt_and_manifest_pin",
            manifest.get("attempt_id") == ATTEMPT_ID
            and manifest.get("schema") == "vey.neutral.qasper.paper-acquisition.v1"
            and manifest.get("status") == "acquired_and_sealed_metadata_only",
        )
        self.inputs["manifest"] = {"sha256": manifest_hash, "bytes": len(manifest_bytes)}
        self.verify_snapshots(manifest)
        plan, protocol = self.verify_plan(manifest)
        receipt, previous_receipt = self.verify_receipts(manifest, plan)
        self.verify_raw_and_metadata(manifest, plan, receipt, previous_receipt)
        connection, database_before = self.open_database(manifest)
        os.environ["TMPDIR"] = str(self.report_dir)
        os.environ["SQLITE_TMPDIR"] = str(self.report_dir)
        scratch_path = self.report_dir / "scratch.sqlite3"
        scratch_fd = os.open(
            scratch_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.close(scratch_fd)
        scratch = sqlite3.connect(scratch_path)
        scratch.execute("PRAGMA journal_mode=OFF")
        scratch.execute("PRAGMA synchronous=OFF")
        scratch.execute("PRAGMA temp_store=FILE")
        scratch.execute("PRAGMA cache_size=-32768")
        scratch.execute("PRAGMA mmap_size=0")
        page_size = int(scratch.execute("PRAGMA page_size").fetchone()[0])
        max_pages = MAX_SQLITE_BYTES // page_size
        if int(scratch.execute(f"PRAGMA max_page_count={max_pages}").fetchone()[0]) > max_pages:
            raise VerificationError("bounded_complete_character_5gram_join")
        scratch.execute(
            "CREATE TABLE docs(doc_id INTEGER PRIMARY KEY,doc_hash TEXT UNIQUE NOT NULL,normalized_text TEXT NOT NULL,representative_group TEXT NOT NULL,gram_count INTEGER NOT NULL DEFAULT 0)"
        )
        try:
            data = self.reconstruct_source(manifest, plan, receipt, connection, scratch)
            self.verify_census_and_descendants(manifest, connection, data)
            near_summary, edges, exclusions = near_duplicate_edges(scratch, data)
            self.verify_db_docs(data, connection, scratch)
            self.check("native_schema_census_and_source_descendants", True, {
                "source_counts": data["counts"],
                "schema_census": data["census_value"],
                "schema_census_sha256": canonical_sha256(data["census_value"]),
                "source_partition_hashes": data["partition_hashes"],
            })
            self.check("strict_json_and_database_row_reconstruction", True, {
                "papers": sum(row["papers"] for row in data["counts"].values()),
                "questions": sum(row["questions"] for row in data["counts"].values()),
                "annotations": sum(row["annotations"] for row in data["counts"].values()),
            })
            self.check("paper_identity_and_document_fingerprints", True, {
                "unique_groups": len(data["groups"]), "unique_normalized_documents": len(data["docs"]),
                "full_document_fields_include_figures": True,
                "normalized_grouping_text_includes_caption": True,
                "model_state_fields": ["title", "abstract", "full_text"],
            })
            self.verify_grouping_and_allocation(manifest, connection, data, near_summary, edges, exclusions, protocol)
            database_after_hash, database_after_size = hash_file(self.attempt / "private.sqlite3")
            self.inputs["database_after"] = {"sha256": database_after_hash, "bytes": database_after_size}
            self.check(
                "database_unchanged_after_read",
                (database_after_hash, database_after_size) == database_before,
            )
        finally:
            scratch.close()
            connection.close()
            try:
                scratch_path.unlink()
            except FileNotFoundError:
                pass

    def verify_snapshots(self, manifest: Mapping[str, Any]) -> None:
        frozen, frozen_hash, frozen_size = read_json_file(self.attempt / "frozen_inputs.json")
        snapshot = manifest.get("frozen_inputs", {}).get("snapshot_metadata")
        self.check(
            "frozen_historical_snapshots",
            isinstance(snapshot, dict) and frozen == snapshot
            and manifest.get("frozen_inputs", {}).get("repository_commit") == EXPECTED_ACQUISITION_COMMIT
            and frozen.get("repository_commit") == EXPECTED_ACQUISITION_COMMIT,
        )
        files = frozen.get("files")
        if not isinstance(files, dict) or len(files) != 4:
            raise VerificationError("frozen_historical_snapshots")
        for relative, record in files.items():
            if relative not in {
                "research/endgame/neutral_acquire.py",
                "research/endgame/neutral_benchmark_protocol.json",
                "research/endgame/neutral_paper_acquire.py",
                "research/endgame/neutral_paper_acquisition_sources.json",
            }:
                raise VerificationError("frozen_historical_snapshots")
            target = self.attempt / "frozen_inputs" / relative
            ensure_plain_path(target)
            digest, size = hash_file(target)
            if (digest, size) != (record.get("sha256"), record.get("bytes")):
                raise VerificationError("frozen_historical_snapshots")
            self.inputs[f"frozen:{relative}"] = {"sha256": digest, "bytes": size}
        for filename, key in (("source_plan_snapshot.json", "research/endgame/neutral_paper_acquisition_sources.json"),
                              ("protocol_snapshot.json", "research/endgame/neutral_benchmark_protocol.json")):
            value, digest, size = read_json_file(self.attempt / filename)
            frozen_path = self.attempt / "frozen_inputs" / key
            frozen_value = strict_json_bytes(frozen_path.read_bytes())
            if value != frozen_value:
                raise VerificationError("frozen_historical_snapshots")
            self.inputs[filename] = {"sha256": digest, "bytes": size}
        ensure_plain_path(self.attempt / "attempt_context.json")
        context = strict_json_bytes((self.attempt / "attempt_context.json").read_bytes())
        if context.get("private_root") != str(self.attempt.parent.parent) or context.get("source_id") != SOURCE_ID or context.get("source_revision") != PINNED_REVISION:
            raise VerificationError("frozen_historical_snapshots")
        if context.get("frozen_inputs") != frozen:
            raise VerificationError("frozen_historical_snapshots")
        self.inputs["frozen_inputs_record"] = {"sha256": frozen_hash, "bytes": frozen_size}

    def verify_plan(self, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        plan_bytes = (self.attempt / "frozen_inputs/research/endgame/neutral_paper_acquisition_sources.json").read_bytes()
        protocol_bytes = (self.attempt / "frozen_inputs/research/endgame/neutral_benchmark_protocol.json").read_bytes()
        plan = strict_json_bytes(plan_bytes)
        protocol = strict_json_bytes(protocol_bytes)
        snapshot = manifest["frozen_inputs"]["snapshot_metadata"]["files"]
        plan_path = "research/endgame/neutral_paper_acquisition_sources.json"
        protocol_path = "research/endgame/neutral_benchmark_protocol.json"
        self.check(
            "source_plan_and_protocol_pins",
            plan["source"]["id"] == SOURCE_ID
            and plan["source"]["hf_revision"] == PINNED_REVISION
            and plan["source"]["loader_version"] == "0.3.0"
            and protocol.get("schema") == "vey.endgame.neutral-benchmark-protocol.v2"
            and protocol["allocation_parameters"]["seed"] == SEED
            and protocol["allocation_parameters"]["caps_independent_groups_per_source"]
            == {"train": 4096, "dev": 1024, "calibration": 1024, "confirmation": 8192}
            and manifest.get("source", {}).get("id") == SOURCE_ID
            and manifest.get("source", {}).get("pinned_revision") == PINNED_REVISION
            and manifest.get("source", {}).get("loader_version") == "0.3.0"
            and manifest.get("execution", {}).get("source_loader_executed") is False
            and manifest.get("execution", {}).get("archive_member_filesystem_extraction") is False
            and manifest.get("execution", {}).get("bundled_evaluator_executed") is False
            and manifest.get("execution", {}).get("model_or_gpu_calls") == 0
            and manifest.get("execution", {}).get("training_or_quality_evaluation") is False
            and manifest.get("execution", {}).get("answer_values_or_text_emitted") is False
            and manifest.get("execution", {}).get("token_length_census_or_eligibility_claim") is False
            and plan.get("resource_limits") == {
                "cpu_threads_max": 4,
                "blas_threads": 1,
                "nice": 10,
                "ionice_class": 2,
                "ionice_priority": 7,
                "compressed_bytes_per_archive_max": MAX_ARCHIVE_BYTES,
                "selected_uncompressed_bytes_total_max": MAX_SELECTED_TOTAL_BYTES,
                "minimum_available_ram_gib": 8,
                "minimum_free_disk_gib": 20,
                "gpu_use": False,
            }
            and manifest["frozen_inputs"]["source_plan_sha256"] == snapshot[plan_path]["sha256"]
            and manifest["frozen_inputs"]["neutral_protocol_sha256"] == snapshot[protocol_path]["sha256"]
            and manifest["frozen_inputs"]["acquisition_script_sha256"] == snapshot["research/endgame/neutral_paper_acquire.py"]["sha256"]
            and manifest["frozen_inputs"]["shared_helper_sha256"] == snapshot["research/endgame/neutral_acquire.py"]["sha256"],
        )
        if plan.get("root") != str(self.attempt.parent.parent):
            raise VerificationError("source_plan_and_protocol_pins")
        self.check(
            "declared_schema_expectation",
            manifest.get("expected_source_schema") == {
                "basis": "features declared by the pinned QASPER loader; source payloads are preserved even when nullable or structurally absent",
                "paper_fields": ["id", "title", "abstract", "full_text", "qas", "figures_and_tables"],
                "full_text_fields": ["section_name", "paragraphs"],
                "question_fields": list(QUESTION_FIELDS),
                "annotation_fields": list(ANNOTATION_FIELDS),
                "answer_fields": list(ANSWER_FIELDS),
                "figures_and_tables_fields": list(FIGURE_FIELDS),
                "annotation_census_rule": "counts raw field presence and JSON type only; explicit null and absent remain distinct; no target-value coercion or usable-Boolean coverage claim",
            },
        )
        self.inputs["source_plan"] = {"sha256": sha256_bytes(plan_bytes), "bytes": len(plan_bytes)}
        self.inputs["protocol"] = {"sha256": sha256_bytes(protocol_bytes), "bytes": len(protocol_bytes)}
        return plan, protocol

    def verify_receipts(self, manifest: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt, receipt_hash, receipt_size = read_json_file(self.attempt / "raw_source_manifest.json")
        reuse = plan.get("raw_reuse")
        if not isinstance(reuse, dict) or reuse.get("preparse_manifest_sha256") != EXPECTED_REUSE_RECEIPT_SHA256:
            raise VerificationError("preparse_receipt_and_raw_reuse_origin")
        prior_attempt = Path(str(reuse.get("attempt", "")))
        if prior_attempt != self.attempt.parent / "20261004T073820.781460Z-591540-6f09fcb3":
            raise VerificationError("preparse_receipt_and_raw_reuse_origin")
        ensure_plain_path(prior_attempt, directory=True)
        previous, previous_hash, previous_size = read_json_file(prior_attempt / "raw_source_manifest.json")
        self.check(
            "preparse_receipt_and_raw_reuse_origin",
            previous_hash == EXPECTED_REUSE_RECEIPT_SHA256
            and receipt.get("schema") == "vey.neutral.qasper.preparse-custody.v1"
            and receipt.get("all_archive_and_selected_member_hashes_frozen_before_parsing") is True
            and receipt.get("source_rows_parsed") == 0
            and receipt.get("model_or_gpu_calls") == 0
            and receipt.get("repository_commit") == EXPECTED_ACQUISITION_COMMIT,
            {"current_receipt_sha256": receipt_hash, "current_receipt_bytes": receipt_size,
             "raw_reuse_receipt_sha256": previous_hash, "raw_reuse_receipt_bytes": previous_size},
        )
        if receipt.get("frozen_input_record") != manifest.get("frozen_inputs", {}).get("snapshot_metadata"):
            raise VerificationError("preparse_receipt_and_raw_reuse_origin")
        self.inputs["raw_source_manifest"] = {"sha256": receipt_hash, "bytes": receipt_size}
        self.inputs["raw_reuse_receipt"] = {"sha256": previous_hash, "bytes": previous_size}
        return receipt, previous

    def verify_raw_and_metadata(
        self, manifest: Mapping[str, Any], plan: Mapping[str, Any], receipt: Mapping[str, Any], previous: Mapping[str, Any]
    ) -> None:
        source = plan["source"]
        raw_record = receipt.get("source_metadata")
        metadata = manifest["source"]["metadata_fingerprints"]
        prior_metadata = previous.get("source_metadata", {})
        if raw_record != metadata:
            raise VerificationError("raw_artifact_hashes_and_metadata_license")
        reuse_index: dict[tuple[str, str], Mapping[str, Any]] = {}
        for old in previous.get("archive_artifacts", []):
            reuse_index[("archive", str(old.get("requested_url")))] = old
        for kind in ("loader", "dataset_card"):
            record = prior_metadata.get(kind)
            if isinstance(record, dict):
                reuse_index[(kind, str(record.get("requested_url")))] = record
        current_archives = manifest.get("archive_artifacts", [])
        receipt_archives = receipt.get("archive_artifacts", [])
        if current_archives != receipt_archives or len(current_archives) != len(source.get("archives", [])):
            raise VerificationError("raw_artifact_hashes_and_metadata_license")
        current_reused: list[tuple[str, str, str, int]] = []
        for archive_spec, record in zip(source["archives"], current_archives):
            url = validate_url(str(archive_spec["url"]))
            name = PurePosixPath(urllib.parse.urlsplit(url).path).name
            if record.get("requested_url") != url or record.get("archive_name") != name:
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            final_url = str(record.get("final_url", ""))
            validate_url(final_url)
            if (urllib.parse.urlsplit(final_url).hostname or "").lower() not in {
                "huggingface.co", "qasper-dataset.s3.us-west-2.amazonaws.com"
            }:
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            path = self.attempt / "raw" / "archives" / name
            if record.get("relative_path") != str(path):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            ensure_plain_path(path)
            digest, size = hash_file(path, MAX_ARCHIVE_BYTES)
            if (digest, size) != (record.get("sha256"), record.get("bytes")):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            old = reuse_index.get(("archive", url))
            if (
                old is None
                or record.get("reused_from") != old.get("relative_path")
                or record.get("final_url") != old.get("final_url")
                or record.get("sha256") != old.get("sha256")
                or record.get("bytes") != old.get("bytes")
                or record.get("reused_from_preparse_sha256") != EXPECTED_REUSE_RECEIPT_SHA256
            ):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            old_path = Path(str(old.get("relative_path")))
            ensure_plain_path(old_path)
            old_digest, old_size = hash_file(old_path, MAX_ARCHIVE_BYTES)
            if (old_digest, old_size) != (old.get("sha256"), old.get("bytes")):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            verify_hardlink_reuse(path, old_path)
            current_reused.append(("archive", name, digest, size))
            self.inputs[f"archive:{name}"] = {"sha256": digest, "bytes": size}
            self.inputs[f"raw_reuse:archive:{name}"] = {"sha256": old_digest, "bytes": old_size}
        metadata_specs = (
            ("loader", "qasper.py", str(source["loader_url"])),
            ("dataset_card", "README.md", str(source["card_url"])),
        )
        metadata_bytes: dict[str, bytes] = {}
        for kind, filename, url_value in metadata_specs:
            url = validate_url(url_value)
            record = metadata[kind]
            path = self.attempt / "raw" / "source_metadata" / filename
            if record.get("requested_url") != url or record.get("relative_path") != str(path):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            final_url = str(record.get("final_url", ""))
            validate_url(final_url)
            if (urllib.parse.urlsplit(final_url).hostname or "").lower() not in {
                "huggingface.co", "qasper-dataset.s3.us-west-2.amazonaws.com"
            }:
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            ensure_plain_path(path)
            digest, size = hash_file(path, MAX_METADATA_BYTES)
            if (digest, size) != (record.get("sha256"), record.get("bytes")):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            old = reuse_index.get((kind, url))
            if (
                old is None
                or record.get("reused_from") != old.get("relative_path")
                or record.get("final_url") != old.get("final_url")
                or record.get("sha256") != old.get("sha256")
                or record.get("bytes") != old.get("bytes")
                or record.get("reused_from_preparse_sha256") != EXPECTED_REUSE_RECEIPT_SHA256
            ):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            old_path = Path(str(old.get("relative_path")))
            ensure_plain_path(old_path)
            old_digest, old_size = hash_file(old_path, MAX_METADATA_BYTES)
            if (old_digest, old_size) != (old.get("sha256"), old.get("bytes")):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
            verify_hardlink_reuse(path, old_path)
            metadata_bytes[kind] = path.read_bytes()
            current_reused.append((kind, filename, digest, size))
            self.inputs[f"metadata:{filename}"] = {"sha256": digest, "bytes": size}
            self.inputs[f"raw_reuse:{kind}:{filename}"] = {"sha256": old_digest, "bytes": old_size}
        for current in current_archives:
            if current.get("reused_from_preparse_sha256") != EXPECTED_REUSE_RECEIPT_SHA256:
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
        loader = metadata_bytes["loader"].decode("utf-8")
        card = metadata_bytes["dataset_card"].decode("utf-8")
        expected_bindings = {
            "_VERSION": "0.3.0",
            "_LICENSE": "CC BY 4.0",
            "_URL_TRAIN_DEV": source["archives"][0]["url"],
            "_URL_TEST": source["archives"][1]["url"],
        }
        bindings = {}
        for node in ast.parse(loader).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in expected_bindings:
                        bindings[target.id] = ast.literal_eval(node.value)
        if bindings != expected_bindings:
            raise VerificationError("raw_artifact_hashes_and_metadata_license")
        frontmatter = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", card, re.DOTALL)
        if frontmatter is None or re.search(
            r'(?m)^license:\s*(?:\n\s*-\s*)?["\']?cc-by-4\.0["\']?\s*(?:\n|$)',
            frontmatter.group(1),
        ) is None or re.search(
            r'https://creativecommons\.org/licenses/by/4\.0/?(?:[)\s"<>]|$)', card
        ) is None:
            raise VerificationError("raw_artifact_hashes_and_metadata_license")
        claims = metadata.get("checked_claims", {})
        if claims.get("license_url") != "https://creativecommons.org/licenses/by/4.0" or claims.get("loader_version") != "0.3.0":
            raise VerificationError("raw_artifact_hashes_and_metadata_license")
        citations = manifest.get("source", {}).get("source_grant_citations", [])
        citation_by_url = {row.get("url"): row for row in citations if isinstance(row, dict)}
        for kind, url_value in (("loader", source["loader_url"]), ("dataset_card", source["card_url"])):
            citation = citation_by_url.get(url_value)
            if citation is None or citation.get("sha256") != metadata[kind].get("sha256"):
                raise VerificationError("raw_artifact_hashes_and_metadata_license")
        if not any(row.get("license_text_url") == "https://creativecommons.org/licenses/by/4.0" for row in citations if isinstance(row, dict)):
            raise VerificationError("raw_artifact_hashes_and_metadata_license")
        self.check("raw_artifact_hashes_and_metadata_license", True, {
            "archive_count": len(current_archives), "metadata_object_count": 2,
            "raw_reuse_objects": len(current_reused), "metadata_urls_and_license_declaration_checked": True,
        })

    def open_database(self, manifest: Mapping[str, Any]) -> tuple[sqlite3.Connection, tuple[str, int]]:
        record = manifest.get("private_database", {})
        path = self.attempt / "private.sqlite3"
        if record.get("relative_path") != "private.sqlite3":
            raise VerificationError("readonly_database_fingerprint")
        ensure_plain_path(path)
        if record.get("git_tracked") is not False:
            raise VerificationError("readonly_database_fingerprint")
        digest, size = hash_file(path)
        if (digest, size) != (record.get("sha256"), record.get("bytes")):
            raise VerificationError("readonly_database_fingerprint")
        uri = f"file:{urllib.parse.quote(str(path), safe='/')}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA mmap_size=0")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            connection.close()
            raise VerificationError("readonly_database_fingerprint")
        self.check(
            "readonly_database_fingerprint",
            True,
            {"sha256": digest, "bytes": size, "mode": "read_only_immutable"},
        )
        self.inputs["private_database"] = {"sha256": digest, "bytes": size, "mode": "read_only_immutable"}
        return connection, (digest, size)

    def reconstruct_source(
        self, manifest: Mapping[str, Any], plan: Mapping[str, Any], receipt: Mapping[str, Any],
        connection: sqlite3.Connection, scratch: sqlite3.Connection,
    ) -> dict[str, Any]:
        archive_specs = plan["source"]["archives"]
        selected_list = manifest.get("selected_member_artifacts", [])
        selected_records = {record.get("member_name"): record for record in selected_list}
        selected_by_partition = {
            partition: record
            for partition, record in receipt.get("selected_member_artifacts", {}).items()
        }
        if len(selected_records) != 3 or set(selected_by_partition) != set(PARTITIONS):
            raise VerificationError("archive_member_safety_and_hashes")
        partition_to_name: dict[str, str] = {}
        expected_names: set[str] = set()
        archive_records = manifest.get("archive_artifacts", [])
        if len(archive_records) != 2:
            raise VerificationError("archive_member_safety_and_hashes")
        tar_results: dict[str, Any] = {}
        for archive_spec, record in zip(archive_specs, archive_records):
            expected = {str(name): str(partition) for name, partition in archive_spec["members"].items()}
            for member_name, partition in expected.items():
                if partition in partition_to_name or member_name in expected_names:
                    raise VerificationError("archive_member_safety_and_hashes")
                partition_to_name[partition] = member_name
                expected_names.add(member_name)
            archive_name = PurePosixPath(urllib.parse.urlsplit(str(archive_spec["url"])).path).name
            tar_results[archive_name] = inspect_archive(
                self.attempt / "raw" / "archives" / archive_name, record, expected, selected_by_partition
            )
        if set(partition_to_name) != set(PARTITIONS) or expected_names != set(selected_records):
            raise VerificationError("archive_member_safety_and_hashes")
        selected_total = 0
        for partition, member_name in partition_to_name.items():
            record = selected_by_partition[partition]
            path = self.attempt / "raw" / "selected_members" / f"{partition}.json"
            if record.get("relative_path") != str(path) or record.get("member_name") != member_name:
                raise VerificationError("archive_member_safety_and_hashes")
            ensure_plain_path(path)
            digest, size = hash_file(path, MAX_SELECTED_TOTAL_BYTES)
            if digest != record.get("sha256") or size != record.get("actual_bytes"):
                raise VerificationError("archive_member_safety_and_hashes")
            selected_total += size
            self.inputs[f"selected_member:{partition}"] = {"sha256": digest, "bytes": size}
        if selected_total > MAX_SELECTED_TOTAL_BYTES:
            raise VerificationError("archive_member_safety_and_hashes")
        expected_receipt_selected = {
            partition: selected_records[member_name]
            for partition, member_name in partition_to_name.items()
        }
        if receipt.get("selected_member_artifacts") != expected_receipt_selected:
            raise VerificationError("archive_member_safety_and_hashes")
        self.check("archive_member_safety_and_hashes", True, {
            "archive_count": len(tar_results), "archive_scan_summaries": tar_results,
            "selected_member_count": len(selected_by_partition),
            "selected_total_bytes": selected_total, "filesystem_extraction": False,
            "tar_member_types_checked": True,
        })
        census = Census()
        data: dict[str, Any] = {
            "census": census,
            "counts": {part: {"papers": 0, "questions": 0, "annotations": 0, "labeled_annotations": 0} for part in PARTITIONS},
            "descendants": {part: {"papers": [], "questions": [], "annotations": []} for part in PARTITIONS},
            "groups": {}, "docs": {}, "keys": defaultdict(lambda: defaultdict(set)),
            "tar_results": tar_results,
        }
        for partition in PARTITIONS:
            seen_ids: set[str] = set()
            path = self.attempt / "raw" / "selected_members" / f"{partition}.json"
            for paper_id, paper in stream_top_level_object(path):
                if paper_id in seen_ids:
                    raise VerificationError("strict_json_and_database_row_reconstruction")
                seen_ids.add(paper_id)
                observe_source_record(connection, partition, paper_id, paper, census, scratch, data)
        scratch.commit()
        for table, expected_count in (
            ("papers", sum(value["papers"] for value in data["counts"].values())),
            ("questions", sum(value["questions"] for value in data["counts"].values())),
            ("annotations", sum(value["annotations"] for value in data["counts"].values())),
        ):
            if connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] != expected_count:
                raise VerificationError("strict_json_and_database_row_reconstruction")
        return data

    def verify_census_and_descendants(
        self, manifest: Mapping[str, Any], connection: sqlite3.Connection, data: dict[str, Any]
    ) -> None:
        census_value = data["census"].serialized()
        db_parents: dict[tuple[str, str], int] = {
            (part, scope): int(count)
            for part, scope, count in connection.execute(
                "SELECT source_partition,scope,parent_objects FROM schema_parents ORDER BY source_partition,scope"
            )
        }
        db_fields: dict[tuple[str, str, str, str], int] = {
            (part, scope, field, kind): int(count)
            for part, scope, field, kind, count in connection.execute(
                "SELECT source_partition,scope,field_name,json_type,value_count FROM schema_counts ORDER BY source_partition,scope,field_name,json_type"
            )
        }
        expected_parents = dict(data["census"].parents)
        expected_fields = dict(data["census"].fields)
        if db_parents != expected_parents or db_fields != expected_fields:
            raise VerificationError("native_schema_census_and_source_descendants")
        manifest_partition = manifest.get("source_partition_counts_and_descendant_hashes", {}).get("by_official_partition", {})
        partition_hashes: dict[str, Any] = {}
        for partition in PARTITIONS:
            for kind in ("papers", "questions", "annotations"):
                entries = data["descendants"][partition][kind]
                entries.sort(key=lambda item: item[0])
                data["descendants"][partition][kind] = [item[1] for item in entries]
            row = data["counts"][partition]
            expected = {
                "papers": row["papers"], "questions": row["questions"],
                "answer_annotation_records": row["annotations"],
                "annotations_with_nonnull_source_outcome_fields": row["labeled_annotations"],
                "paper_descendant_sha256": hash_rows(data["descendants"][partition]["papers"]),
                "question_descendant_sha256": hash_rows(data["descendants"][partition]["questions"]),
                "annotation_descendant_sha256": hash_rows(data["descendants"][partition]["annotations"]),
            }
            if manifest_partition.get(partition) != expected:
                raise VerificationError("native_schema_census_and_source_descendants")
            partition_hashes[partition] = expected
        if manifest.get("source_partition_counts_and_descendant_hashes", {}).get("schema_census") != census_value:
            raise VerificationError("native_schema_census_and_source_descendants")
        if manifest.get("annotation_schema_census_sha256") != canonical_sha256(census_value):
            raise VerificationError("native_schema_census_and_source_descendants")
        data["census_value"] = census_value
        data["partition_hashes"] = partition_hashes

    def verify_db_docs(self, data: Mapping[str, Any], connection: sqlite3.Connection, scratch: sqlite3.Connection) -> None:
        expected_docs = data["docs"]
        db_docs = {
            row[0]: (row[1], row[2], int(row[3]))
            for row in connection.execute(
                "SELECT doc_hash,normalized_text,representative_group,gram_count FROM docs ORDER BY doc_hash"
            )
        }
        for doc_hash, record in expected_docs.items():
            representative = min(record["groups"])
            value = db_docs.get(doc_hash)
            if value is None or value[:2] != (record["text"], representative):
                raise VerificationError("paper_identity_and_document_fingerprints")
        if set(db_docs) != set(expected_docs):
            raise VerificationError("paper_identity_and_document_fingerprints")
        for doc_hash, text, representative, grams in scratch.execute(
            "SELECT doc_hash,normalized_text,representative_group,gram_count FROM docs ORDER BY doc_hash"
        ):
            db_value = db_docs[doc_hash]
            if db_value != (text, representative, int(grams)):
                raise VerificationError("bounded_complete_character_5gram_join")
        expected_doc_groups = sorted(
            (doc_hash, group_id)
            for doc_hash, record in expected_docs.items()
            for group_id in record["groups"]
        )
        actual_doc_groups = [tuple(row) for row in connection.execute("SELECT doc_hash,group_id FROM doc_groups ORDER BY doc_hash,group_id")]
        if actual_doc_groups != expected_doc_groups:
            raise VerificationError("paper_identity_and_document_fingerprints")

    def verify_grouping_and_allocation(
        self, manifest: Mapping[str, Any], connection: sqlite3.Connection, data: dict[str, Any],
        near_summary: Mapping[str, Any], edges: Mapping[str, Sequence[Any]], exclusions: Sequence[Sequence[Any]],
        protocol: Mapping[str, Any],
    ) -> None:
        groups = data["groups"]
        component_by_group, component_groups = connected_components(sorted(groups), edges)
        for group_id, record in groups.items():
            record["component_id"] = component_by_group[group_id]
        allocation, components, group_rows, component_rows = allocate_groups(
            groups, component_by_group, component_groups, protocol
        )
        edge_rows = [list(row) for _, row in sorted(edges.items())]
        actual_edges = [list(row) for row in connection.execute(
            "SELECT edge_key,group_a,group_b,reason,evidence_hash_a,evidence_hash_b,intersection_size,union_size FROM edges ORDER BY edge_key"
        )]
        if actual_edges != edge_rows:
            raise VerificationError("group_edges_and_edge_hashes")
        edge_reasons = dict(Counter(row[3] for row in edge_rows))
        edge_hash = hash_rows(edge_rows)
        source_group_rows: list[list[Any]] = []
        group_part_rows: list[list[Any]] = []
        for group_id, record in sorted(groups.items()):
            lineage_json = canonical_json(record["lineage"])
            hashes_json = canonical_json(record["input_hashes"])
            expected = [
                group_id, SOURCE_ID, PINNED_REVISION, record["official_partition"], lineage_json,
                hashes_json, record["rank"], record["component_id"], record["final_split"],
                record["allocation_reason"], record["exclusion_reason"],
            ]
            source_group_rows.append(expected)
            for partition, counts in sorted(record["partitions"].items()):
                group_part_rows.append([
                    group_id, partition, counts["paper_count"], counts["question_count"],
                    counts["annotation_count"], counts["labeled_annotation_count"],
                ])
        actual_source_groups = [list(row) for row in connection.execute(
            "SELECT group_id,source_id,pinned_revision,official_partition,lineage_ids_json,input_hashes_json,rank_sha256,component_id,final_split,allocation_reason,exclusion_reason FROM source_groups ORDER BY group_id"
        )]
        actual_group_parts = [list(row) for row in connection.execute(
            "SELECT group_id,official_partition,paper_count,question_count,annotation_count,nonnull_outcome_annotation_count FROM group_partitions ORDER BY group_id,official_partition"
        )]
        if actual_source_groups != source_group_rows or actual_group_parts != group_part_rows:
            raise VerificationError("group_edges_and_edge_hashes")
        actual_exclusions = [list(row) for row in connection.execute(
            "SELECT doc_hash,representative_group,normalized_codepoints,reason FROM search_exclusions ORDER BY doc_hash"
        )]
        if actual_exclusions != [list(row) for row in exclusions]:
            raise VerificationError("bounded_complete_character_5gram_join")
        component_db_rows: list[list[Any]] = []
        component_manifest_rows: list[list[Any]] = []
        for component_id, component in sorted(components.items()):
            groups_json = canonical_json(component["groups"])
            papers = sum(count["paper_count"] for gid in component["groups"] for count in groups[gid]["partitions"].values())
            questions = sum(count["question_count"] for gid in component["groups"] for count in groups[gid]["partitions"].values())
            annotations = sum(count["annotation_count"] for gid in component["groups"] for count in groups[gid]["partitions"].values())
            anchor = component["anchor"]
            component_db_rows.append([
                component_id, groups_json, len(component["groups"]), papers, questions, annotations,
                component["final_split"], component["reason"], anchor[0] if anchor else None,
                anchor[1] if anchor else None,
            ])
            component_manifest_rows.append([
                component_id, groups_json, component["final_split"], component["reason"],
            ])
            if component_id != canonical_sha256(component["groups"]):
                raise VerificationError("components_and_component_hashes")
        actual_components = [list(row) for row in connection.execute(
            "SELECT component_id,groups_json,group_count,paper_count,question_count,annotation_count,final_split,allocation_reason,rank_anchor_sha256,rank_anchor_group_id FROM components ORDER BY component_id"
        )]
        if actual_components != component_db_rows:
            raise VerificationError("components_and_component_hashes")
        grouping = manifest.get("grouping", {})
        grouping_hashes = {
            "source_group_count": len(groups),
            "connected_component_count": len(components),
            "edge_count": len(edge_rows),
            "edge_reason_counts": edge_reasons,
            "near_duplicate_search": dict(near_summary),
            "source_groups_sha256": hash_rows(source_group_rows),
            "edges_sha256": edge_hash,
            "components_sha256": hash_rows(component_manifest_rows),
            "grouping_inputs": [
                "canonical arXiv versionless identifiers parsed from the original QASPER paper key",
                "Unicode NFKC/casefold/whitespace-normalized complete titles",
                "exact full-document field-content SHA-256 over title, abstract, full_text, and figures_and_tables",
                "exact normalized full-document text and verified complete character-5gram-set Jaccard >= 0.90",
            ],
            "never_grouped_by": ["question wording", "question IDs", "annotation IDs", "worker IDs", "answer values", "evidence wording"],
        }
        if grouping != grouping_hashes:
            raise VerificationError("group_edges_and_edge_hashes")
        if allocation != manifest.get("allocation"):
            raise VerificationError("allocation_membership_reasons_caps_and_hashes")
        for component_id, component in components.items():
            if any(groups[group_id]["final_split"] != component["final_split"] for group_id in component["groups"]):
                raise VerificationError("allocation_membership_reasons_caps_and_hashes")
        self.check("bounded_complete_character_5gram_join", True, {
            "documents": near_summary["document_count"],
            "character_5gram_postings": near_summary["gram_posting_count"],
            "candidate_pairs": near_summary["unique_candidate_pairs_verified"],
            "threshold_rejections": near_summary["threshold_rejections"],
            "accepted_near_duplicate_edges": near_summary["accepted_near_duplicate_edges"],
            "raw_prefix_pair_bound": near_summary["raw_prefix_pair_bound"],
            "candidate_work_cap": MAX_PREFIX_PAIR_WORK,
        })
        self.check("group_edges_and_edge_hashes", True, {
            "source_groups": len(groups), "edges": len(edge_rows), "edge_sha256": edge_hash,
            "edge_reason_counts": edge_reasons,
        })
        self.check("components_and_component_hashes", True, {
            "components": len(components), "components_sha256": hash_rows(component_manifest_rows),
            "ordered_component_ids_sha256": canonical_sha256(sorted(components)),
            "component_assignment_rows_sha256": allocation["component_assignment_rows_sha256"],
        })
        reason_counts = Counter(record["allocation_reason"] or record["exclusion_reason"] for record in groups.values())
        split_counts = {split: value for split, value in allocation["counts_by_final_split"].items()}
        self.check("allocation_membership_reasons_caps_and_hashes", True, {
            "split_counts": split_counts, "group_split_hashes": allocation["group_split_hashes"],
            "group_assignment_rows_sha256": allocation["group_assignment_rows_sha256"],
            "component_assignment_rows_sha256": allocation["component_assignment_rows_sha256"],
            "allocation_reason_counts": dict(sorted(reason_counts.items())),
            "unused_component_count": allocation["unused_component_count"],
            "caps": allocation["caps_independent_groups_per_source"],
            "label_presence_only_for_eligibility": True,
        })
        self.results["grouping"] = grouping_hashes
        self.results["allocation"] = allocation
        self.results["reason_inventory"] = {
            "edge": dict(sorted(edge_reasons.items())),
            "source_group": dict(sorted(reason_counts.items())),
            "component": dict(sorted(Counter(component["reason"] for component in components.values()).items())),
            "search_exclusion": dict(sorted(Counter(row[3] for row in exclusions).items())),
        }


def create_report_dir(attempt: Path) -> Path:
    verification = attempt / "verification"
    if verification.exists() and verification.is_symlink():
        raise VerificationError("attempt_and_manifest_pin")
    verification.mkdir(mode=0o700, exist_ok=True)
    if stat.S_IMODE(verification.stat().st_mode) & 0o077:
        raise VerificationError("attempt_and_manifest_pin")
    for _ in range(12):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        target = verification / f"{stamp}-{os.getpid()}-{os.urandom(4).hex()}"
        try:
            target.mkdir(mode=0o700)
            return target
        except FileExistsError:
            continue
    raise VerificationError("attempt_and_manifest_pin")


def write_report(verifier: Verifier, report_dir: Path, status: str, error_check: str | None) -> Path:
    report = {
        "schema": "vey.neutral.qasper.paper-verification.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempt_id": ATTEMPT_ID,
        "attempt_path": str(verifier.attempt),
        "manifest_sha256_expected": EXPECTED_MANIFEST_SHA256,
        "input_hashes_and_sizes": verifier.inputs,
        "resource_guards": verifier.resource_info,
        "checks": verifier.checks,
        "results": verifier.results,
        "error_check": error_check,
        "exact_completeness_limits": [
            "Custody and row-identity reconstruction only; no token-length/count or tokenizer-eligibility result.",
            "No Boolean/outcome quality, certification, confidence-interval, statistical-independence, or competitor claim.",
            "No source text, question wording, answer values, evidence, or worker identifiers are emitted in this report or stdout.",
            "The source is public/exposed; this verification does not establish universally fresh confirmation or unknown pretraining exclusion.",
            "Underlying full-paper republication rights remain subject to review; verification does not authorize redistribution.",
            "Character-5gram joining verifies the specified bounded exact algorithm over this pinned corpus; it is not a general proof that all semantic duplicates are detected.",
        ],
    }
    destination = report_dir / "report.json"
    payload = (canonical_json(report) + "\n").encode("utf-8")
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    directory_fd = os.open(report_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently verify the pinned, private QASPER custody attempt.")
    parser.add_argument("--attempt", type=Path, required=True, help="sealed private attempt directory")
    args = parser.parse_args(argv)
    attempt = args.attempt.absolute()
    try:
        ensure_plain_path(attempt, directory=True)
        if attempt.resolve(strict=True) != EXPECTED_ATTEMPT or stat.S_IMODE(attempt.stat().st_mode) & 0o077:
            raise VerificationError("attempt_and_manifest_pin")
        resource_info = set_resource_guards()
        os.umask(0o077)
        report_dir = create_report_dir(attempt)
    except VerificationError as exc:
        print(canonical_json({"status": "failed", "check": exc.check}), file=sys.stderr)
        return 2
    verifier = Verifier(attempt, report_dir, resource_info)
    status = "verified"
    error_check = None
    try:
        verifier.run()
    except VerificationError as exc:
        status = "failed"
        error_check = exc.check
        if verifier.checks.get(exc.check, {}).get("status") == "not_run":
            verifier.checks[exc.check] = {"status": "fail"}
    except Exception:
        status = "failed"
        error_check = "unhandled_integrity_or_io_error"
    try:
        report_path = write_report(verifier, report_dir, status, error_check)
    except OSError:
        print(canonical_json({"status": "failed", "check": "evidence_write_failed"}), file=sys.stderr)
        return 3
    output = {
        "status": status,
        "report_path": str(report_path),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "checks_passed": sum(1 for check in verifier.checks.values() if check["status"] == "pass"),
        "checks_total": len(verifier.checks),
    }
    print(canonical_json(output))
    return 0 if status == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
