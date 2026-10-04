#!/usr/bin/env python3
"""Independently verify the retained neutral-source acquisition custody tree.

This program uses only Python's standard library. It does not import or execute
neutral_acquire.py, model loaders, or data loaders. Source payloads and native
annotations remain in the private custody tree and are never printed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import gzip
import hashlib
import json
import os
import resource
import sqlite3
import stat
import sys
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

SOURCE_BANK = "PolyAI/banking77"
SOURCE_MASSIVE = "AmazonScience/massive"
SOURCE_IDS = (SOURCE_BANK, SOURCE_MASSIVE)
PARTITIONS = ("train", "validation", "test")
SPLITS = ("train", "dev", "calibration", "confirmation", "unused")
SEED = "vey-neutral-v2:2026-10-04:1729"
PURPOSE = "neutral-acquisition"
CAPS = {"train": 4096, "dev": 1024, "calibration": 1024, "confirmation": 8192}
ACQUISITION_MANIFEST_SHA256 = "05c7365315ea62b2d84a0b5793ab4b55cc9be419827cfbe3d95d57242bf48c17"
JACCARD_NUMERATOR = 9
JACCARD_DENOMINATOR = 10
MAX_INPUT_CHARS = 16384
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
MAX_TAR_MEMBERS = 10000
MAX_TAR_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TAR_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
MAX_NGRAM_POSTINGS = 50_000_000
MAX_PREFIX_PAIR_WORK = 100_000_000
MAX_SQLITE_BYTES = 8 * 1024 * 1024 * 1024
MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

class VerificationError(RuntimeError):
    """A fail-closed custody inconsistency."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_input(text: str) -> str:
    require(len(text) <= MAX_INPUT_CHARS, "source input exceeds the fixed normalization bound")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key")
        result[key] = value
    return result


def _no_json_constant(value: str) -> None:
    raise VerificationError("non-finite JSON value is not permitted")


def strict_json_bytes(data: bytes) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError("invalid UTF-8 JSON in retained custody metadata") from exc


def jsonl_records(stream: BinaryIO, *, max_line: int = MAX_JSONL_LINE_BYTES) -> Iterator[dict[str, Any]]:
    while True:
        line = stream.readline(max_line + 1)
        if not line:
            return
        require(len(line) <= max_line, "JSONL line exceeds the fixed parser bound")
        require(bool(line.strip()), "empty JSONL record")
        value = strict_json_bytes(line)
        require(isinstance(value, dict), "JSONL record is not an object")
        yield value


def safe_relative(value: str) -> tuple[str, ...]:
    require(isinstance(value, str) and value and "\\" not in value and "\x00" not in value,
            "unsafe relative custody path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts),
            "unsafe relative custody path")
    return path.parts


def secure_file(root: Path, relative: str) -> Path:
    parts = safe_relative(relative)
    cursor = root
    for part in parts[:-1]:
        cursor = cursor / part
        try:
            st = cursor.lstat()
        except OSError as exc:
            raise VerificationError("required custody directory is unavailable") from exc
        require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode),
                "custody path contains a non-directory or symlink")
        require(stat.S_IMODE(st.st_mode) & 0o077 == 0,
                "custody directory permissions expose private material")
    target = cursor / parts[-1]
    try:
        st = target.lstat()
    except OSError as exc:
        raise VerificationError("required custody file is unavailable") from exc
    require(stat.S_ISREG(st.st_mode) and not stat.S_ISLNK(st.st_mode),
            "custody file is not a regular non-symlink file")
    require(stat.S_IMODE(st.st_mode) & 0o077 == 0,
            "custody file permissions expose private material")
    return target


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK_BYTES)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def check_hash(path: Path, expected_hash: str, expected_size: int | None = None) -> dict[str, Any]:
    actual_hash, actual_size = hash_file(path)
    require(actual_hash == expected_hash, "custody SHA-256 mismatch")
    if expected_size is not None:
        require(actual_size == expected_size, "custody byte-size mismatch")
    return {"sha256": actual_hash, "size_bytes": actual_size}


def secure_directory(path: Path, *, create: bool = False) -> None:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        st = path.lstat()
    except OSError as exc:
        raise VerificationError("required private output directory is unavailable") from exc
    require(stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode),
            "private output path is not a plain directory")
    require(stat.S_IMODE(st.st_mode) & 0o077 == 0,
            "private output directory permissions are not restrictive")


def set_resource_guards() -> dict[str, int | str]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    require(hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"),
            "cannot enforce the four-CPU affinity limit")
    allowed = sorted(os.sched_getaffinity(0))
    require(bool(allowed), "no CPU affinity is available")
    os.sched_setaffinity(0, set(allowed[:4]))
    affinity = len(os.sched_getaffinity(0))
    require(0 < affinity <= 4, "effective CPU affinity exceeds the fixed four-CPU limit")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = MEMORY_LIMIT_BYTES if soft == resource.RLIM_INFINITY else min(soft, MEMORY_LIMIT_BYTES)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    require(limit >= 512 * 1024 * 1024, "available address-space limit is too small")
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return {"cpu_affinity_count": affinity, "address_space_limit_bytes": limit, "blas_threads": 1}


def db_guard(db: sqlite3.Connection) -> None:
    page_count, page_size = db.execute("PRAGMA page_count").fetchone()[0], db.execute("PRAGMA page_size").fetchone()[0]
    require(page_count * page_size <= MAX_SQLITE_BYTES, "private scratch SQLite database exceeded its fixed size limit")


def init_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE raw_rows(
          row_key TEXT PRIMARY KEY, source_id TEXT NOT NULL, official_partition TEXT NOT NULL,
          locale TEXT NOT NULL, original_id TEXT NOT NULL, raw_input_sha256 TEXT NOT NULL,
          normalized_sha256 TEXT NOT NULL, source_file TEXT NOT NULL, line_number INTEGER NOT NULL,
          payload_sha256 TEXT NOT NULL, group_id TEXT, component_id TEXT, final_split TEXT
        );
        CREATE INDEX raw_source_original ON raw_rows(source_id,original_id);
        CREATE INDEX raw_group ON raw_rows(group_id);
        CREATE INDEX raw_norm ON raw_rows(normalized_sha256);
        CREATE TABLE massive_base(
          original_id TEXT PRIMARY KEY, official_partition TEXT NOT NULL,
          intent_sha256 TEXT NOT NULL, scenario_sha256 TEXT NOT NULL
        );
        CREATE TABLE massive_locale_ids(original_id TEXT NOT NULL,locale TEXT NOT NULL,
          PRIMARY KEY(original_id,locale)) WITHOUT ROWID;
        CREATE TABLE source_groups(
          group_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, pinned_revision TEXT NOT NULL,
          official_partition TEXT NOT NULL, lineage_json TEXT NOT NULL, input_hashes_json TEXT NOT NULL,
          rank_sha256 TEXT NOT NULL, component_id TEXT, final_split TEXT,
          allocation_reason TEXT, exclusion_reason TEXT, component_member_count INTEGER
        );
        CREATE INDEX groups_source_split ON source_groups(source_id,final_split,rank_sha256,group_id);
        CREATE TABLE group_docs(doc_hash TEXT NOT NULL,group_id TEXT NOT NULL,
          PRIMARY KEY(doc_hash,group_id)) WITHOUT ROWID;
        CREATE INDEX group_docs_by_group ON group_docs(group_id,doc_hash);
        CREATE TABLE docs(doc_hash TEXT PRIMARY KEY,normalized_text TEXT NOT NULL,
          representative_group TEXT,gram_count INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID;
        CREATE TABLE doc_grams(doc_hash TEXT NOT NULL,gram TEXT NOT NULL,
          PRIMARY KEY(doc_hash,gram)) WITHOUT ROWID;
        CREATE INDEX grams_by_token ON doc_grams(gram,doc_hash);
        CREATE TABLE gram_frequency(gram TEXT PRIMARY KEY,document_frequency INTEGER NOT NULL) WITHOUT ROWID;
        CREATE INDEX gram_frequency_order ON gram_frequency(document_frequency,gram);
        CREATE TABLE prefixes(doc_hash TEXT NOT NULL,gram TEXT NOT NULL,
          PRIMARY KEY(doc_hash,gram)) WITHOUT ROWID;
        CREATE INDEX prefixes_by_token ON prefixes(gram,doc_hash);
        CREATE TABLE expected_edges(
          edge_key TEXT PRIMARY KEY,group_a TEXT NOT NULL,group_b TEXT NOT NULL,reason TEXT NOT NULL,
          hash_a TEXT NOT NULL,hash_b TEXT,intersection_size INTEGER,union_size INTEGER
        );
        CREATE TABLE exclusions(doc_hash TEXT PRIMARY KEY,group_id TEXT NOT NULL,
          codepoint_count INTEGER NOT NULL,reason TEXT NOT NULL) WITHOUT ROWID;
        CREATE TABLE components(component_id TEXT PRIMARY KEY,final_split TEXT,
          allocation_reason TEXT,exclusion_reason TEXT);
        CREATE TABLE component_groups(component_id TEXT NOT NULL,group_id TEXT NOT NULL,
          PRIMARY KEY(component_id,group_id)) WITHOUT ROWID;
    """)
    return db


def insert_raw_row(db: sqlite3.Connection, row: Sequence[Any], normalized_text: str) -> None:
    db.execute("""INSERT INTO raw_rows(row_key,source_id,official_partition,locale,original_id,
        raw_input_sha256,normalized_sha256,source_file,line_number,payload_sha256)
        VALUES(?,?,?,?,?,?,?,?,?,?)""", row)
    doc_hash = row[6]
    db.execute("INSERT OR IGNORE INTO docs(doc_hash,normalized_text) VALUES(?,?)", (doc_hash, normalized_text))
    stored = db.execute("SELECT normalized_text FROM docs WHERE doc_hash=?", (doc_hash,)).fetchone()[0]
    require(stored == normalized_text, "normalized-input SHA-256 collision")


def csv_rows(path: Path, filename: str, source_id: str, partition: str,
             db: sqlite3.Connection, expected_count: int) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            require(reader.fieldnames is not None and len(reader.fieldnames) == 2
                    and set(reader.fieldnames) == {"text", "category"},
                    "Banking77 CSV schema differs from the frozen source")
            for item in reader:
                count += 1
                require(None not in item and isinstance(item.get("text"), str)
                        and isinstance(item.get("category"), str) and bool(item["category"]),
                        "Banking77 CSV row has an invalid shape")
                text = item["text"]
                normalized = normalize_input(text)
                row_key = f"banking77:{partition}:{count:08d}"
                payload = {"text": text, "category": item["category"]}
                raw_row = (
                    row_key, source_id, partition, "en", row_key,
                    sha256_bytes(text.encode("utf-8")), sha256_bytes(normalized.encode("utf-8")),
                    filename, count + 1, sha256_bytes(canonical_json(payload).encode("utf-8")),
                )
                insert_raw_row(db, raw_row, normalized)
                if count % 10000 == 0:
                    db.commit()
                    db_guard(db)
    except (UnicodeError, csv.Error) as exc:
        raise VerificationError("Banking77 CSV could not be parsed as strict UTF-8 CSV") from exc
    require(count == expected_count, "Banking77 partition row count differs from frozen inventory")
    db.commit()
    return count


def canonical_partition(value: Any) -> str:
    require(isinstance(value, str), "MASSIVE partition field is not text")
    value = value.casefold()
    if value in ("dev", "validation"):
        return "validation"
    require(value in ("train", "test"), "MASSIVE partition field is outside the frozen native partitions")
    return value


def add_massive_record(db: sqlite3.Connection, record: Mapping[str, Any], locale: str,
                       member_name: str, line_number: int) -> tuple[str, bool, bool, int]:
    required = ("id", "locale", "partition", "intent", "scenario", "utt")
    require(all(key in record for key in required), "MASSIVE record is missing a required native field")
    original_id = record["id"]
    require(isinstance(original_id, str) and bool(original_id), "MASSIVE ID is invalid")
    require(record["locale"] == locale, "MASSIVE record locale differs from its archive member")
    partition = canonical_partition(record["partition"])
    text = record["utt"]
    require(isinstance(text, str), "MASSIVE utterance is not text")
    normalized = normalize_input(text)
    require(record["intent"] is not None and record["scenario"] is not None,
            "MASSIVE native intent/scenario field is absent")
    judgments_present = "judgments" in record
    judgments = record["judgments"] if judgments_present else None
    judgments_null = judgments_present and judgments is None
    if judgments_present and not judgments_null:
        require(isinstance(judgments, list), "MASSIVE judgments field is neither an array nor null")
        require(all(isinstance(item, dict) for item in judgments),
                "MASSIVE judgment entries are not native objects")
        judgment_record_count = len(judgments)
    else:
        judgment_record_count = 0
    intent_hash = sha256_bytes(canonical_json(record["intent"]).encode("utf-8"))
    scenario_hash = sha256_bytes(canonical_json(record["scenario"]).encode("utf-8"))
    base = db.execute("SELECT official_partition,intent_sha256,scenario_sha256 FROM massive_base WHERE original_id=?",
                      (original_id,)).fetchone()
    expected = (partition, intent_hash, scenario_hash)
    if base is None:
        db.execute("INSERT INTO massive_base VALUES(?,?,?,?)", (original_id, *expected))
    else:
        require(base == expected, "MASSIVE locale descendants disagree on native partition or source identity fields")
    db.execute("INSERT INTO massive_locale_ids VALUES(?,?)", (original_id, locale))
    row_key = f"massive:{locale}:{original_id}"
    raw_row = (
        row_key, SOURCE_MASSIVE, partition, locale, original_id,
        sha256_bytes(text.encode("utf-8")), sha256_bytes(normalized.encode("utf-8")),
        member_name, line_number,
        sha256_bytes(canonical_json(dict(record)).encode("utf-8")),
    )
    insert_raw_row(db, raw_row, normalized)
    return partition, judgments_present, judgments_null, judgment_record_count


def strict_tar_member_name(name: str) -> str:
    require(bool(name) and "\x00" not in name and "\\" not in name and not name.startswith("/"),
            "MASSIVE tar contains an unsafe path")
    parts = name.split("/")
    require(all(part != ".." for part in parts), "MASSIVE tar contains a traversal path")
    require(not (parts and len(parts[0]) >= 2 and parts[0][1] == ":"),
            "MASSIVE tar contains a drive-qualified path")
    normalized = "/".join(part for part in parts if part not in ("", "."))
    require(bool(normalized), "MASSIVE tar contains an empty canonical path")
    return normalized


def selected_locale(path: str, locales: set[str]) -> str | None:
    pure = PurePosixPath(path)
    return pure.stem if pure.suffix == ".jsonl" and pure.stem in locales else None


def verify_tar_and_parse(root: Path, archive_path: Path, locale_order: Sequence[str],
                         inventory: Mapping[str, Any], member_index_path: Path,
                         db: sqlite3.Connection) -> dict[str, Any]:
    locale_set = set(locale_order)
    seen_paths: set[str] = set()
    seen_locales: set[str] = set()
    member_count = regular_count = directory_count = total_unpacked = selected_count = 0
    row_counts: dict[str, Counter[str]] = {locale: Counter() for locale in locale_order}
    judgment_metadata_counts: dict[str, dict[str, dict[str, int]]] = {
        locale: {
            partition: {
                "row_count": 0,
                "judgments_field_present_rows": 0,
                "judgments_field_missing_rows": 0,
                "judgments_field_null_rows": 0,
                "rows_with_judgments": 0,
                "judgment_record_count": 0,
            }
            for partition in PARTITIONS
        }
        for locale in locale_order
    }
    expected_members = iter(jsonl_records(gzip.open(member_index_path, "rb")))
    try:
        with tarfile.open(archive_path, mode="r|gz") as archive:
            for member in archive:
                member_count += 1
                require(member_count <= MAX_TAR_MEMBERS, "MASSIVE tar exceeds the fixed member count")
                name = strict_tar_member_name(member.name)
                require(name not in seen_paths, "MASSIVE tar has duplicate canonical member paths")
                seen_paths.add(name)
                locale = selected_locale(name, locale_set)
                executable = bool(member.mode & 0o111)
                if member.isdir():
                    directory_count += 1
                    observed = {"path": name, "size_bytes": 0, "sha256": None,
                                "member_type": "directory", "selected_locale": None, "executable": executable}
                elif member.isfile():
                    regular_count += 1
                    require(0 <= member.size <= MAX_TAR_MEMBER_BYTES, "MASSIVE tar member exceeds its byte bound")
                    total_unpacked += member.size
                    require(total_unpacked <= MAX_TAR_UNPACKED_BYTES, "MASSIVE tar exceeds its total uncompressed byte bound")
                    require(not executable, "MASSIVE tar contains an executable regular file")
                    source = archive.extractfile(member)
                    require(source is not None, "MASSIVE regular tar member could not be read")
                    digest = hashlib.sha256()
                    actual_size = 0
                    try:
                        if locale is None:
                            while True:
                                block = source.read(CHUNK_BYTES)
                                if not block:
                                    break
                                actual_size += len(block)
                                require(actual_size <= member.size, "MASSIVE member expanded past its declared size")
                                digest.update(block)
                        else:
                            require(locale not in seen_locales, "MASSIVE contains duplicate selected locale members")
                            seen_locales.add(locale)
                            selected_count += 1
                            line_number = 0
                            while True:
                                line = source.readline(MAX_JSONL_LINE_BYTES + 1)
                                if not line:
                                    break
                                require(len(line) <= MAX_JSONL_LINE_BYTES, "MASSIVE JSONL row exceeds its fixed byte bound")
                                actual_size += len(line)
                                require(actual_size <= member.size, "MASSIVE selected member expanded past its declared size")
                                digest.update(line)
                                require(bool(line.strip()), "MASSIVE JSONL contains an empty source row")
                                line_number += 1
                                record = strict_json_bytes(line)
                                require(isinstance(record, dict), "MASSIVE JSONL row is not an object")
                                part, present, null, judgment_records = add_massive_record(
                                    db, record, locale, name, line_number)
                                row_counts[locale][part] += 1
                                stats = judgment_metadata_counts[locale][part]
                                stats["row_count"] += 1
                                stats["judgments_field_present_rows"] += int(present)
                                stats["judgments_field_missing_rows"] += int(not present)
                                stats["judgments_field_null_rows"] += int(null)
                                stats["rows_with_judgments"] += int(judgment_records > 0)
                                stats["judgment_record_count"] += judgment_records
                                if line_number % 10000 == 0:
                                    db.commit()
                                    db_guard(db)
                        require(actual_size == member.size, "MASSIVE tar member byte count differs from its header")
                    finally:
                        source.close()
                    observed = {"path": name, "size_bytes": actual_size, "sha256": digest.hexdigest(),
                                "member_type": "regular_file", "selected_locale": locale, "executable": executable}
                else:
                    raise VerificationError("MASSIVE tar contains a nonregular, non-directory member")
                try:
                    pinned = next(expected_members)
                except StopIteration as exc:
                    raise VerificationError("MASSIVE member index ended before the archive") from exc
                require(pinned == observed, "retained MASSIVE tar-member index differs from independent archive scan")
                require(set(pinned) == {"path", "size_bytes", "sha256", "member_type", "selected_locale", "executable"},
                        "retained MASSIVE tar-member row has an unexpected schema")
        try:
            next(expected_members)
            raise VerificationError("retained MASSIVE member index contains extra entries")
        except StopIteration:
            pass
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise VerificationError("MASSIVE archive stream could not be verified") from exc
    require(seen_locales == locale_set and selected_count == len(locale_order),
            "MASSIVE archive does not contain exactly one selected member per frozen locale")
    split_sizes = inventory["sources"]
    massive_inventory = next(item for item in split_sizes if item.get("id") == SOURCE_MASSIVE)
    expected_sizes = massive_inventory["split_sizes"]["per_locale"]
    for locale in locale_order:
        for partition in PARTITIONS:
            require(row_counts[locale][partition] == int(expected_sizes[partition]),
                    "MASSIVE locale/partition count differs from frozen inventory")
    db.commit()
    db_guard(db)
    return {
        "member_count": member_count,
        "regular_file_count": regular_count,
        "directory_count": directory_count,
        "unpacked_regular_bytes": total_unpacked,
        "selected_jsonl_member_count": selected_count,
        "selected_locale_count": len(seen_locales),
        "locale_row_counts": {locale: dict(sorted(row_counts[locale].items())) for locale in locale_order},
        "native_judgment_metadata_counts": judgment_metadata_counts,
    }


def check_frozen_plan(root: Path, manifest: Mapping[str, Any], source_files: Mapping[str, Any]) -> tuple[
        dict[str, Any], dict[str, Any], list[str], dict[str, Any], dict[str, dict[str, Any]]]:
    pins = manifest.get("source_pins")
    require(isinstance(pins, dict) and pins, "acquisition manifest lacks pinned snapshot records")
    reported_pins = source_files.get("protocol_and_source_input_hashes")
    require(reported_pins == pins, "source-file manifest and acquisition manifest disagree on frozen snapshots")
    parsed: dict[str, Any] = {}
    snapshot_hashes: dict[str, dict[str, Any]] = {}
    producer_bytes: bytes | None = None
    for relative, pin in sorted(pins.items()):
        require(isinstance(pin, dict) and isinstance(pin.get("path"), str), "invalid frozen snapshot record")
        path = secure_file(root, pin["path"])
        snapshot_hashes[pin["path"]] = check_hash(path, pin["sha256"], int(pin["size_bytes"]))
        if relative.endswith(".json"):
            parsed[relative] = strict_json_bytes(path.read_bytes())
        elif relative == "research/endgame/neutral_acquire.py":
            producer_bytes = path.read_bytes()
    required = {
        "research/endgame/protocol.json", "research/endgame/neutral_benchmark_protocol.json",
        "research/endgame/neutral_acquisition_sources.json", "research/endgame/neutral_source_inventory.json",
        "research/competitors/COMPETITOR_MATRIX.json", "research/competitors/LAYA.json",
        "research/competitors/JEV.json", "research/endgame/neutral_acquire.py",
    }
    require(set(parsed) | {"research/endgame/neutral_acquire.py"} == required and producer_bytes is not None,
            "frozen snapshot set is incomplete")
    plan = parsed["research/endgame/neutral_acquisition_sources.json"]
    inventory = parsed["research/endgame/neutral_source_inventory.json"]
    protocol = parsed["research/endgame/neutral_benchmark_protocol.json"]
    require(plan.get("schema") == "vey.neutral.source-acquisition.v1"
            and protocol.get("schema") == "vey.endgame.neutral-benchmark-protocol.v2",
            "frozen acquisition plan or protocol schema differs")
    require(plan.get("parent_protocol_git") == "5b3fe4c", "frozen acquisition parent-protocol pin differs")
    require(Path(plan["output_root"]).resolve() == root, "CLI root differs from frozen output root")
    plan_sources = {item.get("id"): item for item in plan.get("sources", [])}
    inventory_sources = {item.get("id"): item for item in inventory.get("sources", [])}
    require(set(plan_sources) == set(SOURCE_IDS) and set(inventory_sources) >= set(SOURCE_IDS),
            "frozen plan does not contain exactly the approved sources")
    locales = plan_sources[SOURCE_MASSIVE].get("fixed_locales")
    inventory_locales = inventory_sources[SOURCE_MASSIVE].get("languages", [])
    require(isinstance(locales, list) and len(locales) == 51 and len(set(locales)) == 51
            and set(locales) == set(inventory_locales) - {"zh-TW"} and "ca-ES" in locales and "zh-TW" not in locales
            and plan_sources[SOURCE_MASSIVE].get("raw_release") == "1.1",
            "frozen MASSIVE locale scope differs")
    require(plan_sources[SOURCE_BANK].get("raw_revision") == "57ec275d8078af65b7731c2a98be812d844a6d6b"
            and plan_sources[SOURCE_MASSIVE].get("hf_loader_revision") == "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
            "frozen source revision differs")
    for source_id in SOURCE_IDS:
        source_record = inventory_sources[source_id]
        require(source_record.get("license_class") == "shipping-train"
                and source_record.get("shipping_training_allowed_now") is True
                and plan_sources[source_id].get("hf_loader_revision") == source_record.get("revision"),
                "frozen inventory grant or source revision differs")
    allocation = protocol.get("allocation_parameters")
    require(isinstance(allocation, dict) and allocation.get("seed") == SEED
            and allocation.get("caps_independent_groups_per_source") == CAPS,
            "frozen allocation seed or component caps differ")
    require(source_files.get("parent_protocol_git") == plan["parent_protocol_git"],
            "source-file manifest parent-protocol pin differs")
    # Parse, never execute, the historical producer snapshot to confirm only the
    # fixed parser/index work bounds used by this independent verifier.
    try:
        tree = ast.parse(producer_bytes.decode("utf-8"))
    except (UnicodeError, SyntaxError) as exc:
        raise VerificationError("pinned producer snapshot is not parseable Python") from exc
    constant_names = {"JACCARD_NUMERATOR", "JACCARD_DENOMINATOR", "MAX_INPUT_CHARS",
                      "MAX_JSONL_LINE_BYTES", "MAX_TAR_MEMBERS", "MAX_TAR_MEMBER_BYTES",
                      "MAX_TAR_UNPACKED_BYTES", "MAX_NGRAM_POSTINGS", "MAX_PREFIX_PAIR_WORK"}
    constants: dict[str, Any] = {}
    def integer_bound(node: ast.AST, depth: int = 0) -> int:
        require(depth <= 16, "producer bound expression exceeds depth cap")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = node.value
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            value = integer_bound(node.left, depth + 1) * integer_bound(node.right, depth + 1)
        else:
            raise VerificationError("producer bound is not an integer literal/product")
        require(0 <= value <= 2**40, "producer bound outside bounded integer range")
        return value
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in constant_names:
                constants[name] = integer_bound(node.value)
    expected_constants = {
        "JACCARD_NUMERATOR": JACCARD_NUMERATOR, "JACCARD_DENOMINATOR": JACCARD_DENOMINATOR,
        "MAX_INPUT_CHARS": MAX_INPUT_CHARS, "MAX_JSONL_LINE_BYTES": MAX_JSONL_LINE_BYTES,
        "MAX_TAR_MEMBERS": MAX_TAR_MEMBERS, "MAX_TAR_MEMBER_BYTES": MAX_TAR_MEMBER_BYTES,
        "MAX_TAR_UNPACKED_BYTES": MAX_TAR_UNPACKED_BYTES,
        "MAX_NGRAM_POSTINGS": MAX_NGRAM_POSTINGS, "MAX_PREFIX_PAIR_WORK": MAX_PREFIX_PAIR_WORK,
    }
    require(constants == expected_constants, "pinned producer bounds differ from independent verifier bounds")
    return plan, inventory, list(locales), allocation, snapshot_hashes


def verify_input_hashes(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = secure_file(root, "metadata/acquisition_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = sha256_bytes(manifest_bytes)
    require(manifest_hash == ACQUISITION_MANIFEST_SHA256,
            "acquisition manifest differs from the sealed-root digest pin")
    manifest_size = len(manifest_bytes)
    manifest = strict_json_bytes(manifest_bytes)
    require(manifest.get("schema") == "vey.neutral.acquisition-manifest.v1", "unexpected acquisition manifest schema")
    source_meta_path = secure_file(root, "metadata/source_files.json")
    source_files = strict_json_bytes(source_meta_path.read_bytes())
    require(source_files.get("schema") == "vey.neutral.acquired-source-files.v1", "unexpected source-file manifest schema")
    expected_meta_hash = manifest.get("outputs", {}).get("metadata/source_files.json")
    require(check_hash(source_meta_path, expected_meta_hash)["sha256"] == expected_meta_hash,
            "source-file manifest hash differs from acquisition manifest")
    require(source_files.get("raw_source_manifest_sha256") == expected_meta_hash
            or manifest.get("raw_source_manifest_sha256") == expected_meta_hash,
            "raw source-manifest pin differs")
    input_hashes: dict[str, Any] = {"metadata/acquisition_manifest.json": {"sha256": manifest_hash,
                                                                                "size_bytes": manifest_size}}
    output_hashes: dict[str, Any] = {}
    output_map = manifest.get("outputs")
    require(isinstance(output_map, dict) and output_map, "acquisition manifest has no output hash map")
    for relative, expected in sorted(output_map.items()):
        path = secure_file(root, relative)
        output_hashes[relative] = check_hash(path, expected)
        input_hashes[relative] = output_hashes[relative]
    require(output_hashes.get("metadata/source_files.json", {}).get("sha256") == expected_meta_hash,
            "source-file manifest hash is inconsistent")
    source_records = source_files.get("source_artifacts")
    manifest_records = manifest.get("source_artifact_hashes")
    require(isinstance(source_records, list) and isinstance(manifest_records, dict)
            and all(isinstance(item, dict) for item in source_records),
            "raw source artifact records are incomplete")
    by_id = {item.get("artifact_id"): item for item in source_records}
    require(len(by_id) == len(source_records), "raw source manifest contains duplicate artifact IDs")
    require(set(by_id) == set(manifest_records), "raw source artifact indexes differ")
    expected_paths = {
        "banking77_train_csv": "raw/banking77/train.csv", "banking77_test_csv": "raw/banking77/test.csv",
        "banking77_license": "raw/banking77/LICENSE", "banking77_readme": "raw/banking77/README.md",
        "massive_archive_1_1": "raw/massive/amazon-massive-dataset-1.1.tar.gz",
        "massive_license": "raw/massive/LICENSE", "massive_notice": "raw/massive/NOTICE.md",
        "massive_readme": "raw/massive/README.md",
    }
    require(set(by_id) == set(expected_paths), "raw source scope differs from the frozen acquisition set")
    raw_hashes: dict[str, Any] = {}
    pin_fields = {"path", "actual_sha256", "size_bytes", "expected_sha256", "source_id"}
    receipt_fields = {"artifact_id", "actual_sha256", "expected_sha256", "expected_sha256_source",
                      "kind", "size_bytes", "source_checksum_was_pinned_before_acquisition",
                      "source_id", "source_url"}
    for artifact_id, record in by_id.items():
        pin = manifest_records[artifact_id]
        require(isinstance(pin, dict) and pin_fields <= set(pin)
                and receipt_fields <= set(record),
                "raw source artifact metadata omits required or explicitly nullable fields")
        relative = pin["path"]
        require(relative == expected_paths[artifact_id]
                and record["actual_sha256"] == pin["actual_sha256"]
                and record["size_bytes"] == pin["size_bytes"]
                and record["expected_sha256"] == pin["expected_sha256"]
                and record["source_id"] == pin["source_id"],
                "raw source manifest differs from acquisition manifest")
        path = secure_file(root, relative)
        actual = check_hash(path, pin["actual_sha256"], int(pin["size_bytes"]))
        if pin["expected_sha256"] is not None:
            require(pin["expected_sha256"] == actual["sha256"], "pinned upstream source checksum differs")
        raw_hashes[relative] = actual
        input_hashes[relative] = actual
    require(source_files.get("massive_archive", {}).get("actual_archive_sha256_frozen_before_parsing")
            == manifest_records["massive_archive_1_1"]["actual_sha256"],
            "MASSIVE archive pre-parse digest differs")
    member_record = source_files.get("massive_archive", {}).get("member_hash_manifest", {})
    member_output = output_hashes.get("private/massive_tar_members.jsonl.gz", {})
    require(member_record.get("path") == "private/massive_tar_members.jsonl.gz"
            and member_record.get("sha256") == member_output.get("sha256")
            and member_record.get("size_bytes") == member_output.get("size_bytes"),
            "MASSIVE member-index digest differs between manifests")
    return manifest, source_files, input_hashes, {"outputs": output_hashes, "raw": raw_hashes}


def verify_raw_artifact_plan(source_files: Mapping[str, Any], plan: Mapping[str, Any],
                             inventory: Mapping[str, Any]) -> None:
    planned = {item["id"]: item for item in plan["sources"]}
    inventoried = {item["id"]: item for item in inventory["sources"]}
    bank_primary = {item["url"]: item["sha256"]
                    for item in inventoried[SOURCE_BANK]["primary_sources"]}
    massive_primary = {item["url"]: item["sha256"]
                       for item in inventoried[SOURCE_MASSIVE]["primary_sources"]}
    bank = planned[SOURCE_BANK]
    massive = planned[SOURCE_MASSIVE]
    readme_url = (f"https://huggingface.co/datasets/AmazonScience/massive/resolve/"
                  f"{massive['hf_loader_revision']}/README.md")
    expected = {
        "banking77_train_csv": (SOURCE_BANK, bank["train_url"], None, "dataset_csv"),
        "banking77_test_csv": (SOURCE_BANK, bank["test_url"], None, "dataset_csv"),
        "banking77_license": (SOURCE_BANK, bank["license_url"], bank["license_sha256_inventory"], "metadata"),
        "banking77_readme": (SOURCE_BANK, bank["attribution_url"],
                             bank_primary.get(bank["attribution_url"]), "metadata"),
        "massive_archive_1_1": (SOURCE_MASSIVE, massive["archive_url"], None, "dataset_archive"),
        "massive_license": (SOURCE_MASSIVE, massive["license_url"],
                            massive_primary.get(massive["license_url"]), "metadata"),
        "massive_notice": (SOURCE_MASSIVE, massive["notice_url"],
                           massive_primary.get(massive["notice_url"]), "metadata"),
        "massive_readme": (SOURCE_MASSIVE, readme_url, massive_primary.get(readme_url), "metadata"),
    }
    artifacts = {item.get("artifact_id"): item for item in source_files.get("source_artifacts", [])}
    require(set(artifacts) == set(expected), "source-file artifact set differs from frozen plan")
    for artifact_id, (source_id, url, checksum, kind) in expected.items():
        record = artifacts[artifact_id]
        require(isinstance(record, dict) and
                {"artifact_id", "source_id", "source_url", "expected_sha256", "kind",
                 "source_checksum_was_pinned_before_acquisition", "expected_sha256_source"} <= set(record),
                "source-file artifact record omits required or explicitly nullable provenance fields")
        require(record["artifact_id"] == artifact_id and record["source_id"] == source_id
                and record["source_url"] == url and record["expected_sha256"] == checksum
                and record["kind"] == kind
                and record["source_checksum_was_pinned_before_acquisition"] is (checksum is not None)
                and record["expected_sha256_source"] ==
                ("committed metadata pin" if checksum is not None else None),
                "raw source provenance differs from frozen source plan or inventory")


def source_revisions(plan: Mapping[str, Any]) -> dict[str, str]:
    rows = {item["id"]: item for item in plan["sources"]}
    return {SOURCE_BANK: str(rows[SOURCE_BANK]["raw_revision"]),
            SOURCE_MASSIVE: str(rows[SOURCE_MASSIVE]["hf_loader_revision"])}


def make_group(db: sqlite3.Connection, source_id: str, revision: str, partition: str,
               lineage: list[str], input_hashes: list[str]) -> str:
    gid = canonical_sha256([source_id, revision, sorted(lineage), sorted(set(input_hashes))])
    rank = hashlib.sha256(f"{SEED}\n{PURPOSE}\n{source_id}\n{gid}".encode("utf-8")).hexdigest()
    db.execute("INSERT INTO source_groups(group_id,source_id,pinned_revision,official_partition,lineage_json,input_hashes_json,rank_sha256) VALUES(?,?,?,?,?,?,?)",
               (gid, source_id, revision, partition, canonical_json(sorted(lineage)),
                canonical_json(sorted(set(input_hashes))), rank))
    return gid


def build_source_groups(db: sqlite3.Connection, plan: Mapping[str, Any]) -> None:
    revisions = source_revisions(plan)
    for row_key, partition, norm_hash in db.execute(
            "SELECT row_key,official_partition,normalized_sha256 FROM raw_rows WHERE source_id=? ORDER BY row_key",
            (SOURCE_BANK,)):
        gid = make_group(db, SOURCE_BANK, revisions[SOURCE_BANK], partition, [row_key], [norm_hash])
        db.execute("UPDATE raw_rows SET group_id=? WHERE row_key=?", (gid, row_key))
    db.commit()
    cursor = db.execute("SELECT original_id,official_partition,normalized_sha256 FROM raw_rows "
                        "WHERE source_id=? ORDER BY original_id,normalized_sha256", (SOURCE_MASSIVE,))
    current_id: str | None = None
    current_partition: str | None = None
    hashes: list[str] = []
    def finish() -> None:
        if current_id is None:
            return
        gid = make_group(db, SOURCE_MASSIVE, revisions[SOURCE_MASSIVE], current_partition or "", [current_id], hashes)
        db.execute("UPDATE raw_rows SET group_id=? WHERE source_id=? AND original_id=?",
                   (gid, SOURCE_MASSIVE, current_id))
    for original_id, partition, norm_hash in cursor:
        if original_id != current_id:
            finish()
            current_id, current_partition, hashes = original_id, partition, []
        require(partition == current_partition, "MASSIVE lineage crosses native partitions")
        if not hashes or hashes[-1] != norm_hash:
            hashes.append(norm_hash)
    finish()
    db.execute("INSERT INTO group_docs SELECT DISTINCT normalized_sha256,group_id FROM raw_rows")
    db.execute("UPDATE docs SET representative_group=(SELECT MIN(group_id) FROM group_docs WHERE group_docs.doc_hash=docs.doc_hash)")
    db.commit()
    db_guard(db)


def add_expected_edge(db: sqlite3.Connection, group_a: str, group_b: str, reason: str,
                      hash_for_a: str, hash_for_b: str | None = None,
                      intersection: int | None = None, union: int | None = None) -> None:
    require(group_a != group_b, "internal edge construction received a self-edge")
    if group_a > group_b:
        group_a, group_b = group_b, group_a
        hash_for_a, hash_for_b = hash_for_b, hash_for_a
    key = canonical_sha256([group_a, group_b, reason, hash_for_a, hash_for_b])
    prior = db.execute("SELECT group_a,group_b,reason,hash_a,hash_b,intersection_size,union_size "
                       "FROM expected_edges WHERE edge_key=?", (key,)).fetchone()
    value = (group_a, group_b, reason, hash_for_a, hash_for_b, intersection, union)
    if prior is None:
        db.execute("INSERT INTO expected_edges VALUES(?,?,?,?,?,?,?,?)", (key, *value))
    else:
        require(prior == value, "duplicate edge identity has conflicting evidence")


def exact_duplicate_edges(db: sqlite3.Connection) -> int:
    current_hash: str | None = None
    representative: str | None = None
    count = 0
    for doc_hash, group_id in db.execute("SELECT doc_hash,group_id FROM group_docs ORDER BY doc_hash,group_id"):
        if doc_hash != current_hash:
            current_hash, representative = doc_hash, group_id
            continue
        add_expected_edge(db, representative or group_id, group_id, "exact_normalized_input", doc_hash)
        count += 1
    db.commit()
    return count


def build_gram_prefix_index(db: sqlite3.Connection) -> dict[str, Any]:
    short_count = posting_count = 0
    postings_complete = True
    for index, (doc_hash, text, representative) in enumerate(
            db.execute("SELECT doc_hash,normalized_text,representative_group FROM docs ORDER BY doc_hash"), start=1):
        grams = {text[pos:pos + 5] for pos in range(max(0, len(text) - 4))}
        if not grams:
            short_count += 1
            db.execute("INSERT INTO exclusions VALUES(?,?,?,?)",
                       (doc_hash, representative, len(text), "empty_raw_character_5gram_set"))
        else:
            posting_count += len(grams)
            if postings_complete and posting_count <= MAX_NGRAM_POSTINGS:
                db.executemany("INSERT INTO doc_grams VALUES(?,?)", ((doc_hash, gram) for gram in grams))
            else:
                postings_complete = False
        if index % 5000 == 0:
            db.commit()
            db_guard(db)
    db.commit()
    doc_count = int(db.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
    if not postings_complete:
        return {"document_count": doc_count, "short_document_count": short_count,
                "gram_posting_count": posting_count, "prefix_token_count": 0,
                "raw_prefix_pair_bound": None, "complete": False,
                "reason": "fixed 5-gram posting work cap exceeded"}
    db.execute("UPDATE docs SET gram_count=(SELECT COUNT(*) FROM doc_grams WHERE doc_grams.doc_hash=docs.doc_hash)")
    db.execute("INSERT INTO gram_frequency SELECT gram,COUNT(*) FROM doc_grams GROUP BY gram")
    current_doc: str | None = None
    size = prefix_limit = position = 0
    prefix_count = 0
    query = db.execute("""SELECT g.doc_hash,g.gram,f.document_frequency,d.gram_count
        FROM doc_grams g JOIN gram_frequency f ON f.gram=g.gram JOIN docs d ON d.doc_hash=g.doc_hash
        ORDER BY g.doc_hash,f.document_frequency,g.gram""")
    batch: list[tuple[str, str]] = []
    for doc_hash, gram, _frequency, gram_count in query:
        if doc_hash != current_doc:
            current_doc = doc_hash
            size = int(gram_count)
            threshold_count = (JACCARD_NUMERATOR * size + JACCARD_DENOMINATOR - 1) // JACCARD_DENOMINATOR
            prefix_limit = size - threshold_count + 1
            position = 0
        position += 1
        if position <= prefix_limit:
            batch.append((doc_hash, gram))
            prefix_count += 1
        if len(batch) >= 10000:
            db.executemany("INSERT INTO prefixes VALUES(?,?)", batch)
            db.commit()
            db_guard(db)
            batch.clear()
    if batch:
        db.executemany("INSERT INTO prefixes VALUES(?,?)", batch)
    db.commit()
    bound = int(db.execute("SELECT COALESCE(SUM(n*(n-1)/2),0) FROM (SELECT COUNT(*) n FROM prefixes GROUP BY gram)").fetchone()[0])
    db_guard(db)
    complete = bound <= MAX_PREFIX_PAIR_WORK
    reason = None if complete else "fixed exact-prefix candidate work cap exceeded"
    return {"document_count": doc_count, "short_document_count": short_count,
            "gram_posting_count": posting_count, "prefix_token_count": prefix_count,
            "raw_prefix_pair_bound": bound, "complete": complete, "reason": reason}


def near_duplicate_edges(db: sqlite3.Connection, index: Mapping[str, Any]) -> dict[str, Any]:
    if not index["complete"]:
        return {"candidate_count": None, "accepted_count": None, "rejected_count": None,
                "complete": False, "reason": index["reason"]}
    accepted = rejected = candidate_count = 0
    intersections = db.execute("""SELECT c.doc_a,c.doc_b,COUNT(*),
            d_a.gram_count,d_a.representative_group,d_b.gram_count,d_b.representative_group
        FROM (
          SELECT p1.doc_hash AS doc_a,p2.doc_hash AS doc_b
          FROM prefixes p1 JOIN prefixes p2 ON p1.gram=p2.gram AND p1.doc_hash<p2.doc_hash
          JOIN docs a ON a.doc_hash=p1.doc_hash JOIN docs b ON b.doc_hash=p2.doc_hash
          WHERE a.representative_group<>b.representative_group
            AND 10*MIN(a.gram_count,b.gram_count)>=9*MAX(a.gram_count,b.gram_count)
          GROUP BY p1.doc_hash,p2.doc_hash
        ) c
        JOIN doc_grams a ON a.doc_hash=c.doc_a
        JOIN doc_grams b ON b.doc_hash=c.doc_b AND b.gram=a.gram
        JOIN docs d_a ON d_a.doc_hash=c.doc_a JOIN docs d_b ON d_b.doc_hash=c.doc_b
        GROUP BY c.doc_a,c.doc_b,d_a.gram_count,d_a.representative_group,
                 d_b.gram_count,d_b.representative_group
        ORDER BY c.doc_a,c.doc_b""")
    for doc_a, doc_b, intersection, size_a, group_a, size_b, group_b in intersections:
        candidate_count += 1
        require(candidate_count <= MAX_PREFIX_PAIR_WORK, "exact prefix candidate count exceeds fixed work cap")
        union = int(size_a) + int(size_b) - int(intersection)
        if JACCARD_DENOMINATOR * int(intersection) >= JACCARD_NUMERATOR * union:
            add_expected_edge(db, group_a, group_b, "verified_char5gram_jaccard_at_least_0.90",
                              doc_a, doc_b, int(intersection), union)
            accepted += 1
        else:
            rejected += 1
        if candidate_count % 10000 == 0:
            db.commit()
            db_guard(db)
    db.commit()
    return {"candidate_count": candidate_count, "accepted_count": accepted,
            "rejected_count": rejected, "complete": True, "reason": None}


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in self.parent}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def join(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def validate_edge_record(db: sqlite3.Connection, record: Mapping[str, Any]) -> tuple[str, str, str]:
    fields = {"group_a", "group_b", "reason", "normalized_input_sha256_a", "normalized_input_sha256_b",
              "char5gram_intersection", "char5gram_union"}
    require(set(record) == fields, "group-edge row has an unexpected schema")
    group_a, group_b, reason = record["group_a"], record["group_b"], record["reason"]
    hash_a, hash_b = record["normalized_input_sha256_a"], record["normalized_input_sha256_b"]
    require(isinstance(group_a, str) and isinstance(group_b, str) and group_a < group_b,
            "group edge endpoints are not canonical")
    require(db.execute("SELECT 1 FROM source_groups WHERE group_id=?", (group_a,)).fetchone() is not None
            and db.execute("SELECT 1 FROM source_groups WHERE group_id=?", (group_b,)).fetchone() is not None,
            "group edge references an unknown lineage group")
    key = canonical_sha256([group_a, group_b, reason, hash_a, hash_b])
    if reason == "exact_normalized_input":
        require(isinstance(hash_a, str) and hash_b is None and record["char5gram_intersection"] is None
                and record["char5gram_union"] is None, "exact-input edge carries invalid evidence")
        for gid in (group_a, group_b):
            require(db.execute("SELECT 1 FROM group_docs WHERE doc_hash=? AND group_id=?", (hash_a, gid)).fetchone()
                    is not None, "exact-input edge evidence does not occur in both groups")
    elif reason == "verified_char5gram_jaccard_at_least_0.90":
        require(isinstance(hash_a, str) and isinstance(hash_b, str) and hash_a != hash_b,
                "near-duplicate edge lacks two distinct document hashes")
        require(db.execute("SELECT 1 FROM docs WHERE doc_hash=? AND representative_group=?", (hash_a, group_a)).fetchone()
                is not None and db.execute("SELECT 1 FROM docs WHERE doc_hash=? AND representative_group=?",
                                           (hash_b, group_b)).fetchone() is not None,
                "near-duplicate evidence hashes do not identify the edge endpoint documents")
        text_a = db.execute("SELECT normalized_text FROM docs WHERE doc_hash=?", (hash_a,)).fetchone()[0]
        text_b = db.execute("SELECT normalized_text FROM docs WHERE doc_hash=?", (hash_b,)).fetchone()[0]
        grams_a = {text_a[pos:pos + 5] for pos in range(max(0, len(text_a) - 4))}
        grams_b = {text_b[pos:pos + 5] for pos in range(max(0, len(text_b) - 4))}
        intersection = len(grams_a & grams_b)
        union = len(grams_a) + len(grams_b) - intersection
        require(union > 0 and 10 * intersection >= 9 * union
                and record["char5gram_intersection"] == intersection
                and record["char5gram_union"] == union,
                "recorded near-duplicate edge fails exact character-5gram Jaccard verification")
    else:
        raise VerificationError("group edge uses an unknown reason")
    expected = db.execute("SELECT group_a,group_b,reason,hash_a,hash_b,intersection_size,union_size "
                          "FROM expected_edges WHERE edge_key=?", (key,)).fetchone()
    if expected is not None:
        observed = (group_a, group_b, reason, hash_a, hash_b,
                    record["char5gram_intersection"], record["char5gram_union"])
        require(observed == expected, "recorded group edge differs from independently reconstructed evidence")
    elif reason == "exact_normalized_input" or db.execute(
            "SELECT complete FROM temp_verifier_state").fetchone()[0]:
        raise VerificationError("recorded edge is absent from the complete independently reconstructed graph")
    return key, group_a, group_b


def derive_components(db: sqlite3.Connection, complete_graph: bool) -> dict[str, Any]:
    group_ids = [row[0] for row in db.execute("SELECT group_id FROM source_groups ORDER BY group_id")]
    forest = UnionFind(group_ids)
    for left, right in db.execute("SELECT group_a,group_b FROM recorded_edges ORDER BY group_a,group_b"):
        forest.join(left, right)
    roots: dict[str, list[str]] = defaultdict(list)
    for gid in group_ids:
        roots[forest.find(gid)].append(gid)
    components: dict[str, list[str]] = {}
    for members in roots.values():
        members.sort()
        cid = canonical_sha256(members)
        components[cid] = members
        db.execute("INSERT INTO components(component_id) VALUES(?)", (cid,))
        db.executemany("INSERT INTO component_groups VALUES(?,?)", ((cid, gid) for gid in members))
        db.executemany("UPDATE source_groups SET component_id=? WHERE group_id=?", ((cid, gid) for gid in members))
    db.commit()
    db_guard(db)
    return {"components": components, "union_find_group_count": len(group_ids), "graph_complete": complete_graph}


def allocation_reconstruction(db: sqlite3.Connection, components: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    rows = {gid: {"source": source, "partition": partition, "rank": rank}
            for gid, source, partition, rank in db.execute(
                "SELECT group_id,source_id,official_partition,rank_sha256 FROM source_groups")}
    members: dict[str, dict[str, list[str]]] = {}
    for cid, gids in components.items():
        source_members: dict[str, list[str]] = defaultdict(list)
        for gid in gids:
            source_members[rows[gid]["source"]].append(gid)
        members[cid] = dict(source_members)

    def anchor(cid: str, partition: str) -> tuple[str, str, str] | None:
        options = [(rows[gid]["rank"], rows[gid]["source"], gid)
                   for gids in members[cid].values() for gid in gids
                   if rows[gid]["partition"] == partition]
        return min(options) if options else None

    confirmation: list[tuple[tuple[str, str, str], str]] = []
    validation: list[tuple[tuple[str, str, str], str]] = []
    train_only: list[str] = []
    for cid in components:
        if any(rows[gid]["partition"] == "test" for gids in members[cid].values() for gid in gids):
            confirmation.append((anchor(cid, "test") or ("f" * 64, "", ""), cid))
        elif any(rows[gid]["partition"] == "validation" for gids in members[cid].values() for gid in gids):
            validation.append((anchor(cid, "validation") or ("f" * 64, "", ""), cid))
        else:
            train_only.append(cid)

    assigned: dict[str, str] = {}
    reasons: dict[str, str] = {}
    unused: dict[str, str] = {}
    anchors: dict[str, tuple[str, str, str]] = {}
    used: dict[str, dict[str, int]] = {source: {split: 0 for split in CAPS} for source in SOURCE_IDS}

    def fits(cid: str, split: str) -> bool:
        return all(used[source][split] < CAPS[split] for source in members[cid])

    def commit(cid: str, split: str, reason: str, chosen: tuple[str, str, str] | None) -> None:
        assigned[cid] = split
        reasons[cid] = reason
        if chosen is not None:
            anchors[cid] = chosen
        for source in members[cid]:
            used[source][split] += 1

    for chosen, cid in sorted(confirmation):
        if fits(cid, "confirmation"):
            commit(cid, "confirmation", "official_test_or_component_connected_to_official_test", chosen)
        else:
            unused[cid] = "confirmation_cap_reached_for_at_least_one_connected_source"

    positions: dict[tuple[str, str], int] = {}
    per_source_validation: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for _outer, cid in validation:
        for source, gids in members[cid].items():
            eligible = [gid for gid in gids if rows[gid]["partition"] == "validation"]
            if eligible:
                best = min((rows[gid]["rank"], gid) for gid in eligible)
                per_source_validation[source].append((best[0], best[1], cid))
    for source, ordered in per_source_validation.items():
        for ordinal, (_rank, _gid, cid) in enumerate(sorted(ordered)):
            positions[(source, cid)] = ordinal
    ranked_validation: list[tuple[tuple[str, str, str], str, str]] = []
    for _outer, cid in validation:
        choices: list[tuple[str, str, str, int]] = []
        for source, gids in members[cid].items():
            ordinal = positions.get((source, cid))
            eligible = [gid for gid in gids if rows[gid]["partition"] == "validation"]
            if ordinal is None or ordinal >= 2 * max(CAPS["dev"], CAPS["calibration"]) or not eligible:
                continue
            rank, gid = min((rows[gid]["rank"], gid) for gid in eligible)
            choices.append((rank, source, gid, ordinal))
        if not choices:
            unused[cid] = "validation_rank_overflow_after_two_caps"
            continue
        rank, source, gid, ordinal = min(choices)
        ranked_validation.append(((rank, source, gid), cid, "dev" if ordinal % 2 == 0 else "calibration"))
    for chosen, cid, split in sorted(ranked_validation):
        if fits(cid, split):
            commit(cid, split, "ranked_official_validation_alternating_dev_calibration", chosen)
        else:
            unused[cid] = f"{split}_cap_reached_for_at_least_one_connected_source"

    per_source_train: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for cid in train_only:
        for source, gids in members[cid].items():
            eligible = [gid for gid in gids if rows[gid]["partition"] == "train"]
            if eligible:
                rank, gid = min((rows[gid]["rank"], gid) for gid in eligible)
                per_source_train[source].append((rank, gid, cid))
    priority = {"calibration": 0, "dev": 1, "train": 2}
    recommendations: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for source, candidates in per_source_train.items():
        for ordinal, (rank, gid, cid) in enumerate(sorted(candidates)):
            slot = ordinal % 10
            desired = "dev" if slot in (0, 1) else "calibration" if slot in (2, 3) else "train"
            recommendations[cid].append((desired, rank, source, gid))
    train_candidates = []
    for cid, suggestions in recommendations.items():
        desired, rank, source, gid = min(suggestions, key=lambda item: (priority[item[0]], item[1], item[2], item[3]))
        train_candidates.append(((rank, source, gid), cid, desired))
    for chosen, cid, split in sorted(train_candidates, key=lambda item: (item[0], item[1])):
        if fits(cid, split):
            commit(cid, split, "official_train_sha_rank_mod_10", chosen)
        else:
            unused[cid] = f"{split}_cap_reached_for_at_least_one_connected_source"

    for cid in components:
        if cid not in assigned and cid not in unused:
            unused[cid] = "no_eligible_source_partition"
        if cid in assigned:
            db.execute("UPDATE components SET final_split=?,allocation_reason=? WHERE component_id=?",
                       (assigned[cid], reasons[cid], cid))
            for gid in components[cid]:
                db.execute("UPDATE source_groups SET final_split=?,allocation_reason=?,exclusion_reason=NULL WHERE group_id=?",
                           (assigned[cid], reasons[cid], gid))
        else:
            db.execute("UPDATE components SET final_split='unused',exclusion_reason=? WHERE component_id=?",
                       (unused[cid], cid))
            for gid in components[cid]:
                db.execute("UPDATE source_groups SET final_split='unused',allocation_reason=NULL,exclusion_reason=? WHERE group_id=?",
                           (unused[cid], gid))
    db.commit()
    return {"assigned": assigned, "reasons": reasons, "unused": unused,
            "anchors": anchors, "used": used, "members": members}


def audit_group_and_edge_artifacts(root: Path, db: sqlite3.Connection, locale_order: Sequence[str],
                                   graph_complete: bool) -> dict[str, int]:
    db.execute("CREATE TABLE recorded_edges(edge_key TEXT PRIMARY KEY,group_a TEXT NOT NULL,group_b TEXT NOT NULL)")
    edge_path = secure_file(root, "private/group_edges.jsonl.gz")
    edge_count = exact_count = near_count = 0
    try:
        with gzip.open(edge_path, "rb") as stream:
            for record in jsonl_records(stream):
                key, left, right = validate_edge_record(db, record)
                inserted = db.execute("INSERT OR IGNORE INTO recorded_edges VALUES(?,?,?)",
                                      (key, left, right))
                require(inserted.rowcount == 1, "recorded edge artifact contains a duplicate edge")
                edge_count += 1
                exact_count += record["reason"] == "exact_normalized_input"
                near_count += record["reason"] == "verified_char5gram_jaccard_at_least_0.90"
                if edge_count % 10000 == 0:
                    db.commit()
                    db_guard(db)
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise VerificationError("group-edge artifact cannot be streamed") from exc
    if graph_complete:
        expected_count = int(db.execute("SELECT COUNT(*) FROM expected_edges").fetchone()[0])
        require(edge_count == expected_count, "recorded graph omits or adds reconstructed edges")
        missing = int(db.execute(
            "SELECT COUNT(*) FROM expected_edges e LEFT JOIN recorded_edges r USING(edge_key) "
            "WHERE r.edge_key IS NULL").fetchone()[0])
        extra = int(db.execute(
            "SELECT COUNT(*) FROM recorded_edges r LEFT JOIN expected_edges e USING(edge_key) "
            "WHERE e.edge_key IS NULL").fetchone()[0])
        require(missing == 0 and extra == 0,
                "recorded graph edge-key set differs from reconstruction")
    else:
        exact_expected = int(db.execute("SELECT COUNT(*) FROM expected_edges WHERE reason='exact_normalized_input'").fetchone()[0])
        exact_seen = int(db.execute(
            "SELECT COUNT(*) FROM recorded_edges r JOIN expected_edges e USING(edge_key) "
            "WHERE e.reason='exact_normalized_input'").fetchone()[0])
        require(exact_seen == exact_expected and exact_count == exact_expected,
                "exact-input star edges differ from independent reconstruction")
    return {"recorded_edge_count": edge_count, "exact_edge_count": int(exact_count),
            "near_duplicate_edge_count": int(near_count)}


def _group_record_expected(db: sqlite3.Connection, row: Sequence[Any], locale_order: Sequence[str]) -> dict[str, Any]:
    gid, source, revision, partition, lineage_json, hashes_json, rank, cid, split, reason, exclusion, member_count = row
    case = "CASE locale " + " ".join(f"WHEN ? THEN {i}" for i, _ in enumerate(locale_order)) + " ELSE 1000 END"
    parameters: list[Any] = [gid]
    parameters.extend(locale_order)
    row_keys = [value[0] for value in db.execute(
        f"SELECT row_key FROM raw_rows WHERE group_id=? ORDER BY {case},official_partition,line_number,row_key", parameters)]
    return {"group_id": gid, "source_id": source, "pinned_revision_for_group_id": revision,
            "official_partition": partition, "original_lineage_ids": json.loads(lineage_json),
            "normalized_input_sha256s": json.loads(hashes_json), "rank_sha256": rank,
            "component_id": cid, "component_member_group_count": member_count,
            "final_split": split, "allocation_reason": reason, "exclusion_reason": exclusion,
            "ordered_source_row_keys": row_keys}


def compare_group_artifact(root: Path, db: sqlite3.Connection, locale_order: Sequence[str]) -> int:
    path = secure_file(root, "private/groups.jsonl.gz")
    seen = 0
    query = db.execute("""SELECT group_id,source_id,pinned_revision,official_partition,lineage_json,input_hashes_json,
        rank_sha256,component_id,final_split,allocation_reason,exclusion_reason,component_member_count
        FROM source_groups ORDER BY CASE source_id WHEN ? THEN 0 ELSE 1 END,
        CASE final_split WHEN 'train' THEN 0 WHEN 'dev' THEN 1 WHEN 'calibration' THEN 2
        WHEN 'confirmation' THEN 3 ELSE 4 END,rank_sha256,group_id""", (SOURCE_BANK,))
    try:
        with gzip.open(path, "rb") as stream:
            records = jsonl_records(stream)
            for row, actual in zip(query, records):
                expected = _group_record_expected(db, row, locale_order)
                require(actual == expected, "group artifact differs from reconstructed source lineage or allocation")
                seen += 1
            require(seen == int(db.execute("SELECT COUNT(*) FROM source_groups").fetchone()[0]),
                    "group artifact count differs from reconstructed lineage groups")
            require(next(records, None) is None, "group artifact contains extra records")
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise VerificationError("group artifact cannot be streamed") from exc
    return seen


def ordered_group_rows(db: sqlite3.Connection, locale_order: Sequence[str]) -> Iterator[tuple[Any, ...]]:
    case = "CASE locale " + " ".join(f"WHEN ? THEN {i}" for i, _ in enumerate(locale_order)) + " ELSE 1000 END"
    for source in SOURCE_IDS:
        for split in SPLITS:
            groups = db.execute("SELECT group_id FROM source_groups WHERE source_id=? AND final_split=? "
                                "ORDER BY rank_sha256,group_id", (source, split))
            for (gid,) in groups:
                rows = db.execute(f"""SELECT r.row_key,r.source_id,r.official_partition,r.locale,r.original_id,
                    r.raw_input_sha256,r.normalized_sha256,r.source_file,r.line_number,r.payload_sha256,
                    r.group_id,g.component_id,g.final_split
                    FROM raw_rows r JOIN source_groups g USING(group_id) WHERE r.group_id=?
                    ORDER BY {case},r.official_partition,r.line_number,r.row_key""", (gid, *locale_order))
                yield from rows


def compare_source_rows(root: Path, db: sqlite3.Connection, locale_order: Sequence[str]) -> int:
    path = secure_file(root, "private/source_rows.jsonl.gz")
    count = 0
    try:
        expected_rows = ordered_group_rows(db, locale_order)
        with gzip.open(path, "rb") as stream:
            actual_rows = jsonl_records(stream)
            for raw in expected_rows:
                actual = next(actual_rows, None)
                require(actual is not None, "source-row artifact ended before raw descendants")
                (row_key, source, partition, locale, original_id, input_sha, norm_sha,
                 source_file, line_number, payload_sha, group_id, component_id, split) = raw
                fields = {"row_key", "source_id", "official_partition", "locale", "original_utterance_id",
                          "input_text", "normalized_input_sha256", "source_file", "source_line_number",
                          "group_id", "component_id", "final_split", "source_record_with_original_labels_and_annotations"}
                require(set(actual) == fields, "source-row record has an unexpected schema")
                original_output = original_id if source == SOURCE_MASSIVE else None
                require(actual["row_key"] == row_key and actual["source_id"] == source
                        and actual["official_partition"] == partition and actual["locale"] == locale
                        and actual["original_utterance_id"] == original_output
                        and actual["normalized_input_sha256"] == norm_sha
                        and actual["source_file"] == source_file and actual["source_line_number"] == line_number
                        and actual["group_id"] == group_id and actual["component_id"] == component_id
                        and actual["final_split"] == split,
                        "source-row identity or provenance differs from raw native records")
                text = actual["input_text"]
                payload = actual["source_record_with_original_labels_and_annotations"]
                require(isinstance(text, str) and sha256_bytes(text.encode("utf-8")) == input_sha,
                        "source-row input bytes differ from raw native source")
                require(isinstance(payload, dict) and sha256_bytes(canonical_json(payload).encode("utf-8")) == payload_sha,
                        "full source-row payload differs from raw native source")
                if source == SOURCE_BANK:
                    require(payload.get("text") == text, "Banking77 payload/input identity differs")
                else:
                    require(payload.get("utt") == text and payload.get("id") == original_id
                            and payload.get("locale") == locale,
                            "MASSIVE payload/input or locale-ID identity differs")
                count += 1
            require(next(actual_rows, None) is None, "source-row artifact contains extra descendants")
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        raise VerificationError("source-row artifact cannot be streamed") from exc
    require(count == int(db.execute("SELECT COUNT(*) FROM raw_rows").fetchone()[0]),
            "source-row artifact count differs from raw source rows")
    return count


def compare_exclusions(root: Path, db: sqlite3.Connection) -> int:
    path = secure_file(root, "private/near_duplicate_search_exclusions.jsonl.gz")
    expected = db.execute("SELECT doc_hash,group_id,codepoint_count,reason FROM exclusions ORDER BY doc_hash")
    count = 0
    with gzip.open(path, "rb") as stream:
        actual_iter = jsonl_records(stream)
        for doc_hash, group_id, codepoints, reason in expected:
            actual = next(actual_iter, None)
            require(actual == {"normalized_input_sha256": doc_hash, "representative_group_id": group_id,
                               "normalized_codepoint_count": codepoints, "reason": reason,
                               "retained_for_exact_input_and_source_lineage_grouping": True,
                               "excluded_from_source_rows_or_allocation": False},
                    "short-input search exclusion differs from independent inventory")
            count += 1
        require(next(actual_iter, None) is None, "search-exclusion artifact contains extra records")
    return count


def compare_unused(root: Path, db: sqlite3.Connection, allocation: Mapping[str, Any],
                   components: Mapping[str, Sequence[str]]) -> int:
    path = secure_file(root, "private/unused_groups.jsonl.gz")
    expected_items = sorted(allocation["unused"].items())
    count = 0
    with gzip.open(path, "rb") as stream:
        actual_iter = jsonl_records(stream)
        for cid, reason in expected_items:
            actual = next(actual_iter, None)
            gids = list(components[cid])
            source_ids = sorted({row[0] for row in db.execute(
                "SELECT DISTINCT source_id FROM source_groups WHERE component_id=?", (cid,))})
            require(actual == {"component_id": cid, "reason": reason, "group_ids": gids, "source_ids": source_ids},
                    "unused-component reason or membership differs from reconstruction")
            count += 1
        require(next(actual_iter, None) is None, "unused-component artifact contains extra records")
    return count


def component_and_split_checks(db: sqlite3.Connection, components: Mapping[str, Sequence[str]],
                               allocation: Mapping[str, Any]) -> dict[str, Any]:
    require(int(db.execute("SELECT COUNT(*) FROM components").fetchone()[0]) == len(components),
            "component count differs from independently closed graph")
    for cid, gids in components.items():
        stored_groups = [row[0] for row in db.execute(
            "SELECT group_id FROM component_groups WHERE component_id=? ORDER BY group_id", (cid,))]
        require(stored_groups == list(gids), "component closure membership is inconsistent")
        if cid in allocation["assigned"]:
            split = allocation["assigned"][cid]
            reason = allocation["reasons"][cid]
            expected_exclusion = None
        else:
            split = "unused"
            reason = None
            expected_exclusion = allocation["unused"][cid]
        for gid in gids:
            row = db.execute("SELECT final_split,allocation_reason,exclusion_reason,component_member_count "
                             "FROM source_groups WHERE group_id=?", (gid,)).fetchone()
            require(row == (split, reason, expected_exclusion, len(gids)),
                    "group split, reason, or component membership count differs")
        assigned_splits = {row[0] for row in db.execute(
            "SELECT DISTINCT final_split FROM source_groups WHERE component_id=?", (cid,))}
        require(assigned_splits == {split}, "connected component leaks across final splits")
    hashes: dict[str, Any] = {}
    row_counts: dict[str, Any] = {}
    group_counts: dict[str, Any] = {}
    component_counts: dict[str, Any] = {}
    for source in SOURCE_IDS:
        hashes[source] = {}
        row_counts[source] = {}
        group_counts[source] = {}
        component_counts[source] = {}
        for split in SPLITS:
            group_ids = [row[0] for row in db.execute(
                "SELECT group_id FROM source_groups WHERE source_id=? AND final_split=? ORDER BY rank_sha256,group_id",
                (source, split))]
            component_ids = list(dict.fromkeys(row[0] for gid in group_ids for row in db.execute(
                "SELECT component_id FROM source_groups WHERE group_id=?", (gid,))))
            hashes[source][split] = {"source_lineage_group_count": len(group_ids),
                                     "connected_component_count": len(component_ids),
                                     "ordered_group_ids_sha256": canonical_sha256(group_ids),
                                     "ordered_component_ids_sha256": canonical_sha256(component_ids)}
            group_counts[source][split] = len(group_ids)
            component_counts[source][split] = len(component_ids)
            row_counts[source][split] = int(db.execute(
                "SELECT COUNT(*) FROM raw_rows r JOIN source_groups g USING(group_id) "
                "WHERE r.source_id=? AND g.final_split=?", (source, split)).fetchone()[0])
    global_components = {split: int(db.execute("SELECT COUNT(*) FROM components WHERE final_split=?", (split,)).fetchone()[0])
                         for split in SPLITS}
    return {"group_split_hashes": hashes, "source_row_counts_by_final_split": row_counts,
            "source_group_counts_by_final_split": group_counts,
            "source_component_counts_by_final_split": component_counts,
            "global_component_counts_by_final_split": global_components,
            "source_group_count": int(db.execute("SELECT COUNT(*) FROM source_groups").fetchone()[0]),
            "component_count": len(components)}


def compare_historical_report(root: Path, report: Mapping[str, Any], reconstructed: Mapping[str, Any]) -> dict[str, Any]:
    allocation = report.get("allocation", {})
    require(allocation.get("seed") == SEED and allocation.get("purpose") == PURPOSE
            and allocation.get("caps_independent_groups_per_source") == CAPS
            and allocation.get("labels_or_outcomes_used_for_grouping_or_allocation") is False
            and allocation.get("source_partitions_and_exposure_are_preserved") is True,
            "historical allocation metadata differs from frozen protocol")
    counts = allocation.get("counts_and_hashes", {})
    require(counts.get("source_group_count") == reconstructed["source_group_count"]
            and counts.get("component_count") == reconstructed["component_count"],
            "historical source-group or component member count differs")
    require(counts.get("global_component_counts_by_final_split") == reconstructed["global_component_counts_by_final_split"],
            "historical global component counts differ")
    require(counts.get("source_row_counts_by_final_split") == reconstructed["source_row_counts_by_final_split"],
            "historical row counts differ")
    historical = counts.get("group_split_hashes", {})
    for source in SOURCE_IDS:
        for split in SPLITS:
            old = historical.get(source, {}).get(split, {})
            new = reconstructed["group_split_hashes"][source][split]
            require(old.get("ordered_group_ids_sha256") == new["ordered_group_ids_sha256"]
                    and old.get("independent_group_count") == new["source_lineage_group_count"],
                    "historical membership count/hash differs from independently reconstructed lineage members")
    require(all("independent_group_count" in historical.get(source, {}).get(split, {})
                for source in SOURCE_IDS for split in SPLITS),
            "historical report lacks its lineage-member count field")
    return {"historical_member_count_field": "independent_group_count",
            "validated_as": "source-lineage group member count, not allocation-unit/component count",
            "independence_label_accepted": False,
            "component_count_validated_separately": reconstructed["component_count"],
            "count_unit_correction_used_as_verification_input": False}


def check_source_alignment(db: sqlite3.Connection, locales: Sequence[str], inventory: Mapping[str, Any]) -> dict[str, Any]:
    locale_count = len(locales)
    original_count = int(db.execute("SELECT COUNT(*) FROM massive_base").fetchone()[0])
    mismatch = int(db.execute("SELECT COUNT(*) FROM massive_base b WHERE "
                              "(SELECT COUNT(*) FROM massive_locale_ids m WHERE m.original_id=b.original_id) != ?",
                              (locale_count,)).fetchone()[0])
    require(mismatch == 0, "MASSIVE source IDs do not align across all frozen locales")
    extra_locale = int(db.execute("SELECT COUNT(*) FROM massive_locale_ids WHERE locale NOT IN (" +
                                  ",".join("?" for _ in locales) + ")", tuple(locales)).fetchone()[0])
    require(extra_locale == 0, "MASSIVE contains a locale outside the frozen locale list")
    source_records = {item["id"]: item for item in inventory["sources"]}
    bank_sizes = source_records[SOURCE_BANK]["split_sizes"]
    bank_expected = {"train": int(bank_sizes["train"]), "test": int(bank_sizes["test"])}
    observed = {part: int(db.execute("SELECT COUNT(*) FROM raw_rows WHERE source_id=? AND official_partition=?",
                                      (SOURCE_BANK, part)).fetchone()[0]) for part in ("train", "test")}
    require(observed == bank_expected, "Banking77 raw partition counts differ from frozen inventory")
    massive_rows = int(db.execute("SELECT COUNT(*) FROM raw_rows WHERE source_id=?", (SOURCE_MASSIVE,)).fetchone()[0])
    require(original_count == 16521 and massive_rows == 842571
            and sum(observed.values()) == 13083 and massive_rows + sum(observed.values()) == 855654,
            "independent raw source row or ID counts differ from the frozen custody scope")
    return {"massive_original_ids": original_count, "massive_rows": massive_rows,
            "massive_locale_descendants_per_id": locale_count,
            "massive_locale_id_alignment_mismatches": mismatch, "banking77_rows_by_native_partition": observed}


def check_native_judgment_metadata(report: Mapping[str, Any], locales: Sequence[str],
                                   observed: Mapping[str, Mapping[str, Mapping[str, int]]]) -> dict[str, Any]:
    sources = report.get("sources")
    require(isinstance(sources, dict), "historical report lacks native source metadata")
    massive = sources.get(SOURCE_MASSIVE)
    require(isinstance(massive, dict)
            and massive.get("judgment_values_are_not_mapped_to_model_primitives_or_gold") is True,
            "MASSIVE judgment annotations are not marked as unmapped native metadata")
    reported = massive.get("judgment_metadata_by_locale_partition")
    require(isinstance(reported, dict) and set(reported) == set(locales),
            "MASSIVE native judgment metadata omits or adds locales")
    for locale in locales:
        raw_locale = observed[locale]
        report_locale = reported[locale]
        expected_partitions = {part for part in PARTITIONS if raw_locale[part]["row_count"] > 0}
        require(isinstance(report_locale, dict) and set(report_locale) == expected_partitions,
                "MASSIVE native judgment metadata partition coverage differs")
        for partition in expected_partitions:
            raw_counts = raw_locale[partition]
            entry = report_locale[partition]
            require(isinstance(entry, dict), "MASSIVE native judgment summary is not an object")
            for field in ("row_count", "judgments_field_present_rows", "judgments_field_missing_rows",
                          "judgments_field_null_rows", "rows_with_judgments", "judgment_record_count"):
                require(type(entry.get(field)) is int and entry[field] == raw_counts[field],
                        "MASSIVE native judgment count differs from the raw source rows")
            require(entry.get("native_values_are_unmapped_source_observations") is True
                    and entry.get("no_rating_distribution_invented_when_source_values_are_missing") is True,
                    "MASSIVE rating values are not retained as unmapped native observations")
    english_records = sum(observed["en-US"][partition]["judgment_record_count"] for partition in PARTITIONS)
    require(english_records == 0, "English MASSIVE unexpectedly contains native quality-rating records")
    return {
        "judgment_record_counts_by_locale_partition": {
            locale: {partition: observed[locale][partition]["judgment_record_count"]
                     for partition in PARTITIONS if observed[locale][partition]["row_count"] > 0}
            for locale in locales
        },
        "english_judgment_record_count": english_records,
    }


def membership_statistics(db: sqlite3.Connection, reconstructed: Mapping[str, Any]) -> dict[str, Any]:
    return {"source_rows": int(db.execute("SELECT COUNT(*) FROM raw_rows").fetchone()[0]),
            "source_rows_by_source": {source: int(db.execute("SELECT COUNT(*) FROM raw_rows WHERE source_id=?",
                                                             (source,)).fetchone()[0]) for source in SOURCE_IDS},
            "source_groups": int(db.execute("SELECT COUNT(*) FROM source_groups").fetchone()[0]),
            "source_groups_by_source": {source: int(db.execute("SELECT COUNT(*) FROM source_groups WHERE source_id=?",
                                                               (source,)).fetchone()[0]) for source in SOURCE_IDS},
            "components": reconstructed["component_count"],
            "source_lineage_groups_by_final_split": reconstructed["source_group_counts_by_final_split"],
            "connected_components_by_final_split": reconstructed["source_component_counts_by_final_split"],
            "source_rows_by_final_split": reconstructed["source_row_counts_by_final_split"],
            "global_components_by_final_split": reconstructed["global_component_counts_by_final_split"]}


def ensure_manifest_artifact_size_report(report: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
    records = report.get("private_artifacts")
    require(isinstance(records, dict) and set(records) ==
            {"edges", "groups", "rows_and_labels", "search_exclusions", "unused"},
            "historical report omits or adds required private artifact receipts")
    for value in records.values():
        require(isinstance(value, dict) and {"path", "sha256", "size_bytes"} <= set(value),
                "historical private artifact receipt omits required digest metadata")
        relative = value.get("path")
        require(relative in outputs and value.get("sha256") == outputs[relative]["sha256"]
                and value.get("size_bytes") == outputs[relative]["size_bytes"],
                "historical private artifact digest or size differs")


def verify(root_arg: str) -> tuple[dict[str, Any], Path]:
    os.umask(0o077)
    guards = set_resource_guards()
    unresolved_root = Path(os.path.abspath(Path(root_arg).expanduser()))
    cursor = Path(unresolved_root.anchor)
    for part in unresolved_root.parts[1:]:
        cursor = cursor / part
        try:
            component_stat = cursor.lstat()
        except OSError as exc:
            raise VerificationError("data-root path component is unavailable") from exc
        require(not stat.S_ISLNK(component_stat.st_mode), "data-root path contains a symlink")
    root = unresolved_root.resolve(strict=True)
    require(root.is_dir(), "data root is not a plain directory")
    secure_directory(root)
    manifest, source_files, input_hashes, hash_sets = verify_input_hashes(root)
    plan, inventory, locales, _allocation_parameters, snapshots = check_frozen_plan(root, manifest, source_files)
    input_hashes.update(snapshots)
    verify_raw_artifact_plan(source_files, plan, inventory)
    repo_root = Path(__file__).resolve().parents[2]
    require(root != repo_root and repo_root not in root.parents,
            "private source root must remain outside the code repository")
    archive_path = secure_file(root, "raw/massive/amazon-massive-dataset-1.1.tar.gz")
    member_index = secure_file(root, "private/massive_tar_members.jsonl.gz")
    scratch_parent = root / "verification"
    secure_directory(scratch_parent, create=True)
    scratch_context = tempfile.TemporaryDirectory(prefix=".neutral-verify-", dir=scratch_parent)
    try:
        scratch = Path(scratch_context.name)
        db = init_db(scratch / "reconstruction.sqlite3")
        source_records = {item["id"]: item for item in inventory["sources"]}
        bank_sizes = source_records[SOURCE_BANK]["split_sizes"]
        csv_train = secure_file(root, "raw/banking77/train.csv")
        csv_test = secure_file(root, "raw/banking77/test.csv")
        bank_rows = {
            "train": csv_rows(csv_train, "train.csv", SOURCE_BANK, "train", db, int(bank_sizes["train"])),
            "test": csv_rows(csv_test, "test.csv", SOURCE_BANK, "test", db, int(bank_sizes["test"])),
        }
        tar_summary = verify_tar_and_parse(root, archive_path, locales, inventory, member_index, db)
        expected_tar_summary = source_files["massive_archive"]["member_inspection"]
        for key in ("member_count", "regular_file_count", "directory_count", "unpacked_regular_bytes",
                    "selected_jsonl_member_count", "selected_locale_count"):
            require(tar_summary[key] == expected_tar_summary[key], "MASSIVE archive inspection summary differs")
        require(expected_tar_summary.get("partition_routing") ==
                "Native JSON row partition; no split-specific archive files."
                and expected_tar_summary.get("remote_code_executed") is False,
                "MASSIVE archive scope metadata differs")
        alignment = check_source_alignment(db, locales, inventory)
        build_source_groups(db, plan)
        source_group_count = int(db.execute("SELECT COUNT(*) FROM source_groups").fetchone()[0])
        require(source_group_count == int(db.execute("SELECT COUNT(DISTINCT row_key) FROM raw_rows WHERE source_id=?",
                                                     (SOURCE_BANK,)).fetchone()[0]) + alignment["massive_original_ids"],
                "source-lineage group construction count differs")
        group_counts_by_source = {source: int(db.execute("SELECT COUNT(*) FROM source_groups WHERE source_id=?",
                                                         (source,)).fetchone()[0]) for source in SOURCE_IDS}
        require(source_group_count == 29604 and group_counts_by_source == {SOURCE_BANK: 13083, SOURCE_MASSIVE: 16521},
                "independent source-lineage group counts differ from the frozen custody scope")
        exact_edge_count = exact_duplicate_edges(db)
        index = build_gram_prefix_index(db)
        near = near_duplicate_edges(db, index)
        graph_complete = bool(index["complete"] and near["complete"])
        # The recorded edge table is built only from the raw reconstructed rows.
        db.execute("CREATE TABLE temp_verifier_state(complete INTEGER NOT NULL)")
        db.execute("INSERT INTO temp_verifier_state VALUES(?)", (int(graph_complete),))
        edge_summary = audit_group_and_edge_artifacts(root, db, locales, graph_complete)
        db.commit()
        report = strict_json_bytes(secure_file(root, "private/acquisition_report.json").read_bytes())
        ensure_manifest_artifact_size_report(report, hash_sets["outputs"])
        native_metadata = check_native_judgment_metadata(
            report, locales, tar_summary["native_judgment_metadata_counts"])
        grouping = report.get("grouping", {})
        historical_search = grouping.get("near_duplicate_search", {})
        require(grouping.get("edge_counts", {}).get("exact_normalized_input_edges") == exact_edge_count
                and grouping.get("edge_counts", {}).get("all_verified_edges") == edge_summary["recorded_edge_count"],
                "historical edge counts differ from independent edge audit")
        if graph_complete:
            require(historical_search.get("document_count") == index["document_count"]
                    and historical_search.get("short_document_count") == index["short_document_count"]
                    and historical_search.get("gram_posting_count") == index["gram_posting_count"]
                    and historical_search.get("prefix_token_count") == index["prefix_token_count"]
                    and historical_search.get("raw_prefix_pair_bound") == index["raw_prefix_pair_bound"]
                    and historical_search.get("unique_candidate_pairs_verified") == near["candidate_count"]
                    and historical_search.get("accepted_near_duplicate_edges") == near["accepted_count"]
                    and historical_search.get("threshold_rejections") == near["rejected_count"],
                    "historical exact-prefix search counters differ from independent reconstruction")
            require(edge_summary["near_duplicate_edge_count"] == near["accepted_count"],
                    "near-duplicate edge count differs from independently reconstructed threshold matches")
        else:
            require(edge_summary["near_duplicate_edge_count"] >= 0,
                    "recorded near-duplicate edges could not be checked")
        components_result = derive_components(db, graph_complete)
        components = components_result["components"]
        require(len(components) == 22169, "independent connected-component count differs from the frozen custody scope")
        assigned = allocation_reconstruction(db, components)
        for cid, gids in components.items():
            db.execute("UPDATE source_groups SET component_member_count=? WHERE component_id=?", (len(gids), cid))
        db.commit()
        reconstructed = component_and_split_checks(db, components, assigned)
        historical_units = compare_historical_report(root, report, reconstructed)
        source_members = compare_group_artifact(root, db, locales)
        source_rows = compare_source_rows(root, db, locales)
        exclusions = compare_exclusions(root, db)
        unused_count = compare_unused(root, db, assigned, components)
        require(tar_summary["selected_locale_count"] == 51 and alignment["massive_locale_id_alignment_mismatches"] == 0,
                "MASSIVE locale alignment gate failed")
        checks = {
            "manifest_outputs_raw_artifacts_and_snapshots_hashed": True,
            "frozen_plan_and_work_bounds_verified_without_execution": True,
            "private_path_permissions_and_symlink_safety": True,
            "massive_archive_members_and_selected_member_bytes_verified": True,
            "banking77_raw_rows_and_provenance_reconstructed": True,
            "massive_native_rows_partitions_and_locale_id_alignment_reconstructed": True,
            "full_source_payload_identity_against_source_rows": True,
            "native_massive_judgment_metadata_reconstructed_and_unmapped": True,
            "source_group_ids_and_seed_ranks_reconstructed": True,
            "recorded_edges_valid_and_component_closure_reconstructed": True,
            "near_duplicate_search_completeness_independently_proved": graph_complete,
            "allocation_order_membership_reasons_and_unused_groups_reconstructed": True,
            "no_cross_split_component_leakage": True,
            "historical_independence_label_rejected": True,
        }
        results = {
            "schema": "vey.neutral.acquisition-independent-verification.v1",
            "verification_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"),
            "source_custody_fully_verified": all(checks.values()),
            "checks": checks,
            "input_hashes": input_hashes,
            "output_hashes": hash_sets["outputs"],
            "verifier_sha256": hash_file(Path(__file__).resolve())[0],
            "resource_guards": guards,
            "counts": {**membership_statistics(db, reconstructed),
                       "banking77_rows_by_native_partition": bank_rows,
                       "massive_original_ids": alignment["massive_original_ids"],
                       "massive_locale_descendants_per_id": alignment["massive_locale_descendants_per_id"],
                       "massive_tar_member_count": tar_summary["member_count"],
                       "massive_native_judgment_record_counts_by_locale_partition":
                           native_metadata["judgment_record_counts_by_locale_partition"],
                       "english_massive_judgment_record_count": native_metadata["english_judgment_record_count"],
                       "selected_locale_count": tar_summary["selected_locale_count"],
                       "recorded_edge_count": edge_summary["recorded_edge_count"],
                       "exact_duplicate_edges": edge_summary["exact_edge_count"],
                       "near_duplicate_edges": edge_summary["near_duplicate_edge_count"],
                       "near_duplicate_candidate_count": near["candidate_count"],
                       "near_duplicate_candidate_work_bound": index["raw_prefix_pair_bound"],
                       "short_exact_only_documents": exclusions,
                       "unused_components": unused_count,
                       "groups_artifact_records": source_members,
                       "source_rows_artifact_records": source_rows},
            "allocation": {**reconstructed, **historical_units},
            "near_duplicate_completeness": {
                "independently_proved": graph_complete,
                "method": "bounded exact global-document-frequency ordered character-5gram prefix join, exact length bound, exact set intersection and integer 0.90 Jaccard test",
                "completeness_argument": "For token sets ordered by global document frequency then token, prefix length m-ceil(0.9*m)+1 guarantees any pair meeting Jaccard 0.90 shares an indexed prefix token; the integer length bound prunes only impossible pairs, and each candidate is checked by full set intersection.",
                "threshold_test": "10*intersection >= 9*union",
                "raw_prefix_pair_bound": index["raw_prefix_pair_bound"],
                "candidate_work_cap": MAX_PREFIX_PAIR_WORK,
                "posting_cap": MAX_NGRAM_POSTINGS,
                "reason_if_incomplete": index["reason"],
                "inputs_with_no_5grams_remain_in_exact_hash_and_lineage_graph": True,
            },
            "scope_and_limitations": {
                "human_author_or_template_independence_established": False,
                "banking77_lineage": "deterministic source-file record ordinals; upstream author/template lineage is unavailable here",
                "massive_lineage": "original utterance IDs align locale descendants; this does not establish author/template independence",
                "native_source_annotations": "retained payloads are byte-identity checked and left unmapped; no source-native values are converted into model targets or primitives",
                "models_or_loaders_executed": False,
                "count_unit_correction_used_as_independent_evidence": False,
                "historical_independent_group_count": "validated only as source-lineage group membership, not independent allocation units",
                "current_producer_source_required_to_match_historical_snapshot": False,
            },
            "claims": {"capabilities_earned": [], "model_outputs_acquired": False,
                       "quality_metrics_measured": False, "noninferiority_established": False,
                       "statistical_independence_established": False, "legal_opinion": False},
        }
        db.close()
    finally:
        scratch_context.cleanup()
    output_dir = root / "verification"
    secure_directory(output_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    base = f"neutral_acquisition_verification_{stamp}"
    body = (canonical_json(results) + "\n").encode("utf-8")
    suffix = 0
    while True:
        name = f"{base}{'' if suffix == 0 else f'-{suffix}'}.json"
        destination = output_dir / name
        try:
            fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            suffix += 1
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        dir_fd = os.open(output_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    return results, destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="actual private neutral acquisition root")
    args = parser.parse_args()
    try:
        result, path = verify(args.root)
    except VerificationError as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("verification failed: unexpected bounded parser, filesystem, or SQLite error; no source values emitted",
              file=sys.stderr)
        return 2
    digest, size = hash_file(path)
    print(canonical_json({"verification_path": str(path), "verification_sha256": digest,
                          "verification_size_bytes": size,
                          "source_custody_fully_verified": result["source_custody_fully_verified"]}))
    return 0 if result["source_custody_fully_verified"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
