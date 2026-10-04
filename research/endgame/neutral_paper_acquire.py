#!/usr/bin/env python3
"""Acquire and seal the pinned QASPER source without running its loader.

Run only after this script and its frozen inputs have been committed:
    python research/endgame/neutral_paper_acquire.py acquire

All source rows and annotation payloads remain in the private data root named
by the committed source plan. Standard output contains metadata only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Sequence

import neutral_acquire as shared


SCRIPT_SCHEMA = "vey.neutral.qasper.paper-acquisition.v1"
SOURCE_ID = "allenai/qasper"
PINNED_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"
SOURCE_URL_HOSTS = frozenset({"huggingface.co", "qasper-dataset.s3.us-west-2.amazonaws.com"})
PARTITIONS = ("train", "validation", "test")
OUTCOME_FIELDS = ("unanswerable", "extractive_spans", "yes_no", "free_form_answer")
PAPER_FIELDS = ("id", "title", "abstract", "full_text", "qas", "figures_and_tables")
QA_FIELDS = (
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
MAX_SINGLE_PAPER_CHARS = 128 * 1024 * 1024
MAX_DOCUMENT_CHARS = 8 * 1024 * 1024
MAX_TAR_MEMBERS = 10000
MAX_TAR_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 30
NETWORK_OBJECT_DEADLINE_SECONDS = 1800
GIB = 1024 ** 3


class AcquisitionError(shared.AcquisitionError):
    """A fail-closed QASPER acquisition or metadata-integrity error."""


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> urllib.request.Request | None:
        _validate_url(new_url)
        if urllib.parse.urlsplit(request.full_url).scheme != "https":
            raise AcquisitionError("refusing a redirect from a non-HTTPS request")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_url(url: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as exc:
        raise AcquisitionError("source metadata contains a malformed URL") from exc
    if (
        parsed.scheme != "https"
        or host not in SOURCE_URL_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise AcquisitionError("source URL is outside the anonymous HTTPS allowlist")
    return parsed


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        SafeRedirectHandler(),
    )


def _read_git(repo: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AcquisitionError("could not verify committed source-plan and code fingerprints") from exc
    if result.returncode != 0:
        raise AcquisitionError("source plan and acquisition code must be committed before acquisition")
    return result.stdout.strip()


def _assert_committed_inputs(repo: Path, relative_paths: Sequence[str]) -> str:
    root = _read_git(repo, ["rev-parse", "--show-toplevel"])
    if Path(root).resolve() != repo.resolve():
        raise AcquisitionError("repository root could not be verified")
    _read_git(repo, ["ls-files", "--error-unmatch", "--", *relative_paths])
    _read_git(repo, ["diff", "--quiet", "HEAD", "--", *relative_paths])
    _read_git(repo, ["diff", "--cached", "--quiet", "HEAD", "--", *relative_paths])
    return _read_git(repo, ["rev-parse", "HEAD"])


def _strict_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=shared._json_object_no_duplicate_keys,
            parse_constant=shared._reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AcquisitionError(f"{label} is not valid strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise AcquisitionError(f"{label} must be a JSON object")
    return value


def _load_frozen_inputs(repo: Path) -> tuple[dict[str, bytes], dict[str, Any], str]:
    relative_paths = (
        "research/endgame/neutral_benchmark_protocol.json",
        "research/endgame/neutral_paper_acquisition_sources.json",
        "research/endgame/neutral_paper_acquire.py",
        "research/endgame/neutral_acquire.py",
    )
    commit = _assert_committed_inputs(repo, relative_paths)
    blobs: dict[str, bytes] = {}
    parsed: dict[str, Any] = {}
    for relative in relative_paths:
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            raise AcquisitionError("a frozen input is absent or is not a regular file")
        data = path.read_bytes()
        blobs[relative] = data
        if relative.endswith(".json"):
            parsed[relative] = _strict_json_bytes(data, relative)
    protocol = parsed["research/endgame/neutral_benchmark_protocol.json"]
    plan = parsed["research/endgame/neutral_paper_acquisition_sources.json"]
    _validate_frozen_plan(plan, protocol)
    return blobs, parsed, commit


def _validate_frozen_plan(plan: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != "vey.endgame.neutral-benchmark-protocol.v2":
        raise AcquisitionError("unexpected neutral benchmark protocol schema")
    if plan.get("schema_version") != 1:
        raise AcquisitionError("unexpected QASPER acquisition plan schema")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise AcquisitionError("QASPER source pin is absent")
    if (
        source.get("id") != SOURCE_ID
        or source.get("hf_revision") != PINNED_REVISION
        or source.get("loader_version") != "0.3.0"
    ):
        raise AcquisitionError("QASPER source revision or loader version differs from the frozen pin")
    if Path(str(plan.get("root", ""))).is_absolute() is False:
        raise AcquisitionError("QASPER private data root must be an absolute path")
    archives = source.get("archives")
    if not isinstance(archives, list) or len(archives) != 2:
        raise AcquisitionError("QASPER archive registry must contain the two pinned archives")
    member_partitions: dict[str, str] = {}
    for archive in archives:
        if not isinstance(archive, dict) or not isinstance(archive.get("members"), dict):
            raise AcquisitionError("QASPER archive registry is malformed")
        _validate_url(str(archive.get("url", "")))
        for name, partition in archive["members"].items():
            if not isinstance(name, str) or partition not in PARTITIONS or name in member_partitions:
                raise AcquisitionError("QASPER selected-member map is malformed")
            member_partitions[name] = partition
    if set(member_partitions.values()) != set(PARTITIONS):
        raise AcquisitionError("QASPER archive pins do not cover train, validation, and test")
    _validate_url(str(source.get("loader_url", "")))
    _validate_url(str(source.get("card_url", "")))
    if PINNED_REVISION not in str(source["loader_url"]) or PINNED_REVISION not in str(source["card_url"]):
        raise AcquisitionError("QASPER metadata URLs are not pinned to the selected revision")
    limits = plan.get("resource_limits")
    expected_limits = {
        "cpu_threads_max": 4,
        "blas_threads": 1,
        "nice": 10,
        "ionice_class": 2,
        "ionice_priority": 7,
        "compressed_bytes_per_archive_max": 536870912,
        "selected_uncompressed_bytes_total_max": 2147483648,
        "minimum_available_ram_gib": 8,
        "minimum_free_disk_gib": 20,
        "gpu_use": False,
    }
    if limits != expected_limits:
        raise AcquisitionError("QASPER resource limits differ from the committed source plan")
    allocation = protocol.get("allocation_parameters")
    if not isinstance(allocation, dict):
        raise AcquisitionError("neutral protocol allocation parameters are absent")
    if allocation.get("seed") != "vey-neutral-v2:2026-10-04:1729":
        raise AcquisitionError("neutral allocation seed differs from the frozen protocol")
    caps = allocation.get("caps_independent_groups_per_source")
    if caps != {"train": 4096, "dev": 1024, "calibration": 1024, "confirmation": 8192}:
        raise AcquisitionError("neutral independent-group caps differ from the frozen protocol")


def _safe_private_root(root: Path) -> None:
    if not root.is_absolute():
        raise AcquisitionError("private data root must be absolute")
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise AcquisitionError("refusing a symlink in the private data path")
    shared.make_private_directory(root)


def _apply_resource_guards(root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    limits = plan["resource_limits"]
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[key] = ""
    cpu_count: int | None = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            raise AcquisitionError("no CPU affinity is available")
        selected = set(allowed[: int(limits["cpu_threads_max"])])
        os.sched_setaffinity(0, selected)
        cpu_count = len(selected)
    if cpu_count is None or cpu_count > int(limits["cpu_threads_max"]):
        raise AcquisitionError("could not enforce the four-CPU acquisition limit")
    try:
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if current_nice < int(limits["nice"]):
            os.nice(int(limits["nice"]) - current_nice)
        applied_nice = os.getpriority(os.PRIO_PROCESS, 0)
    except (AttributeError, OSError) as exc:
        raise AcquisitionError("could not apply the required nice priority") from exc
    if applied_nice < int(limits["nice"]):
        raise AcquisitionError("required nice priority was not applied")
    try:
        ionice = subprocess.run(
            ["ionice", "-c", str(limits["ionice_class"]), "-n", str(limits["ionice_priority"]), "-p", str(os.getpid())],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AcquisitionError("could not apply required ionice priority") from exc
    if ionice.returncode != 0:
        raise AcquisitionError("required ionice class/priority was not applied")
    available_bytes = _available_ram_bytes()
    minimum_ram = int(limits["minimum_available_ram_gib"]) * GIB
    if available_bytes < minimum_ram:
        raise AcquisitionError("available RAM is below the committed acquisition minimum")
    free_bytes = shutil.disk_usage(root).free
    minimum_disk = int(limits["minimum_free_disk_gib"]) * GIB
    if free_bytes < minimum_disk:
        raise AcquisitionError("free disk space is below the committed acquisition minimum")
    return {
        "cpu_affinity_count": cpu_count,
        "blas_threads": 1,
        "nice_value": applied_nice,
        "ionice_class": int(limits["ionice_class"]),
        "ionice_priority": int(limits["ionice_priority"]),
        "available_ram_bytes_at_start": available_bytes,
        "minimum_available_ram_bytes": minimum_ram,
        "free_disk_bytes_at_start": free_bytes,
        "minimum_free_disk_bytes": minimum_disk,
        "network_concurrency": 1,
        "gpu_access_requested": False,
    }


def _available_ram_bytes() -> int:
    try:
        with Path("/proc/meminfo").open("r", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    fields = line.split()
                    return int(fields[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError) as exc:
        raise AcquisitionError("could not determine available system RAM") from exc


def _create_attempt(root: Path) -> tuple[str, Path]:
    attempts = root / "attempts"
    shared.make_private_directory(attempts)
    for _ in range(8):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        attempt_id = f"{stamp}-{os.getpid()}-{secrets.token_hex(4)}"
        path = attempts / attempt_id
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            continue
        if path.is_symlink() or not path.is_dir() or (path.stat().st_mode & 0o077):
            raise AcquisitionError("attempt directory is not private")
        shared.fsync_directory(attempts)
        return attempt_id, path
    raise AcquisitionError("could not create a unique timestamped attempt directory")


def _snapshot_frozen_inputs(attempt: Path, blobs: Mapping[str, bytes], commit: str) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for relative, data in sorted(blobs.items()):
        destination = attempt / "frozen_inputs" / relative
        shared.write_exclusive(destination, data)
        records[relative] = {"sha256": shared.sha256_bytes(data), "bytes": len(data)}
    record = {
        "repository_commit": commit,
        "files": records,
        "frozen_at_utc": _now_utc(),
    }
    shared.write_json_immutable(attempt / "frozen_inputs.json", record)
    return record


def _download_file(
    url: str,
    destination: Path,
    max_bytes: int,
    *,
    deadline_seconds: int = NETWORK_OBJECT_DEADLINE_SECONDS,
) -> dict[str, Any]:
    _validate_url(url)
    shared.make_private_directory(destination.parent)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "vey-neutral-paper-acquire/1", "Accept-Encoding": "identity"},
        method="GET",
    )
    opener = _opener()
    started = time.monotonic()
    try:
        response = opener.open(request, timeout=NETWORK_TIMEOUT_SECONDS)
    except (OSError, urllib.error.URLError) as exc:
        raise AcquisitionError("anonymous source request failed") from exc
    with response:
        final_url = response.geturl()
        _validate_url(final_url)
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                announced = int(content_length)
            except ValueError as exc:
                raise AcquisitionError("source response has an invalid Content-Length") from exc
            if announced < 0 or announced > max_bytes:
                raise AcquisitionError("source response exceeds its pinned byte limit")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(destination, flags, 0o600)
        except FileExistsError as exc:
            raise AcquisitionError("refusing to overwrite an acquisition artifact") from exc
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb", closefd=False) as output:
                while True:
                    if time.monotonic() - started > deadline_seconds:
                        raise AcquisitionError("source request exceeded its bounded deadline")
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    size += len(block)
                    if size > max_bytes:
                        raise AcquisitionError("source response exceeded its pinned byte limit")
                    digest.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
        finally:
            os.close(fd)
        if content_length is not None and size != int(content_length):
            raise AcquisitionError("source response length differs from Content-Length")
        shared.fsync_directory(destination.parent)
    return {
        "requested_url": url,
        "final_url": final_url,
        "sha256": digest.hexdigest(),
        "bytes": size,
        "relative_path": str(destination),
    }


def _check_pinned_metadata(source: Mapping[str, Any], loader: bytes, card: bytes) -> dict[str, Any]:
    try:
        loader_text = loader.decode("utf-8")
        card_text = card.decode("utf-8")
    except UnicodeError as exc:
        raise AcquisitionError("pinned QASPER metadata is not UTF-8") from exc
    required_loader_fragments = (
        '_VERSION = "0.3.0"',
        '_LICENSE = "CC BY 4.0"',
        str(source["archives"][0]["url"]),
        str(source["archives"][1]["url"]),
        "qasper-train-v0.3.json",
        "qasper-dev-v0.3.json",
        "qasper-test-v0.3.json",
        "highlighted_evidence",
        "yes_no",
    )
    if not all(fragment in loader_text for fragment in required_loader_fragments):
        raise AcquisitionError("pinned QASPER loader no longer matches the source schema or grant")
    if "license:\n- cc-by-4.0" not in card_text or "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0)" not in card_text:
        raise AcquisitionError("pinned QASPER card no longer declares the recorded CC-BY-4.0 grant")
    if "@inproceedings{Dasigi2021ADO" not in loader_text or "QASPER" not in card_text:
        raise AcquisitionError("pinned QASPER citation metadata is absent")
    return {
        "loader_version": "0.3.0",
        "loader_license_declaration": "CC BY 4.0",
        "card_license_declaration": "cc-by-4.0",
        "dataset_citation": "Dasigi et al. (2021), A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
        "license_url": "https://creativecommons.org/licenses/by/4.0",
        "checked_schema_fields": list(PAPER_FIELDS[1:]) + list(QA_FIELDS) + list(ANSWER_FIELDS),
    }


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _text_values(item)


def _document_text(paper: Mapping[str, Any]) -> str:
    pieces: list[str] = []
    for field in ("title", "abstract"):
        value = paper.get(field)
        if isinstance(value, str):
            pieces.append(value)
    full_text = paper.get("full_text")
    if isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, dict):
                pieces.extend(_text_values(section.get("section_name")))
                pieces.extend(_text_values(section.get("paragraphs")))
    elif isinstance(full_text, dict):
        names = full_text.get("section_name")
        paragraphs = full_text.get("paragraphs")
        if isinstance(names, list) or isinstance(paragraphs, list):
            names_list = names if isinstance(names, list) else []
            paragraphs_list = paragraphs if isinstance(paragraphs, list) else []
            for index in range(max(len(names_list), len(paragraphs_list))):
                if index < len(names_list):
                    pieces.extend(_text_values(names_list[index]))
                if index < len(paragraphs_list):
                    pieces.extend(_text_values(paragraphs_list[index]))
    figures = paper.get("figures_and_tables")
    if isinstance(figures, list):
        for figure in figures:
            if isinstance(figure, dict):
                pieces.extend(_text_values(figure.get("caption")))
    return "\n".join(pieces)


def _canonical_arxiv_id(paper_id: str) -> str | None:
    match = ARXIV_ID.search(paper_id)
    if match is None:
        return None
    value = match.group(0).lower()
    return re.sub(r"v\d+$", "", value)


def _json_field(obj: Mapping[str, Any], field: str) -> tuple[int, str | None]:
    if field not in obj:
        return 0, None
    return 1, shared.canonical_json(obj[field])


class SchemaCensus:
    def __init__(self) -> None:
        self.parents: Counter[tuple[str, str]] = Counter()
        self.fields: Counter[tuple[str, str, str, str]] = Counter()

    def observe(self, partition: str, scope: str, obj: Mapping[str, Any], fields: Sequence[str]) -> None:
        self.parents[(partition, scope)] += 1
        for field in fields:
            kind = shared._value_type(obj[field]) if field in obj else "missing"
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


def _stream_json_object(path: Path) -> Iterator[tuple[str, Any]]:
    decoder = json.JSONDecoder(
        object_pairs_hook=shared._json_object_no_duplicate_keys,
        parse_constant=shared._reject_json_constant,
    )
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise AcquisitionError("selected QASPER member cannot be opened") from exc
    with stream:
        buffer = ""
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

        def skip_ws(position: int) -> int:
            nonlocal buffer
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return position
                fill()

        def char_at(position: int) -> str | None:
            while position >= len(buffer) and not eof:
                fill()
            return buffer[position] if position < len(buffer) else None

        def decode(position: int) -> tuple[Any, int]:
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    if end - position > MAX_SINGLE_PAPER_CHARS:
                        raise AcquisitionError("one QASPER paper JSON object exceeds the parser bound")
                    if end == len(buffer) and not eof:
                        fill()
                        continue
                    return value, end
                except AcquisitionError:
                    raise
                except json.JSONDecodeError as exc:
                    if not eof:
                        if len(buffer) - position > MAX_SINGLE_PAPER_CHARS:
                            raise AcquisitionError("one QASPER paper JSON object exceeds the parser bound") from exc
                        fill()
                        continue
                    raise AcquisitionError("selected QASPER member contains invalid strict JSON") from exc
                except ValueError as exc:
                    raise AcquisitionError("selected QASPER member contains invalid strict JSON") from exc

        position = skip_ws(0)
        if char_at(position) != "{":
            raise AcquisitionError("selected QASPER member must be a top-level JSON object")
        position += 1
        seen_keys: set[str] = set()
        while True:
            position = skip_ws(position)
            current = char_at(position)
            if current == "}":
                if seen_keys:
                    raise AcquisitionError("selected QASPER member has a trailing mapping comma")
                position += 1
                position = skip_ws(position)
                while not eof:
                    if position < len(buffer):
                        raise AcquisitionError("selected QASPER member has trailing data")
                    fill()
                    position = skip_ws(0)
                if position < len(buffer):
                    raise AcquisitionError("selected QASPER member has trailing data")
                return
            if current is None:
                raise AcquisitionError("selected QASPER member has an unterminated top-level object")
            key, end_key = decode(position)
            if not isinstance(key, str) or key in seen_keys:
                raise AcquisitionError("selected QASPER member has a non-string or duplicate paper key")
            seen_keys.add(key)
            colon = skip_ws(end_key)
            if char_at(colon) != ":":
                raise AcquisitionError("selected QASPER member has malformed paper mapping syntax")
            value_position = skip_ws(colon + 1)
            value, end_value = decode(value_position)
            delimiter_position = skip_ws(end_value)
            delimiter = char_at(delimiter_position)
            if delimiter not in (",", "}"):
                raise AcquisitionError("selected QASPER member has malformed paper mapping syntax")
            next_position = delimiter_position + 1
            yield key, value
            if delimiter == ",":
                buffer = buffer[next_position:]
                position = 0
                continue
            buffer = buffer[next_position:]
            position = skip_ws(0)
            while not eof:
                if position < len(buffer):
                    raise AcquisitionError("selected QASPER member has trailing data")
                fill()
                position = skip_ws(0)
            if position < len(buffer):
                raise AcquisitionError("selected QASPER member has trailing data")
            return


def _initialize_database(path: Path) -> sqlite3.Connection:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    os.close(fd)
    connection = sqlite3.connect(path)
    os.chmod(path, 0o600)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.executescript(
        """
        CREATE TABLE papers(
          paper_pk INTEGER PRIMARY KEY,
          source_partition TEXT NOT NULL,
          paper_id TEXT NOT NULL,
          canonical_arxiv_id TEXT,
          title_json TEXT,
          abstract_json TEXT,
          full_text_json TEXT,
          figures_json TEXT,
          model_state_json TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          full_document_sha256 TEXT NOT NULL,
          normalized_document_sha256 TEXT NOT NULL,
          normalized_document_text TEXT NOT NULL,
          normalized_title TEXT,
          normalized_title_sha256 TEXT,
          group_id TEXT NOT NULL,
          question_count INTEGER NOT NULL,
          annotation_count INTEGER NOT NULL,
          nonnull_outcome_annotation_count INTEGER NOT NULL
        );
        CREATE INDEX papers_partition ON papers(source_partition,paper_id);
        CREATE INDEX papers_group ON papers(group_id);
        CREATE INDEX papers_arxiv ON papers(canonical_arxiv_id,group_id);
        CREATE INDEX papers_title ON papers(normalized_title_sha256,group_id);
        CREATE INDEX papers_document ON papers(full_document_sha256,group_id);
        CREATE TABLE questions(
          question_pk INTEGER PRIMARY KEY,
          paper_pk INTEGER NOT NULL REFERENCES papers(paper_pk),
          ordinal INTEGER NOT NULL,
          question_id_present INTEGER NOT NULL,
          question_id_json TEXT,
          question_type TEXT NOT NULL,
          question_json TEXT NOT NULL,
          annotation_count INTEGER NOT NULL
        );
        CREATE INDEX questions_paper ON questions(paper_pk,ordinal);
        CREATE TABLE annotations(
          annotation_pk INTEGER PRIMARY KEY,
          paper_pk INTEGER NOT NULL REFERENCES papers(paper_pk),
          question_pk INTEGER NOT NULL REFERENCES questions(question_pk),
          ordinal INTEGER NOT NULL,
          annotation_type TEXT NOT NULL,
          annotation_id_json TEXT,
          worker_id_json TEXT,
          annotation_json TEXT NOT NULL,
          answer_present INTEGER NOT NULL,
          answer_json TEXT,
          unanswerable_present INTEGER NOT NULL,
          unanswerable_json TEXT,
          extractive_spans_present INTEGER NOT NULL,
          extractive_spans_json TEXT,
          yes_no_present INTEGER NOT NULL,
          yes_no_json TEXT,
          free_form_answer_present INTEGER NOT NULL,
          free_form_answer_json TEXT,
          evidence_present INTEGER NOT NULL,
          evidence_json TEXT,
          highlighted_evidence_present INTEGER NOT NULL,
          highlighted_evidence_json TEXT,
          nonnull_outcome_annotation INTEGER NOT NULL
        );
        CREATE INDEX annotations_question ON annotations(question_pk,ordinal);
        CREATE TABLE source_groups(
          group_id TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          pinned_revision TEXT NOT NULL,
          official_partition TEXT NOT NULL,
          lineage_ids_json TEXT NOT NULL,
          input_hashes_json TEXT NOT NULL,
          rank_sha256 TEXT NOT NULL,
          component_id TEXT,
          final_split TEXT,
          allocation_reason TEXT,
          exclusion_reason TEXT
        );
        CREATE INDEX source_groups_component ON source_groups(component_id,group_id);
        CREATE TABLE group_partitions(
          group_id TEXT NOT NULL REFERENCES source_groups(group_id),
          official_partition TEXT NOT NULL,
          paper_count INTEGER NOT NULL,
          question_count INTEGER NOT NULL,
          annotation_count INTEGER NOT NULL,
          nonnull_outcome_annotation_count INTEGER NOT NULL,
          PRIMARY KEY(group_id,official_partition)
        ) WITHOUT ROWID;
        CREATE TABLE doc_groups(
          doc_hash TEXT NOT NULL,
          group_id TEXT NOT NULL REFERENCES source_groups(group_id),
          PRIMARY KEY(doc_hash,group_id)
        ) WITHOUT ROWID;
        CREATE INDEX doc_groups_group ON doc_groups(group_id,doc_hash);
        CREATE TABLE docs(
          doc_id INTEGER PRIMARY KEY,
          doc_hash TEXT NOT NULL UNIQUE,
          normalized_text TEXT NOT NULL,
          representative_group TEXT NOT NULL,
          gram_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE doc_grams(
          doc_id INTEGER NOT NULL,
          gram TEXT NOT NULL,
          PRIMARY KEY(doc_id,gram)
        ) WITHOUT ROWID;
        CREATE INDEX doc_grams_by_gram ON doc_grams(gram,doc_id);
        CREATE TABLE gram_frequency(
          gram TEXT PRIMARY KEY,
          document_frequency INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX gram_frequency_order ON gram_frequency(document_frequency,gram);
        CREATE TABLE prefixes(
          doc_id INTEGER NOT NULL,
          gram TEXT NOT NULL,
          PRIMARY KEY(doc_id,gram)
        ) WITHOUT ROWID;
        CREATE INDEX prefixes_by_gram ON prefixes(gram,doc_id);
        CREATE TABLE search_exclusions(
          doc_hash TEXT PRIMARY KEY,
          representative_group TEXT NOT NULL,
          normalized_codepoints INTEGER NOT NULL,
          reason TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE edges(
          edge_key TEXT PRIMARY KEY,
          group_a TEXT NOT NULL,
          group_b TEXT NOT NULL,
          reason TEXT NOT NULL,
          evidence_hash_a TEXT NOT NULL,
          evidence_hash_b TEXT,
          intersection_size INTEGER,
          union_size INTEGER
        );
        CREATE INDEX edges_groups ON edges(group_a,group_b);
        CREATE TABLE schema_parents(
          source_partition TEXT NOT NULL,
          scope TEXT NOT NULL,
          parent_objects INTEGER NOT NULL,
          PRIMARY KEY(source_partition,scope)
        ) WITHOUT ROWID;
        CREATE TABLE schema_counts(
          source_partition TEXT NOT NULL,
          scope TEXT NOT NULL,
          field_name TEXT NOT NULL,
          json_type TEXT NOT NULL,
          value_count INTEGER NOT NULL,
          PRIMARY KEY(source_partition,scope,field_name,json_type)
        ) WITHOUT ROWID;
        CREATE TABLE components(
          component_id TEXT PRIMARY KEY,
          groups_json TEXT NOT NULL,
          group_count INTEGER NOT NULL,
          paper_count INTEGER NOT NULL,
          question_count INTEGER NOT NULL,
          annotation_count INTEGER NOT NULL,
          final_split TEXT NOT NULL,
          allocation_reason TEXT NOT NULL,
          rank_anchor_sha256 TEXT,
          rank_anchor_group_id TEXT
        );
        CREATE TABLE acquisition_metadata(
          key TEXT PRIMARY KEY,
          value_json TEXT NOT NULL
        ) WITHOUT ROWID;
        """
    )
    shared.fsync_directory(path.parent)
    return connection


def _insert_schema_census(connection: sqlite3.Connection, census: SchemaCensus) -> None:
    connection.executemany(
        "INSERT INTO schema_parents(source_partition,scope,parent_objects) VALUES(?,?,?)",
        ((part, scope, count) for (part, scope), count in sorted(census.parents.items())),
    )
    connection.executemany(
        "INSERT INTO schema_counts(source_partition,scope,field_name,json_type,value_count) VALUES(?,?,?,?,?)",
        ((part, scope, field, kind, count) for (part, scope, field, kind), count in sorted(census.fields.items())),
    )


def _insert_paper(
    connection: sqlite3.Connection,
    census: SchemaCensus,
    partition: str,
    paper_id: str,
    paper: Mapping[str, Any],
    pinned_revision: str,
) -> dict[str, int]:
    census.observe(partition, "paper", paper, PAPER_FIELDS[1:])
    raw_qas = paper.get("qas")
    qas = raw_qas if isinstance(raw_qas, list) else []
    question_count = len(qas)
    annotation_count = 0
    nonnull_outcome_count = 0
    question_staging: list[tuple[Any, int, list[tuple[Any, int, bool]]]] = []
    for q_index, question in enumerate(qas):
        answer_staging: list[tuple[Any, int, bool]] = []
        if isinstance(question, dict):
            census.observe(partition, "question", question, QA_FIELDS)
            raw_answers = question.get("answers")
            answers = raw_answers if isinstance(raw_answers, list) else []
            q_id_present, q_id_json = _json_field(question, "question_id")
        else:
            census.observe_value(partition, "question_item", shared._value_type(question))
            answers = []
            q_id_present, q_id_json = 0, None
        for annotation in answers:
            annotation_count += 1
            if isinstance(annotation, dict):
                census.observe(partition, "annotation", annotation, ANNOTATION_FIELDS)
                answer_present, answer_json = _json_field(annotation, "answer")
                answer = annotation.get("answer")
                has_nonnull_outcome = False
                if isinstance(answer, dict):
                    census.observe(partition, "answer", answer, ANSWER_FIELDS)
                    has_nonnull_outcome = any(field in answer and answer[field] is not None for field in OUTCOME_FIELDS)
                else:
                    census.observe_value(partition, "answer_value", shared._value_type(answer) if answer_present else "missing")
                if has_nonnull_outcome:
                    nonnull_outcome_count += 1
                answer_staging.append((annotation, answer_present, has_nonnull_outcome))
            else:
                census.observe_value(partition, "annotation_item", shared._value_type(annotation))
                answer_staging.append((annotation, 0, False))
        question_staging.append((question, q_id_present, answer_staging))

    figures = paper.get("figures_and_tables")
    if isinstance(figures, list):
        for figure in figures:
            if isinstance(figure, dict):
                census.observe(partition, "figure_or_table", figure, FIGURE_FIELDS)
            else:
                census.observe_value(partition, "figure_or_table_item", shared._value_type(figure))
    full_text = paper.get("full_text")
    if isinstance(full_text, list):
        for section in full_text:
            if isinstance(section, dict):
                census.observe(partition, "section", section, SECTION_FIELDS)
            else:
                census.observe_value(partition, "section_item", shared._value_type(section))
    elif isinstance(full_text, dict):
        census.observe(partition, "full_text_columnar", full_text, SECTION_FIELDS)

    doc_text = _document_text(paper)
    normalized_document = _normalize_text(doc_text)
    if len(normalized_document) > MAX_DOCUMENT_CHARS:
        raise AcquisitionError("normalized QASPER full-document text exceeds the grouping bound")
    document_fields = {
        field: paper[field]
        for field in ("title", "abstract", "full_text", "figures_and_tables")
        if field in paper
    }
    full_document_hash = shared.sha256_bytes(shared.canonical_json(document_fields).encode("utf-8"))
    normalized_document_hash = shared.sha256_bytes(normalized_document.encode("utf-8"))
    title = paper.get("title")
    normalized_title = _normalize_text(title) if isinstance(title, str) else ""
    title_hash = shared.sha256_bytes(normalized_title.encode("utf-8")) if normalized_title else None
    canonical_arxiv = _canonical_arxiv_id(paper_id)
    lineage_ids = [f"qasper-paper:{paper_id.casefold()}"]
    if canonical_arxiv is not None:
        lineage_ids.append(f"arxiv:{canonical_arxiv}")
    lineage_ids = sorted(set(lineage_ids))
    input_hashes = [normalized_document_hash]
    group_id = shared.canonical_sha256([SOURCE_ID, pinned_revision, lineage_ids, input_hashes])
    rank_hash = shared._rank_hash(SOURCE_ID, group_id)
    existing = connection.execute("SELECT normalized_text FROM docs WHERE doc_hash=?", (normalized_document_hash,)).fetchone()
    if existing is not None and existing[0] != normalized_document:
        raise AcquisitionError("SHA-256 collision detected among normalized full-document texts")
    priority = {"train": 0, "validation": 1, "test": 2}
    group_partition = connection.execute("SELECT official_partition FROM source_groups WHERE group_id=?", (group_id,)).fetchone()
    if group_partition is None:
        connection.execute(
            "INSERT INTO source_groups(group_id,source_id,pinned_revision,official_partition,lineage_ids_json,input_hashes_json,rank_sha256) VALUES(?,?,?,?,?,?,?)",
            (group_id, SOURCE_ID, pinned_revision, partition, shared.canonical_json(lineage_ids), shared.canonical_json(input_hashes), rank_hash),
        )
    elif priority[partition] > priority[group_partition[0]]:
        connection.execute("UPDATE source_groups SET official_partition=? WHERE group_id=?", (partition, group_id))
    connection.execute(
        "INSERT INTO group_partitions(group_id,official_partition,paper_count,question_count,annotation_count,nonnull_outcome_annotation_count) VALUES(?,?,1,?,?,?) "
        "ON CONFLICT(group_id,official_partition) DO UPDATE SET paper_count=paper_count+1,question_count=question_count+excluded.question_count,annotation_count=annotation_count+excluded.annotation_count,nonnull_outcome_annotation_count=nonnull_outcome_annotation_count+excluded.nonnull_outcome_annotation_count",
        (group_id, partition, question_count, annotation_count, nonnull_outcome_count),
    )
    doc = connection.execute("SELECT doc_id FROM docs WHERE doc_hash=?", (normalized_document_hash,)).fetchone()
    if doc is None:
        connection.execute(
            "INSERT INTO docs(doc_hash,normalized_text,representative_group) VALUES(?,?,?)",
            (normalized_document_hash, normalized_document, group_id),
        )
    else:
        connection.execute(
            "UPDATE docs SET representative_group=MIN(representative_group,?) WHERE doc_hash=?",
            (group_id, normalized_document_hash),
        )
    connection.execute("INSERT OR IGNORE INTO doc_groups(doc_hash,group_id) VALUES(?,?)", (normalized_document_hash, group_id))

    state = {field: paper[field] for field in ("title", "abstract", "full_text") if field in paper}
    title_present, title_json = _json_field(paper, "title")
    abstract_present, abstract_json = _json_field(paper, "abstract")
    full_text_present, full_text_json = _json_field(paper, "full_text")
    figures_present, figures_json = _json_field(paper, "figures_and_tables")
    del title_present, abstract_present, full_text_present, figures_present
    paper_cursor = connection.execute(
        "INSERT INTO papers(source_partition,paper_id,canonical_arxiv_id,title_json,abstract_json,full_text_json,figures_json,model_state_json,payload_json,full_document_sha256,normalized_document_sha256,normalized_document_text,normalized_title,normalized_title_sha256,group_id,question_count,annotation_count,nonnull_outcome_annotation_count) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            partition, paper_id, canonical_arxiv, title_json, abstract_json, full_text_json, figures_json,
            shared.canonical_json(state), shared.canonical_json(paper), full_document_hash,
            normalized_document_hash, normalized_document, normalized_title or None, title_hash, group_id,
            question_count, annotation_count, nonnull_outcome_count,
        ),
    )
    paper_pk = int(paper_cursor.lastrowid)
    for q_index, (question, question_id_present, answer_staging) in enumerate(question_staging):
        if isinstance(question, dict):
            _, question_id_json = _json_field(question, "question_id")
        else:
            question_id_json = None
        question_cursor = connection.execute(
            "INSERT INTO questions(paper_pk,ordinal,question_id_present,question_id_json,question_type,question_json,annotation_count) VALUES(?,?,?,?,?,?,?)",
            (paper_pk, q_index, question_id_present, question_id_json, shared._value_type(question), shared.canonical_json(question), len(answer_staging)),
        )
        question_pk = int(question_cursor.lastrowid)
        for answer_index, (annotation, answer_present, has_nonnull_outcome) in enumerate(answer_staging):
            ann_obj = annotation if isinstance(annotation, dict) else {}
            answer_obj = ann_obj.get("answer") if isinstance(ann_obj.get("answer"), dict) else {}
            answer_present_actual, answer_json = _json_field(ann_obj, "answer")
            fields = {
                field: _json_field(answer_obj, field)
                for field in ANSWER_FIELDS
            }
            ann_id = ann_obj.get("annotation_id") if "annotation_id" in ann_obj else None
            worker_id = ann_obj.get("worker_id") if "worker_id" in ann_obj else None
            connection.execute(
                "INSERT INTO annotations(paper_pk,question_pk,ordinal,annotation_type,annotation_id_json,worker_id_json,annotation_json,answer_present,answer_json,unanswerable_present,unanswerable_json,extractive_spans_present,extractive_spans_json,yes_no_present,yes_no_json,free_form_answer_present,free_form_answer_json,evidence_present,evidence_json,highlighted_evidence_present,highlighted_evidence_json,nonnull_outcome_annotation) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    paper_pk, question_pk, answer_index, shared._value_type(annotation),
                    shared.canonical_json(ann_id) if isinstance(annotation, dict) and "annotation_id" in ann_obj else None,
                    shared.canonical_json(worker_id) if isinstance(annotation, dict) and "worker_id" in ann_obj else None,
                    shared.canonical_json(annotation), answer_present_actual, answer_json,
                    *fields["unanswerable"], *fields["extractive_spans"], *fields["yes_no"],
                    *fields["free_form_answer"], *fields["evidence"], *fields["highlighted_evidence"],
                    int(has_nonnull_outcome),
                ),
            )
    return {
        "papers": 1,
        "questions": question_count,
        "annotations": annotation_count,
        "nonnull_outcome_annotations": nonnull_outcome_count,
    }


def _group_edges_from_paper_keys(connection: sqlite3.Connection) -> None:
    relations = (
        ("canonical_arxiv_id", "same_canonical_arxiv_version_family"),
        ("normalized_title_sha256", "same_normalized_complete_title"),
        ("full_document_sha256", "same_exact_full_document_sha256"),
    )
    for column, reason in relations:
        cursor = connection.execute(
            f"SELECT DISTINCT {column},group_id FROM papers WHERE {column} IS NOT NULL AND {column}<>'' ORDER BY {column},group_id"
        )
        current_key: str | None = None
        representative: str | None = None
        for key, group_id in cursor:
            if key != current_key:
                current_key, representative = key, group_id
                continue
            if representative != group_id:
                evidence_hash = shared.sha256_bytes(str(key).encode("utf-8"))
                shared._add_group_edge(connection, representative or group_id, group_id, reason, evidence_hash, evidence_hash)
    cursor = connection.execute("SELECT doc_hash,group_id FROM doc_groups ORDER BY doc_hash,group_id")
    current_hash: str | None = None
    representative = None
    for doc_hash, group_id in cursor:
        if doc_hash != current_hash:
            current_hash, representative = doc_hash, group_id
            continue
        if representative != group_id:
            shared._add_group_edge(connection, representative or group_id, group_id, "exact_normalized_full_document_text", doc_hash)
    connection.commit()


def _manifest_hash(connection: sqlite3.Connection, query: str, parameters: Sequence[Any] = ()) -> str:
    rows = [list(row) for row in connection.execute(query, parameters)]
    return shared.canonical_sha256(rows)


def _allocate_components(
    connection: sqlite3.Connection,
    component_id_by_group: Mapping[str, str],
    component_groups: Mapping[str, Sequence[str]],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    allocation_parameters = protocol["allocation_parameters"]
    seed = str(allocation_parameters["seed"])
    caps = {key: int(value) for key, value in allocation_parameters["caps_independent_groups_per_source"].items()}
    group_rows: dict[str, dict[str, Any]] = {}
    partitions_by_group: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for group_id, rank_hash, partition in connection.execute(
        "SELECT group_id,rank_sha256,official_partition FROM source_groups ORDER BY group_id"
    ):
        group_rows[group_id] = {"group_id": group_id, "rank_sha256": rank_hash}
    for row in connection.execute(
        "SELECT group_id,official_partition,paper_count,question_count,annotation_count,nonnull_outcome_annotation_count FROM group_partitions ORDER BY group_id,official_partition"
    ):
        partitions_by_group[row[0]][row[1]] = {
            "paper_count": int(row[2]),
            "question_count": int(row[3]),
            "annotation_count": int(row[4]),
            "labeled_annotation_count": int(row[5]),
        }
    components: dict[str, dict[str, Any]] = {
        component_id: {
            "groups": list(groups),
            "partitions": defaultdict(lambda: {"paper_count": 0, "question_count": 0, "annotation_count": 0, "labeled_annotation_count": 0}),
        }
        for component_id, groups in component_groups.items()
    }
    for group_id, partitions in partitions_by_group.items():
        component = components[component_id_by_group[group_id]]
        for partition, counts in partitions.items():
            target = component["partitions"][partition]
            for field, value in counts.items():
                target[field] += value

    has_labeled_test = any(
        values.get("test", {}).get("labeled_annotation_count", 0) > 0
        for values in (component["partitions"] for component in components.values())
    )
    assigned: dict[str, str] = {}
    reasons: dict[str, str] = {}
    anchors: dict[str, tuple[str, str]] = {}
    unused: dict[str, str] = {}
    used = {split: 0 for split in ("train", "dev", "calibration", "confirmation")}

    def ranked_group(component_id: str, partition: str, *, labeled_only: bool = True) -> tuple[str, str] | None:
        candidates: list[tuple[str, str]] = []
        for group_id in components[component_id]["groups"]:
            stats = partitions_by_group[group_id].get(partition)
            if stats is None or (labeled_only and stats["labeled_annotation_count"] == 0):
                continue
            candidates.append((group_rows[group_id]["rank_sha256"], group_id))
        return min(candidates) if candidates else None

    def component_fits(split: str) -> bool:
        return used[split] < caps[split]

    def assign(component_id: str, split: str, reason: str, anchor: tuple[str, str] | None) -> None:
        assigned[component_id] = split
        reasons[component_id] = reason
        if anchor is not None:
            anchors[component_id] = anchor
        used[split] += 1

    test_candidates: list[tuple[tuple[str, str], str]] = []
    validation_candidates: list[tuple[tuple[str, str], str]] = []
    train_candidates: list[tuple[tuple[str, str], str]] = []
    for component_id, component in components.items():
        parts = component["partitions"]
        if "test" in parts:
            test_anchor = ranked_group(component_id, "test")
            if test_anchor is not None:
                test_candidates.append((test_anchor, component_id))
            else:
                unused[component_id] = "official_test_component_without_nonnull_source_annotation"
            continue
        if "validation" in parts:
            validation_anchor = ranked_group(component_id, "validation")
            if validation_anchor is None:
                unused[component_id] = "official_validation_component_without_nonnull_source_annotation"
            elif has_labeled_test:
                validation_candidates.append((validation_anchor, component_id))
            else:
                test_candidates.append((validation_anchor, component_id))
            continue
        if "train" in parts:
            train_anchor = ranked_group(component_id, "train")
            if train_anchor is not None:
                train_candidates.append((train_anchor, component_id))
            else:
                unused[component_id] = "official_train_component_without_nonnull_source_annotation"
        else:
            unused[component_id] = "no_eligible_official_partition"

    for anchor, component_id in sorted(test_candidates):
        if component_fits("confirmation"):
            reason = "official_labeled_test_confirmation_only" if has_labeled_test and "test" in components[component_id]["partitions"] else "official_labeled_validation_confirmation_no_labeled_test"
            assign(component_id, "confirmation", reason, anchor)
        else:
            unused[component_id] = "confirmation_cap_reached"

    for ordinal, (anchor, component_id) in enumerate(sorted(validation_candidates)):
        split = "dev" if ordinal % 2 == 0 else "calibration"
        if component_fits(split):
            assign(component_id, split, "ranked_official_validation_alternating_dev_calibration", anchor)
        else:
            unused[component_id] = f"{split}_cap_reached"

    priority = {"calibration": 0, "dev": 1, "train": 2}
    recommendations: list[tuple[tuple[str, str], str, str]] = []
    for ordinal, (anchor, component_id) in enumerate(sorted(train_candidates)):
        slot = ordinal % 10
        desired = "dev" if slot in (0, 1) else "calibration" if slot in (2, 3) else "train"
        recommendations.append((anchor, component_id, desired))
    for anchor, component_id, split in sorted(recommendations, key=lambda item: (item[0], item[1])):
        if component_fits(split):
            assign(component_id, split, "ranked_official_train_sha_rank_mod_10", anchor)
        else:
            unused[component_id] = f"{split}_cap_reached"

    for component_id, component in components.items():
        if component_id not in assigned and component_id not in unused:
            unused[component_id] = "no_eligible_source_partition"
        final_split = assigned.get(component_id, "unused")
        reason = reasons[component_id] if component_id in assigned else unused[component_id]
        anchor = anchors.get(component_id)
        group_ids = list(component["groups"])
        papers, questions, annotations = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(question_count),0),COALESCE(SUM(annotation_count),0) FROM papers WHERE group_id IN (SELECT value FROM json_each(?))",
            (shared.canonical_json(group_ids),),
        ).fetchone()
        connection.execute(
            "INSERT INTO components(component_id,groups_json,group_count,paper_count,question_count,annotation_count,final_split,allocation_reason,rank_anchor_sha256,rank_anchor_group_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (component_id, shared.canonical_json(group_ids), len(group_ids), int(papers), int(questions), int(annotations), final_split, reason, anchor[0] if anchor else None, anchor[1] if anchor else None),
        )
        for group_id in group_ids:
            connection.execute(
                "UPDATE source_groups SET final_split=?,allocation_reason=?,exclusion_reason=? WHERE group_id=?",
                (final_split, reason if final_split != "unused" else None, reason if final_split == "unused" else None, group_id),
            )
    connection.commit()
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "dev", "calibration", "confirmation", "unused"):
        row = connection.execute(
            "SELECT COUNT(DISTINCT g.component_id),COUNT(DISTINCT g.group_id),COUNT(DISTINCT p.paper_pk),COALESCE(SUM(p.question_count),0),COALESCE(SUM(p.annotation_count),0) "
            "FROM source_groups g LEFT JOIN papers p ON p.group_id=g.group_id WHERE g.final_split=?",
            (split,),
        ).fetchone()
        split_counts[split] = {
            "connected_components": int(row[0]),
            "source_groups": int(row[1]),
            "papers": int(row[2]),
            "questions": int(row[3]),
            "answer_annotation_records": int(row[4]),
        }
    group_split_hashes = {}
    for split in ("train", "dev", "calibration", "confirmation", "unused"):
        ids = [row[0] for row in connection.execute(
            "SELECT group_id FROM source_groups WHERE final_split=? ORDER BY rank_sha256,group_id", (split,)
        )]
        group_split_hashes[split] = {"source_group_count": len(ids), "ordered_group_ids_sha256": shared.canonical_sha256(ids)}
    assignment_rows_hash = _manifest_hash(
        connection,
        "SELECT group_id,component_id,rank_sha256,final_split,allocation_reason,exclusion_reason FROM source_groups ORDER BY group_id",
    )
    component_rows_hash = _manifest_hash(
        connection,
        "SELECT component_id,groups_json,final_split,allocation_reason,rank_anchor_sha256,rank_anchor_group_id FROM components ORDER BY component_id",
    )
    return {
        "seed": seed,
        "purpose": shared.ALLOCATION_PURPOSE,
        "caps_independent_groups_per_source": caps,
        "official_partition_precedence": "labeled official test -> confirmation; otherwise labeled official validation -> confirmation; with labeled test, validation alternates dev/calibration; train-only labeled components use rank mod 10; unlabeled or overflow components remain unused",
        "allocation_uses": "source partition, source annotation-presence schema, and input-only group/rank hashes; no answer values, question text, worker IDs, or model outputs",
        "has_labeled_test_partition": has_labeled_test,
        "counts_by_final_split": split_counts,
        "group_split_hashes": group_split_hashes,
        "group_assignment_rows_sha256": assignment_rows_hash,
        "component_assignment_rows_sha256": component_rows_hash,
        "unused_component_count": len(unused),
    }


def _source_partition_summary(connection: sqlite3.Connection, census: SchemaCensus) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for partition in PARTITIONS:
        row = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(question_count),0),COALESCE(SUM(annotation_count),0),COALESCE(SUM(nonnull_outcome_annotation_count),0) FROM papers WHERE source_partition=?",
            (partition,),
        ).fetchone()
        paper_ids = [list(item) for item in connection.execute(
            "SELECT paper_id,full_document_sha256,group_id FROM papers WHERE source_partition=? ORDER BY paper_id,full_document_sha256,group_id",
            (partition,),
        )]
        question_ids = [list(item) for item in connection.execute(
            "SELECT p.paper_id,q.ordinal,q.question_id_present,q.question_id_json FROM questions q JOIN papers p ON p.paper_pk=q.paper_pk WHERE p.source_partition=? ORDER BY p.paper_id,p.full_document_sha256,q.ordinal",
            (partition,),
        )]
        annotation_ids = [list(item) for item in connection.execute(
            "SELECT p.paper_id,q.ordinal,a.ordinal,a.annotation_type,a.annotation_id_json,a.worker_id_json FROM annotations a JOIN questions q ON q.question_pk=a.question_pk JOIN papers p ON p.paper_pk=a.paper_pk WHERE p.source_partition=? ORDER BY p.paper_id,p.full_document_sha256,q.ordinal,a.ordinal",
            (partition,),
        )]
        counts[partition] = {
            "papers": int(row[0]),
            "questions": int(row[1]),
            "answer_annotation_records": int(row[2]),
            "annotations_with_nonnull_source_outcome_fields": int(row[3]),
            "paper_descendant_sha256": shared.canonical_sha256(paper_ids),
            "question_descendant_sha256": shared.canonical_sha256(question_ids),
            "annotation_descendant_sha256": shared.canonical_sha256(annotation_ids),
        }
    return {"by_official_partition": counts, "schema_census": census.serialized()}


def _grouping_summary(connection: sqlite3.Connection, near_duplicate_summary: Mapping[str, Any]) -> dict[str, Any]:
    edge_reasons = {
        reason: int(count)
        for reason, count in connection.execute("SELECT reason,COUNT(*) FROM edges GROUP BY reason ORDER BY reason")
    }
    return {
        "source_group_count": int(connection.execute("SELECT COUNT(*) FROM source_groups").fetchone()[0]),
        "connected_component_count": int(connection.execute("SELECT COUNT(*) FROM components").fetchone()[0]),
        "edge_count": int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]),
        "edge_reason_counts": edge_reasons,
        "near_duplicate_search": dict(near_duplicate_summary),
        "source_groups_sha256": _manifest_hash(
            connection,
            "SELECT group_id,source_id,pinned_revision,official_partition,lineage_ids_json,input_hashes_json,rank_sha256,component_id,final_split,allocation_reason,exclusion_reason FROM source_groups ORDER BY group_id",
        ),
        "edges_sha256": _manifest_hash(
            connection,
            "SELECT edge_key,group_a,group_b,reason,evidence_hash_a,evidence_hash_b,intersection_size,union_size FROM edges ORDER BY edge_key",
        ),
        "components_sha256": _manifest_hash(
            connection,
            "SELECT component_id,groups_json,final_split,allocation_reason FROM components ORDER BY component_id",
        ),
        "grouping_inputs": [
            "canonical arXiv versionless identifiers parsed from the original QASPER paper key",
            "Unicode NFKC/casefold/whitespace-normalized complete titles",
            "exact full-document field-content SHA-256 over title, abstract, full_text, and figures_and_tables",
            "exact normalized full-document text and verified complete character-5gram-set Jaccard >= 0.90",
        ],
        "never_grouped_by": ["question wording", "question IDs", "annotation IDs", "worker IDs", "answer values", "evidence wording"],
    }


def _scan_tar_archive(
    archive_path: Path,
    expected_names: Mapping[str, str],
    max_archive_bytes: int,
) -> tuple[dict[str, tarfile.TarInfo], dict[str, Any]]:
    archive_digest, compressed_size = shared.hash_file(archive_path)
    if compressed_size > max_archive_bytes:
        raise AcquisitionError("downloaded QASPER archive exceeds its compressed-byte limit")
    selected: dict[str, tarfile.TarInfo] = {}
    seen_names: set[str] = set()
    member_count = 0
    declared_uncompressed = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                member_count += 1
                if member_count > MAX_TAR_MEMBERS:
                    raise AcquisitionError("QASPER tar archive exceeds its member-count limit")
                try:
                    normalized_name = shared._normalize_member_name(member.name)
                except shared.AcquisitionError as exc:
                    raise AcquisitionError("QASPER tar archive contains an unsafe member path") from exc
                if normalized_name in seen_names:
                    raise AcquisitionError("QASPER tar archive contains duplicate normalized member paths")
                seen_names.add(normalized_name)
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise AcquisitionError("QASPER tar archive contains a link or special filesystem member")
                if not (member.isfile() or member.isdir()):
                    raise AcquisitionError("QASPER tar archive contains an unsupported member type")
                if getattr(member, "issparse", lambda: False)():
                    raise AcquisitionError("QASPER tar archive contains a sparse file member")
                if member.size < 0:
                    raise AcquisitionError("QASPER tar archive contains an invalid member size")
                if member.isfile():
                    declared_uncompressed += member.size
                    if declared_uncompressed > MAX_TAR_UNCOMPRESSED_BYTES:
                        raise AcquisitionError("QASPER tar archive exceeds its bounded declared-size limit")
                if normalized_name in expected_names:
                    if not member.isfile():
                        raise AcquisitionError("a selected QASPER member is not a regular file")
                    selected[normalized_name] = member
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise AcquisitionError("QASPER tar archive could not be safely inspected") from exc
    if set(selected) != set(expected_names):
        raise AcquisitionError("QASPER tar archive is missing or changes a selected member name")
    return selected, {
        "archive_sha256": archive_digest,
        "compressed_bytes": compressed_size,
        "tar_member_count": member_count,
        "declared_uncompressed_file_bytes": declared_uncompressed,
        "selected_member_header_bytes": {name: int(member.size) for name, member in selected.items()},
    }


def _stream_selected_members(
    archive_path: Path,
    selected: Mapping[str, tarfile.TarInfo],
    partitions_by_name: Mapping[str, str],
    destination_dir: Path,
    remaining_budget: int,
) -> tuple[dict[str, Any], int]:
    results: dict[str, Any] = {}
    used = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            ordered = sorted(selected.items(), key=lambda item: (item[1].offset_data, item[0]))
            for member_name, info in ordered:
                if info.size > remaining_budget - used:
                    raise AcquisitionError("selected QASPER member sizes exceed the committed uncompressed-byte budget")
                partition = partitions_by_name[member_name]
                destination = destination_dir / "selected_members" / f"{partition}.json"
                shared.make_private_directory(destination.parent)
                try:
                    source_stream = archive.extractfile(info)
                except (tarfile.TarError, OSError) as exc:
                    raise AcquisitionError("selected QASPER member could not be opened as a stream") from exc
                if source_stream is None:
                    raise AcquisitionError("selected QASPER member has no regular-file stream")
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                fd = os.open(destination, flags, 0o600)
                digest = hashlib.sha256()
                actual_size = 0
                try:
                    with source_stream, os.fdopen(fd, "wb", closefd=False) as output:
                        while True:
                            block = source_stream.read(CHUNK_BYTES)
                            if not block:
                                break
                            actual_size += len(block)
                            if actual_size > int(info.size) or actual_size > remaining_budget - used:
                                raise AcquisitionError("selected QASPER member exceeded its declared or total size")
                            output.write(block)
                            digest.update(block)
                        output.flush()
                        os.fsync(output.fileno())
                finally:
                    os.close(fd)
                if actual_size != int(info.size):
                    raise AcquisitionError("selected QASPER member size differs from its tar header")
                used += actual_size
                results[partition] = {
                    "member_name": member_name,
                    "tar_header_bytes": int(info.size),
                    "actual_bytes": actual_size,
                    "sha256": digest.hexdigest(),
                    "relative_path": str(destination),
                }
        shared.fsync_directory(destination_dir / "selected_members")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise AcquisitionError("selected QASPER member bytes could not be safely streamed") from exc
    return results, used


def _write_metadata_artifacts(
    attempt: Path,
    source: Mapping[str, Any],
    max_metadata_bytes: int,
) -> tuple[dict[str, Any], bytes, bytes]:
    metadata_dir = attempt / "raw" / "source_metadata"
    loader_path = metadata_dir / "qasper.py"
    card_path = metadata_dir / "README.md"
    loader_record = _download_file(str(source["loader_url"]), loader_path, max_metadata_bytes, deadline_seconds=300)
    card_record = _download_file(str(source["card_url"]), card_path, max_metadata_bytes, deadline_seconds=300)
    loader = loader_path.read_bytes()
    card = card_path.read_bytes()
    metadata_claims = _check_pinned_metadata(source, loader, card)
    return {
        "loader": {**loader_record, "sha256": shared.sha256_bytes(loader), "bytes": len(loader)},
        "dataset_card": {**card_record, "sha256": shared.sha256_bytes(card), "bytes": len(card)},
        "checked_claims": metadata_claims,
    }, loader, card


def _parse_members(
    connection: sqlite3.Connection,
    member_records: Mapping[str, Mapping[str, Any]],
    partition_members: Mapping[str, Path],
    pinned_revision: str,
) -> tuple[SchemaCensus, dict[str, dict[str, int]]]:
    census = SchemaCensus()
    counts = {partition: Counter() for partition in PARTITIONS}
    seen_paper_ids: dict[str, set[str]] = {partition: set() for partition in PARTITIONS}
    for partition in PARTITIONS:
        member_path = partition_members[partition]
        recorded = member_records.get(partition)
        if recorded is None or not member_path.is_file():
            raise AcquisitionError("not all selected QASPER member bytes were saved before parsing")
        digest, size = shared.hash_file(member_path)
        if (digest, size) != (recorded["sha256"], recorded["actual_bytes"]):
            raise AcquisitionError("a selected QASPER member changed before parsing")
        for paper_id, paper in _stream_json_object(member_path):
            if paper_id in seen_paper_ids[partition]:
                raise AcquisitionError("duplicate paper source key in selected QASPER member")
            seen_paper_ids[partition].add(paper_id)
            if not isinstance(paper, dict):
                census.observe_value(partition, "paper_item", shared._value_type(paper))
                raise AcquisitionError("a QASPER paper mapping value is not a JSON object")
            row = _insert_paper(connection, census, partition, paper_id, paper, pinned_revision)
            for key, value in row.items():
                counts[partition][key] += value
            if sum(counts[partition].values()) % 500 == 0:
                connection.commit()
                shared._maybe_guard_sqlite_size(connection)
    _insert_schema_census(connection, census)
    connection.commit()
    shared._maybe_guard_sqlite_size(connection)
    return census, {partition: dict(values) for partition, values in counts.items()}


def _source_rights_and_exposure(plan: Mapping[str, Any], source: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_grant_citations": [
            {
                "claim": "Pinned QASPER dataset card declares CC-BY-4.0",
                "url": str(source["card_url"]),
                "sha256": metadata["dataset_card"]["sha256"],
                "license_text_url": "https://creativecommons.org/licenses/by/4.0",
            },
            {
                "claim": "Pinned dataset loader declares CC BY 4.0 for the dataset and provides the QASPER paper citation",
                "url": str(source["loader_url"]),
                "sha256": metadata["loader"]["sha256"],
                "citation": metadata["checked_claims"]["dataset_citation"],
            },
        ],
        "grant_as_recorded_in_source_plan": source["license_class"],
        "permitted_scope_for_this_cli": "local source custody, paper/question/annotation schema census, input-only grouping, and partition metadata only",
        "raw_paper_redistribution": "not performed or authorized by this workflow",
        "underlying_full_paper_republication_rights": "remain subject to review; no independent legal conclusion is made",
        "exposure": {
            "source_statement": source["exposure"],
            "public_replication_risk": "QASPER is public and overlaps LongBench/SCROLLS may exist; this is exposed replication evidence, not universally fresh confirmation",
            "model_training_membership": "unknown",
            "selection_process": "source partitions and deterministic group manifests are frozen before any model execution; public/pretraining exposure is not claimed absent",
        },
        "workflow_scope": plan["source"]["use_now"],
        "model_visible_substantive_fields": ["title", "abstract", "full_text"],
        "model_state_excludes": [
            "questions", "answer annotations", "unanswerable", "extractive_spans", "yes_no",
            "free_form_answer", "evidence", "highlighted_evidence", "figures_and_tables", "worker identifiers",
        ],
    }


def _schema_expectation() -> dict[str, Any]:
    return {
        "basis": "features declared by the pinned QASPER loader; source payloads are preserved even when nullable or structurally absent",
        "paper_fields": list(PAPER_FIELDS),
        "full_text_fields": list(SECTION_FIELDS),
        "question_fields": list(QA_FIELDS),
        "annotation_fields": list(ANNOTATION_FIELDS),
        "answer_fields": list(ANSWER_FIELDS),
        "figures_and_tables_fields": list(FIGURE_FIELDS),
        "annotation_census_rule": "counts raw field presence and JSON type only; explicit null and absent remain distinct; no target-value coercion or usable-Boolean coverage claim",
    }


def _freeze_manifest_inputs(
    attempt: Path,
    frozen_record: Mapping[str, Any],
    root: Path,
    source: Mapping[str, Any],
    plan: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    shared.write_json_immutable(attempt / "source_plan_snapshot.json", plan)
    shared.write_json_immutable(attempt / "protocol_snapshot.json", protocol)
    shared.write_json_immutable(attempt / "attempt_context.json", {
        "attempt_created_at_utc": _now_utc(),
        "private_root": str(root),
        "source_id": source["id"],
        "source_revision": source["hf_revision"],
        "frozen_inputs": frozen_record,
    })


def _record_failure(attempt: Path, phase: str, exc: BaseException, completed: Mapping[str, Any]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    message = str(exc).replace("\n", " ")[:500]
    failure = {
        "failed_at_utc": _now_utc(),
        "failure_record_timestamp": stamp,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "message": message,
        "completed_artifact_metadata": dict(completed),
    }
    try:
        shared.write_json_immutable(attempt / f"failure-{stamp}.json", failure)
    except Exception:
        pass


def _acquire() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    blobs, parsed, commit = _load_frozen_inputs(repo)
    plan = parsed["research/endgame/neutral_paper_acquisition_sources.json"]
    protocol = parsed["research/endgame/neutral_benchmark_protocol.json"]
    source = plan["source"]
    root = Path(plan["root"])
    _safe_private_root(root)
    attempt_id, attempt = _create_attempt(root)
    completed: dict[str, Any] = {}
    phase = "freeze"
    try:
        frozen_record = _snapshot_frozen_inputs(attempt, blobs, commit)
        _freeze_manifest_inputs(attempt, frozen_record, root, source, plan, protocol)
        os.umask(0o077)
        resource_info = _apply_resource_guards(root, plan)
        completed["resource_guards"] = resource_info

        phase = "source_metadata"
        metadata_info, loader_bytes, card_bytes = _write_metadata_artifacts(
            attempt,
            source,
            int(plan["resource_limits"].get("metadata_bytes_max", MAX_METADATA_BYTES)),
        )
        completed["source_metadata"] = metadata_info

        phase = "archive_download"
        partition_members: dict[str, Path] = {}
        archive_records: list[dict[str, Any]] = []
        selected_member_headers: dict[str, dict[str, tarfile.TarInfo]] = {}
        selected_name_to_partition: dict[str, str] = {}
        max_archive_bytes = int(plan["resource_limits"]["compressed_bytes_per_archive_max"])
        for index, archive_spec in enumerate(source["archives"]):
            archive_url = str(archive_spec["url"])
            archive_name = PurePosixPath(urllib.parse.urlsplit(archive_url).path).name
            if not archive_name or "/" in archive_name or "\\" in archive_name:
                raise AcquisitionError("pinned archive URL has an unsafe filename")
            archive_path = attempt / "raw" / "archives" / archive_name
            record = _download_file(archive_url, archive_path, max_archive_bytes)
            expected_members = {str(name): str(partition) for name, partition in archive_spec["members"].items()}
            scan_members, scan_record = _scan_tar_archive(archive_path, expected_members, max_archive_bytes)
            if scan_record["archive_sha256"] != record["sha256"] or scan_record["compressed_bytes"] != record["bytes"]:
                raise AcquisitionError("downloaded QASPER archive changed before tar inspection")
            for member_name, partition in expected_members.items():
                if member_name in selected_name_to_partition:
                    raise AcquisitionError("selected QASPER member is duplicated across archives")
                selected_name_to_partition[member_name] = partition
            selected_member_headers[archive_name] = scan_members
            record.update({
                "archive_name": archive_name,
                "selected_members": scan_record["selected_member_header_bytes"],
                "tar_member_count": scan_record["tar_member_count"],
                "declared_uncompressed_file_bytes": scan_record["declared_uncompressed_file_bytes"],
                "selected_member_header_bytes": scan_record["selected_member_header_bytes"],
            })
            archive_records.append(record)
        if set(selected_name_to_partition.values()) != set(PARTITIONS):
            raise AcquisitionError("all pinned QASPER partitions must be selected exactly once")
        total_selected_header_bytes = sum(
            int(member.size)
            for archive_name, members in selected_member_headers.items()
            for member in members.values()
        )
        selected_total_cap = int(plan["resource_limits"]["selected_uncompressed_bytes_total_max"])
        if total_selected_header_bytes > selected_total_cap:
            raise AcquisitionError("selected QASPER member headers exceed the committed total byte limit")

        phase = "selected_member_acquisition"
        selected_member_records: dict[str, Any] = {}
        consumed = 0
        for archive_record in archive_records:
            archive_name = str(archive_record["archive_name"])
            expected = {
                name: selected_name_to_partition[name]
                for name in selected_member_headers[archive_name]
            }
            stream_records, stream_bytes = _stream_selected_members(
                attempt / "raw" / "archives" / archive_name,
                selected_member_headers[archive_name],
                expected,
                attempt / "raw",
                selected_total_cap - consumed,
            )
            consumed += stream_bytes
            selected_member_records.update(stream_records)
        if set(selected_member_records) != set(PARTITIONS) or consumed != total_selected_header_bytes:
            raise AcquisitionError("not every selected QASPER member was completely acquired and hashed")
        for partition in PARTITIONS:
            path = attempt / "raw" / "selected_members" / f"{partition}.json"
            partition_members[partition] = path
        completed["archives"] = archive_records
        completed["selected_members"] = selected_member_records
        shared.write_json_immutable(attempt / "raw_source_manifest.json", {
            "schema": "vey.neutral.qasper.preparse-custody.v1",
            "repository_commit": commit,
            "frozen_input_record": frozen_record,
            "source_metadata": metadata_info,
            "archive_artifacts": archive_records,
            "selected_member_artifacts": selected_member_records,
            "all_archive_and_selected_member_hashes_frozen_before_parsing": True,
            "source_rows_parsed": 0,
            "model_or_gpu_calls": 0,
        })

        phase = "database_parse_and_census"
        database_path = attempt / "private.sqlite3"
        connection = _initialize_database(database_path)
        try:
            census, parse_counts = _parse_members(connection, selected_member_records, partition_members, str(source["hf_revision"]))
            completed["parse_counts"] = parse_counts
            phase = "grouping_and_allocation"
            _group_edges_from_paper_keys(connection)
            near_duplicate_summary = shared._near_duplicate_edges(connection)
            component_id_by_group, component_groups = shared._connected_components(connection)
            allocation_summary = _allocate_components(connection, component_id_by_group, component_groups, protocol)
            partition_summary = _source_partition_summary(connection, census)
            grouping_summary = _grouping_summary(connection, near_duplicate_summary)
            schema_summary = census.serialized()
            connection.execute("PRAGMA optimize")
            connection.commit()
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            if quick_check != "ok":
                raise AcquisitionError("private QASPER acquisition database failed its SQLite integrity check")
            connection.close()
        except BaseException:
            connection.close()
            raise
        os.chmod(database_path, 0o600)
        shared.fsync_directory(attempt)
        database_hash, database_size = shared.hash_file(database_path)
        completed["database"] = {"sha256": database_hash, "bytes": database_size}

        phase = "manifest_seal"
        plan_sha = shared.sha256_bytes(blobs["research/endgame/neutral_paper_acquisition_sources.json"])
        protocol_sha = shared.sha256_bytes(blobs["research/endgame/neutral_benchmark_protocol.json"])
        script_sha = shared.sha256_bytes(blobs["research/endgame/neutral_paper_acquire.py"])
        helper_sha = shared.sha256_bytes(blobs["research/endgame/neutral_acquire.py"])
        rights_exposure = _source_rights_and_exposure(plan, source, metadata_info)
        annotation_census_sha = shared.canonical_sha256(schema_summary)
        manifest = {
            "schema": SCRIPT_SCHEMA,
            "status": "acquired_and_sealed_metadata_only",
            "attempt_id": attempt_id,
            "created_at_utc": _now_utc(),
            "frozen_inputs": {
                "repository_commit": commit,
                "source_plan_sha256": plan_sha,
                "neutral_protocol_sha256": protocol_sha,
                "acquisition_script_sha256": script_sha,
                "shared_helper_sha256": helper_sha,
                "snapshot_metadata": frozen_record,
            },
            "source": {
                "id": SOURCE_ID,
                "pinned_revision": source["hf_revision"],
                "loader_version": source["loader_version"],
                "metadata_fingerprints": metadata_info,
                "source_grant_citations": rights_exposure["source_grant_citations"],
            },
            "archive_artifacts": archive_records,
            "selected_member_artifacts": [selected_member_records[part] for part in PARTITIONS],
            "private_database": {
                "relative_path": "private.sqlite3",
                "sha256": database_hash,
                "bytes": database_size,
                "contains_source_rows_labels_annotations_and_grouping_records": True,
                "git_tracked": False,
            },
            "expected_source_schema": _schema_expectation(),
            "source_partition_counts_and_descendant_hashes": partition_summary,
            "annotation_schema_census_sha256": annotation_census_sha,
            "grouping": grouping_summary,
            "allocation": allocation_summary,
            "rights_and_exposure": rights_exposure,
            "resource_guards": resource_info,
            "execution": {
                "source_loader_executed": False,
                "archive_member_filesystem_extraction": False,
                "bundled_evaluator_executed": False,
                "model_or_gpu_calls": 0,
                "training_or_quality_evaluation": False,
                "answer_values_or_text_emitted": False,
                "token_length_census_or_eligibility_claim": False,
                "raw_paper_redistribution": False,
            },
        }
        manifest_path = attempt / "manifest.json"
        manifest_hash = shared.write_json_immutable(manifest_path, manifest)
        output = {
            "status": "sealed",
            "manifest_relative_path": str(manifest_path.relative_to(root)),
            "manifest_sha256": manifest_hash,
            "source_partition_counts": {
                partition: {
                    key: value
                    for key, value in partition_summary["by_official_partition"][partition].items()
                    if key.endswith("papers") or key == "questions" or key == "answer_annotation_records" or key == "annotations_with_nonnull_source_outcome_fields"
                }
                for partition in PARTITIONS
            },
            "schema_census": schema_summary,
            "archive_and_member_hashes": {
                "archives": [{"sha256": item["sha256"], "bytes": item["bytes"]} for item in archive_records],
                "members": [{"partition": part, "sha256": selected_member_records[part]["sha256"], "bytes": selected_member_records[part]["actual_bytes"]} for part in PARTITIONS],
            },
            "grouping_counts_and_hashes": {
                "source_group_count": grouping_summary["source_group_count"],
                "connected_component_count": grouping_summary["connected_component_count"],
                "edge_count": grouping_summary["edge_count"],
                "source_groups_sha256": grouping_summary["source_groups_sha256"],
                "edges_sha256": grouping_summary["edges_sha256"],
            },
            "allocation_counts_and_hashes": allocation_summary,
            "exposure": rights_exposure["exposure"],
        }
        return output
    except BaseException as exc:
        _record_failure(attempt, phase, exc, completed)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire and seal the pinned QASPER source without executing its loader or evaluator.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("acquire", help="download, hash, parse, group, and allocate the pinned QASPER source")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "acquire":
        return 2
    try:
        result = _acquire()
    except (AcquisitionError, OSError, sqlite3.Error, tarfile.TarError, urllib.error.URLError) as exc:
        print(f"QASPER acquisition stopped safely: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"QASPER acquisition stopped safely: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(shared.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
