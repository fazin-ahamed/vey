#!/usr/bin/env python3
"""Acquire and freeze the prospective neutral benchmark sources.

Usage (only after the parent has committed the protocol, source pins, and this
script):
    python research/endgame/neutral_acquire.py acquire
    python research/endgame/neutral_acquire.py acquire --resume

The command downloads only the Banking77 CSVs at the pinned raw Git commit and
the pinned-name MASSIVE 1.1 S3 archive, plus their pinned license/citation
metadata. It never downloads model weights, runs remote code, or starts model
training/evaluation. All source rows, labels, annotations, IDs, and group
manifests are written beneath the configured private data root, never Git.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import resource
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


SCRIPT_SCHEMA = "vey.neutral.acquisition-manifest.v1"
ALLOCATION_SEED = "vey-neutral-v2:2026-10-04:1729"
ALLOCATION_PURPOSE = "neutral-acquisition"
JACCARD_NUMERATOR = 9
JACCARD_DENOMINATOR = 10
MAX_OBJECT_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_TOTAL_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_TAR_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
MAX_TAR_MEMBER_BYTES = 256 * 1024 * 1024
MAX_TAR_MEMBERS = 10000
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
MAX_INPUT_CHARS = 16384
MAX_PREFIX_PAIR_WORK = 100_000_000
MAX_NGRAM_POSTINGS = 50_000_000
MAX_SQLITE_BYTES = 8 * 1024 * 1024 * 1024
MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 30
NETWORK_OBJECT_DEADLINE_SECONDS = 1800
CHUNK_BYTES = 1024 * 1024

SOURCE_IDS = ("PolyAI/banking77", "AmazonScience/massive")
PARTITIONS = ("train", "validation", "test")
SPLITS = ("train", "dev", "calibration", "confirmation")
CAPS = {"train": 4096, "dev": 1024, "calibration": 1024, "confirmation": 8192}
MASSIVE_JUDGMENT_FIELDS = (
    "intent_score",
    "slots_score",
    "grammar_score",
    "spelling_score",
    "language_identification",
    "worker_id",
)
MASSIVE_NATIVE_VALUE_FIELDS = frozenset(
    {"intent_score", "slots_score", "grammar_score", "spelling_score", "language_identification"}
)
MASSIVE_README_JUDGMENT_METADATA = {
    "intent_score": {
        "documented_native_values": {"0": "No", "1": "Yes", "2": "reasonable"},
        "type": "nominal/source annotation, not ordinal",
    },
    "slots_score": {
        "documented_native_values": {"0": "No", "1": "Yes", "2": "noSlots/N/A"},
        "type": "nominal/source annotation, not ordinal",
    },
    "grammar_score": {
        "documented_native_scale": [0, 1, 2, 3, 4],
        "documented_construct": "naturalness",
    },
    "spelling_score": {
        "documented_native_values": [0, 1, 2],
        "type": "categorical source values; category mapping not inferred",
    },
}
ALLOWED_HOSTS = {
    "amazon-massive-nlu-dataset.s3.amazonaws.com",
    "raw.githubusercontent.com",
}


class AcquisitionError(RuntimeError):
    """A fail-closed acquisition or metadata-integrity error."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def make_private_directory(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise AcquisitionError("refusing a symlink in the private data path")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or not path.is_dir():
        raise AcquisitionError("private data path is not a plain directory")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise AcquisitionError("private data directory permissions must be 0700")


def write_exclusive(path: Path, data: bytes) -> None:
    make_private_directory(path.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def write_json_immutable(path: Path, value: Any) -> str:
    data = (canonical_json(value) + "\n").encode("utf-8")
    digest = sha256_bytes(data)
    if path.exists():
        existing_hash, existing_size = hash_file(path)
        if existing_size != len(data) or existing_hash != digest:
            raise AcquisitionError("refusing to replace a different metadata artifact")
        return digest
    write_exclusive(path, data)
    return digest


def publish_immutable(source: Path, destination: Path) -> tuple[str, int]:
    digest, size = hash_file(source)
    make_private_directory(destination.parent)
    if destination.exists():
        other_hash, other_size = hash_file(destination)
        if (other_hash, other_size) != (digest, size):
            raise AcquisitionError("refusing to replace a different private artifact")
        return digest, size
    os.link(source, destination)
    fsync_directory(destination.parent)
    return digest, size


def _set_resource_guards() -> dict[str, Any]:
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = "1"
    try:
        nice_value = os.nice(10)
    except (AttributeError, OSError) as exc:
        raise AcquisitionError("could not apply the required nice(10) priority") from exc
    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed:
            os.sched_setaffinity(0, set(allowed[:4]))
            affinity_count = len(allowed[:4])
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = MEMORY_LIMIT_BYTES
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if limit < 512 * 1024 * 1024:
        raise AcquisitionError("existing address-space limit is too small for bounded parsing")
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    return {
        "nice_value": nice_value,
        "cpu_affinity_count_max": affinity_count or 1,
        "network_concurrency": 1,
        "blas_threads": 1,
        "address_space_limit_bytes": limit,
    }


def _safe_url(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    host = (parsed.hostname or "").lower()
    allowed = host in ALLOWED_HOSTS or host == "huggingface.co" or host.endswith(
        ".huggingface.co"
    ) or host.endswith(".hf.co")
    if parsed.scheme != "https" or not allowed or parsed.username or parsed.password:
        raise AcquisitionError("source URL is not an allowed anonymous HTTPS endpoint")
    if parsed.port not in (None, 443):
        raise AcquisitionError("nonstandard HTTPS port is not allowed")
    return parsed


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
        _safe_url(new_url)
        if urllib.parse.urlsplit(request.full_url).scheme != "https":
            raise AcquisitionError("refusing an insecure redirect")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        SafeRedirectHandler(),
    )


def _append_state(path: Path, state: Mapping[str, Any], *, create: bool = False) -> None:
    make_private_directory(path.parent)
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    if create:
        flags |= os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        payload = (canonical_json(state) + "\n").encode("utf-8")
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_resume_state(path: Path, part: Path, source_url: str) -> dict[str, Any]:
    if not path.exists() or not part.exists():
        raise AcquisitionError("resume requested without a recorded raw partial and state log")
    state_lines: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise AcquisitionError("partial-state log has an uncommitted trailing record")
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise AcquisitionError("invalid partial-state record")
                state_lines.append(item)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError("could not verify the raw partial-state log") from exc
    if not state_lines:
        raise AcquisitionError("raw partial has no durable prefix-hash record")
    state = state_lines[-1]
    if state.get("source_url") != source_url:
        raise AcquisitionError("raw partial belongs to a different pinned URL")
    if not state.get("etag"):
        raise AcquisitionError("raw partial lacks a stable ETag and cannot be resumed")
    if any(
        item.get("source_url") != source_url
        or item.get("etag") != state.get("etag")
        or item.get("bytes", -1) < 0
        for item in state_lines
    ):
        raise AcquisitionError("raw partial-state history is inconsistent")
    if any(
        state_lines[index]["bytes"] > state_lines[index + 1]["bytes"]
        for index in range(len(state_lines) - 1)
    ):
        raise AcquisitionError("raw partial-state byte offsets are not monotone")
    size = part.stat().st_size
    if size != state.get("bytes"):
        raise AcquisitionError("partial has bytes beyond its last exact recorded hash")
    digest, hashed_size = hash_file(part)
    if hashed_size != size or digest != state.get("prefix_sha256"):
        raise AcquisitionError("raw partial prefix does not match its recorded SHA-256")
    return state


def _artifact_specs(
    plan: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sources = {source["id"]: source for source in plan["sources"]}
    inv_sources = {source["id"]: source for source in inventory["sources"]}
    source_records = {source_id: inv_sources[source_id] for source_id in SOURCE_IDS}
    bank = sources["PolyAI/banking77"]
    massive = sources["AmazonScience/massive"]
    bank_primary = {item["url"]: item["sha256"] for item in source_records["PolyAI/banking77"]["primary_sources"]}
    massive_primary = {item["url"]: item["sha256"] for item in source_records["AmazonScience/massive"]["primary_sources"]}
    massive_readme_url = (
        "https://huggingface.co/datasets/AmazonScience/massive/resolve/"
        f"{massive['hf_loader_revision']}/README.md"
    )

    def spec(
        artifact_id: str,
        source_id: str,
        url: str,
        relative_path: str,
        expected_sha256: str | None,
        max_bytes: int | None = None,
        kind: str = "metadata",
    ) -> dict[str, Any]:
        _safe_url(url)
        return {
            "artifact_id": artifact_id,
            "source_id": source_id,
            "url": url,
            "relative_path": relative_path,
            "expected_sha256": expected_sha256,
            "max_bytes": max_bytes if max_bytes is not None else (
                MAX_OBJECT_BYTES if kind.startswith("dataset_") else MAX_METADATA_BYTES
            ),
            "kind": kind,
        }

    return [
        spec("banking77_train_csv", "PolyAI/banking77", bank["train_url"], "raw/banking77/train.csv", None, kind="dataset_csv"),
        spec("banking77_test_csv", "PolyAI/banking77", bank["test_url"], "raw/banking77/test.csv", None, kind="dataset_csv"),
        spec("banking77_license", "PolyAI/banking77", bank["license_url"], "raw/banking77/LICENSE", bank["license_sha256_inventory"]),
        spec("banking77_readme", "PolyAI/banking77", bank["attribution_url"], "raw/banking77/README.md", bank_primary.get(bank["attribution_url"])),
        spec("massive_archive_1_1", "AmazonScience/massive", massive["archive_url"], "raw/massive/amazon-massive-dataset-1.1.tar.gz", None, kind="dataset_archive"),
        spec("massive_license", "AmazonScience/massive", massive["license_url"], "raw/massive/LICENSE", massive_primary.get(massive["license_url"])),
        spec("massive_notice", "AmazonScience/massive", massive["notice_url"], "raw/massive/NOTICE.md", massive_primary.get(massive["notice_url"])),
        spec(
            "massive_readme",
            "AmazonScience/massive",
            massive_readme_url,
            "raw/massive/README.md",
            massive_primary.get(massive_readme_url),
        ),
    ]


def _load_frozen_inputs(repo: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    relatives = (
        "research/endgame/protocol.json",
        "research/endgame/neutral_benchmark_protocol.json",
        "research/endgame/neutral_acquisition_sources.json",
        "research/endgame/neutral_source_inventory.json",
        "research/competitors/COMPETITOR_MATRIX.json",
        "research/competitors/LAYA.json",
        "research/competitors/JEV.json",
        "research/endgame/neutral_acquire.py",
    )
    blobs: dict[str, bytes] = {}
    parsed: dict[str, Any] = {}
    for relative in relatives:
        path = repo / relative
        if not path.is_file() or path.is_symlink():
            raise AcquisitionError("a required frozen protocol/source metadata file is absent or unsafe")
        data = path.read_bytes()
        blobs[relative] = data
        if relative.endswith(".json"):
            try:
                parsed[relative] = json.loads(data.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise AcquisitionError("a required frozen JSON metadata file is invalid") from exc
    return blobs, parsed


def _check_frozen_metadata(parsed: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any]]:
    protocol = parsed["research/endgame/neutral_benchmark_protocol.json"]
    plan = parsed["research/endgame/neutral_acquisition_sources.json"]
    inventory = parsed["research/endgame/neutral_source_inventory.json"]
    if protocol.get("schema") != "vey.endgame.neutral-benchmark-protocol.v2":
        raise AcquisitionError("unexpected frozen neutral protocol schema")
    if plan.get("schema") != "vey.neutral.source-acquisition.v1":
        raise AcquisitionError("unexpected frozen acquisition-plan schema")
    if plan.get("parent_protocol_git") != "5b3fe4c":
        raise AcquisitionError("acquisition plan does not pin the required parent protocol commit")
    plan_sources = {item.get("id"): item for item in plan.get("sources", [])}
    inv_sources = {item.get("id"): item for item in inventory.get("sources", [])}
    if set(plan_sources) != set(SOURCE_IDS) or not set(SOURCE_IDS).issubset(inv_sources):
        raise AcquisitionError("only the two explicitly approved source records may be acquired")
    for source_id in SOURCE_IDS:
        record = inv_sources[source_id]
        if record.get("license_class") != "shipping-train" or record.get("shipping_training_allowed_now") is not True:
            raise AcquisitionError("the committed inventory no longer records a shipping-train source grant")
        if plan_sources[source_id].get("hf_loader_revision") != record.get("revision"):
            raise AcquisitionError("the acquisition pin disagrees with the frozen source inventory revision")
    massive_plan = plan_sources["AmazonScience/massive"]
    locales = massive_plan.get("fixed_locales")
    if not isinstance(locales, list) or len(locales) != 51 or len(set(locales)) != 51:
        raise AcquisitionError("the frozen MASSIVE locale list must contain exactly 51 unique locales")
    if "ca-ES" not in locales or "zh-TW" in locales or massive_plan.get("excluded_locale") != "zh-TW; locale variant not counted as an additional language; ca-ES requires release1.1":
        raise AcquisitionError("the fixed MASSIVE locale scope no longer matches the committed source pin")
    inventory_locales = inv_sources["AmazonScience/massive"].get("languages", [])
    if set(locales) != set(inventory_locales) - {"zh-TW"}:
        raise AcquisitionError("the fixed 51 MASSIVE locales do not match the inspected source inventory")
    allocation = protocol.get("allocation_parameters")
    if not isinstance(allocation, dict) or not isinstance(allocation.get("seed"), str):
        raise AcquisitionError("the frozen protocol lacks its allocation seed")
    caps = allocation.get("caps_independent_groups_per_source")
    if not isinstance(caps, dict) or set(caps) != set(SPLITS):
        raise AcquisitionError("the frozen protocol allocation caps are incomplete")
    if any(not isinstance(caps[split], int) or caps[split] <= 0 for split in SPLITS):
        raise AcquisitionError("the frozen protocol contains an invalid independent-group cap")
    return plan, inventory, list(locales), allocation


def _assert_committed(repo: Path, relative_paths: Iterable[str], required_commit: str) -> None:
    def run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:

        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AcquisitionError("could not verify the required parent commit before acquisition") from exc

    root_result = run_git(["rev-parse", "--show-toplevel"])
    if root_result.returncode != 0 or Path(root_result.stdout.strip()).resolve() != repo.resolve():
        raise AcquisitionError("repository root could not be verified")
    if run_git(["cat-file", "-e", f"{required_commit}^{{commit}}"]).returncode != 0:
        raise AcquisitionError("the parent protocol commit is not present in repository history")
    if run_git(["merge-base", "--is-ancestor", required_commit, "HEAD"]).returncode != 0:
        raise AcquisitionError("the parent protocol commit is not an ancestor of HEAD")
    paths = list(relative_paths)
    tracked = run_git(["ls-files", "--error-unmatch", "--", *paths])
    if tracked.returncode != 0:
        raise AcquisitionError("protocol, source pins, and acquisition code must be committed before acquisition")
    if run_git(["diff", "--quiet", "HEAD", "--", *paths]).returncode != 0:
        raise AcquisitionError("frozen protocol, source pins, and acquisition code have local modifications")
    if run_git(["diff", "--cached", "--quiet", "HEAD", "--", *paths]).returncode != 0:
        raise AcquisitionError("frozen protocol, source pins, and acquisition code have staged modifications")


def _read_existing_artifact_record(root: Path, spec: Mapping[str, Any]) -> dict[str, Any] | None:
    destination = root / spec["relative_path"]
    sidecar = root / "raw" / "provenance" / f"{spec['artifact_id']}.json"
    if destination.exists() != sidecar.exists():
        raise AcquisitionError("raw artifact and its immutable SHA-256 record are incomplete")
    if not destination.exists():
        return None
    if destination.is_symlink() or sidecar.is_symlink():
        raise AcquisitionError("refusing symlinked raw evidence")
    try:
        record = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcquisitionError("raw artifact provenance record is unreadable") from exc
    digest, size = hash_file(destination)
    if digest != record.get("actual_sha256") or size != record.get("size_bytes"):
        raise AcquisitionError("raw bytes no longer match their recorded SHA-256")
    if record.get("source_url") != spec["url"] or record.get("artifact_id") != spec["artifact_id"]:
        raise AcquisitionError("raw artifact provenance does not match the frozen source URL")
    if spec.get("expected_sha256") and digest != spec["expected_sha256"]:
        raise AcquisitionError("raw metadata no longer matches its pinned inventory checksum")
    if size > spec["max_bytes"]:
        raise AcquisitionError("an existing raw artifact exceeds its acquisition byte cap")
    return record


def _download_artifact(
    root: Path,
    spec: Mapping[str, Any],
    completed_bytes: int,
    resume: bool,
) -> dict[str, Any]:
    existing = _read_existing_artifact_record(root, spec)
    if existing is not None:
        return existing
    destination = root / spec["relative_path"]
    if destination.exists() or destination.is_symlink():
        raise AcquisitionError("unrecorded raw bytes already occupy the pinned artifact path")
    provenance_dir = root / "raw" / "provenance"
    make_private_directory(provenance_dir)
    sidecar = provenance_dir / f"{spec['artifact_id']}.json"
    part = destination.with_name(destination.name + ".part")
    state_path = destination.with_name(destination.name + ".part.state.jsonl")
    download_log = destination.with_name(destination.name + ".download.jsonl")
    if sidecar.exists() or sidecar.is_symlink():
        raise AcquisitionError("orphaned raw provenance record prevents a safe overwrite")
    if download_log.exists() or download_log.is_symlink():
        raise AcquisitionError("a prior download log exists without its final artifact record")
    part_exists = part.exists() or part.is_symlink()
    state_exists = state_path.exists() or state_path.is_symlink()
    if part_exists != state_exists:
        raise AcquisitionError("unknown raw partial or state evidence; refusing to truncate or advance it")
    if part_exists and (part.is_symlink() or state_path.is_symlink()):
        raise AcquisitionError("symlinked raw partial state is not trusted")
    if part_exists and not resume:
        raise AcquisitionError("raw partial exists; use --resume only after checking the recorded prefix hash")
    if not part_exists and resume:
        resume = False
    make_private_directory(destination.parent)

    offset = 0
    prefix_digest = hashlib.sha256()
    previous_state: dict[str, Any] | None = None
    if resume:
        previous_state = _read_resume_state(state_path, part, spec["url"])
        offset = int(previous_state["bytes"])
        with part.open("rb") as prefix:
            while True:
                block = prefix.read(CHUNK_BYTES)
                if not block:
                    break
                prefix_digest.update(block)
        if offset > spec["max_bytes"] or completed_bytes + offset > MAX_TOTAL_DOWNLOAD_BYTES:
            raise AcquisitionError("raw partial exceeds the configured total byte cap")
        if not previous_state.get("etag"):
            raise AcquisitionError("raw partial lacks an ETag and cannot be safely resumed")

    request = urllib.request.Request(
        spec["url"],
        headers={
            "Accept": "application/octet-stream, text/plain, */*",
            "User-Agent": "vey-neutral-acquisition/1.0",
            **({"Range": f"bytes={offset}-", "If-Range": previous_state["etag"]} if resume else {}),
        },
    )
    opener = _opener()
    started = time.monotonic()
    try:
        response = opener.open(request, timeout=NETWORK_TIMEOUT_SECONDS)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AcquisitionError("anonymous source download failed before a response was received") from exc
    with response:
        _safe_url(response.geturl())
        status = response.getcode()
        headers = response.headers
        etag = headers.get("ETag")
        last_modified = headers.get("Last-Modified")
        content_length_header = headers.get("Content-Length")
        response_length = None
        if content_length_header is not None:
            try:
                response_length = int(content_length_header)
            except ValueError as exc:
                raise AcquisitionError("source returned an invalid Content-Length") from exc
            if response_length < 0:
                raise AcquisitionError("source returned a negative Content-Length")
        if resume:
            if status != 206:
                raise AcquisitionError("server did not honor the exact byte-range resume request")
            if etag != previous_state.get("etag"):
                raise AcquisitionError("source ETag changed; refusing to append different raw bytes")
            content_range = headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
            if not match or int(match.group(1)) != offset:
                raise AcquisitionError("range response did not begin at the recorded raw offset")
            range_end = int(match.group(2))
            range_total = match.group(3)
            if range_total == "*":
                raise AcquisitionError("resumed response does not declare the complete source entity length")
            total_length = int(range_total)
            if range_end < offset or range_end != total_length - 1:
                raise AcquisitionError("range response does not extend exactly through the complete source entity")
            if response_length is not None and response_length != range_end - offset + 1:
                raise AcquisitionError("range response Content-Length disagrees with its declared byte range")
            entity_length = previous_state.get("content_length")
            if entity_length is not None and total_length != int(entity_length):
                raise AcquisitionError("range response total length differs from the recorded source entity")
            if total_length > spec["max_bytes"] or completed_bytes + total_length > MAX_TOTAL_DOWNLOAD_BYTES:
                raise AcquisitionError("resumed source entity exceeds a configured byte cap")
        else:
            if status != 200:
                raise AcquisitionError("source did not return a complete HTTP 200 response")
            total_length = response_length
            if total_length is not None and total_length > spec["max_bytes"]:
                raise AcquisitionError("source object exceeds the per-artifact streaming cap")
            if total_length is not None and completed_bytes + total_length > MAX_TOTAL_DOWNLOAD_BYTES:
                raise AcquisitionError("source objects exceed the total streaming-download cap")
            fd = os.open(part, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            _append_state(
                state_path,
                {
                    "source_url": spec["url"],
                    "bytes": 0,
                    "prefix_sha256": hashlib.sha256(b"").hexdigest(),
                    "etag": etag,
                    "last_modified": last_modified,
                    "content_length": total_length,
                },
                create=True,
            )
        if response_length is not None:
            expected_response = total_length - offset if total_length is not None else response_length
            if response_length != expected_response:
                raise AcquisitionError("HTTP body length disagrees with the pinned/resumed entity length")
        current_size = offset
        open_mode = "ab" if resume else "ab"
        try:
            with part.open(open_mode) as out:
                while True:
                    if time.monotonic() - started > NETWORK_OBJECT_DEADLINE_SECONDS:
                        raise AcquisitionError("bounded source-download deadline exceeded")
                    block = response.read(CHUNK_BYTES)
                    if not block:
                        break
                    current_size += len(block)
                    if current_size > spec["max_bytes"]:
                        raise AcquisitionError("source object exceeded its per-artifact streaming cap")
                    if completed_bytes + current_size > MAX_TOTAL_DOWNLOAD_BYTES:
                        raise AcquisitionError("source objects exceeded the total streaming-download cap")
                    out.write(block)
                    prefix_digest.update(block)
                    out.flush()
                    os.fsync(out.fileno())
                    _append_state(
                        state_path,
                        {
                            "source_url": spec["url"],
                            "bytes": current_size,
                            "prefix_sha256": prefix_digest.hexdigest(),
                            "etag": etag,
                            "last_modified": last_modified,
                            "content_length": total_length,
                        },
                    )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AcquisitionError("bounded source download was interrupted; raw partial is preserved") from exc
        if total_length is not None and current_size != total_length:
            raise AcquisitionError("downloaded byte count does not match the server's entity length")
        if current_size == 0:
            raise AcquisitionError("source returned an empty raw artifact")

    actual_sha = prefix_digest.hexdigest()
    if spec.get("expected_sha256") and actual_sha != spec["expected_sha256"]:
        raise AcquisitionError("downloaded metadata differs from its frozen inventory checksum")
    if destination.exists() or destination.is_symlink():
        raise AcquisitionError("refusing to overwrite an existing raw source artifact")
    os.link(part, destination)
    os.unlink(part)
    if download_log.exists():
        raise AcquisitionError("refusing to replace a prior raw download log")
    os.rename(state_path, download_log)
    fsync_directory(destination.parent)
    record = {
        "artifact_id": spec["artifact_id"],
        "source_id": spec["source_id"],
        "kind": spec["kind"],
        "source_url": spec["url"],
        "resolved_url": response.geturl(),
        "size_bytes": current_size,
        "actual_sha256": actual_sha,
        "expected_sha256": spec.get("expected_sha256"),
        "expected_sha256_source": "committed metadata pin" if spec.get("expected_sha256") else None,
        "etag": etag,
        "last_modified": last_modified,
        "etag_is_cryptographic_checksum": False,
        "source_checksum_was_pinned_before_acquisition": bool(spec.get("expected_sha256")),
        "download_log": download_log.relative_to(root).as_posix(),
    }
    write_json_immutable(sidecar, record)
    return record




def _snapshot_inputs(root: Path, blobs: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, data in sorted(blobs.items()):
        destination = root / "metadata" / "pinned_inputs" / relative
        digest = sha256_bytes(data)
        if destination.exists():
            existing_digest, existing_size = hash_file(destination)
            if existing_digest != digest or existing_size != len(data):
                raise AcquisitionError("existing frozen metadata snapshot does not match the repository")
        else:
            write_exclusive(destination, data)
        result[relative] = {
            "path": destination.relative_to(root).as_posix(),
            "size_bytes": len(data),
            "sha256": digest,
        }
    return result


def _csv_rows(path: Path, source_id: str, partition: str, connection: sqlite3.Connection) -> int:
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames is None or set(reader.fieldnames) != {"text", "category"} or len(reader.fieldnames) != 2:
                raise AcquisitionError("Banking77 CSV headers do not match the pinned text/category schema")
            for row in reader:
                count += 1
                if None in row or row.get("text") is None or row.get("category") is None:
                    raise AcquisitionError("Banking77 CSV row has missing or extra fields")
                text = row["text"]
                label = row["category"]
                if not isinstance(text, str) or not isinstance(label, str) or not label:
                    raise AcquisitionError("Banking77 input or source label has an invalid type")
                if len(text) > MAX_INPUT_CHARS:
                    raise AcquisitionError("Banking77 input exceeds the private-parser character bound")
                normalized = normalize_input(text)
                norm_hash = sha256_bytes(normalized.encode("utf-8"))
                row_key = f"banking77:{partition}:{count:08d}"
                payload = {"text": text, "category": label}
                connection.execute(
                    "INSERT INTO rows(row_key,source_id,official_partition,locale,original_id,input_text,normalized_text,norm_hash,source_path,line_number,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        row_key,
                        source_id,
                        partition,
                        "en",
                        row_key,
                        text,
                        normalized,
                        norm_hash,
                        path.name,
                        count + 1,
                        canonical_json(payload),
                    ),
                )
    except (UnicodeError, csv.Error) as exc:
        raise AcquisitionError("pinned Banking77 CSV could not be parsed as strict UTF-8 CSV") from exc
    return count


def normalize_input(text: str) -> str:
    if len(text) > MAX_INPUT_CHARS:
        raise AcquisitionError("source input exceeds the configured normalization bound")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _json_object_no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _strict_json_line(line: bytes) -> dict[str, Any]:
    try:
        decoded = line.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_json_object_no_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AcquisitionError("MASSIVE JSONL contains an invalid or noncanonical JSON row") from exc
    if not isinstance(value, dict):
        raise AcquisitionError("MASSIVE JSONL row is not a JSON object")
    return value


def _canonical_partition(value: Any) -> str:
    if not isinstance(value, str):
        raise AcquisitionError("MASSIVE partition field is not text")
    normalized = value.casefold()
    if normalized in ("dev", "validation"):
        return "validation"
    if normalized in ("train", "test"):
        return normalized
    raise AcquisitionError("MASSIVE contains an unrecognized official partition")


def _value_type(value: Any) -> str:
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


class MassiveMetadataStats:
    def __init__(self) -> None:
        self.data: dict[tuple[str, str], dict[str, Any]] = {}

    def add(self, locale: str, partition: str, record: Mapping[str, Any]) -> None:
        key = (locale, partition)
        stat = self.data.setdefault(
            key,
            {
                "row_count": 0,
                "top_level_fields": {},
                "judgments_field_present_rows": 0,
                "judgments_field_null_rows": 0,
                "rows_with_judgments": 0,
                "judgment_record_count": 0,
                "num_raters_min": None,
                "num_raters_max": None,
                "num_raters_sum": 0,
                "judgment_fields": {},
            },
        )
        stat["row_count"] += 1
        for name, value in record.items():
            field = stat["top_level_fields"].setdefault(name, {"present_rows": 0, "types": set()})
            field["present_rows"] += 1
            field["types"].add(_value_type(value))
        if "judgments" not in record:
            raters = 0
        else:
            stat["judgments_field_present_rows"] += 1
            judgments = record["judgments"]
            if judgments is None:
                stat["judgments_field_null_rows"] += 1
                raters = 0
            elif isinstance(judgments, list):
                raters = len(judgments)
                if raters:
                    stat["rows_with_judgments"] += 1
                stat["judgment_record_count"] += raters
                for judgment in judgments:
                    if not isinstance(judgment, dict):
                        raise AcquisitionError("MASSIVE judgment entries must be JSON objects")
                    for field_name, value in judgment.items():
                        self._add_judgment_field(stat, field_name, value)
            else:
                raise AcquisitionError("MASSIVE judgments field is neither an array nor null")
        stat["num_raters_sum"] += raters
        stat["num_raters_min"] = raters if stat["num_raters_min"] is None else min(stat["num_raters_min"], raters)
        stat["num_raters_max"] = raters if stat["num_raters_max"] is None else max(stat["num_raters_max"], raters)

    @staticmethod
    def _add_judgment_field(stat: dict[str, Any], name: str, value: Any) -> None:
        fields = stat["judgment_fields"]
        field = fields.setdefault(
            name,
            {
                "present_records": 0,
                "null_records": 0,
                "types": set(),
                "native_value_counts": Counter(),
            },
        )
        field["present_records"] += 1
        field["types"].add(_value_type(value))
        if value is None:
            field["null_records"] += 1
            return
        if name in MASSIVE_NATIVE_VALUE_FIELDS:
            field["native_value_counts"][canonical_json(value)] += 1

    def report(self, locale_order: Sequence[str]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for locale in locale_order:
            locale_obj: dict[str, Any] = {}
            for partition in PARTITIONS:
                stat = self.data.get((locale, partition))
                if stat is None:
                    continue
                top_fields = {}
                for name, value in sorted(stat["top_level_fields"].items()):
                    top_fields[name] = {
                        "present_rows": value["present_rows"],
                        "missing_rows": stat["row_count"] - value["present_rows"],
                        "value_types": sorted(value["types"]),
                    }
                judgement_fields: dict[str, Any] = {}
                names = sorted(set(MASSIVE_JUDGMENT_FIELDS) | set(stat["judgment_fields"]))
                for name in names:
                    value = stat["judgment_fields"].get(name)
                    if value is None:
                        judgement_fields[name] = {
                            "present_records": 0,
                            "missing_records": stat["judgment_record_count"],
                            "null_records": 0,
                            "value_types": [],
                            "raw_native_value_counts": {} if name in MASSIVE_NATIVE_VALUE_FIELDS else None,
                        }
                        continue
                    native_counts = (
                        dict(sorted(value["native_value_counts"].items()))
                        if name in MASSIVE_NATIVE_VALUE_FIELDS
                        else None
                    )
                    judgement_fields[name] = {
                        "present_records": value["present_records"],
                        "missing_records": stat["judgment_record_count"] - value["present_records"],
                        "null_records": value["null_records"],
                        "value_types": sorted(value["types"]),
                        "raw_native_value_counts": native_counts,
                    }
                row_count = stat["row_count"]
                locale_obj[partition] = {
                    "row_count": row_count,
                    "top_level_fields": top_fields,
                    "judgments_field_present_rows": stat["judgments_field_present_rows"],
                    "judgments_field_missing_rows": row_count - stat["judgments_field_present_rows"],
                    "judgments_field_null_rows": stat["judgments_field_null_rows"],
                    "rows_with_judgments": stat["rows_with_judgments"],
                    "judgment_record_count": stat["judgment_record_count"],
                    "num_raters_per_utterance": {
                        "minimum": stat["num_raters_min"],
                        "maximum": stat["num_raters_max"],
                        "mean": stat["num_raters_sum"] / row_count if row_count else None,
                        "unit": "number of entries in source judgments array; not a deduplicated worker count",
                    },
                    "judgment_fields": judgement_fields,
                    "native_values_are_unmapped_source_observations": True,
                    "no_rating_distribution_invented_when_source_values_are_missing": True,
                }
            output[locale] = locale_obj
        return output


def _normalize_member_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name or name.startswith("/"):
        raise AcquisitionError("MASSIVE archive contains an unsafe member path")
    pieces = name.split("/")
    if any(piece == ".." for piece in pieces):
        raise AcquisitionError("MASSIVE archive contains a path-traversal member")
    if pieces and re.match(r"^[A-Za-z]:", pieces[0]):
        raise AcquisitionError("MASSIVE archive contains a drive-qualified member path")
    normalized = "/".join(piece for piece in pieces if piece not in ("", "."))
    if not normalized:
        raise AcquisitionError("MASSIVE archive contains an empty member path")
    return normalized


def _selected_member(
    canonical_name: str,
    locale_set: set[str],
) -> str | None:
    path = PurePosixPath(canonical_name)
    return path.stem if path.suffix == ".jsonl" and path.stem in locale_set else None


@dataclass(frozen=True)
class TarMember:
    name: str
    size_bytes: int
    sha256: str | None
    member_type: str
    selected_locale: str | None
    executable: bool


def _gzip_jsonl_writer(path: Path) -> tuple[BinaryIO, gzip.GzipFile]:
    raw = path.open("xb")
    os.chmod(path, 0o600)
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0)
    return raw, compressed


def _scan_massive_archive(
    archive_path: Path,
    expected_archive_sha256: str,
    locale_order: Sequence[str],
    scratch_dir: Path,
) -> tuple[dict[str, TarMember], Path, dict[str, Any]]:
    digest, _ = hash_file(archive_path)
    if digest != expected_archive_sha256:
        raise AcquisitionError("MASSIVE archive hash changed before member inspection")
    locale_set = set(locale_order)
    member_map: dict[str, TarMember] = {}
    selected: dict[str, TarMember] = {}
    selected_locales: set[str] = set()
    total_unpacked = 0
    member_count = 0
    index_temp = scratch_dir / "massive_tar_members.jsonl.gz"
    raw_out, gzip_out = _gzip_jsonl_writer(index_temp)
    try:
        with tarfile.open(archive_path, mode="r|gz") as tf:
            for member in tf:
                member_count += 1
                if member_count > MAX_TAR_MEMBERS:
                    raise AcquisitionError("MASSIVE archive exceeds the member-count inspection cap")
                canonical_name = _normalize_member_name(member.name)
                if canonical_name in member_map:
                    raise AcquisitionError("MASSIVE archive contains duplicate canonical member paths")
                selected_key = _selected_member(canonical_name, locale_set)
                executable = bool(member.mode & 0o111)
                if member.isdir():
                    info = TarMember(canonical_name, 0, None, "directory", None, executable)
                elif member.isfile():
                    if member.size < 0 or member.size > MAX_TAR_MEMBER_BYTES:
                        raise AcquisitionError("MASSIVE archive member exceeds its uncompressed size cap")
                    total_unpacked += member.size
                    if total_unpacked > MAX_TAR_UNPACKED_BYTES:
                        raise AcquisitionError("MASSIVE archive exceeds the total uncompressed size cap")
                    if executable:
                        raise AcquisitionError("MASSIVE archive contains an executable regular member; refusing it")
                    source = tf.extractfile(member)
                    if source is None:
                        raise AcquisitionError("MASSIVE regular archive member could not be read safely")
                    digest_member = hashlib.sha256()
                    actual_size = 0
                    while True:
                        block = source.read(CHUNK_BYTES)
                        if not block:
                            break
                        actual_size += len(block)
                        if actual_size > member.size:
                            raise AcquisitionError("MASSIVE archive member expanded beyond its declared size")
                        digest_member.update(block)
                    if actual_size != member.size:
                        raise AcquisitionError("MASSIVE archive member byte count is inconsistent")
                    locale = selected_key
                    info = TarMember(
                        canonical_name,
                        actual_size,
                        digest_member.hexdigest(),
                        "regular_file",
                        locale,
                        executable,
                    )
                    if selected_key:
                        if selected_key in selected_locales:
                            raise AcquisitionError("MASSIVE contains duplicate locale JSONL sources")
                        selected_locales.add(selected_key)
                        selected[canonical_name] = info
                else:
                    raise AcquisitionError("MASSIVE archive contains a symlink, device, or nonregular member")
                member_map[canonical_name] = info
                gzip_out.write(
                    (canonical_json({
                        "path": info.name,
                        "size_bytes": info.size_bytes,
                        "sha256": info.sha256,
                        "member_type": info.member_type,
                        "selected_locale": info.selected_locale,
                        "executable": info.executable,
                    }) + "\n").encode("utf-8")
                )
    finally:
        gzip_out.close()
        raw_out.flush()
        os.fsync(raw_out.fileno())
        raw_out.close()
    found_selected = {item.selected_locale for item in selected.values()}
    if found_selected != locale_set or len(selected) != len(locale_order):
        raise AcquisitionError("MASSIVE archive does not contain exactly one JSONL file for each frozen locale")
    return selected, index_temp, {
        "member_count": member_count,
        "regular_file_count": sum(item.member_type == "regular_file" for item in member_map.values()),
        "directory_count": sum(item.member_type == "directory" for item in member_map.values()),
        "unpacked_regular_bytes": total_unpacked,
        "selected_jsonl_member_count": len(selected),
        "selected_locale_count": len(locale_order),
        "partition_routing": "Native JSON row partition; no split-specific archive files.",
        "remote_code_executed": False,
    }


def _initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE rows(
          row_key TEXT PRIMARY KEY,
          source_id TEXT NOT NULL,
          official_partition TEXT NOT NULL,
          locale TEXT NOT NULL,
          original_id TEXT NOT NULL,
          input_text TEXT NOT NULL,
          normalized_text TEXT NOT NULL,
          norm_hash TEXT NOT NULL,
          source_path TEXT NOT NULL,
          line_number INTEGER NOT NULL,
          payload_json TEXT NOT NULL,
          group_id TEXT
        );
        CREATE INDEX rows_source_original ON rows(source_id, original_id);
        CREATE INDEX rows_norm_hash ON rows(norm_hash);
        CREATE INDEX rows_group ON rows(group_id);
        CREATE TABLE massive_base(
          original_id TEXT PRIMARY KEY,
          official_partition TEXT NOT NULL,
          intent_sha256 TEXT NOT NULL,
          scenario_sha256 TEXT NOT NULL
        );
        CREATE TABLE massive_locale_ids(
          original_id TEXT NOT NULL,
          locale TEXT NOT NULL,
          PRIMARY KEY(original_id, locale)
        ) WITHOUT ROWID;
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
        CREATE INDEX source_groups_source ON source_groups(source_id, official_partition);
        CREATE TABLE doc_groups(
          doc_hash TEXT NOT NULL,
          group_id TEXT NOT NULL,
          PRIMARY KEY(doc_hash, group_id)
        ) WITHOUT ROWID;
        CREATE INDEX doc_groups_group ON doc_groups(group_id, doc_hash);
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
          PRIMARY KEY(doc_id, gram)
        ) WITHOUT ROWID;
        CREATE INDEX doc_grams_by_gram ON doc_grams(gram, doc_id);
        CREATE TABLE gram_frequency(
          gram TEXT PRIMARY KEY,
          document_frequency INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE prefixes(
          doc_id INTEGER NOT NULL,
          gram TEXT NOT NULL,
          PRIMARY KEY(doc_id, gram)
        ) WITHOUT ROWID;
        CREATE INDEX prefixes_by_gram ON prefixes(gram, doc_id);
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
        CREATE INDEX edges_groups ON edges(group_a, group_b);
        """
    )
    return connection


def _maybe_guard_sqlite_size(connection: sqlite3.Connection) -> None:
    page_count, page_size = connection.execute("PRAGMA page_count").fetchone()
    if page_count * page_size > MAX_SQLITE_BYTES:
        raise AcquisitionError("private grouping work database exceeded its explicit storage cap")


def _insert_massive_record(
    connection: sqlite3.Connection,
    record: Mapping[str, Any],
    locale: str,
    path: str,
    line_number: int,
    stats: MassiveMetadataStats,
) -> str:
    required = ("id", "locale", "partition", "intent", "scenario", "utt")
    if any(field not in record for field in required):
        raise AcquisitionError("MASSIVE row lacks a required source ID, locale, partition, label, or utterance field")
    original_id = record["id"]
    if not isinstance(original_id, str) or not original_id:
        raise AcquisitionError("MASSIVE original-utterance ID is not a nonempty string")
    if record["locale"] != locale:
        raise AcquisitionError("MASSIVE row locale disagrees with its archive member path")
    partition = _canonical_partition(record["partition"])
    if not isinstance(record["utt"], str):
        raise AcquisitionError("MASSIVE utterance input is not text")
    if len(record["utt"]) > MAX_INPUT_CHARS:
        raise AcquisitionError("MASSIVE utterance exceeds the private-parser character bound")
    if record["intent"] is None or record["scenario"] is None:
        raise AcquisitionError("MASSIVE source intent/scenario label is missing")
    input_text = record["utt"]
    normalized = normalize_input(input_text)
    norm_hash = sha256_bytes(normalized.encode("utf-8"))
    intent_hash = sha256_bytes(canonical_json(record["intent"]).encode("utf-8"))
    scenario_hash = sha256_bytes(canonical_json(record["scenario"]).encode("utf-8"))
    row_key = f"massive:{locale}:{original_id}"
    try:
        base = connection.execute(
            "SELECT official_partition,intent_sha256,scenario_sha256 FROM massive_base WHERE original_id=?",
            (original_id,),
        ).fetchone()
        if base is None:
            connection.execute(
                "INSERT INTO massive_base(original_id,official_partition,intent_sha256,scenario_sha256) VALUES(?,?,?,?)",
                (original_id, partition, intent_hash, scenario_hash),
            )
        elif base != (partition, intent_hash, scenario_hash):
            raise AcquisitionError("MASSIVE original ID has inconsistent partition, scenario, or intent across locales")
        connection.execute("INSERT INTO massive_locale_ids(original_id,locale) VALUES(?,?)", (original_id, locale))
        connection.execute(
            "INSERT INTO rows(row_key,source_id,official_partition,locale,original_id,input_text,normalized_text,norm_hash,source_path,line_number,payload_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                row_key,
                "AmazonScience/massive",
                partition,
                locale,
                original_id,
                input_text,
                normalized,
                norm_hash,
                path,
                line_number,
                canonical_json(record),
            ),
        )
    except sqlite3.IntegrityError as exc:
        raise AcquisitionError("MASSIVE contains a duplicate original-ID/locale row") from exc
    stats.add(locale, partition, record)
    return partition


def _parse_massive_member(
    stream: BinaryIO,
    expected: TarMember,
    locale: str,
    connection: sqlite3.Connection,
    stats: MassiveMetadataStats,
) -> dict[str, int]:
    digest = hashlib.sha256()
    bytes_read = 0
    line_number = 0
    counts = {partition: 0 for partition in PARTITIONS}
    while True:
        line = stream.readline(MAX_JSONL_LINE_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_JSONL_LINE_BYTES:
            raise AcquisitionError("MASSIVE JSONL line exceeds the parser's explicit byte bound")
        bytes_read += len(line)
        if bytes_read > expected.size_bytes:
            raise AcquisitionError("MASSIVE JSONL member grew after its pre-parse hash")
        digest.update(line)
        if not line.strip():
            raise AcquisitionError("MASSIVE JSONL contains an empty source row")
        line_number += 1
        record = _strict_json_line(line)
        partition = _insert_massive_record(connection, record, locale, expected.name, line_number, stats)
        counts[partition] += 1
        if line_number % 10000 == 0:
            connection.commit()
            _maybe_guard_sqlite_size(connection)
    if bytes_read != expected.size_bytes or digest.hexdigest() != expected.sha256:
        raise AcquisitionError("MASSIVE JSONL member changed between hash inspection and parsing")
    return counts


def _parse_massive_archive(
    archive_path: Path,
    archive_sha256: str,
    selected: Mapping[str, TarMember],
    locale_order: Sequence[str],
    expected_split_sizes: Mapping[str, int],
    connection: sqlite3.Connection,
) -> tuple[MassiveMetadataStats, dict[str, dict[str, int]]]:
    digest, _ = hash_file(archive_path)
    if digest != archive_sha256:
        raise AcquisitionError("MASSIVE archive hash changed immediately before JSONL parsing")
    stats = MassiveMetadataStats()
    observed_rows: dict[str, dict[str, int]] = {locale: {p: 0 for p in PARTITIONS} for locale in locale_order}
    seen: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as tf:
        for member in tf:
            canonical_name = _normalize_member_name(member.name)
            expected = selected.get(canonical_name)
            if expected is None:
                continue
            if not member.isfile() or member.mode & 0o111 or member.size != expected.size_bytes:
                raise AcquisitionError("selected MASSIVE member metadata changed after pre-parse inspection")
            source = tf.extractfile(member)
            if source is None:
                raise AcquisitionError("selected MASSIVE JSONL member could not be read safely")
            count = _parse_massive_member(
                source,
                expected,
                expected.selected_locale or "",
                connection,
                stats,
            )
            observed_rows[expected.selected_locale or ""] = count
            seen.add(canonical_name)
    if seen != set(selected):
        raise AcquisitionError("one or more pre-hashed MASSIVE JSONL members were not parsed")
    for locale in locale_order:
        for partition in PARTITIONS:
            expected_count = int(expected_split_sizes[partition])
            if observed_rows[locale][partition] != expected_count:
                raise AcquisitionError("MASSIVE observed locale/partition row count differs from pinned metadata")
    locale_count = len(locale_order)
    mismatch = connection.execute(
        "SELECT COUNT(*) FROM massive_base b WHERE (SELECT COUNT(*) FROM massive_locale_ids m WHERE m.original_id=b.original_id) != ?",
        (locale_count,),
    ).fetchone()[0]
    if mismatch:
        raise AcquisitionError("MASSIVE original utterance IDs do not have all 51 fixed locale descendants")
    duplicate_or_unexpected = connection.execute(
        "SELECT COUNT(*) FROM massive_locale_ids WHERE locale NOT IN (" + ",".join("?" for _ in locale_order) + ")",
        tuple(locale_order),
    ).fetchone()[0]
    if duplicate_or_unexpected:
        raise AcquisitionError("MASSIVE contains an ID outside the exact frozen locale set")
    connection.commit()
    return stats, observed_rows


def _add_group_edge(
    connection: sqlite3.Connection,
    group_a: str,
    group_b: str,
    reason: str,
    evidence_hash_a: str,
    evidence_hash_b: str | None = None,
    intersection_size: int | None = None,
    union_size: int | None = None,
) -> None:
    if group_a == group_b:
        return
    if group_a <= group_b:
        first, second = group_a, group_b
        first_hash, second_hash = evidence_hash_a, evidence_hash_b
    else:
        first, second = group_b, group_a
        first_hash, second_hash = evidence_hash_b, evidence_hash_a
    identity = [first, second, reason, first_hash, second_hash]
    edge_key = canonical_sha256(identity)
    connection.execute(
        "INSERT OR IGNORE INTO edges(edge_key,group_a,group_b,reason,evidence_hash_a,evidence_hash_b,intersection_size,union_size) VALUES(?,?,?,?,?,?,?,?)",
        (edge_key, first, second, reason, first_hash, second_hash, intersection_size, union_size),
    )


def _source_revision(source_id: str, plan_sources: Mapping[str, Mapping[str, Any]]) -> str:
    source = plan_sources[source_id]
    if source_id == "PolyAI/banking77":
        return str(source["raw_revision"])
    return str(source["hf_loader_revision"])


def _build_source_groups(
    connection: sqlite3.Connection,
    plan_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    for source_id in SOURCE_IDS:
        revision = _source_revision(source_id, plan_sources)
        if source_id == "PolyAI/banking77":
            records = connection.execute(
                "SELECT row_key,official_partition,norm_hash FROM rows WHERE source_id=? ORDER BY row_key",
                (source_id,),
            ).fetchall()
            for row_key, partition, norm_hash in records:
                lineage_ids = [row_key]
                input_hashes = [norm_hash]
                group_id = canonical_sha256([source_id, revision, lineage_ids, input_hashes])
                rank_hash = _rank_hash(source_id, group_id)
                connection.execute(
                    "INSERT INTO source_groups(group_id,source_id,pinned_revision,official_partition,lineage_ids_json,input_hashes_json,rank_sha256) VALUES(?,?,?,?,?,?,?)",
                    (group_id, source_id, revision, partition, canonical_json(lineage_ids), canonical_json(input_hashes), rank_hash),
                )
                connection.execute("UPDATE rows SET group_id=? WHERE row_key=?", (group_id, row_key))
        else:
            cursor = connection.execute(
                "SELECT original_id,MIN(official_partition),GROUP_CONCAT(DISTINCT norm_hash) FROM rows WHERE source_id=? GROUP BY original_id ORDER BY original_id",
                (source_id,),
            ).fetchall()
            for original_id, partition, _ in cursor:
                hashes = [row[0] for row in connection.execute(
                    "SELECT DISTINCT norm_hash FROM rows WHERE source_id=? AND original_id=? ORDER BY norm_hash",
                    (source_id, original_id),
                )]
                lineage_ids = [original_id]
                group_id = canonical_sha256([source_id, revision, lineage_ids, hashes])
                rank_hash = _rank_hash(source_id, group_id)
                connection.execute(
                    "INSERT INTO source_groups(group_id,source_id,pinned_revision,official_partition,lineage_ids_json,input_hashes_json,rank_sha256) VALUES(?,?,?,?,?,?,?)",
                    (group_id, source_id, revision, partition, canonical_json(lineage_ids), canonical_json(hashes), rank_hash),
                )
                connection.execute(
                    "UPDATE rows SET group_id=? WHERE source_id=? AND original_id=?",
                    (group_id, source_id, original_id),
                )
        connection.commit()

    connection.execute(
        "INSERT INTO doc_groups(doc_hash,group_id) SELECT DISTINCT norm_hash,group_id FROM rows WHERE group_id IS NOT NULL"
    )
    collisions = connection.execute(
        "SELECT COUNT(*) FROM (SELECT norm_hash,COUNT(DISTINCT normalized_text) n FROM rows GROUP BY norm_hash HAVING n>1)"
    ).fetchone()[0]
    if collisions:
        raise AcquisitionError("SHA-256 collision detected among normalized source inputs")
    cursor = connection.execute(
        "SELECT doc_hash,MIN(group_id) FROM doc_groups GROUP BY doc_hash ORDER BY doc_hash"
    )
    for doc_hash, representative in cursor:
        text_row = connection.execute(
            "SELECT normalized_text FROM rows WHERE norm_hash=? ORDER BY row_key LIMIT 1",
            (doc_hash,),
        ).fetchone()
        if text_row is None:
            raise AcquisitionError("normalized input index references a missing source row")
        connection.execute(
            "INSERT INTO docs(doc_hash,normalized_text,representative_group) VALUES(?,?,?)",
            (doc_hash, text_row[0], representative),
        )
    # Exact duplicate edges use a star per exact normalized-text hash. This is a
    # complete connectivity certificate without quadratic cliques for repeated boilerplate.
    cursor = connection.execute(
        "SELECT doc_hash,group_id FROM doc_groups ORDER BY doc_hash,group_id"
    )
    current_hash: str | None = None
    representative: str | None = None
    for doc_hash, group_id in cursor:
        if doc_hash != current_hash:
            current_hash = doc_hash
            representative = group_id
            continue
        _add_group_edge(connection, representative or group_id, group_id, "exact_normalized_input", doc_hash)
    connection.commit()


def _fivegrams(text: str) -> set[str]:
    return {text[index : index + 5] for index in range(max(0, len(text) - 4))}


def _prepare_prefix_index(connection: sqlite3.Connection) -> dict[str, int]:
    short_docs = 0
    gram_postings = 0
    text_cursor = connection.execute("SELECT doc_id,normalized_text FROM docs ORDER BY doc_id")
    for doc_id, text in text_cursor:
        grams = _fivegrams(text)
        if not grams:
            short_docs += 1
            doc_hash, group_id = connection.execute(
                "SELECT doc_hash,representative_group FROM docs WHERE doc_id=?",
                (doc_id,),
            ).fetchone()
            connection.execute(
                "INSERT INTO search_exclusions(doc_hash,representative_group,normalized_codepoints,reason) VALUES(?,?,?,?)",
                (doc_hash, group_id, len(text), "empty_raw_character_5gram_set"),
            )
            continue
        gram_postings += len(grams)
        if gram_postings > MAX_NGRAM_POSTINGS:
            raise AcquisitionError("complete character-5gram index exceeds its explicit storage cap")
        connection.executemany(
            "INSERT INTO doc_grams(doc_id,gram) VALUES(?,?)",
            ((doc_id, gram) for gram in grams),
        )
        if doc_id % 10000 == 0:
            connection.commit()
            _maybe_guard_sqlite_size(connection)
    text_cursor.close()
    connection.commit()
    connection.execute(
        "UPDATE docs SET gram_count=(SELECT COUNT(*) FROM doc_grams WHERE doc_grams.doc_id=docs.doc_id)"
    )
    connection.execute(
        "INSERT INTO gram_frequency(gram,document_frequency) SELECT gram,COUNT(*) FROM doc_grams GROUP BY gram"
    )
    connection.execute("CREATE INDEX gram_frequency_order ON gram_frequency(document_frequency,gram)")
    prefix_cursor = connection.execute(
        "SELECT dg.doc_id,dg.gram,gf.document_frequency,d.gram_count "
        "FROM doc_grams dg JOIN gram_frequency gf ON gf.gram=dg.gram "
        "JOIN docs d ON d.doc_id=dg.doc_id "
        "ORDER BY dg.doc_id,gf.document_frequency,dg.gram"
    )
    batch: list[tuple[int, str]] = []
    current_doc: int | None = None
    position = 0
    prefix_limit = 0
    for doc_id, gram, _, gram_count in prefix_cursor:
        if doc_id != current_doc:
            current_doc = doc_id
            position = 0
            ceil_threshold = (JACCARD_NUMERATOR * gram_count + JACCARD_DENOMINATOR - 1) // JACCARD_DENOMINATOR
            prefix_limit = gram_count - ceil_threshold + 1
        position += 1
        if position <= prefix_limit:
            batch.append((doc_id, gram))
        if len(batch) >= 10000:
            connection.executemany("INSERT INTO prefixes(doc_id,gram) VALUES(?,?)", batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO prefixes(doc_id,gram) VALUES(?,?)", batch)
    connection.commit()
    if connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0:
        return {
            "document_count": 0,
            "short_document_count": 0,
            "gram_posting_count": 0,
            "gram_posting_cap": MAX_NGRAM_POSTINGS,
            "prefix_token_count": 0,
            "raw_prefix_pair_bound": 0,
        }
    prefix_tokens = connection.execute("SELECT COUNT(*) FROM prefixes").fetchone()[0]
    raw_pair_bound = connection.execute(
        "SELECT COALESCE(SUM(n*(n-1)/2),0) FROM (SELECT COUNT(*) n FROM prefixes GROUP BY gram)"
    ).fetchone()[0]
    _maybe_guard_sqlite_size(connection)
    return {
        "document_count": connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0],
        "short_document_count": short_docs,
        "gram_posting_count": gram_postings,
        "gram_posting_cap": MAX_NGRAM_POSTINGS,
        "prefix_token_count": prefix_tokens,
        "raw_prefix_pair_bound": int(raw_pair_bound),
    }


def _near_duplicate_edges(connection: sqlite3.Connection) -> dict[str, int]:
    index_summary = _prepare_prefix_index(connection)
    if index_summary["raw_prefix_pair_bound"] > MAX_PREFIX_PAIR_WORK:
        raise AcquisitionError(
            "complete Jaccard prefix join exceeds its explicit exact-work cap; no edges were skipped or approximated"
        )
    candidate_query = (
        "SELECT p1.doc_id,p2.doc_id "
        "FROM prefixes p1 JOIN prefixes p2 ON p1.gram=p2.gram AND p1.doc_id<p2.doc_id "
        "JOIN docs a ON a.doc_id=p1.doc_id JOIN docs b ON b.doc_id=p2.doc_id "
        "WHERE a.representative_group<>b.representative_group "
        "AND 10*MIN(a.gram_count,b.gram_count)>=9*MAX(a.gram_count,b.gram_count) "
        "GROUP BY p1.doc_id,p2.doc_id ORDER BY p1.doc_id,p2.doc_id"
    )
    last_pair: tuple[int, int] | None = None
    candidate_count = 0
    verified_count = 0
    rejected_count = 0
    cache: dict[int, frozenset[str]] = {}
    cache_order: list[int] = []

    def grams_for(doc_id: int) -> frozenset[str]:
        if doc_id in cache:
            return cache[doc_id]
        text = connection.execute("SELECT normalized_text FROM docs WHERE doc_id=?", (doc_id,)).fetchone()[0]
        result = frozenset(_fivegrams(text))
        cache[doc_id] = result
        cache_order.append(doc_id)
        if len(cache_order) > 128:
            expired = cache_order.pop(0)
            cache.pop(expired, None)
        return result

    cursor = connection.execute(candidate_query)
    for doc_a, doc_b in cursor:
        pair = (doc_a, doc_b)
        if pair == last_pair:
            continue
        last_pair = pair
        candidate_count += 1
        if candidate_count > MAX_PREFIX_PAIR_WORK:
            raise AcquisitionError("complete Jaccard candidate count exceeded the exact-work cap")
        set_a = grams_for(doc_a)
        set_b = grams_for(doc_b)
        intersection = len(set_a & set_b)
        union = len(set_a) + len(set_b) - intersection
        if not union:
            rejected_count += 1
            continue
        if JACCARD_DENOMINATOR * intersection < JACCARD_NUMERATOR * union:
            rejected_count += 1
            continue
        row_a = connection.execute(
            "SELECT doc_hash,representative_group FROM docs WHERE doc_id=?", (doc_a,)
        ).fetchone()
        row_b = connection.execute(
            "SELECT doc_hash,representative_group FROM docs WHERE doc_id=?", (doc_b,)
        ).fetchone()
        _add_group_edge(
            connection,
            row_a[1],
            row_b[1],
            "verified_char5gram_jaccard_at_least_0.90",
            row_a[0],
            row_b[0],
            intersection,
            union,
        )
        verified_count += 1
        if verified_count % 10000 == 0:
            connection.commit()
            _maybe_guard_sqlite_size(connection)
    connection.commit()
    return {
        **index_summary,
        "unique_candidate_pairs_verified": candidate_count,
        "threshold_rejections": rejected_count,
        "accepted_near_duplicate_edges": verified_count,
        "candidate_generation": "complete Jaccard prefix filter with document-frequency token order, exact length bound, and exact set-Jaccard verification",
        "candidate_join_pair_work_cap": MAX_PREFIX_PAIR_WORK,
        "threshold_exact_integer_test": "10*intersection >= 9*union",
        "inputs_shorter_than_5_codepoints": "retained for exact-input and source-lineage grouping; they have no raw character-5gram set and are counted explicitly, not silently discarded",
    }


def _connected_components(connection: sqlite3.Connection) -> tuple[dict[str, str], dict[str, list[str]]]:
    group_ids = [row[0] for row in connection.execute("SELECT group_id FROM source_groups ORDER BY group_id")]
    parent = {group_id: group_id for group_id in group_ids}
    size = {group_id: 1 for group_id in group_ids}

    def find(item: str) -> str:
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    for first, second in connection.execute("SELECT group_a,group_b FROM edges ORDER BY edge_key"):
        root_a = find(first)
        root_b = find(second)
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
    for group_id, component_id in group_to_component.items():
        connection.execute("UPDATE source_groups SET component_id=? WHERE group_id=?", (component_id, group_id))
    connection.commit()
    return group_to_component, component_groups


def _rank_hash(source_id: str, group_id: str) -> str:
    return hashlib.sha256(
        f"{ALLOCATION_SEED}\n{ALLOCATION_PURPOSE}\n{source_id}\n{group_id}".encode("utf-8")
    ).hexdigest()


def _allocate_components(
    connection: sqlite3.Connection,
    component_groups: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    group_rows: dict[str, dict[str, Any]] = {}
    for row in connection.execute(
        "SELECT group_id,source_id,official_partition,rank_sha256 FROM source_groups ORDER BY group_id"
    ):
        group_rows[row[0]] = {
            "group_id": row[0],
            "source_id": row[1],
            "official_partition": row[2],
            "rank_sha256": row[3],
        }
    components: dict[str, dict[str, Any]] = {}
    for component_id, gids in component_groups.items():
        rows = [group_rows[group_id] for group_id in gids]
        members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            members[row["source_id"]].append(row)
        components[component_id] = {"groups": list(gids), "members": dict(members)}

    def anchor_for(component: Mapping[str, Any], partition: str) -> tuple[str, str, str] | None:
        candidates = []
        for source_id, rows in component["members"].items():
            for row in rows:
                if row["official_partition"] == partition:
                    candidates.append((row["rank_sha256"], source_id, row["group_id"]))
        return min(candidates) if candidates else None

    used: dict[str, dict[str, int]] = {source_id: {split: 0 for split in SPLITS} for source_id in SOURCE_IDS}
    assigned: dict[str, str] = {}
    reasons: dict[str, str] = {}
    anchors: dict[str, dict[str, Any]] = {}
    unused_reasons: dict[str, str] = {}

    confirmation_candidates = []
    validation_candidates = []
    train_only_ids = []
    for component_id, component in components.items():
        if any(row["official_partition"] == "test" for rows in component["members"].values() for row in rows):
            anchor = anchor_for(component, "test")
            confirmation_candidates.append((anchor or ("f" * 64, "", ""), component_id))
        elif any(row["official_partition"] == "validation" for rows in component["members"].values() for row in rows):
            anchor = anchor_for(component, "validation")
            validation_candidates.append((anchor or ("f" * 64, "", ""), component_id))
        else:
            train_only_ids.append(component_id)

    def fits(component_id: str, split: str) -> bool:
        sources = components[component_id]["members"].keys()
        return all(used[source_id][split] < CAPS[split] for source_id in sources)

    def commit_assignment(component_id: str, split: str, reason: str, anchor: tuple[str, str, str] | None) -> None:
        assigned[component_id] = split
        reasons[component_id] = reason
        if anchor is not None:
            anchors[component_id] = {"rank_sha256": anchor[0], "source_id": anchor[1], "group_id": anchor[2]}
        for source_id in components[component_id]["members"]:
            used[source_id][split] += 1

    for anchor, component_id in sorted(confirmation_candidates):
        if fits(component_id, "confirmation"):
            commit_assignment(component_id, "confirmation", "official_test_or_component_connected_to_official_test", anchor)
        else:
            unused_reasons[component_id] = "confirmation_cap_reached_for_at_least_one_connected_source"

    # Validation candidates alternate by their frozen source-specific SHA rank;
    # cross-source connected units use the earliest-ranked validation member.
    validation_position: dict[tuple[str, str], int] = {}
    per_source_validation: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for _, component_id in validation_candidates:
        component = components[component_id]
        for source_id, rows in component["members"].items():
            candidates = [row for row in rows if row["official_partition"] == "validation"]
            if candidates:
                best = min((row["rank_sha256"], row["group_id"]) for row in candidates)
                per_source_validation[source_id].append((best[0], best[1], component_id))
    for source_id, rows in per_source_validation.items():
        ordered = sorted(rows)
        for ordinal, (rank, group_id, component_id) in enumerate(ordered):
            validation_position[(source_id, component_id)] = ordinal
    ranked_validation: list[tuple[tuple[str, str, str], str, str]] = []
    for _, component_id in validation_candidates:
        component = components[component_id]
        choices: list[tuple[str, str, str, int]] = []
        for source_id in component["members"]:
            ordinal = validation_position.get((source_id, component_id))
            if ordinal is None or ordinal >= 2 * max(CAPS["dev"], CAPS["calibration"]):
                continue
            rows = [row for row in component["members"][source_id] if row["official_partition"] == "validation"]
            if rows:
                rank, group_id = min((row["rank_sha256"], row["group_id"]) for row in rows)
                choices.append((rank, source_id, group_id, ordinal))
        if not choices:
            unused_reasons[component_id] = "validation_rank_overflow_after_two_caps"
            continue
        rank, source_id, group_id, ordinal = min(choices)
        split = "dev" if ordinal % 2 == 0 else "calibration"
        ranked_validation.append(((rank, source_id, group_id), component_id, split))
    for anchor, component_id, split in sorted(ranked_validation):
        if fits(component_id, split):
            commit_assignment(component_id, split, "ranked_official_validation_alternating_dev_calibration", anchor)
        else:
            unused_reasons[component_id] = f"{split}_cap_reached_for_at_least_one_connected_source"

    # Train-only units are ranked once per source. A duplicate crossing source
    # boundaries takes the most protective source-rank assignment (calibration,
    # then dev, then train); capacity overflow is UNUSED, never shifted to train.
    per_source_train: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for component_id in train_only_ids:
        for source_id, rows in components[component_id]["members"].items():
            train_rows = [row for row in rows if row["official_partition"] == "train"]
            if train_rows:
                rank, group_id = min((row["rank_sha256"], row["group_id"]) for row in train_rows)
                per_source_train[source_id].append((rank, group_id, component_id))
    train_recommendations: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for source_id, rows in per_source_train.items():
        for ordinal, (rank, group_id, component_id) in enumerate(sorted(rows)):
            slot = ordinal % 10
            desired = "dev" if slot in (0, 1) else "calibration" if slot in (2, 3) else "train"
            train_recommendations[component_id].append((desired, rank, source_id, group_id))
    candidate_train: list[tuple[tuple[str, str, str], str, str]] = []
    priority = {"calibration": 0, "dev": 1, "train": 2}
    for component_id, recommendations in train_recommendations.items():
        desired, rank, source_id, group_id = min(
            recommendations,
            key=lambda item: (priority[item[0]], item[1], item[2], item[3]),
        )
        candidate_train.append(((rank, source_id, group_id), component_id, desired))
    for anchor, component_id, split in sorted(candidate_train, key=lambda item: (item[0], item[1])):
        if fits(component_id, split):
            commit_assignment(component_id, split, "official_train_sha_rank_mod_10", anchor)
        else:
            unused_reasons[component_id] = f"{split}_cap_reached_for_at_least_one_connected_source"

    for component_id in components:
        if component_id not in assigned and component_id not in unused_reasons:
            unused_reasons[component_id] = "no_eligible_source_partition"
        if component_id in assigned:
            for group_id in components[component_id]["groups"]:
                connection.execute(
                    "UPDATE source_groups SET final_split=?,allocation_reason=?,exclusion_reason=NULL WHERE group_id=?",
                    (assigned[component_id], reasons[component_id], group_id),
                )
        else:
            for group_id in components[component_id]["groups"]:
                connection.execute(
                    "UPDATE source_groups SET final_split='unused',allocation_reason=NULL,exclusion_reason=? WHERE group_id=?",
                    (unused_reasons[component_id], group_id),
                )
    connection.commit()
    return {
        "assigned_components": assigned,
        "assignment_reasons": reasons,
        "assignment_anchors": anchors,
        "unused_component_reasons": unused_reasons,
        "source_split_counts": used,
        "components": components,
        "group_rows": group_rows,
        "validation_alternation": "per-source rank SHA ascending; even rank position dev, odd calibration; cross-source component uses earliest-ranked eligible validation member",
        "train_rule": "per-source eligible train-only component rank SHA ascending, mod10 0/1 dev, 2/3 calibration, 4-9 train; cross-source conflict chooses calibration before dev before train; cap overflow is unused",
        "partition_precedence": "official test-connected component, then official validation-connected component, then official train-only component",
    }


def _ordered_group_ids(
    source_id: str,
    split: str,
    group_rows: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, Mapping[str, Any]],
    assigned: Mapping[str, str],
    component_id_by_group: Mapping[str, str],
) -> list[str]:
    rows = [
        row for group_id, row in group_rows.items()
        if row["source_id"] == source_id
        and assigned.get(component_id_by_group[group_id], "unused") == split
    ]
    rows.sort(key=lambda row: (row["rank_sha256"], row["group_id"]))
    return [row["group_id"] for row in rows]


def _stats_and_split_manifests(
    connection: sqlite3.Connection,
    allocation: Mapping[str, Any],
    component_id_by_group: Mapping[str, str],
) -> tuple[dict[str, Any], dict[tuple[str, str], list[str]]]:
    group_rows = allocation["group_rows"]
    components = allocation["components"]
    assigned = allocation["assigned_components"]
    membership: dict[tuple[str, str], list[str]] = {}
    split_hashes: dict[str, Any] = {}
    for source_id in SOURCE_IDS:
        source_obj: dict[str, Any] = {}
        for split in (*SPLITS, "unused"):
            ordered = _ordered_group_ids(source_id, split, group_rows, components, assigned, component_id_by_group)
            membership[(source_id, split)] = ordered
            source_obj[split] = {
                "independent_group_count": len(ordered),
                "ordered_group_ids_sha256": canonical_sha256(ordered),
            }
        split_hashes[source_id] = source_obj
    row_counts: dict[str, Any] = {}
    for source_id in SOURCE_IDS:
        row_counts[source_id] = {}
        for split in (*SPLITS, "unused"):
            count = connection.execute(
                "SELECT COUNT(*) FROM rows r JOIN source_groups g ON g.group_id=r.group_id WHERE r.source_id=? AND g.final_split=?",
                (source_id, split),
            ).fetchone()[0]
            row_counts[source_id][split] = count
    components_by_split: dict[str, int] = {split: 0 for split in (*SPLITS, "unused")}
    for component_id in components:
        components_by_split[assigned.get(component_id, "unused")] += 1
    return {
        "group_split_hashes": split_hashes,
        "source_row_counts_by_final_split": row_counts,
        "global_component_counts_by_final_split": components_by_split,
        "component_count": len(components),
        "source_group_count": len(group_rows),
    }, membership


def _jsonl_gzip_file(path: Path) -> tuple[BinaryIO, gzip.GzipFile]:
    return _gzip_jsonl_writer(path)


def _write_private_outputs(
    scratch_dir: Path,
    private_dir: Path,
    connection: sqlite3.Connection,
    allocation: Mapping[str, Any],
    membership: Mapping[tuple[str, str], Sequence[str]],
    component_id_by_group: Mapping[str, str],
    locale_order: Sequence[str],
) -> dict[str, dict[str, Any]]:
    locale_order_clause = "CASE locale " + " ".join("WHEN ? THEN ?" for _ in locale_order) + " ELSE 1000 END"
    locale_order_values = [value for index, locale in enumerate(locale_order) for value in (locale, index)]
    outputs: dict[str, dict[str, Any]] = {}
    component_by_group = component_id_by_group
    group_manifest_tmp = scratch_dir / "groups.jsonl.gz"
    raw, compressed = _jsonl_gzip_file(group_manifest_tmp)
    try:
        for source_id in SOURCE_IDS:
            for split in (*SPLITS, "unused"):
                for group_id in membership[(source_id, split)]:
                    row = connection.execute(
                        "SELECT source_id,pinned_revision,official_partition,lineage_ids_json,input_hashes_json,rank_sha256,component_id,final_split,allocation_reason,exclusion_reason FROM source_groups WHERE group_id=?",
                        (group_id,),
                    ).fetchone()
                    row_keys = [
                        item[0]
                        for item in connection.execute(
                            "SELECT row_key FROM rows WHERE group_id=? ORDER BY "
                            + locale_order_clause
                            + ",official_partition,line_number,row_key",
                            (group_id, *locale_order_values),
                        )
                    ]
                    component_id = row[6]
                    component = allocation["components"][component_id]
                    value = {
                        "group_id": group_id,
                        "source_id": row[0],
                        "pinned_revision_for_group_id": row[1],
                        "official_partition": row[2],
                        "original_lineage_ids": json.loads(row[3]),
                        "normalized_input_sha256s": json.loads(row[4]),
                        "rank_sha256": row[5],
                        "component_id": component_id,
                        "component_member_group_count": len(component["groups"]),
                        "final_split": row[7],
                        "allocation_reason": row[8],
                        "exclusion_reason": row[9],
                        "ordered_source_row_keys": row_keys,
                    }
                    compressed.write((canonical_json(value) + "\n").encode("utf-8"))
    finally:
        compressed.close()
        raw.flush()
        os.fsync(raw.fileno())
        raw.close()
    groups_digest, groups_size = publish_immutable(group_manifest_tmp, private_dir / "groups.jsonl.gz")
    outputs["groups"] = {"path": "private/groups.jsonl.gz", "sha256": groups_digest, "size_bytes": groups_size}

    edges_tmp = scratch_dir / "group_edges.jsonl.gz"
    raw, compressed = _jsonl_gzip_file(edges_tmp)
    try:
        for edge in connection.execute(
            "SELECT group_a,group_b,reason,evidence_hash_a,evidence_hash_b,intersection_size,union_size FROM edges ORDER BY group_a,group_b,reason,evidence_hash_a,evidence_hash_b"
        ):
            compressed.write(
                (canonical_json({
                    "group_a": edge[0],
                    "group_b": edge[1],
                    "reason": edge[2],
                    "normalized_input_sha256_a": edge[3],
                    "normalized_input_sha256_b": edge[4],
                    "char5gram_intersection": edge[5],
                    "char5gram_union": edge[6],
                }) + "\n").encode("utf-8")
            )
    finally:
        compressed.close()
        raw.flush()
        os.fsync(raw.fileno())
        raw.close()
    edge_digest, edge_size = publish_immutable(edges_tmp, private_dir / "group_edges.jsonl.gz")
    outputs["edges"] = {"path": "private/group_edges.jsonl.gz", "sha256": edge_digest, "size_bytes": edge_size}
    exclusions_tmp = scratch_dir / "near_duplicate_search_exclusions.jsonl.gz"
    raw, compressed = _jsonl_gzip_file(exclusions_tmp)
    try:
        for doc_hash, group_id, codepoints, reason in connection.execute(
            "SELECT doc_hash,representative_group,normalized_codepoints,reason FROM search_exclusions ORDER BY doc_hash"
        ):
            compressed.write((canonical_json({
                "normalized_input_sha256": doc_hash,
                "representative_group_id": group_id,
                "normalized_codepoint_count": codepoints,
                "reason": reason,
                "retained_for_exact_input_and_source_lineage_grouping": True,
                "excluded_from_source_rows_or_allocation": False,
            }) + "\n").encode("utf-8"))
    finally:
        compressed.close()
        raw.flush()
        os.fsync(raw.fileno())
        raw.close()
    exclusions_digest, exclusions_size = publish_immutable(
        exclusions_tmp,
        private_dir / "near_duplicate_search_exclusions.jsonl.gz",
    )
    outputs["search_exclusions"] = {
        "path": "private/near_duplicate_search_exclusions.jsonl.gz",
        "sha256": exclusions_digest,
        "size_bytes": exclusions_size,
    }

    rows_tmp = scratch_dir / "source_rows.jsonl.gz"
    raw, compressed = _jsonl_gzip_file(rows_tmp)
    try:
        for source_id in SOURCE_IDS:
            for split in (*SPLITS, "unused"):
                for group_id in membership[(source_id, split)]:
                    cursor = connection.execute(
                        "SELECT row_key,official_partition,locale,original_id,input_text,norm_hash,source_path,line_number,payload_json "
                        "FROM rows WHERE group_id=? ORDER BY "
                        + locale_order_clause
                        + ",official_partition,line_number,row_key",
                        (group_id, *locale_order_values),
                    )
                    for row in cursor:
                        component_id = component_by_group[group_id]
                        value = {
                            "row_key": row[0],
                            "source_id": source_id,
                            "official_partition": row[1],
                            "locale": row[2],
                            "original_utterance_id": row[3] if source_id == "AmazonScience/massive" else None,
                            "input_text": row[4],
                            "normalized_input_sha256": row[5],
                            "source_file": row[6],
                            "source_line_number": row[7],
                            "group_id": group_id,
                            "component_id": component_id,
                            "final_split": allocation["assigned_components"].get(component_id, "unused"),
                            "source_record_with_original_labels_and_annotations": json.loads(row[8]),
                        }
                        compressed.write((canonical_json(value) + "\n").encode("utf-8"))
    finally:
        compressed.close()
        raw.flush()
        os.fsync(raw.fileno())
        raw.close()
    rows_digest, rows_size = publish_immutable(rows_tmp, private_dir / "source_rows.jsonl.gz")
    outputs["rows_and_labels"] = {"path": "private/source_rows.jsonl.gz", "sha256": rows_digest, "size_bytes": rows_size}

    unused_tmp = scratch_dir / "unused_groups.jsonl.gz"
    raw, compressed = _jsonl_gzip_file(unused_tmp)
    try:
        for component_id, reason in sorted(allocation["unused_component_reasons"].items()):
            compressed.write((canonical_json({
                "component_id": component_id,
                "reason": reason,
                "group_ids": allocation["components"][component_id]["groups"],
                "source_ids": sorted(allocation["components"][component_id]["members"]),
            }) + "\n").encode("utf-8"))
    finally:
        compressed.close()
        raw.flush()
        os.fsync(raw.fileno())
        raw.close()
    unused_digest, unused_size = publish_immutable(unused_tmp, private_dir / "unused_groups.jsonl.gz")
    outputs["unused"] = {"path": "private/unused_groups.jsonl.gz", "sha256": unused_digest, "size_bytes": unused_size}
    return outputs


def _source_rights(plan_sources: Mapping[str, Mapping[str, Any]], inventory_sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_id in SOURCE_IDS:
        plan = plan_sources[source_id]
        inventory = inventory_sources[source_id]
        result[source_id] = {
            "license_class": inventory["license_class"],
            "license_basis_as_recorded_in_inventory": inventory["license_basis"],
            "source_grant": plan["license_class"],
            "permitted_scope_for_this_script": "local source acquisition, grouped source manifest, and neutral research metadata only; no model training/evaluation is performed here",
            "attribution_or_modification_material": "retained under raw/ and metadata/pinned_inputs/ with exact-byte hashes",
            "legal_opinion": False,
            "upstream_or_privacy_clearance_beyond_the_committed_inventory": "not inferred or represented as independently established",
        }
    return result


def _load_source_split_sizes(inventory_sources: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    bank_sizes = inventory_sources["PolyAI/banking77"]["split_sizes"]
    massive_sizes = inventory_sources["AmazonScience/massive"]["split_sizes"]["per_locale"]
    bank = {"train": int(bank_sizes["train"]), "test": int(bank_sizes["test"])}
    massive = {partition: int(massive_sizes[partition]) for partition in PARTITIONS}
    return bank, massive


def _acquire(args: argparse.Namespace) -> dict[str, Any]:
    global ALLOCATION_SEED, CAPS

    resource_info = _set_resource_guards()
    repo = Path(__file__).resolve().parents[2]
    if repo.is_symlink() or not repo.is_dir():
        raise AcquisitionError("script repository root is unsafe")
    blobs, parsed = _load_frozen_inputs(repo)
    plan, inventory, locales, allocation_parameters = _check_frozen_metadata(parsed)
    ALLOCATION_SEED = allocation_parameters["seed"]
    CAPS = dict(allocation_parameters["caps_independent_groups_per_source"])
    required_commit = plan["parent_protocol_git"]
    _assert_committed(repo, blobs.keys(), required_commit)
    data_root = Path(plan["output_root"]).expanduser()
    if not data_root.is_absolute():
        raise AcquisitionError("frozen data root must be absolute")
    data_root = data_root.resolve(strict=False)
    try:
        data_root.relative_to(repo.resolve())
    except ValueError:
        pass
    else:
        raise AcquisitionError("raw source rows and labels must remain outside the Git repository")
    if data_root == repo.resolve():
        raise AcquisitionError("raw source root cannot be the Git repository")
    if (data_root / "metadata" / "acquisition_manifest.json").exists():
        raise AcquisitionError("complete acquisition manifest already exists; immutable evidence will not be overwritten")
    make_private_directory(data_root)
    old_umask = os.umask(0o077)
    try:
        make_private_directory(data_root / "raw")
        make_private_directory(data_root / "metadata")
        make_private_directory(data_root / "private")
        for path in (data_root / "raw", data_root / "metadata", data_root / "private"):
            make_private_directory(path)
        input_hashes = _snapshot_inputs(data_root, blobs)
        plan_sources = {source["id"]: source for source in plan["sources"]}
        inventory_sources = {source["id"]: source for source in inventory["sources"]}
        specs = _artifact_specs(plan, inventory)
        existing_records = {
            spec["artifact_id"]: record
            for spec in specs
            if (record := _read_existing_artifact_record(data_root, spec)) is not None
        }
        total_so_far = sum(int(record["size_bytes"]) for record in existing_records.values())
        if total_so_far > MAX_TOTAL_DOWNLOAD_BYTES:
            raise AcquisitionError("already acquired source objects exceed the total cap")
        artifact_records = dict(existing_records)
        for spec in specs:
            artifact_id = spec["artifact_id"]
            if artifact_id in artifact_records:
                continue
            record = _download_artifact(data_root, spec, total_so_far, args.resume)
            total_so_far += int(record["size_bytes"])
            if total_so_far > MAX_TOTAL_DOWNLOAD_BYTES:
                raise AcquisitionError("complete source download exceeds the total cap")
            artifact_records[artifact_id] = record
        # Hash all acquired files again before any source row is parsed.
        for spec in specs:
            record = artifact_records[spec["artifact_id"]]
            actual_hash, actual_size = hash_file(data_root / spec["relative_path"])
            if (actual_hash, actual_size) != (record["actual_sha256"], record["size_bytes"]):
                raise AcquisitionError("raw source bytes changed before pre-parse manifest freeze")

        archive_record = artifact_records["massive_archive_1_1"]
        scratch_context = tempfile.TemporaryDirectory(prefix="neutral-acquire-", dir=data_root / "private")
        try:
            scratch_dir = Path(scratch_context.name)
            selected_members, member_index_tmp, tar_summary = _scan_massive_archive(
                data_root / "raw/massive/amazon-massive-dataset-1.1.tar.gz",
                archive_record["actual_sha256"],
                locales,
                scratch_dir,
            )
            member_index_path = data_root / "private" / "massive_tar_members.jsonl.gz"
            member_index_hash, member_index_size = publish_immutable(member_index_tmp, member_index_path)
            inventory_primary = {
                item["url"]: item["sha256"]
                for source_id in SOURCE_IDS
                for item in inventory_sources[source_id].get("primary_sources", [])
            }
            raw_source_manifest = {
                "schema": "vey.neutral.acquired-source-files.v1",
                "parent_protocol_git": plan["parent_protocol_git"],
                "source_plan_parent_protocol_git": plan["parent_protocol_git"],
                "protocol_and_source_input_hashes": input_hashes,
                "source_artifacts": [artifact_records[spec["artifact_id"]] for spec in specs],
                "pinned_inventory_checksums_by_url": inventory_primary,
                "massive_archive": {
                    "release_name": "1.1",
                    "source_filename_checksum_pinned_before_acquisition": False,
                    "actual_archive_sha256_frozen_before_parsing": archive_record["actual_sha256"],
                    "member_inspection": tar_summary,
                    "member_hash_manifest": {
                        "path": "private/massive_tar_members.jsonl.gz",
                        "sha256": member_index_hash,
                        "size_bytes": member_index_size,
                    },
                    "tar_members_are_inspected_and_hashed_but_never_extracted_or_executed": True,
                },
                "download_limits": {
                    "max_object_bytes": MAX_OBJECT_BYTES,
                    "max_total_download_bytes": MAX_TOTAL_DOWNLOAD_BYTES,
                    "max_tar_unpacked_bytes": MAX_TAR_UNPACKED_BYTES,
                    "max_tar_member_bytes": MAX_TAR_MEMBER_BYTES,
                    "http_concurrency": 1,
                    "resume_policy": "raw-only; exact prefix size+SHA-256, same pinned URL and unchanged recorded ETag; incomplete unknown bytes fail closed",
                },
                "rights_and_citation": _source_rights(plan_sources, inventory_sources),
                "legal_note": "Committed inventory and publisher-provided license/NOTICE material are recorded evidence, not legal advice or independent clearance of unknown upstream rights.",
            }
            source_manifest_path = data_root / "metadata" / "source_files.json"
            write_json_immutable(source_manifest_path, raw_source_manifest)

            database_path = scratch_dir / "grouping.sqlite3"
            connection = _initialize_database(database_path)
            try:
                bank_expected, massive_expected = _load_source_split_sizes(inventory_sources)
                banking_counts = {}
                for partition in ("train", "test"):
                    artifact_id = "banking77_train_csv" if partition == "train" else "banking77_test_csv"
                    csv_path = data_root / specs[[s["artifact_id"] for s in specs].index(artifact_id)]["relative_path"]
                    count = _csv_rows(csv_path, "PolyAI/banking77", partition, connection)
                    if count != bank_expected[partition]:
                        raise AcquisitionError("Banking77 observed split count differs from pinned source metadata")
                    banking_counts[partition] = count
                massive_stats, massive_counts = _parse_massive_archive(
                    data_root / "raw/massive/amazon-massive-dataset-1.1.tar.gz",
                    archive_record["actual_sha256"],
                    selected_members,
                    locales,
                    massive_expected,
                    connection,
                )
                _build_source_groups(connection, plan_sources)
                _maybe_guard_sqlite_size(connection)
                near_summary = _near_duplicate_edges(connection)
                group_to_component, component_groups = _connected_components(connection)
                allocation = _allocate_components(connection, component_groups)
                count_summary, membership = _stats_and_split_manifests(
                    connection,
                    allocation,
                    group_to_component,
                )
                private_outputs = _write_private_outputs(
                    scratch_dir,
                    data_root / "private",
                    connection,
                    allocation,
                    membership,
                    group_to_component,
                    locales,
                )
                judgment_summary = massive_stats.report(locales)
                group_edge_count = connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                exact_edge_count = connection.execute("SELECT COUNT(*) FROM edges WHERE reason='exact_normalized_input'").fetchone()[0]
                near_edge_count = connection.execute(
                    "SELECT COUNT(*) FROM edges WHERE reason='verified_char5gram_jaccard_at_least_0.90'"
                ).fetchone()[0]
                report = {
                    "schema": SCRIPT_SCHEMA,
                    "study_status": "ACQUISITION_AND_INPUT_ONLY_GROUPING; NO MODEL USE",
                    "source_scope": SOURCE_IDS,
                    "sources": {
                        "PolyAI/banking77": {
                            "raw_revision": plan_sources["PolyAI/banking77"]["raw_revision"],
                            "hf_loader_revision_not_used_for_raw_csv": plan_sources["PolyAI/banking77"]["hf_loader_revision"],
                            "official_row_counts": banking_counts,
                            "source_exposure": plan_sources["PolyAI/banking77"]["exposure"],
                            "source_role": "legacy intent/K=77 replication, not fresh transfer",
                            "input_schema": ["text", "category"],
                        },
                        "AmazonScience/massive": {
                            "raw_release": plan_sources["AmazonScience/massive"]["raw_release"],
                            "hf_loader_revision": plan_sources["AmazonScience/massive"]["hf_loader_revision"],
                            "official_row_counts_by_locale_partition": massive_counts,
                            "fixed_locale_count": len(locales),
                            "fixed_locales": locales,
                            "excluded_locale": plan_sources["AmazonScience/massive"]["excluded_locale"],
                            "source_exposure": plan_sources["AmazonScience/massive"]["exposure"],
                            "documentation_example_exposure": {
                                "public_README_examples_seen_before_split_row_acquisition": True,
                                "split_rows_had_not_been_acquired_or_opened_before_that_documentation_exposure": True,
                                "README_examples_are_disclosed_as_public_exposure_not_as_source_split_row_acquisition": True,
                                "README_is_saved_with_its_pinned_byte_hash": True,
                            },
                            "source_role": "public intent/slot label replication only; no invented Boolean, Score, alias, or composition gold",
                            "original_utterance_id_field": "id",
                            "cross_locale_verification": {
                                "all_51_locales_per_original_id": True,
                                "partition_scenario_intent_agree": True,
                                "translations_are_descendants_not_independent_groups": True,
                            },
                            "judgment_metadata_definition_source": "pinned public MASSIVE README; definitions below are native source metadata only",
                            "judgment_native_value_metadata_from_README": MASSIVE_README_JUDGMENT_METADATA,
                            "judgment_values_are_not_mapped_to_model_primitives_or_gold": True,
                            "judgment_metadata_by_locale_partition": judgment_summary,
                        },
                    },
                    "grouping": {
                        "normalization": "Unicode NFKC, casefold, collapse whitespace, strip; punctuation, numbers, and negation preserved",
                        "source_group_id": "SHA256(canonical JSON [source_id,pinned_revision,sorted_original_lineage_ids,sorted_normalized_input_sha256s]); labels and outputs excluded",
                        "near_duplicate_search": near_summary,
                        "edge_counts": {
                            "all_verified_edges": group_edge_count,
                            "exact_normalized_input_edges": exact_edge_count,
                            "verified_near_duplicate_edges": near_edge_count,
                        },
                        "cross_source_english_duplicate_handling": "all Banking77 and all MASSIVE locale texts, including en-US, enter the same exact-hash and complete near-duplicate graph before split allocation",
                        "template_lineage_disclosure": {
                            "banking77": "No stable source utterance lineage ID is present in the pinned CSV; deterministic official-file row ordinals are retained, while paraphrase/customer/template lineage remains unresolved.",
                            "massive": "The source original utterance ID joins the 51 locales; additional scenario/template/author lineage is not fabricated as independent groups.",
                            "independence_claim": False,
                        },
                    },
                    "allocation": {
                        "seed": ALLOCATION_SEED,
                        "purpose": ALLOCATION_PURPOSE,
                        "rank_formula": "SHA256(UTF8(seed+'\\n'+purpose+'\\n'+source_id+'\\n'+group_id)), ascending; tie group_id",
                        "caps_independent_groups_per_source": CAPS,
                        "partition_precedence": allocation["partition_precedence"],
                        "validation_rule": allocation["validation_alternation"],
                        "official_train_rule": allocation["train_rule"],
                        "counts_and_hashes": count_summary,
                        "all_full_ordered_group_ids_and_rows": "private/groups.jsonl.gz and private/source_rows.jsonl.gz",
                        "all_edge_reasons": "private/group_edges.jsonl.gz",
                        "all_unused_reasons": "private/unused_groups.jsonl.gz",
                        "source_partitions_and_exposure_are_preserved": True,
                        "labels_or_outcomes_used_for_grouping_or_allocation": False,
                    },
                    "private_artifacts": private_outputs,
                    "raw_source_manifest": {
                        "path": "metadata/source_files.json",
                        "sha256": sha256_bytes(source_manifest_path.read_bytes()),
                        "massive_tar_member_manifest": {
                            "path": "private/massive_tar_members.jsonl.gz",
                            "sha256": member_index_hash,
                            "size_bytes": member_index_size,
                        },
                    },
                    "resource_guards": resource_info,
                    "claims": {
                        "model_executed": False,
                        "model_outputs_acquired": False,
                        "quality_metrics_measured": False,
                        "capabilities_earned": [],
                        "overall_noninferiority_established": False,
                        "public_tests_are_universally_fresh": False,
                        "source_artifacts_are_model_study_results": False,
                        "raw_rows_and_plaintext_labels_in_git": False,
                    },
                }
                report_path = data_root / "private" / "acquisition_report.json"
                write_json_immutable(report_path, report)
                output_hashes = {
                    "metadata/source_files.json": sha256_bytes(source_manifest_path.read_bytes()),
                    "private/acquisition_report.json": sha256_bytes(report_path.read_bytes()),
                    **{value["path"]: value["sha256"] for value in private_outputs.values()},
                    "private/massive_tar_members.jsonl.gz": member_index_hash,
                }
                final_manifest = {
                    "schema": SCRIPT_SCHEMA,
                    "evidence_class": "MEASURED source acquisition bytes and metadata; prospective neutral grouping/allocation; no model outputs",
                    "parent_protocol_git": plan["parent_protocol_git"],
                    "source_pins": input_hashes,
                    "source_artifact_count": len(artifact_records),
                    "source_artifact_hashes": {
                        artifact_id: {
                            "source_id": record["source_id"],
                            "path": next(spec["relative_path"] for spec in specs if spec["artifact_id"] == artifact_id),
                            "size_bytes": record["size_bytes"],
                            "actual_sha256": record["actual_sha256"],
                            "expected_sha256": record["expected_sha256"],
                            "etag_is_checksum": False,
                        }
                        for artifact_id, record in artifact_records.items()
                    },
                    "raw_source_manifest_sha256": sha256_bytes(source_manifest_path.read_bytes()),
                    "outputs": output_hashes,
                    "report": "private/acquisition_report.json",
                    "claims": report["claims"],
                }
                final_path = data_root / "metadata" / "acquisition_manifest.json"
                final_sha = write_json_immutable(final_path, final_manifest)
            finally:
                connection.close()
        finally:
            scratch_context.cleanup()
    finally:
        os.umask(old_umask)

    public_summary = {
        "status": "acquired_and_grouped",
        "data_root": str(data_root),
        "source_artifact_count": len(artifact_records),
        "source_bytes": total_so_far,
        "massive_release": "1.1",
        "massive_locale_count": len(locales),
        "banking77_rows": banking_counts,
        "massive_rows_per_locale": massive_counts,
        "independent_group_counts_by_source_split": count_summary["group_split_hashes"],
        "verified_edge_counts": {
            "all": group_edge_count,
            "exact_input": exact_edge_count,
            "near_duplicate": near_edge_count,
        },
        "source_sha256s": {artifact_id: record["actual_sha256"] for artifact_id, record in artifact_records.items()},
        "manifest_sha256": final_sha,
        "rights": {source_id: inventory_sources[source_id]["license_class"] for source_id in SOURCE_IDS},
        "source_artifacts_not_model_results": True,
        "model_executed": False,
        "capabilities_earned": [],
        "overall_noninferiority_established": False,
        "resource_guards": resource_info,
    }
    return public_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire only the pinned Banking77 raw CSV and MASSIVE 1.1 archive, then freeze private input-only groups."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire", help="download, hash, parse metadata, group, and allocate the pinned sources")
    acquire.add_argument(
        "--resume",
        action="store_true",
        help="resume only a recorded raw partial whose exact prefix SHA-256 and ETag still match; never resumes parsing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command != "acquire":
        return 2
    try:
        result = _acquire(args)
    except AcquisitionError as exc:
        print(f"acquisition stopped safely: {exc}", file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, tarfile.TarError, urllib.error.URLError) as exc:
        print(f"acquisition stopped safely: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
