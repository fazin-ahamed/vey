#!/usr/bin/env python3
"""Prepare opaque ECA-2 final reviews and build the reviewed, final-only corpus.

This CLI does not review meanings, capture encoder features, fit, or score models.
Existing artifacts are never overwritten, including partial failed-build artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Sequence

import ephemeral_pages_build as base

PROTOCOL_PATH = Path(__file__).with_name("ephemeral_pages_atomic_protocol.json")
OLD_ROOT = Path("/home/fazinahamed/Documents/vey-data/decisionmix/endgame/ephemeral-pages-v1")
SOURCE_SCHEMA = "vey.eca2.authored-final-source.v1"
PREPARE_SCHEMA = "vey.eca2.final-preparation-manifest.v1"
INVENTORY_SCHEMA = "vey.eca2.final-source-inventory.v1"
BUILD_SCHEMA = "vey.eca2.final-build-manifest.v1"
WORLD_SEED = 271828
NAMESPACE = "eca2-child-local:271828"
SOURCE_FILE = "source/eca2_final_source.json"
INVENTORY_FILE = "source/eca2_final_inventory.json"
PREPARE_FILE = "source/eca2_final_prepare_manifest.json"
PACKET_FILE = "audit/eca2_opaque_review_packet.json"
KEY_FILE = "audit/eca2_sealed_target_key.json"
FINAL_FILE = "corpus/final.jsonl"
DECISION_FILE = "corpus/decision_ledger.jsonl"
WORLD_FILE = "corpus/world_ledger.jsonl"
BUILD_FILE = "corpus/eca2_final_build_manifest.json"


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe regular file: {path}")


def _json(path: Path) -> dict[str, Any]:
    _regular(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    _regular(path)
    digest, size = base._sha_file(path)
    return {"sha256": digest, "bytes": size}


def _protocol(root_arg: str | None) -> tuple[Path, dict[str, Any], bytes]:
    protocol = _json(PROTOCOL_PATH)
    raw = PROTOCOL_PATH.read_bytes()
    _, parent_raw, _ = base._protocol()
    if protocol.get("experiment") != "ECA-2 child-local supervision correctness replication":
        raise ValueError("unexpected ECA-2 protocol identity")
    if protocol.get("parent_protocol_sha256") != base._sha_bytes(parent_raw):
        raise ValueError("ECA-1 parent protocol differs from the ECA-2 pin")
    final = protocol.get("new_final", {})
    expected = {"worlds": 320, "world_seed": WORLD_SEED, "grade_and_exact_generator_seed": base.SEED}
    if any(final.get(key) != value for key, value in expected.items()):
        raise ValueError("ECA-2 final schedule differs from the frozen protocol")
    root = Path(protocol["output_root"]).expanduser()
    if root_arg is not None and Path(root_arg).expanduser().absolute() != root.absolute():
        raise ValueError("--root must equal the ECA-2 protocol output_root")
    if root.absolute() == OLD_ROOT.absolute() or root.is_symlink():
        raise ValueError("ECA-2 root must be new and not a symlink")
    return root.absolute(), protocol, raw


def _runtime_pins(protocol_raw: bytes) -> dict[str, Any]:
    return {
        "protocol_sha256": base._sha_bytes(protocol_raw),
        "builder": {"path": str(Path(__file__).resolve()), **_file_record(Path(__file__))},
        "helper_builder": {"path": str(Path(base.__file__).resolve()), **_file_record(Path(base.__file__))},
        "canonical_dependency_sha256": {
            relative: base._sha_file(base.CANONICAL_ROOT / relative)[0]
            for relative in base.EXPECTED_CANONICAL_FILES
        },
        "grade_intervention_amendment": _file_record(base.PROTOCOL_PATH.with_name("ephemeral_pages_grade_amendment.json")),
    }


def _text_hash(text: str) -> str:
    return base._sha_bytes(base._normal_text(text).encode("utf-8"))


def _old_source_inventory() -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    source_path = OLD_ROOT / "source/eca_authored_source_v1.json"
    inventory_path = OLD_ROOT / "source/eca_source_inventory_v1.json"
    source = _json(source_path)
    inventory = _json(inventory_path)
    if source.get("schema") != base.SOURCE_SCHEMA or inventory.get("schema") != base.INVENTORY_SCHEMA:
        raise ValueError("old source/inventory schema mismatch")
    source_pin = _file_record(source_path)
    if inventory.get("source_sha256") != source_pin["sha256"]:
        raise ValueError("old inventory does not bind the retained source")
    if tuple(source.get("families", [])) != base.FAMILIES or len(source.get("properties", [])) != 60:
        raise ValueError("old source must retain all sixty A/B/C properties")
    ids: set[str] = set()
    texts: set[str] = set()
    for prop in source["properties"]:
        ids.add(prop["property_id"])
        texts.update(_text_hash(prop[key]) for key in ("title", "criterion", "description"))
        texts.update(_text_hash(text) for text in prop["grade_rubrics"])
        for collection, id_key in (("page_wordings", "wording_id"), ("questions", "question_id")):
            for item in prop[collection]:
                ids.add(item[id_key])
                texts.add(_text_hash(item["text"]))
    inventory_texts = {
        digest
        for split in inventory["counts_by_split"].values()
        for key in ("unique_page_text_sha256", "unique_question_sha256")
        for digest in split[key]
    }
    texts.update(inventory_texts)
    lineage = {
        "source": {"path": str(source_path), **source_pin},
        "inventory": {"path": str(inventory_path), **_file_record(inventory_path)},
    }
    return lineage, ids, texts, inventory_texts


def _source_inventory(source: Mapping[str, Any], source_sha: str) -> dict[str, Any]:
    old_lineage, old_ids, old_texts, _ = _old_source_inventory()
    if source.get("schema") != SOURCE_SCHEMA or type(source.get("seed")) is not int or source["seed"] != WORLD_SEED:
        raise ValueError("new final source schema/seed mismatch")
    if source.get("grade_order") != list(base.GRADE_ORDER) or source.get("families") != list(base.FAMILIES):
        raise ValueError("new source grade/family order differs from the frozen inventory")
    properties = source.get("properties")
    if not isinstance(properties, list) or len(properties) != 20:
        raise ValueError("new source requires exactly twenty properties")
    if [prop.get("family") for prop in properties if isinstance(prop, dict)] != list(base.FAMILIES):
        raise ValueError("new source requires one property per original family, in original order")
    ids: set[str] = set()
    semantic_hashes: set[str] = set()
    page_hashes: set[str] = set()
    question_hashes: set[str] = set()

    def identifier(value: Any) -> None:
        if not isinstance(value, str) or not value.strip() or value in ids or value in old_ids:
            raise ValueError("new source IDs must be nonempty, unique and disjoint from old source IDs")
        ids.add(value)

    def text(value: Any, *, unique: bool = False) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("new semantic text must be nonempty")
        if re.search(r"\b(?:grade|level)\s*\d|\b[0-4]\b", value, re.IGNORECASE):
            raise ValueError("numeric or literal grade token in new semantic text")
        digest = _text_hash(value)
        if digest in old_texts or (unique and digest in page_hashes | question_hashes):
            raise ValueError("new semantic text overlaps old inventory or duplicates another review item")
        semantic_hashes.add(digest)
        return digest

    for prop in properties:
        identifier(prop.get("property_id"))
        if prop.get("field_code") != "C" or prop.get("grade_order") != list(base.GRADE_ORDER):
            raise ValueError("new final properties require partition C and five ordered grades")
        for key in ("title", "criterion", "description"):
            text(prop.get(key))
        rubrics = prop.get("grade_rubrics")
        if not isinstance(rubrics, list) or len(rubrics) != 5:
            raise ValueError("each new property requires five extent rubrics")
        if len({text(value) for value in rubrics}) != 5:
            raise ValueError("extent rubrics must be distinct within each property")
        wordings = prop.get("page_wordings")
        questions = prop.get("questions")
        if not isinstance(wordings, list) or len(wordings) != 15 or not isinstance(questions, list) or len(questions) != 6:
            raise ValueError("each property requires fifteen pages and six questions")
        page_slots: set[tuple[int, int]] = set()
        question_slots: set[tuple[int, int]] = set()
        for item in wordings:
            if not isinstance(item, dict) or item.get("split") != "final":
                raise ValueError("page wordings must be final-only objects")
            grade, variant = item.get("grade"), item.get("variant")
            if type(grade) is not int or grade not in base.GRADE_ORDER or type(variant) is not int or variant not in (0, 1, 2):
                raise ValueError("page grade/variant is outside the authored schedule")
            page_slots.add((grade, variant))
            identifier(item.get("wording_id"))
            page_hashes.add(text(item.get("text"), unique=True))
        for item in questions:
            if not isinstance(item, dict) or item.get("split") != "final":
                raise ValueError("questions must be final-only objects")
            orientation, variant = item.get("orientation"), item.get("variant")
            if type(orientation) is not int or orientation not in (1, -1) or type(variant) is not int or variant not in (0, 1, 2):
                raise ValueError("question orientation/variant is outside the authored schedule")
            question_slots.add((orientation, variant))
            identifier(item.get("question_id"))
            question_hashes.add(text(item.get("text"), unique=True))
        if page_slots != {(grade, variant) for grade in base.GRADE_ORDER for variant in range(3)}:
            raise ValueError("missing or duplicate property/grade/page variant")
        if question_slots != {(orientation, variant) for orientation in (1, -1) for variant in range(3)}:
            raise ValueError("missing or duplicate property/orientation/question variant")
    return {
        "schema": INVENTORY_SCHEMA, "source_sha256": source_sha,
        "grade_order": list(base.GRADE_ORDER), "families": {family: 1 for family in base.FAMILIES},
        "property_count": 20, "grade_rubric_count": 100, "page_wording_count": 300, "question_count": 120,
        "source_ids": sorted(ids), "semantic_text_sha256": sorted(semantic_hashes),
        "counts_by_split": {"final": {"page_wordings": 300, "questions": 120,
            "unique_page_text_sha256": sorted(page_hashes), "unique_question_sha256": sorted(question_hashes)}},
        "old_source_lineage": old_lineage, "old_source_id_overlap": 0, "old_normalized_text_overlap": 0,
        "meaning_distinctness": "Independent opaque meaning review required; exact-text nonoverlap is not semantic proof.",
    }


def _prepare(root: Path, protocol_raw: bytes) -> dict[str, Any]:
    base._private_directory(root, create=False)
    base._private_directory(root / "source", create=False)
    source = _json(root / SOURCE_FILE)
    source_pin = _file_record(root / SOURCE_FILE)
    inventory = _source_inventory(source, source_pin["sha256"])
    packet, key = base._make_review_files(source, source_pin["sha256"])
    payloads = {INVENTORY_FILE: inventory, PACKET_FILE: packet, KEY_FILE: key}
    destinations = [root / relative for relative in (*payloads, PREPARE_FILE)]
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("prepared final artifacts already exist; preserve lineage rather than overwrite")
    base._private_directory(root / "audit", create=True)
    files = {SOURCE_FILE: source_pin}
    for relative, value in payloads.items():
        encoded = base._canonical_json(value)
        files[relative] = {"sha256": base._write_exclusive(root / relative, encoded), "bytes": len(encoded)}
    manifest = {
        "schema": PREPARE_SCHEMA, "root": str(root), "source_sha256": source_pin["sha256"],
        **_runtime_pins(protocol_raw), "files": files, "old_source_lineage": inventory["old_source_lineage"],
        "audit_item_count": 420, "audit_items_by_type": {"page": 300, "question": 120},
        "world_namespace": NAMESPACE, "world_seed": WORLD_SEED, "grade_and_exact_generator_seed": base.SEED,
    }
    base._write_exclusive(root / PREPARE_FILE, base._canonical_json(manifest))
    return manifest


def _load_prepared(root: Path, protocol_raw: bytes) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for directory in (root, root / "source", root / "audit"):
        base._private_directory(directory, create=False)
    manifest = _json(root / PREPARE_FILE)
    if manifest.get("schema") != PREPARE_SCHEMA or manifest.get("root") != str(root):
        raise ValueError("prepared ECA-2 root/schema mismatch")
    pins = _runtime_pins(protocol_raw)
    if any(manifest.get(key) != value for key, value in pins.items()):
        raise ValueError("protocol, builder, helper, amendment or canonical dependency changed after preparation")
    for relative in (SOURCE_FILE, INVENTORY_FILE, PACKET_FILE, KEY_FILE):
        if manifest.get("files", {}).get(relative) != _file_record(root / relative):
            raise ValueError(f"prepared ECA-2 file hash/size mismatch: {relative}")
    source = _json(root / SOURCE_FILE)
    source_sha = manifest["files"][SOURCE_FILE]["sha256"]
    inventory = _source_inventory(source, source_sha)
    if _json(root / INVENTORY_FILE) != inventory or manifest.get("old_source_lineage") != inventory["old_source_lineage"]:
        raise ValueError("new or retained source inventory changed after preparation")
    expected_packet, expected_key = base._make_review_files(source, source_sha)
    packet = _json(root / PACKET_FILE)
    if packet != expected_packet or _json(root / KEY_FILE) != expected_key:
        raise ValueError("opaque packet/sealed key differs from the pinned source")
    return source, inventory, packet, manifest


def _review_receipt(root: Path, manifest: Mapping[str, Any], packet: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    _regular(receipt_path)
    if receipt_path.absolute().parent != root / "audit":
        raise ValueError("new final receipt must be directly inside the ECA-2 audit directory")
    receipt = _json(receipt_path)
    lineage = {
        "protocol_path": str(PROTOCOL_PATH.resolve()), "protocol_sha256": manifest["protocol_sha256"],
        "source_file": SOURCE_FILE, "packet_file": PACKET_FILE,
        "prepare_manifest_file": PREPARE_FILE,
        "prepare_manifest_sha256": _file_record(root / PREPARE_FILE)["sha256"],
    }
    if receipt.get("eca2_lineage") != lineage:
        raise ValueError("receipt must bind the ECA-2 protocol, source/packet paths and exact preparation manifest")
    attestations = receipt.get("independence", {})
    required = ("independent_of_authorship", "opaque_packet_only", "no_model_outputs_or_weights", "no_old_final_performance")
    if any(attestations.get(key) is not True for key in required):
        raise ValueError("receipt lacks independent, opaque, model-blind meaning review attestations")
    raw_relative = receipt.get("raw_reviews_file")
    if not isinstance(raw_relative, str):
        raise ValueError("receipt must preserve raw reviews")
    raw_path = root / raw_relative
    if Path(raw_relative).is_absolute() or raw_path.parent != root / "audit":
        raise ValueError("raw new-final reviews must be directly inside ECA-2 audit, not old-root lineage")
    _regular(raw_path)
    result = base._validate_receipt(root, manifest["source_sha256"], manifest["files"][PACKET_FILE]["sha256"], packet, receipt_path)
    return {**result, "receipt_file": str(receipt_path.relative_to(root)), "raw_reviews_file": raw_relative,
        "eca2_lineage": lineage, "independence": dict(attestations)}


def _old_ids(path: Path, key: str) -> tuple[set[str], dict[str, Any]]:
    # Project identifiers only: do not deserialize or inspect old final outcomes.
    _regular(path)
    pattern = re.compile(rb'"' + key.encode("ascii") + rb'"\s*:\s*"([^"\\]+)"')
    ids: set[str] = set()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            if not line.strip():
                continue
            matches = pattern.findall(line)
            if len(matches) != 1:
                raise ValueError(f"old ledger must contain one plain {key} per record: {path}")
            identifier = matches[0].decode("utf-8")
            if identifier in ids:
                raise ValueError(f"old ledger contains duplicate {key}: {path}")
            ids.add(identifier)
    if not ids:
        raise ValueError(f"old ID inventory is empty: {path}")
    return ids, {"path": str(path), "sha256": digest.hexdigest(), "bytes": size, "identifier_count": len(ids)}


def _worlds(source: Mapping[str, Any]) -> list[base.World]:
    by_family = {prop["family"]: prop for prop in source["properties"]}
    world_ids = [base._stable_id("world", NAMESPACE, slot, length=28) for slot in range(320)]
    ranked = sorted(world_ids, key=lambda wid: hashlib.sha256(f"eca-world-rank:{NAMESPACE}:{wid}".encode()).hexdigest())
    exposures: Counter[str] = Counter()
    worlds: list[base.World] = []
    for index, world_id in enumerate(ranked):
        families = tuple(base.FAMILIES[(4 * index + offset) % 20] for offset in range(4))
        exposure = {family: exposures[family] for family in families}
        exposures.update(families)
        worlds.append(base.World(world_id, "final", index, index, families,
            tuple(by_family[family] for family in families), exposure))
    if len(set(world_ids)) != 320 or any(exposures[family] != 64 for family in base.FAMILIES):
        raise ValueError("ECA-2 world uniqueness/family balance failed")
    return worlds


def _rows(world: base.World, properties: Mapping[str, Mapping[str, Any]], by_family: Mapping[str, list[Mapping[str, Any]]], protocol_sha: str) -> Iterator[base.DecisionIR]:
    candidates, blocks, owners, grades, fields, exact, ordinals = base._build_world_state(world, world.properties)
    specs, _ = base._query_specs(world, properties, by_family)
    if len(specs) != 11:
        raise ValueError("expected eleven main queries per ECA-2 world")
    present = {prop["property_id"] for prop in world.properties}
    count = 0
    for spec_index, spec in enumerate(specs):
        row = base._make_row(world, spec, candidates, blocks, owners, grades, fields, exact, ordinals,
            variant="base", extra_metadata={"atomic_supervision": "child-local-v1", "protocol_sha256": protocol_sha,
                "world_namespace": NAMESPACE, "world_seed": WORLD_SEED, "grade_and_exact_generator_seed": base.SEED})
        row = base._replace_row(replace(row, source="eca2_authored"))
        row = base._add_parent(row, row.id)
        yield row
        count += 1
        for term_index, term in enumerate(spec.terms):
            if term.property_id not in present:
                continue
            for derived in (base._erase_variant(row, term.property_id),
                base._contradiction_variant(row, term.property_id, "final", properties),
                base._grade_change_variant(row, term_index, properties, "final")):
                yield base._add_parent(derived, row.id)
                count += 1
        for index in (1, 2):
            yield base._add_parent(base._candidate_permutation(row, index), row.id)
            count += 1
        yield base._add_parent(base._rename_candidates(row), row.id)
        yield base._add_parent(base._page_reorder(row), row.id)
        count += 2
        if len(spec.terms) == 2:
            yield base._add_parent(base._question_reorder(row), row.id)
            count += 1
        if spec_index == 0:
            for k in (2, 8, 16, 32, 64):
                yield base._add_parent(base._candidate_count_probe(row, world, k, properties), row.id)
                count += 1
    if count != 98:
        raise ValueError(f"world {world.world_id} emitted {count} rows instead of 98")


def _writer(path: Path) -> Any:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    return os.fdopen(fd, "wb")


def _build(root: Path, protocol_raw: bytes, receipt_path: Path) -> dict[str, Any]:
    source, inventory, packet, prepared = _load_prepared(root, protocol_raw)
    audit = _review_receipt(root, prepared, packet, receipt_path)
    old_world_ids, world_id_pin = _old_ids(OLD_ROOT / "corpus/world_ledger.jsonl", "world_id")
    old_row_ids, row_id_pin = _old_ids(OLD_ROOT / "corpus/decision_ledger.jsonl", "decision_id")
    worlds = _worlds(source)
    if old_world_ids.intersection(world.world_id for world in worlds):
        raise ValueError("ECA-2 worlds overlap an ECA-1 world ID")
    properties, by_family = base._property_maps(source)
    for relative in (FINAL_FILE, DECISION_FILE, WORLD_FILE, BUILD_FILE):
        path = root / relative
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite final corpus artifact: {path}")
    # Only a complete accepted review can reach creation of the corpus directory/writers.
    base._private_directory(root / "corpus", create=True)
    split_hashes = {"final": hashlib.sha256()}
    split_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    question_hashes: dict[str, set[str]] = {"final": set()}
    page_hashes: dict[str, set[str]] = {"final": set()}
    row_ids: set[str] = set()
    decision_hash = hashlib.sha256()
    world_hash = hashlib.sha256()
    grade_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    with ExitStack() as stack:
        final_stream = stack.enter_context(_writer(root / FINAL_FILE))
        decision_stream = stack.enter_context(_writer(root / DECISION_FILE))
        world_stream = stack.enter_context(_writer(root / WORLD_FILE))
        for world in worlds:
            family_counts.update(world.families)
            _, _, _, grades, _, _, _ = base._build_world_state(world, world.properties)
            grade_counts.update(str(grade) for grade in grades.values())
            world_record = {"world_id": world.world_id, "split": "final", "split_index": world.local_index,
                "global_hash_rank": world.rank, "families": list(world.families),
                "field_keys": [prop["property_id"] for prop in world.properties], "field_codes": ["C"] * 4,
                "exposure_index_by_family": dict(world.exposure_by_family), "world_namespace": NAMESPACE}
            encoded = base._canonical_json(world_record)
            world_stream.write(encoded)
            world_hash.update(encoded)
            for row in _rows(world, properties, by_family, prepared["protocol_sha256"]):
                if row.id in old_row_ids:
                    raise ValueError(f"ECA-2 row overlaps ECA-1 decision ID: {row.id}")
                base._emit_row(row, final_stream, decision_stream, split_hashes, split_counts,
                    variant_counts, question_hashes, page_hashes, row_ids, decision_hash)
        for stream in (final_stream, decision_stream, world_stream):
            stream.flush()
            os.fsync(stream.fileno())
    if split_counts["final"] != 320 * 98 or len(row_ids) != 320 * 98:
        raise ValueError("ECA-2 final row count/uniqueness invariant failed")
    old_lineage, _, _, old_texts = _old_source_inventory()
    if old_lineage != inventory["old_source_lineage"]:
        raise ValueError("retained source changed during final build")
    if old_texts.intersection(page_hashes["final"] | question_hashes["final"]):
        raise ValueError("generated ECA-2 semantic inputs overlap the old source inventory")
    outputs = {relative: _file_record(root / relative) for relative in (FINAL_FILE, DECISION_FILE, WORLD_FILE)}
    outputs[FINAL_FILE]["rows"] = outputs[DECISION_FILE]["rows"] = split_counts["final"]
    outputs[WORLD_FILE]["rows"] = 320
    if outputs[FINAL_FILE]["sha256"] != split_hashes["final"].hexdigest() or outputs[DECISION_FILE]["sha256"] != decision_hash.hexdigest() or outputs[WORLD_FILE]["sha256"] != world_hash.hexdigest():
        raise ValueError("streamed outputs changed before final manifest sealing")
    manifest = {"schema": BUILD_SCHEMA, "atomic_supervision": "child-local-v1", **_runtime_pins(protocol_raw),
        "source_sha256": prepared["source_sha256"],
        "source_inventory_sha256": prepared["files"][INVENTORY_FILE]["sha256"],
        "opaque_packet_sha256": prepared["files"][PACKET_FILE]["sha256"],
        "sealed_key_sha256": prepared["files"][KEY_FILE]["sha256"],
        "prepare_manifest": {"path": PREPARE_FILE, **_file_record(root / PREPARE_FILE)}, "receipt": audit,
        "world_namespace": NAMESPACE, "world_seed": WORLD_SEED, "grade_and_exact_generator_seed": base.SEED,
        "grade_order": {"integer_values": list(base.GRADE_ORDER), "normalized_values": [0.0, 0.25, 0.5, 0.75, 1.0]},
        "world_counts": {"final": 320}, "world_count": 320, "rows_per_world": 98,
        "row_count": split_counts["final"], "row_counts_by_split": dict(split_counts),
        "variant_counts": dict(variant_counts), "family_world_counts_by_split": {"final": dict(family_counts)},
        "page_grade_counts_by_split": {"final": dict(grade_counts)},
        "decision_ids_sha256": decision_hash.hexdigest(), "world_ledger_sha256": world_hash.hexdigest(),
        "split_jsonl_sha256": {"final": split_hashes["final"].hexdigest()},
        "old_id_inventory": {"worlds": world_id_pin, "decisions": row_id_pin}, "old_source_lineage": old_lineage,
        "old_world_id_overlap": 0, "old_row_id_overlap": 0,
        "semantic_text_disjointness": {"old_source_inventory_overlap": 0,
            "hash_normalization": "Unicode casefold and whitespace collapse before SHA-256"},
        "robustness_coverage": {"candidate_permutations_per_main_query": 2, "candidate_rename_per_main_query": 1,
            "page_reorder_per_main_query": 1, "question_reorder_per_composition": 1, "candidate_count_probes": [2, 8, 16, 32, 64]},
        "custody": "No encoder/model execution, feature capture or scoring performed by this builder; final neural access requires a separately sealed corrected selection receipt.",
        "outputs": outputs}
    base._write_exclusive(root / BUILD_FILE, base._canonical_json(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate final source, then exclusively seal opaque packet/key and preparation pins")
    prepare.add_argument("--root", help="Must equal the ECA-2 protocol output_root")
    build = commands.add_parser("build", help="Build final-only IRs after complete independent meaning review")
    build.add_argument("--root", help="Must equal the ECA-2 protocol output_root")
    build.add_argument("--audit-receipt", required=True, help="Accepted receipt directly inside the ECA-2 audit directory")
    args = parser.parse_args(argv)
    root, _, protocol_raw = _protocol(args.root)
    if args.command == "prepare":
        result = _prepare(root, protocol_raw)
    else:
        result = _build(root, protocol_raw, Path(args.audit_receipt).expanduser().absolute())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ECA-2 final custody failure: {exc}", file=sys.stderr)
        raise SystemExit(2)
