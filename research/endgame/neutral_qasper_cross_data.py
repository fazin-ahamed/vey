"""Custody-bound QNATIVE-2 rows and source-token supervision (no model imports)."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from . import native_field_data as native, neutral_qasper_native_compile as projection
except ImportError:
    import native_field_data as native
    import neutral_qasper_native_compile as projection

AMENDMENT = Path(__file__).with_name("neutral_qasper_cross_study_amendment.json")
PHASES = ("fit", "selection", "calibration", "dev")
INTENTS = ("banking77.intent", "massive.intent")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def authority(protocol_path=AMENDMENT):
    cfg = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    require(cfg["schema"] == "vey.neutral.qasper.cross-study-amendment.v1", "Unknown amendment")
    entries = {**cfg["authority"], **cfg["current_authority"],
               "historical_native_arch_protocol": cfg["historical_native_arch_protocol"]}
    for name, entry in entries.items():
        require(pin(entry["path"]) == entry, "Authority changed: " + name)
    original = json.loads(Path(entries["deferred_question_protocol"]["path"]).read_text())
    historical = entries["historical_native_arch_protocol"]
    for entry in original["authority"].values():
        resolved = historical if entry["sha256"] == historical["sha256"] else entry
        require(pin(resolved["path"]) == resolved, "Historical question authority differs")
    return cfg, entries


def original_phase(phase, protocol_path=AMENDMENT):
    require(phase in PHASES, "Sealed or unknown phase denied")
    cfg, pins = authority(protocol_path)
    outer = "dev" if phase == "dev" else "train"
    prov = {}
    all_counts = Counter()
    for item in projection.open_phase(outer, "provenance"):
        require(item["id"] not in prov, "Duplicate provenance")
        all_counts[item["endpoint"]] += 1
        prov[item["id"]] = item
    expected = {e: v["rows"][outer] for e, v in cfg["data"]["question_study"]["endpoints"].items()}
    require(dict(all_counts) == expected, "Outer QASPER counts differ")
    membership = {rid: p for rid, p in prov.items()
                  if ("dev" if outer == "dev" else native.split_component(p["component_id"])) == phase}
    views, targets = {}, {}
    seen_views, seen_targets = set(), set()
    block_pool, paper_blocks = {}, {}
    state_pool = {}
    for view in projection.open_phase(outer, "serving"):
        rid = view["id"]
        require(rid in prov and rid not in seen_views, "Serving join differs")
        seen_views.add(rid)
        if prov[rid]["endpoint"] == "qasper.evidence_retrieval":
            key = (prov[rid]["group_id"], prov[rid]["paper_id"])
            blocks = []
            for candidate in view["candidates"]:
                pair = (candidate["id"], candidate["text"])
                blocks.append(block_pool.setdefault(pair, {"id": pair[0], "text": pair[1]}))
            require(len({b["id"] for b in blocks}) == len(blocks), "Duplicate source block ID")
            if key in paper_blocks:
                require(paper_blocks[key] == blocks, "Paper source blocks differ by question")
            else:
                paper_blocks[key] = blocks
            view["candidates"] = paper_blocks[key]
        if rid in membership:
            view["state"] = state_pool.setdefault(view["state"], view["state"])
            views[rid] = view
    for target in projection.open_phase(outer, "targets"):
        rid = target["id"]
        require(rid in prov and rid not in seen_targets, "Target join differs")
        seen_targets.add(rid)
        if rid in membership:
            targets[rid] = target
    seen_decisions = set()
    for decision in projection.open_phase(outer, "decisions"):
        rid = decision["id"]
        require(rid in prov and rid not in seen_decisions, "Decision join differs")
        seen_decisions.add(rid)
        require(decision["split"] == outer, "Projection phase differs")
    require(seen_views == seen_targets == seen_decisions == set(prov), "Incomplete outer joins")
    split = json.loads(Path(pins["question_split_manifest"]["path"]).read_text())
    entry = split["phases"][phase]["file"]
    require(pin(entry["path"]) == entry, "Question split bytes differ")
    with Path(entry["path"]).open() as stream:
        registered = {r["id"]: r for r in map(json.loads, stream)}
    require(set(registered) == set(membership) == set(views) == set(targets), "Question membership differs")
    rows = []
    for rid in sorted(membership):
        p = membership[rid]
        require(all(registered[rid][k] == p[k] for k in registered[rid]), "Question membership fields differ")
        key = (p["group_id"], p["paper_id"])
        require(key in paper_blocks, "Missing full paper catalogue")
        rows.append({"id": rid, "endpoint": p["endpoint"], "component_id": p["component_id"],
                     "phase": phase, "serving": views[rid], "blocks": paper_blocks[key],
                     "target": targets[rid], "paper_id": p["paper_id"], "group_id": p["group_id"],
                     "question_ordinal": p["question_ordinal"],
                     "input_sha256": value_digest(views[rid])})
    census = cfg["data"]["question_phase_counts"][phase]
    require(len(rows) == census["rows"] and len({r["component_id"] for r in rows}) == census["components"]
            and dict(Counter(r["endpoint"] for r in rows)) == census["endpoint_counts"], "Inner QASPER census differs")
    native_cfg = native.protocol()
    manifest, phases = native.load_dataset(native_cfg["output_root"])
    for state in phases[phase]:
        for decision in state["decisions"]:
            if decision["endpoint"] not in INTENTS:
                continue
            view = {"id": decision["id"], "task": decision["task"], "locale": state["locale"],
                    "state": state["state"], "question": decision["question"],
                    "candidates": [{"id": cid, "text": decision["candidates"][cid]}
                                   for cid in decision["candidate_ids"]]}
            rows.append({"id": decision["id"], "endpoint": decision["endpoint"],
                         "component_id": state["component_id"], "phase": phase,
                         "serving": view, "blocks": [], "target": {"target_available": True,
                         "distribution": decision["target_distribution"], "label_id": decision["gold"]},
                         "paper_id": None, "group_id": state["group_id"], "question_ordinal": None,
                         "input_sha256": value_digest(view)})
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate joined decision")
    return sorted(rows, key=lambda r: r["id"])


def source_windows(tokenizer, question, blocks):
    from tokenizers import Tokenizer
    backend = getattr(tokenizer, "_qnative_window_backend", None)
    if backend is None:
        backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
        backend.no_truncation()
        backend.no_padding()
        tokenizer._qnative_window_backend = backend
    question_encoding = backend.encode(question, add_special_tokens=False)
    qids = question_encoding.ids
    special_count = backend.num_special_tokens_to_add(True)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    sep_id = getattr(tokenizer, "sep_token_id", None)
    require(type(cls_id) is int and type(sep_id) is int and cls_id != sep_id,
            "Tokenizer must expose distinct [CLS]/[SEP] ids")
    capacity = 512 - len(qids) - special_count
    require(capacity > 0, "Question leaves no source token capacity")
    overlap = min(64, capacity - 1)
    windows, census = [], []
    for position, block in enumerate(blocks):
        encoded = backend.encode(block["text"], add_special_tokens=False)
        ids, offsets = encoded.ids, [list(o) for o in encoded.offsets]
        encoded.truncate(capacity, stride=overlap, direction="right")
        chunks = [encoded, *encoded.overflowing]
        covered = set()
        for count, chunk in enumerate(chunks):
            start = count * (capacity - overlap)
            end = start + len(chunk.ids)
            require(chunk.ids == ids[start:end] and [list(o) for o in chunk.offsets] == offsets[start:end],
                    "Source window token identity changed")
            pair = backend.post_process(question_encoding, chunk, add_special_tokens=True)
            # DeBERTa pair layout is [CLS] question [SEP] chunk [SEP]; specials and
            # question tokens both carry sequence_id None, so positions are derived
            # from the validated exact template, never from sequence ids.
            q0, q1 = 1, 1 + len(qids)
            s0, s1 = q1 + 1, q1 + 1 + len(chunk.ids)
            require(pair.ids[0] == cls_id and pair.ids[q1] == sep_id and pair.ids[s1] == sep_id
                    and len(pair.ids) == s1 + 1 and len(pair.ids) <= 512, "Pair template layout differs")
            question_positions = list(range(q0, q1))
            positions = list(range(s0, s1))
            require(pair.ids[q0:q1] == qids, "Question tokens changed")
            require([pair.ids[i] for i in positions] == ids[start:end], "Pair source tokens changed")
            require(len(pair.ids) <= 512 and len(positions) == end - start, "Invalid pair layout")
            indices = list(range(start, end))
            covered.update(indices)
            windows.append({"block_id": block["id"], "block_position": position,
                            "window_index": count, "source_token_indices": indices,
                            "offsets": offsets[start:end], "source_positions": positions,
                            "input_ids": pair.ids, "attention_mask": pair.attention_mask,
                            "question_input_ids": qids, "question_positions": question_positions,
                            "question_token_count": len(qids), "source_capacity": capacity, "overlap": overlap,
                            "pair_special_tokens": special_count,
                            "input_sha256": value_digest([pair.ids, pair.attention_mask])})
        require(covered == set(range(len(ids))), "Source token coverage incomplete")
        census.append({"block_id": block["id"], "block_position": position,
                       "source_token_offsets": offsets, "source_token_ids": ids, "source_token_count": len(ids),
                       "covered_source_tokens": len(covered), "window_count": len(chunks)})
    return windows, census


def retrieval_target(target, block_ids):
    masses = [0.0] * len(block_ids)
    positions = {cid: i for i, cid in enumerate(block_ids)}
    matched = 0
    for rating in target.get("ratings", []):
        for item in rating.get("matches", []):
            ids = item.get("block_ids", [])
            if ids:
                require(all(cid in positions for cid in ids), "Target block absent")
                matched += 1
                for cid in ids:
                    masses[positions[cid]] += 1 / len(ids)
    return [p / matched for p in masses] if matched else None


def bio_supervision(target, windows, census):
    """Annotation means on observed tokens only; unsupported spans never imply O."""
    by_block = {b["block_id"]: b for b in census}
    annotation_support = []
    sums = {b: [[0., 0., 0.] for _ in c["source_token_offsets"]] for b, c in by_block.items()}
    denominators = {b: [0] * len(c["source_token_offsets"]) for b, c in by_block.items()}
    for rating in target.get("ratings", []):
        items = rating.get("matches", [])
        positives, support = {}, []
        for item in items:
            supported, occurrence_support = [], []
            for occurrence in item.get("occurrences", []):
                bid, start, end = occurrence["block_id"], occurrence["start"], occurrence["end"]
                offsets = by_block[bid]["source_token_offsets"]
                indices = [i for i, (a, b) in enumerate(offsets) if a >= start and b <= end and b > a]
                aligned = bool(indices) and offsets[indices[0]][0] == start and offsets[indices[-1]][1] == end
                contained = aligned and any(w["block_id"] == bid and set(indices).issubset(w["source_token_indices"])
                                            for w in windows)
                occurrence_support.append({**occurrence, "source_token_indices": indices,
                                           "token_boundary_supported": aligned, "window_supported": contained})
                if contained:
                    supported.append({**occurrence, "source_token_indices": indices})
                    labels = positives.setdefault(bid, {})
                    for j, index in enumerate(indices):
                        labels[index] = 1 if j == 0 or labels.get(index) == 1 else 2
            support.append({"source_index": item["source_index"], "text": item["text"],
                            "source_occurrences": item.get("occurrences", []), "supported_occurrences": supported,
                            "occurrence_support": occurrence_support})
        full = bool(items) and all(item["supported_occurrences"] for item in support)
        usable = bool(positives)
        if rating.get("observed") and usable:
            for bid in sums:
                labels = positives.get(bid, {})
                for index in range(len(sums[bid])):
                    if full or index in labels:
                        sums[bid][index][labels.get(index, 0)] += 1
                        denominators[bid][index] += 1
        annotation_support.append({"annotation_ordinal": rating["annotation_ordinal"],
                                   "observed": rating.get("observed", False), "native_item_count": len(items),
                                   "full_support": full, "positive_only": usable and not full,
                                   "items": support})
    distributions = {bid: [[v / denominators[bid][i] for v in values] if denominators[bid][i] else None
                           for i, values in enumerate(tokens)] for bid, tokens in sums.items()}
    items = [item for annotation in annotation_support for item in annotation["items"]]
    counts = {"native_item_count": len(items), "supported_items": sum(bool(i["supported_occurrences"]) for i in items),
              "source_unmatched_items": sum(not i["source_occurrences"] for i in items),
              "token_boundary_unsupported_items": sum(bool(i["source_occurrences"]) and not any(
                  o["token_boundary_supported"] for o in i["occurrence_support"]) for i in items),
              "window_length_unsupported_items": sum(any(o["token_boundary_supported"] for o in i["occurrence_support"])
                  and not i["supported_occurrences"] for i in items)}
    return {"annotations": annotation_support, "token_distributions": distributions,
            "token_annotation_counts": denominators, "support_counts": counts,
            "supervised_source_tokens": sum(sum(n > 0 for n in counts) for counts in denominators.values())}


TRANSPORT_AMENDMENT = AMENDMENT.with_name("neutral_qasper_cross_transport_input_amendment.json")


def materialize(out, protocol_path=AMENDMENT):
    import gzip
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=False)
    cfg, pins = authority(protocol_path)
    transport = json.loads(TRANSPORT_AMENDMENT.read_text())
    require(pin(protocol_path) == transport["scientific_amendment"], "Transport science binding differs")
    manifest = {"schema": "vey.qnative.cross.dataset.v1", "protocol_sha256": pin(protocol_path)["sha256"],
                "transport_protocol_sha256": pin(TRANSPORT_AMENDMENT)["sha256"], "source_pins": pins,
                "construction_sources": {p.name: pin(p) for p in (Path(__file__), Path(native.__file__), Path(projection.__file__))},
                "phases": {}, "sealed_access": False, "model_loads": 0, "model_forwards": 0}
    papers = {}
    try:
        for phase in PHASES:
            rows = original_phase(phase, protocol_path)
            path = directory / (phase + ".jsonl.gz")
            with path.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
                for row in rows:
                    stored = {**row, "serving": dict(row["serving"])}
                    if row["blocks"]:
                        identity = value_digest([row["serving"]["state"], row["blocks"]])
                        papers.setdefault(identity, {"id": identity, "state": row["serving"]["state"], "blocks": row["blocks"]})
                        stored["paper_ref"] = identity
                        del stored["serving"]["state"], stored["blocks"]
                    stream.write((canonical(stored) + "\n").encode())
            manifest["phases"][phase] = {**pin(path), "rows": len(rows),
                                         "components": len({r["component_id"] for r in rows}),
                                         "endpoint_counts": dict(Counter(r["endpoint"] for r in rows))}
        path = directory / "papers.jsonl.gz"
        with path.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            for identity in sorted(papers):
                stream.write((canonical(papers[identity]) + "\n").encode())
        manifest["papers"] = {**pin(path), "rows": len(papers)}
        with (directory / "dataset_manifest.json").open("x") as stream:
            stream.write(canonical(manifest) + "\n")
    except BaseException as exc:
        with (directory / "failure.json").open("x") as stream:
            stream.write(canonical({"status": "failed", "error": str(exc), "manifest": manifest}) + "\n")
        raise
    return manifest


def derived_authority(root, protocol_path=AMENDMENT):
    import os
    root = Path(root)
    inventory_path = os.environ.get("QNATIVE_TRANSPORT_MANIFEST")
    require(bool(inventory_path), "Committed explicit transport inventory required")
    require(pin(inventory_path)["sha256"] == os.environ.get("QNATIVE_TRANSPORT_MANIFEST_SHA256"),
            "Explicit committed transport inventory fingerprint required")
    inventory = json.loads(Path(inventory_path).read_text())
    # Inventory is supplied as a committed source by the isolated parent transport.
    entries = inventory["derived_dataset"]
    for name, filename in (("manifest", "dataset_manifest.json"), ("verification_receipt", "verification_receipt.json")):
        actual = pin(root / filename)
        require(actual["sha256"] == entries[name]["sha256"] and actual["bytes"] == entries[name]["bytes"],
                "Transport dataset custody differs: " + name)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    receipt = json.loads((root / "verification_receipt.json").read_text())
    require(manifest["schema"] == "vey.qnative.cross.dataset.v1", "Unknown derived dataset")
    require(manifest["protocol_sha256"] == pin(protocol_path)["sha256"]
            and manifest["transport_protocol_sha256"] == pin(TRANSPORT_AMENDMENT)["sha256"], "Derived protocol binding differs")
    require(receipt.get("status") == "PASS" and receipt["manifest"]["sha256"] == entries["manifest"]["sha256"],
            "Independent dataset PASS receipt missing or wrong")
    require(receipt["schema"] == "vey.qnative.cross.dataset-verification.v1"
            and receipt["protocol_sha256"] == manifest["protocol_sha256"]
            and receipt["transport_protocol_sha256"] == manifest["transport_protocol_sha256"]
            and receipt["source_pins"] == manifest["source_pins"]
            and receipt["all_rows_independently_reconstructed"] is True
            and receipt["sealed_labels_accessed"] is False
            and receipt["model_loads"] == receipt["model_forwards"] == 0
            and set(receipt["phases"]) == set(manifest["phases"]) == set(PHASES), "Derived reconstruction receipt incomplete")
    cfg = json.loads(Path(protocol_path).read_text())
    expected_pins = {**cfg["authority"], **cfg["current_authority"],
                     "historical_native_arch_protocol": cfg["historical_native_arch_protocol"]}
    require(manifest["source_pins"] == expected_pins, "Derived scientific provenance differs")
    for phase, entry in manifest["phases"].items():
        require(set(entry) == {"path", "bytes", "sha256", "rows", "components", "endpoint_counts"}
                and entry["rows"] > 0 and entry["components"] > 0 and bool(entry["endpoint_counts"]),
                "Derived phase pin shape differs: " + phase)
        for key in entry["endpoint_counts"]:
            require(key in ("qasper.yes_no", "qasper.answerability", "qasper.evidence_retrieval",
                             "qasper.extractive_answer", "banking77.intent", "massive.intent"),
                    "Derived endpoint unknown: " + key)
        for inner in entry.values():
            if isinstance(inner, int):
                require(inner >= 0, "Negative derived phase count")
        actual = pin(root / Path(entry["path"]).name)
        require(actual["bytes"] == entry["bytes"] and actual["sha256"] == entry["sha256"], "Derived input bytes differ")
    for name, entry in manifest["construction_sources"].items():
        actual = pin(Path(__file__).with_name(name))
        require(actual["bytes"] == entry["bytes"] and actual["sha256"] == entry["sha256"],
                "Derived construction source identity differs: " + name)
    return json.loads(Path(protocol_path).read_text()), manifest["source_pins"], manifest


def study_authority(protocol_path=AMENDMENT):
    import os
    root = os.environ.get("QNATIVE_DATASET_ROOT")
    if root:
        cfg, pins, _ = derived_authority(root, protocol_path)
        return cfg, pins
    return authority(protocol_path)


def load_phase(phase, protocol_path=AMENDMENT):
    import gzip
    import os
    require(phase in PHASES, "Sealed or unknown phase denied")
    root = os.environ.get("QNATIVE_DATASET_ROOT")
    if not root:
        return original_phase(phase, protocol_path)
    _, _, manifest = derived_authority(root, protocol_path)
    papers = {}
    with gzip.open(Path(root) / "papers.jsonl.gz", "rt", encoding="utf-8") as stream:
        for paper in map(json.loads, stream):
            require(paper["id"] == value_digest([paper["state"], paper["blocks"]]) and paper["id"] not in papers,
                    "Derived paper identity differs")
            papers[paper["id"]] = paper
    require(len(papers) == manifest["papers"]["rows"], "Derived paper census differs")
    rows = []
    with gzip.open(Path(root) / (phase + ".jsonl.gz"), "rt", encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            if "paper_ref" in row:
                paper = papers[row.pop("paper_ref")]
                row["serving"]["state"] = paper["state"]
                row["blocks"] = paper["blocks"]
            require(row["phase"] == phase and row["input_sha256"] == value_digest(row["serving"]), "Derived serving binding differs")
            rows.append(row)
    entry = manifest["phases"][phase]
    require(len(rows) == entry["rows"] and len({r["id"] for r in rows}) == len(rows)
            and len({r["component_id"] for r in rows}) == entry["components"]
            and dict(Counter(r["endpoint"] for r in rows)) == entry["endpoint_counts"], "Derived phase census differs")
    return rows
