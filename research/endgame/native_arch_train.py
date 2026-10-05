#!/usr/bin/env python3
"""Run the preregistered NATIVE-2 cross / dual / Evidence-Pages architecture study."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import random
import tempfile
import torch

try:
    from . import native_field_data as data
    from . import native_field_train as native
    from .native_arch_model import (
        ARMS, ArchEvalCache, CANDIDATE_COUNT_GRID, ENCODER_PREFIXES, REPEATED_STATE_REPEATS,
        build_arch, canonical_modules, gold_rank, load_arch_encoder, nested_subsets,
        sha256_file,
    )
    from .native_field_model import EMPTY_EVIDENCE, donor_plan, native_readout
except ImportError:
    import native_field_data as data
    import native_field_train as native
    from native_arch_model import (
        ARMS, ArchEvalCache, CANDIDATE_COUNT_GRID, ENCODER_PREFIXES, REPEATED_STATE_REPEATS,
        build_arch, canonical_modules, gold_rank, load_arch_encoder, nested_subsets,
        sha256_file,
    )
    from native_field_model import EMPTY_EVIDENCE, donor_plan, native_readout


write_row = native.write_row


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "native_arch_protocol.json"
SOURCE_FILES = ("native_arch_model.py", "native_arch_train.py")
REUSED_SOURCES = ("native_field_model.py", "native_field_train.py", "native_field_data.py")


def arch_protocol():
    cfg = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if cfg.get("schema") != "vey.native-arch.protocol.v1":
        raise RuntimeError("unexpected NATIVE-2 protocol schema")
    if tuple(cfg["arms"]) != ARMS:
        raise RuntimeError("NATIVE-2 architecture arms differ from implementation")
    if cfg["training"]["page_tokens"] != 64 or cfg["training"]["top_k"] != 4:
        raise RuntimeError("NATIVE-2 page/top-k recipe differs from implementation")
    return cfg


def resource_config(cfg, arch):
    """Give the reused NATIVE-1 resource guard the NATIVE-2 microbatch and resource policy."""
    return {**cfg, "resources": arch["resources"],
            "training": {**cfg["training"], "seed": arch["training"]["seed"],
                         "batch_states": arch["training"]["batch_decisions"]}}


class Resources(native.Resources):
    def guard(self):
        if self.device.startswith("cuda"):
            device = torch.device(self.device)
            free_before = torch.cuda.mem_get_info(device)[0]
            threshold = float(self.cfg["pause_GPU_free_MiB_below"]) * 1024 ** 2
            reserved = torch.cuda.memory_reserved(device)
            allocated = torch.cuda.memory_allocated(device)
            if free_before < threshold and reserved > allocated:
                torch.cuda.empty_cache()
                self.events.append({
                    "kind": "reclaimed_idle_CUDA_cache",
                    "GPU_free_before_bytes": free_before,
                    "GPU_free_after_bytes": torch.cuda.mem_get_info(device)[0],
                    "own_allocated_bytes": allocated,
                    "own_reserved_before_bytes": reserved,
                    "own_reserved_after_bytes": torch.cuda.memory_reserved(device),
                })
        return super().guard()


def group_records(records, endpoints):
    grouped = []
    allowed = set(endpoints)
    for record in records:
        decisions = [decision for decision in record["decisions"]
                     if decision["endpoint"] in allowed]
        if decisions:
            grouped.append({**record, "decisions": decisions})
    if not grouped:
        raise ValueError("no eligible records in phase")
    return grouped


def optimizer(model):
    """Head lr0.001, encoder lr0.00002; one AdamW, weight_decay0.01."""
    encoder = list(model.transformer().parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    heads = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    if not heads or not encoder:
        raise ValueError("architecture lacks a separable head or encoder")
    groups = [{"params": heads, "lr": 0.001}, {"params": encoder, "lr": 0.00002}]
    return torch.optim.AdamW(groups, weight_decay=0.01)


def forward_scores(model, tokenizer, records, max_tokens):
    scores = model.score(tokenizer, records, max_tokens)
    if len(scores) != len(records):
        raise RuntimeError("architecture returned a different record count")
    for index, record in enumerate(records):
        if set(scores[index]) != {decision["endpoint"] for decision in record["decisions"]}:
            raise RuntimeError("architecture returned unexpected endpoints")
        for decision in record["decisions"]:
            value = scores[index][decision["endpoint"]]
            if value.ndim != 1 or value.shape[0] != len(decision["candidate_ids"]):
                raise RuntimeError("architecture returned a wrong candidate width")
    return scores


def decision_loss(scores, records):
    """Soft-target CE summed over decisions; identical math to native_field_train.decision_loss."""
    loss = None
    count = 0
    for index, record in enumerate(records):
        for decision in record["decisions"]:
            logits = scores[index][decision["endpoint"]]
            targets = torch.tensor(decision["target_distribution"], dtype=logits.dtype,
                                   device=logits.device)
            value = -(targets * torch.log_softmax(logits, dim=-1)).sum()
            loss = value if loss is None else loss + value
            count += 1
    if loss is None or count == 0:
        raise ValueError("no valid decisions in state batch")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite native soft-target loss")
    return loss, count


def collect_logits(model, tokenizer, records, max_tokens, resources):
    model.eval()
    rows = []
    with torch.no_grad():
        start = 0
        while start < len(records):
            resources.guard()
            batch = records[start:start + resources.microbatch_states]
            try:
                scores = forward_scores(model, tokenizer, batch, max_tokens)
            except torch.cuda.OutOfMemoryError:
                scores = None
                resources.reduce("eval")
                continue
            for index, record in enumerate(batch):
                for decision in record["decisions"]:
                    logits = scores[index][decision["endpoint"]].detach().cpu().tolist()
                    rows.append({**native.identity(record, decision), "logits": logits,
                                 "raw_probs": native.probabilities(logits),
                                 "target_distribution": decision["target_distribution"],
                                 "gold": decision.get("gold"),
                                 "observed_raters": decision.get("observed_raters")})
            start += len(batch)
    return rows


def probe_records(records, catalogues):
    """One record per eligible endpoint so restore/selection probes cover both heads."""
    selected, covered = [], set()
    for record in records:
        added = {decision["endpoint"] for decision in record["decisions"]} - covered
        if added:
            selected.append(record)
            covered.update(added)
        if covered == set(catalogues):
            break
    if covered != set(catalogues):
        raise ValueError("phase lacks a record for every eligible endpoint")
    return selected


def train_epoch(model, tokenizer, records, cfg, resources, opt, epoch, rng):
    order = list(records)
    rng.shuffle(order)
    effective = int(cfg["training"]["effective_batch_decisions"])
    max_tokens = int(cfg["training"]["max_tokens"])
    blocks = []
    total_loss, total_decisions = 0.0, 0
    model.train()
    for block_index, block in enumerate(native.chunks(order, effective)):
        denominator = sum(len(record["decisions"]) for record in block)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
        while True:
            opt.zero_grad(set_to_none=True)
            raw_loss = 0.0
            scores = loss = None
            try:
                for batch in native.chunks(block, resources.microbatch_states):
                    resources.guard()
                    scores = forward_scores(model, tokenizer, batch, max_tokens)
                    loss, _ = decision_loss(scores, batch)
                    (loss / denominator).backward()
                    raw_loss += float(loss.detach())
                    scores = loss = None
            except torch.cuda.OutOfMemoryError:
                scores = loss = None
                opt.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                resources.reduce("epoch%d/block%d" % (epoch, block_index))
                continue
            opt.step()
            break
        blocks.append({"states": len(block), "valid_decisions": denominator,
                       "normalized_loss": raw_loss / denominator,
                       "microbatch_states": resources.microbatch_states})
        total_loss += raw_loss
        total_decisions += denominator
    return {"valid_decisions": total_decisions, "states": len(records),
            "optimizer_updates": len(blocks),
            "mean_decision_loss": total_loss / total_decisions, "effective_batches": blocks}


def cache_smoke(cache, records):
    controls = []
    seen = set()
    for record in records:
        for decision in record["decisions"]:
            if decision["endpoint"] in seen:
                continue
            seen.add(decision["endpoint"])
            before = cache.counters()
            first = cache.read(record["state"], decision)
            middle = cache.counters()
            repeated = cache.read(record["state"], decision)
            after = cache.counters()
            if first != repeated:
                raise RuntimeError("repeated cached read changed architecture outputs")
            if after != middle:
                raise RuntimeError("repeated cached read re-encoded state or candidates")
            controls.append({**native.identity(record, decision),
                             "state_encode_calls_before": before["state_encode_calls"],
                             "state_encode_calls_after_first": middle["state_encode_calls"],
                             "state_encode_calls_after_repeat": after["state_encode_calls"],
                             "encoder_forward_calls_before": before["encoder_forward_calls"],
                             "encoder_forward_calls_after_first": middle["encoder_forward_calls"],
                             "encoder_forward_calls_after_repeat": after["encoder_forward_calls"],
                             "first": first, "repeated": repeated})
    if seen != {"banking77.intent", "massive.intent"}:
        raise RuntimeError("cache smoke did not exercise both eligible endpoints")
    return controls


def neural_smoke(cfg, arch, catalogues, fit, arm, device, directory, resources):
    native.seed(arch)
    max_tokens = int(arch["training"]["max_tokens"])
    tokenizer, encoder, custody = load_arch_encoder(cfg, catalogues, device)
    model = build_arch(encoder, catalogues, arch, arm).to(device=device, dtype=torch.float32)
    initial_model_hash = native.fingerprint(model.state_dict())
    encoder_before = native.fingerprint(model.transformer().state_dict())
    records = native.smoke_records(fit, catalogues)
    opt = optimizer(model)
    model.train()
    denominator = sum(len(record["decisions"]) for record in records)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
    total = 0.0
    while True:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        scores = loss = None
        try:
            for batch in native.chunks(records, resources.microbatch_states):
                resources.guard()
                scores = forward_scores(model, tokenizer, batch, max_tokens)
                loss, _ = decision_loss(scores, batch)
                (loss / denominator).backward()
                total += float(loss.detach())
                scores = loss = None
        except torch.cuda.OutOfMemoryError:
            scores = loss = None
            opt.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            resources.reduce("smoke")
            continue
        break
    if not math.isfinite(total / denominator):
        raise RuntimeError("nonfinite smoke loss")
    gradients = native.gradient_statistics(model)
    head_names = {}
    for endpoint in catalogues:
        candidates = [name for name, stats in gradients.items()
                      if stats["nonzero"] > 0 and not name.startswith("scorer.pool.encoder.")
                      and not name.startswith("encoder.") and "pool.encoder" not in name]
        if not candidates:
            raise RuntimeError("no nonzero active head gradient: " + endpoint)
        head_names[endpoint] = sorted(candidates)[0]
    encoder_names = [name for name, stats in gradients.items() if stats["nonzero"] > 0
                     and ("encoder." in name)]
    if not encoder_names:
        raise RuntimeError("smoke has no nonzero encoder gradient")
    named = sorted(set(head_names.values()))
    preferred = [name for name in encoder_names if "encoder.layer.0.output.dense.weight" in name]
    named.append(preferred[0] if preferred else sorted(encoder_names)[0])
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().cpu().clone() for name in named}
    opt.step()
    deltas = {}
    for name in named:
        difference = parameters[name].detach().cpu() - before[name]
        deltas[name] = {"l2": float(torch.linalg.vector_norm(difference)),
                        "max_abs": float(difference.abs().max()),
                        "nonzero": int(torch.count_nonzero(difference))}
        if not math.isfinite(deltas[name]["l2"]) or deltas[name]["nonzero"] == 0:
            raise RuntimeError("smoke optimizer did not change active parameter: " + name)
    encoder_after = native.fingerprint(model.transformer().state_dict())
    if encoder_after == encoder_before:
        raise RuntimeError("smoke encoder liveness violated: encoder unchanged")
    model.eval()
    saved_logits = collect_logits(model, tokenizer, records[:1], max_tokens, resources)
    with tempfile.TemporaryDirectory(prefix="smoke-weights-", dir=directory) as temporary:
        checkpoint = Path(temporary) / "smoke.safetensors"
        saved_hash = native.save_checkpoint(model, checkpoint)
        with torch.no_grad():
            parameters[named[0]].flatten()[0].add_(0.125)
        perturbed_hash = native.fingerprint(model.state_dict())
        restore = native.restore_checkpoint(model, checkpoint, saved_hash)
        restored_logits = collect_logits(model, tokenizer, records[:1], max_tokens, resources)
        restore.update({"saved_logits": saved_logits, "restored_logits": restored_logits,
                        "perturbed_fingerprint": perturbed_hash,
                        "max_abs_difference": native.numeric_difference(saved_logits, restored_logits)})
    temperatures = {endpoint: 1.0 for endpoint in catalogues}
    cache = ArchEvalCache(model, tokenizer, temperatures, max_tokens, resources.guard)
    controls = cache_smoke(cache, records)
    proof = {"schema": "vey.native-arch.smoke.v1", "arm": arm,
             "actual_fit_state_ids": [record["id"] for record in records],
             "actual_fit_decision_ids": [decision["id"] for record in records
                                         for decision in record["decisions"]],
             "valid_decisions": denominator, "loss": total / denominator,
             "gradient_statistics": gradients, "active_head_parameter_names": head_names,
             "active_encoder_parameter_name": named[-1], "parameter_deltas": deltas,
             "encoder_before_sha256": encoder_before, "encoder_after_sha256": encoder_after,
             "initial_model_sha256": initial_model_hash, "strict_restore": restore,
             "cache_controls": controls, "cache_counters": cache.counters(),
             "smoke_weights_discarded": True, "custody": custody}
    del cache, opt, model, encoder, tokenizer, parameters, before
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return proof


def dev_metrics(rows, endpoints):
    raw, calibrated = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["task"].lower() != "choice":
            continue
        raw[row["endpoint"]].append(int(row["raw_answer"] == row["gold"]))
        calibrated[row["endpoint"]].append(int(row["answer"] == row["gold"]))
    raw_accuracy = {endpoint: math.fsum(values) / len(values) for endpoint, values in raw.items()}
    calibrated_accuracy = {endpoint: math.fsum(values) / len(values)
                           for endpoint, values in calibrated.items()}
    return {"raw": raw_accuracy, "calibrated": calibrated_accuracy,
            "macro_raw": math.fsum(raw_accuracy.values()) / len(raw_accuracy),
            "macro_calibrated": math.fsum(calibrated_accuracy.values()) / len(calibrated_accuracy),
            "endpoints": list(endpoints)}


def prediction_controls(directory, cache, dev, fit, catalogues, arm):
    endpoints = list(catalogues)
    plan = donor_plan(dev, endpoints)
    indexed = {decision["id"]: (record, decision) for record in dev
               for decision in record["decisions"]}
    counts = {endpoint: {"total": 0, "available": 0, "unavailable": 0, "unique_donors": 0}
              for endpoint in endpoints}
    donor_sets = defaultdict(set)
    path_counts = {}
    dev_rows = []

    def counters():
        return cache.counters()

    def receipt(name, before):
        after = counters()
        path_counts[name] = {key + "_before": value for key, value in before.items()}
        path_counts[name].update({key + "_after": value for key, value in after.items()})

    paths = {name: directory / (name + ".jsonl")
             for name in ("dev", "masked", "swapped", "candidate_count",
                          "coordinate_reversal", "repeated_state")}
    before = counters()
    with paths["dev"].open("x", encoding="utf-8") as dev_stream, \
            paths["coordinate_reversal"].open("x", encoding="utf-8") as reversal_stream:
        for record in dev:
            for decision in record["decisions"]:
                output = cache.read(record["state"], decision)
                write_row(dev_stream, {**native.identity(record, decision), **output})
                dev_rows.append({**native.identity(record, decision), **output,
                                 "gold": decision.get("gold")})
                calls_before = cache.counters()["state_encode_calls"]
                reversed_output = cache.read(record["state"], decision, reverse=True)
                calls_after = cache.counters()["state_encode_calls"]
                if calls_before != calls_after:
                    raise RuntimeError("coordinate reversal encoded state again")
                for key in ("logits", "raw_probs", "probs"):
                    left = dict(zip(output["candidate_ids"], output[key]))
                    right = dict(zip(reversed_output["candidate_ids"], reversed_output[key]))
                    if left != right:
                        raise RuntimeError("coordinate reversal changed " + key + " by native ID")
                for key in ("answer", "ties", "raw_answer", "raw_ties"):
                    if key in output and output[key] != reversed_output[key]:
                        raise RuntimeError("coordinate reversal changed " + key)
                write_row(reversal_stream, {**native.identity(record, decision),
                          "candidate_ids": reversed_output["candidate_ids"], "reversed": True,
                          "logits": reversed_output["logits"], "raw_probs": reversed_output["raw_probs"],
                          "probs": reversed_output["probs"], "answer": reversed_output["answer"],
                          "ties": reversed_output["ties"], "raw_answer": reversed_output["raw_answer"],
                          "raw_ties": reversed_output["raw_ties"],
                          "state_encode_calls_before": calls_before,
                          "state_encode_calls_after": calls_after})
    receipt("dev", before)
    before = counters()
    with paths["masked"].open("x", encoding="utf-8") as stream:
        for record in dev:
            for decision in record["decisions"]:
                output = cache.read(EMPTY_EVIDENCE, decision)
                write_row(stream, {**native.identity(record, decision), **output})
    receipt("masked", before)
    before = counters()
    with paths["swapped"].open("x", encoding="utf-8") as stream:
        for record in dev:
            for decision in record["decisions"]:
                endpoint = decision["endpoint"]
                counts[endpoint]["total"] += 1
                donor_id = plan[decision["id"]]
                if donor_id is None:
                    counts[endpoint]["unavailable"] += 1
                    write_row(stream, {**native.identity(record, decision), "donor_id": None,
                              "reason": "no different-component different-native-target dev donor"})
                    continue
                donor_record, _ = indexed[donor_id]
                output = cache.read(donor_record["state"], decision)
                write_row(stream, {**native.identity(record, decision), **output,
                          "donor_id": donor_id})
                counts[endpoint]["available"] += 1
                donor_sets[endpoint].add(donor_id)
    receipt("swapped", before)
    if arm in ("dual", "pages"):
        if path_counts["swapped"]["state_encode_calls_before"] != path_counts["swapped"]["state_encode_calls_after"]:
            raise RuntimeError("swapped donor read encoded a state again")
        if path_counts["swapped"]["encoder_forward_calls_before"] != path_counts["swapped"]["encoder_forward_calls_after"]:
            raise RuntimeError("swapped donor read re-ran the encoder")
    for endpoint in counts:
        counts[endpoint]["unique_donors"] = len(donor_sets[endpoint])
    before = counters()
    candidate_rows = 0
    with paths["candidate_count"].open("x", encoding="utf-8") as stream:
        for record in dev:
            for decision in record["decisions"]:
                if decision["task"].lower() != "choice":
                    raise RuntimeError("candidate-count screen supports native choice fields only")
                gold = decision.get("gold")
                subsets = nested_subsets(decision["candidate_ids"], record["component_id"])
                for k, subset in sorted(subsets.items()):
                    subset_decision = {**decision, "candidate_ids": subset}
                    logits = cache.field_logits(record["state"], subset_decision)
                    raw = native.probabilities(logits)
                    probs = native.probabilities(logits, cache.temperatures[decision["endpoint"]])
                    rank = gold_rank(subset, probs, gold)
                    row = {**native.identity(record, decision), "k": k, "subset_ids": subset,
                           "candidate_ids": list(subset), "logits": list(logits),
                           "raw_probs": raw, "probs": probs, "gold_rank": rank,
                           "gold_in_shortlist": bool(gold in subset)}
                    for prefix, masses in (("", probs), ("raw_", raw)):
                        readout = native_readout(subset, masses, "choice", list(subset))
                        row.update({prefix + name: value for name, value in readout.items()})
                    write_row(stream, row)
                    candidate_rows += 1
    receipt("candidate_count", before)
    state = min(dev, key=lambda record: record["id"])
    native_decision = min(state["decisions"], key=lambda decision: decision["id"])
    foreign_sources = [(record, decision) for record in fit
                       for decision in record["decisions"]
                       if decision["endpoint"] != native_decision["endpoint"]]
    if not foreign_sources:
        raise RuntimeError("repeated-state control lacks a FIT catalogue from the other endpoint")
    foreign_source = min(foreign_sources, key=lambda item: item[1]["id"])
    sources = ((state, native_decision), foreign_source)
    if {decision["endpoint"] for _, decision in sources} != set(endpoints):
        raise RuntimeError("repeated-state control did not cover both public catalogues")
    repeated_rows = 0
    repeated_state = {
        "state_id": state["id"], "native_state_decision_id": native_decision["id"],
        "repeats": REPEATED_STATE_REPEATS, "catalogue_count": len(sources),
        "catalogue_source_decision_ids": [decision["id"] for _, decision in sources],
        "catalogue_endpoints": [decision["endpoint"] for _, decision in sources],
        "foreign_catalogue_semantic_metrics": False, "learned_retrieval_credit": 0,
    }
    repeated_cache = ArchEvalCache(model=cache.model, tokenizer=cache.tokenizer,
                                   temperatures=cache.temperatures, max_tokens=cache.max_tokens,
                                   guard=cache.guard)
    before = counters()
    first_outputs = {}
    with paths["repeated_state"].open("x", encoding="utf-8") as stream:
        for repeat in range(REPEATED_STATE_REPEATS):
            for source_index, (source_record, source_decision) in enumerate(sources):
                decision = {key: source_decision[key]
                            for key in ("endpoint", "task", "candidate_ids", "candidates")}
                calls_before = repeated_cache.counters()
                output = repeated_cache.read(state["state"], decision)
                calls_after = repeated_cache.counters()
                if repeat > 0:
                    if calls_after != calls_before:
                        raise RuntimeError("repeated public-catalogue read reran the encoder")
                    if output != first_outputs[decision["endpoint"]]:
                        raise RuntimeError("repeated public-catalogue read changed outputs")
                else:
                    first_outputs[decision["endpoint"]] = output
                row = {**native.identity(source_record, source_decision), **output,
                       "state_id": state["id"],
                       "native_state_decision_id": native_decision["id"],
                       "catalogue_source_decision_id": source_decision["id"],
                       "probe_kind": "native" if source_index == 0 else "unlabelled_foreign_catalogue",
                       "semantic_metric_eligible": source_index == 0, "repeat_index": repeat}
                for key in ("state_encode_calls", "encoder_forward_calls", "encoded_states"):
                    row[key + "_before"] = calls_before[key]
                    row[key + "_after"] = calls_after[key]
                write_row(stream, row)
                repeated_rows += 1
            if repeat == 0:
                first_round = repeated_cache.counters()
                for key in ("state_encode_calls", "encoder_forward_calls", "encoded_states"):
                    repeated_state["first_round_" + key] = first_round[key]
    final_round = repeated_cache.counters()
    repeated_state.update({
        "state_encodes_total": final_round["state_encode_calls"],
        "encoder_forwards_total": final_round["encoder_forward_calls"],
        "encoded_states_total": final_round["encoded_states"],
        "extra_state_encodes": final_round["state_encode_calls"] - first_round["state_encode_calls"],
        "extra_encoder_forwards": final_round["encoder_forward_calls"] - first_round["encoder_forward_calls"],
        "extra_encoded_states": final_round["encoded_states"] - first_round["encoded_states"],
    })
    if any(repeated_state[key] != 0 for key in
           ("extra_state_encodes", "extra_encoder_forwards", "extra_encoded_states")):
        raise RuntimeError("repeated-state control observed extra encoder work")
    expected_exposure = sum(len(decision["candidate_ids"]) for _, decision in sources) if arm == "cross" else 1
    expected_state_calls = len(sources) if arm == "cross" else 1
    expected_forward_calls = len(sources) if arm == "cross" else 1 + len(sources)
    if (first_round["encoded_states"] != expected_exposure
            or first_round["state_encode_calls"] != expected_state_calls
            or first_round["encoder_forward_calls"] != expected_forward_calls):
        raise RuntimeError("two-catalogue first-round state exposure differs from architecture")
    receipt("repeated_state", before)
    path_counts["repeated_state"].update(
        {"fresh_cache_" + key: value for key, value in final_round.items()})
    return {"coordinate_reversal_rows": sum(len(record["decisions"]) for record in dev),
            "candidate_count_rows": candidate_rows, "repeated_state_rows": repeated_rows,
            "repeated_state": repeated_state, "dev_metrics": dev_metrics(dev_rows, endpoints),
            "swap_denominators": counts, "path_counts": path_counts,
            "candidate_count_grid": sorted(CANDIDATE_COUNT_GRID),
            "candidate_count_membership_rule": "input-only component-seeded whole-catalogue shuffle",
            "candidate_count_learned_retrieval_credit": 0,
            "donor_rule": "endpoint sorted-ID next cyclic different component and gold; reuse allowed",
            "state_mask_text": EMPTY_EVIDENCE, "cache_counters": cache.counters(),
            "model_identity": cache.identity}


def metadata_base(cfg, arch, manifest, root, arm, mode, resources):
    canonical_root, canonical_files = canonical_modules()[2]["root"], canonical_modules()[2]["files"]
    return {"schema": "vey.native-arch.arm.v1", "arm": arm, "mode": mode, "architecture": arm,
            "protocol_sha256": sha256_file(PROTOCOL),
            "dataset_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
            "dataset_root": str(root), "dataset_phases": manifest["phases"],
            "source_exports": manifest["source_exports"],
            "source_files": {name: sha256_file(HERE / name) for name in SOURCE_FILES},
            "reused_native_field_sources": {name: sha256_file(HERE / name) for name in REUSED_SOURCES},
            "canonical_dependencies": {"root": canonical_root, "files": canonical_files},
            "eligible_endpoints": manifest["eligible_endpoints"],
            "ineligible_endpoints": manifest["ineligible_endpoints"],
            "catalogues": manifest["catalogues"], "recipe": arch["training"],
            "state_mask_text": EMPTY_EVIDENCE,
            "candidate_count_grid": sorted(CANDIDATE_COUNT_GRID),
            "repeated_state_repeats": REPEATED_STATE_REPEATS,
            "pooled_reference": "NATIVE-1 full arm dev/masked/swapped.jsonl",
            "no_dev_selection_or_calibration": True, "sealed_phases_accessed": False,
            "resource_events": resources.events}


def run(root: str | Path, arm: str, mode: str, device: str):
    cfg = data.protocol()
    arch = arch_protocol()
    supplied_root = Path(root)
    root = Path(cfg["output_root"])
    if supplied_root.resolve() != root.resolve():
        raise RuntimeError("Unregistered native architecture dataset root")
    manifest, phases = data.load_dataset(root)
    endpoints = manifest["eligible_endpoints"]
    catalogues = {endpoint: manifest["catalogues"][endpoint] for endpoint in endpoints}
    phases = {phase: group_records(records, endpoints) for phase, records in phases.items()}
    directory = root / (arm if mode == "train" else arm + "-smoke")
    directory.mkdir(exist_ok=False)
    history = {"schema": "vey.native-arch.history.v1", "arm": arm, "epochs": []}
    liveness = {"schema": "vey.native-arch.liveness.v1", "arm": arm}
    with Resources(resource_config(cfg, arch), device) as resources:
        metadata = metadata_base(cfg, arch, manifest, root, arm, mode, resources)
        try:
            liveness["smoke"] = neural_smoke(cfg, arch, catalogues, phases["fit"], arm, device,
                                             directory, resources)
            if mode == "train":
                native.seed(arch)
                tokenizer, encoder, custody = load_arch_encoder(cfg, catalogues, device)
                model = build_arch(encoder, catalogues, arch, arm).to(device=device,
                                                                      dtype=torch.float32)
                fresh_hash = native.fingerprint(model.state_dict())
                if fresh_hash != liveness["smoke"]["initial_model_sha256"]:
                    raise RuntimeError("real arm was not freshly seeded after discarded smoke")
                liveness["fresh_training_initial_sha256"] = fresh_hash
                max_tokens = int(arch["training"]["max_tokens"])
                opt = optimizer(model)
                rng = random.Random(int(arch["training"]["seed"]))
                probe = probe_records(phases["selection"], catalogues)
                best_rank, best_epoch, best_hash, selected_rows = None, None, None, None
                checkpoint = directory / "selected.safetensors"
                with (directory / "selection.jsonl").open("x", encoding="utf-8") as selection_stream:
                    for epoch in range(int(arch["training"]["epochs"]) + 1):
                        fit = None if epoch == 0 else train_epoch(
                            model, tokenizer, phases["fit"], arch, resources, opt, epoch, rng)
                        rows = collect_logits(model, tokenizer, phases["selection"], max_tokens,
                                              resources)
                        metrics = native.selection_metrics(rows)
                        for row in rows:
                            write_row(selection_stream, {"epoch": epoch, **row})
                        selection_stream.flush()
                        candidate_hash = native.fingerprint(model.state_dict())
                        rank = (-metrics["macro_source_intent_accuracy"],
                                metrics["mean_endpoint_nll"], epoch)
                        if best_rank is None or rank < best_rank:
                            best_rank, best_epoch = rank, epoch
                            best_hash = native.save_checkpoint(model, checkpoint)
                            selected_rows = collect_logits(model, tokenizer, probe,
                                                           max_tokens, resources)
                        history["epochs"].append({"epoch": epoch, "fit": fit, "selection": metrics,
                                                  "state_dict_fingerprint": candidate_hash,
                                                  "selected_epoch_so_far": best_epoch})
                        history["selected_epoch"] = best_epoch
                        history["selected_fingerprint"] = best_hash
                        native.write_json(directory / "history.json", history)
                before_restore = native.fingerprint(model.state_dict())
                trained_encoder_hash = native.fingerprint(model.transformer().state_dict())
                strict = native.restore_checkpoint(model, checkpoint, best_hash)
                restored_rows = collect_logits(model, tokenizer, probe, max_tokens, resources)
                strict.update({"before_restore_fingerprint": before_restore, "selected_epoch": best_epoch,
                               "probe_state_ids": [record["id"] for record in probe],
                               "selected_rows": selected_rows, "restored_rows": restored_rows,
                               "max_abs_difference": native.numeric_difference(selected_rows, restored_rows)})
                liveness["selected_restore"] = strict
                selected_encoder_hash = native.fingerprint(model.transformer().state_dict())
                liveness["encoder_training_before_sha256"] = custody["encoder_initial_sha256"]
                liveness["encoder_training_after_sha256"] = trained_encoder_hash
                liveness["encoder_selected_sha256"] = selected_encoder_hash
                if trained_encoder_hash == custody["encoder_initial_sha256"]:
                    raise RuntimeError("encoder unchanged after actual training")
                model.zero_grad(set_to_none=True)
                del opt
                calibration = collect_logits(model, tokenizer, phases["calibration"], max_tokens,
                                             resources)
                with (directory / "calibration.jsonl").open("x", encoding="utf-8") as stream:
                    for row in calibration:
                        write_row(stream, row)
                temperatures, temperature_evidence = native.fit_temperatures(calibration, catalogues)
                del calibration
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                cache = ArchEvalCache(model, tokenizer, temperatures, max_tokens, resources.guard)
                liveness["controls"] = prediction_controls(
                    directory, cache, phases["dev"], phases["fit"], catalogues, arm)
                metadata.update({"custody": custody, "temperatures": temperatures,
                                 "temperature_evidence": temperature_evidence,
                                 "checkpoint": {"path": str(checkpoint),
                                                "sha256": sha256_file(checkpoint),
                                                "schema": "vey.native-field.checkpoint.v1",
                                                "encoder_prefix": ENCODER_PREFIXES[arm],
                                                "encoder_state_dict_fingerprint": selected_encoder_hash,
                                                "state_dict_fingerprint": best_hash,
                                                "selected_epoch": best_epoch},
                                 "dev_metrics": liveness["controls"]["dev_metrics"],
                                 "swap_denominators": liveness["controls"]["swap_denominators"]})
            else:
                metadata.update({"custody": liveness["smoke"]["custody"], "checkpoint": None,
                                 "temperatures": None, "temperature_evidence": None})
            metadata["status"] = "smoke_complete" if mode == "smoke" else "train_complete"
            native.write_json(directory / "history.json", history)
            native.write_json(directory / "liveness.json", liveness)
            metadata["artifact_hashes"] = native.artifact_hashes(directory)
            native.write_json(directory / "metadata.json", metadata)
        except BaseException as exc:
            metadata.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            native.write_json(directory / "history.json", history)
            native.write_json(directory / "liveness.json", liveness)
            metadata["artifact_hashes"] = native.artifact_hashes(directory)
            native.write_json(directory / "metadata.json", metadata)
            raise
    return directory


def main():
    cfg = data.protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=cfg["output_root"])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "cuda:0"))
    args = parser.parse_args()
    directory = run(args.root, args.arm, args.mode, args.device)
    print(json.dumps({"output": str(directory), "arm": args.arm, "mode": args.mode}, sort_keys=True))


if __name__ == "__main__":
    main()
