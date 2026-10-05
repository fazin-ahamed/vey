#!/usr/bin/env python3
"""Run the preregistered NATIVE-1 prior/frozen/full state-field controls offline."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import AbstractContextManager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import torch
from safetensors.torch import load_file, save_file

try:
    from . import native_field_data as data
    from .native_field_model import (
        EMPTY_EVIDENCE, StateFieldCache, canonical_bytes, donor_plan, encode_texts,
        fingerprint, load_stock, membership, probabilities, sha256_file,
    )
except ImportError:
    import native_field_data as data
    from native_field_model import (
        EMPTY_EVIDENCE, StateFieldCache, canonical_bytes, donor_plan, encode_texts,
        fingerprint, load_stock, membership, probabilities, sha256_file,
    )


HERE = Path(__file__).resolve().parent
GPU_LOCK = Path("/tmp/vey-gpu.lock")
EVAL_ATOL = 2e-5
EVAL_RTOL = 2e-6


def write_json(path, value):
    """Atomic journal replacement within this run's exclusively created directory."""
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def write_row(stream, value):
    stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def meminfo():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


class Resources(AbstractContextManager):
    """Pause this process under contention; never manipulate other processes."""

    def __init__(self, cfg, device):
        self.cfg = cfg["resources"]
        self.device = device
        self.lock = None
        self.events = []
        self.microbatch_states = int(cfg["training"]["batch_states"])
        self.baseline_swap = meminfo()[1]

    def __enter__(self):
        torch.set_num_threads(int(self.cfg["threads"]))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
        desired_nice = int(self.cfg["nice"])
        if current_nice < desired_nice:
            os.nice(desired_nice - current_nice)
        self.events.append({"kind": "priority", "nice": os.getpriority(os.PRIO_PROCESS, 0),
                            "threads": torch.get_num_threads(), "device": self.device})
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            self.lock = GPU_LOCK.open("a+")
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX)
        self.baseline_swap = meminfo()[1]
        try:
            self.guard()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.lock is not None:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()
        return False

    def guard(self):
        paused = False
        while True:
            available, swap = meminfo()
            free_gpu = None
            if self.device.startswith("cuda"):
                free_gpu = int(torch.cuda.mem_get_info(torch.device(self.device))[0])
            snapshot = {"available_RAM_bytes": available, "swap_used_bytes": swap,
                        "swap_growth_bytes": max(0, swap - self.baseline_swap),
                        "GPU_free_bytes": free_gpu}
            reasons = []
            if available < float(self.cfg["pause_available_RAM_GiB_below"]) * 1024 ** 3:
                reasons.append("available_RAM")
            if swap - self.baseline_swap > float(self.cfg["pause_swap_growth_MiB_above"]) * 1024 ** 2:
                reasons.append("swap_growth")
            if free_gpu is not None and free_gpu < float(self.cfg["pause_GPU_free_MiB_below"]) * 1024 ** 2:
                reasons.append("GPU_free")
            if not reasons:
                if paused or not any(event["kind"] == "startup" for event in self.events):
                    self.events.append({"kind": "resumed" if paused else "startup", **snapshot})
                return snapshot
            if not paused:
                self.events.append({"kind": "pause", "reasons": reasons, **snapshot})
                paused = True
            time.sleep(5)

    def reduce(self, where):
        previous = self.microbatch_states
        if previous <= 1:
            raise RuntimeError("FP32 NATIVE-1 exceeds resources at one state; no recipe substitution")
        self.microbatch_states = max(1, previous // 2)
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        self.events.append({"kind": "OOM_microbatch_reduction", "where": where,
                            "before": previous, "after": self.microbatch_states,
                            "effective_batch_states": 32, "dtype": "float32"})


def seed(cfg):
    value = int(cfg["training"]["seed"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def eligible_records(records, endpoints):
    allowed = set(endpoints)
    return [{**record, "decisions": [decision for decision in record["decisions"]
                                     if decision["endpoint"] in allowed]}
            for record in records if any(d["endpoint"] in allowed for d in record["decisions"])]


def identity(record, decision):
    return {"id": decision["id"], "endpoint": decision["endpoint"], "task": decision["task"],
            "component_id": record["component_id"], "group_id": record["group_id"],
            "locale": record["locale"], "candidate_ids": list(decision["candidate_ids"])}


def fit_prior(records, catalogues):
    counts = Counter()
    totals = {endpoint: [0.0] * len(ids) for endpoint, ids in catalogues.items()}
    for record in records:
        for decision in record["decisions"]:
            endpoint = decision["endpoint"]
            counts[endpoint] += 1
            for index, mass in enumerate(decision["target_distribution"]):
                totals[endpoint][index] += mass
    if any(counts[endpoint] == 0 for endpoint in catalogues):
        raise ValueError("eligible endpoint has no fit decisions")
    return {endpoint: [mass / counts[endpoint] for mass in masses]
            for endpoint, masses in totals.items()}, dict(counts)


class FrozenFeatures:
    def __init__(self, model, tokenizer, cfg, resources):
        if model.arm != "frozen":
            raise ValueError("detached feature cache is frozen-arm only")
        self.model = model
        self.tokenizer = tokenizer
        self.limit = int(cfg["training"]["max_tokens"])
        self.resources = resources
        self.encoder_identity = fingerprint(model.pool.encoder.state_dict())
        self.features = {}

    def get(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text.encode("utf-8") not in self.features))
        start = 0
        while start < len(missing):
            self.resources.guard()
            batch = missing[start:start + self.resources.microbatch_states]
            try:
                encoded = encode_texts(self.model, self.tokenizer, batch, self.limit).detach().cpu()
            except torch.cuda.OutOfMemoryError:
                self.resources.reduce("frozen_feature_cache")
                continue
            if encoded.requires_grad or encoded.grad_fn is not None:
                raise RuntimeError("frozen features must be detached")
            for text, feature in zip(batch, encoded):
                self.features[text.encode("utf-8")] = feature
            start += len(batch)
        device = next(self.model.parameters()).device
        return torch.stack([self.features[text.encode("utf-8")] for text in texts]).to(device)


def forward_records(model, tokenizer, records, cfg, frozen_features=None):
    texts = [record["state"] for record in records]
    features = (frozen_features.get(texts) if frozen_features is not None else
                encode_texts(model, tokenizer, texts, int(cfg["training"]["max_tokens"])))
    if model.arm == "full" and model.training and not features.requires_grad:
        raise RuntimeError("full-arm training features were detached")
    return model.fields(features)


def decision_loss(fields, records):
    loss = None
    count = 0
    for index, record in enumerate(records):
        for decision in record["decisions"]:
            logits = fields[decision["endpoint"]][index]
            targets = torch.tensor(decision["target_distribution"], dtype=logits.dtype, device=logits.device)
            value = -(targets * torch.log_softmax(logits, dim=-1)).sum()
            loss = value if loss is None else loss + value
            count += 1
    if loss is None or count == 0:
        raise ValueError("no valid decisions in state batch")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("nonfinite native soft-target loss")
    return loss, count


def optimizer(model):
    groups = [{"params": list(model.heads.parameters()), "lr": 0.001}]
    if model.arm == "full":
        groups.append({"params": list(model.pool.encoder.parameters()), "lr": 0.00002})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def save_checkpoint(model, path):
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    expected = fingerprint(state)
    temporary = Path(path).with_suffix(".pending.safetensors")
    save_file(state, str(temporary), metadata={"schema": "vey.native-field.checkpoint.v1",
                                             "state_dict_sha256": expected})
    temporary.replace(path)
    return expected


def restore_checkpoint(model, path, expected):
    state = load_file(str(path), device="cpu")
    if fingerprint(state) != expected:
        raise RuntimeError("saved checkpoint tensor fingerprint differs")
    incompatible = model.load_state_dict(state, strict=True)
    actual = fingerprint(model.state_dict())
    if actual != expected:
        raise RuntimeError("strict restored state differs from selected checkpoint")
    return {"expected_fingerprint": expected, "restored_fingerprint": actual,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "checkpoint_sha256": sha256_file(path)}


def probe_logits(model, tokenizer, record, cfg):
    model.eval()
    with torch.no_grad():
        fields = forward_records(model, tokenizer, [record], cfg)
    return {endpoint: tensor[0].detach().cpu().tolist() for endpoint, tensor in fields.items()}


def numeric_difference(left, right):
    if isinstance(left, dict):
        if set(left) != set(right):
            raise RuntimeError("comparison keys differ")
        return max((numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list):
        if len(left) != len(right):
            raise RuntimeError("comparison lengths differ")
        return max((numeric_difference(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if not math.isfinite(left) or not math.isfinite(right):
            raise RuntimeError("nonfinite comparison values")
        if not math.isclose(left, right, abs_tol=EVAL_ATOL, rel_tol=EVAL_RTOL):
            raise RuntimeError(f"eval/cache mismatch: {left} vs {right}")
        return abs(left - right)
    if left != right:
        raise RuntimeError("native semantic output differs")
    return 0.0


def cache_controls(cache, records, catalogues):
    controls = []
    seen = set()
    for record in records:
        for decision in record["decisions"]:
            endpoint = decision["endpoint"]
            if endpoint in seen:
                continue
            seen.add(endpoint)
            ids = catalogues[endpoint]
            before = cache.encoder_calls
            first = cache.read(record["state"], decision, ids)
            after_first = cache.encoder_calls
            repeated = cache.read(record["state"], decision, ids)
            after_repeat = cache.encoder_calls
            if first != repeated or after_repeat != after_first:
                raise RuntimeError("repeated state read changed outputs or encoded again")
            member_output = None
            if decision["task"].lower() == "choice":
                members = sorted(ids)[:max(1, len(ids) // 2)]
                cached_mass = cache.field(record["state"], endpoint)["probs"]
                member_output = membership(ids, cached_mass, members)
            after_membership = cache.encoder_calls
            if after_membership != after_repeat:
                raise RuntimeError("canonical membership encoded the state again")
            uncached = cache.read(record["state"], decision, ids, cached=False)
            after_uncached = cache.encoder_calls
            maximum = numeric_difference(first, uncached)
            controls.append({**identity(record, decision), "state_sha256": hashlib.sha256(record["state"].encode()).hexdigest(),
                             "first": first, "repeated": repeated, "uncached": uncached,
                             "membership": member_output,
                             "encoder_calls_before": before, "encoder_calls_after_first": after_first,
                             "encoder_calls_after_repeat": after_repeat,
                             "encoder_calls_after_membership": after_membership,
                             "encoder_calls_after_uncached": after_uncached,
                             "cached_uncached_max_abs_difference": maximum})
    if seen != set(catalogues):
        raise RuntimeError("cache controls did not exercise every eligible endpoint")
    return controls


def smoke_records(records, endpoints):
    selected, covered = [], set()
    for record in records:
        added = {decision["endpoint"] for decision in record["decisions"]} - covered
        if added:
            selected.append(record)
            covered.update(added)
        if covered == set(endpoints):
            return selected
    raise ValueError("actual fit data lacks an eligible smoke endpoint")


def gradient_statistics(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"nonfinite parameter gradient: {name}")
        result[name] = {"l2": float(torch.linalg.vector_norm(grad)),
                        "max_abs": float(grad.abs().max()), "nonzero": int(torch.count_nonzero(grad))}
    return result


def neural_smoke(cfg, catalogues, fit, arm, device, directory, resources):
    seed(cfg)
    tokenizer, model, custody = load_stock(cfg, catalogues, arm, device)
    initial_model_hash = fingerprint(model.state_dict())
    encoder_before = fingerprint(model.pool.encoder.state_dict())
    records = smoke_records(fit, catalogues)
    opt = optimizer(model)
    model.train()
    frozen_features = FrozenFeatures(model, tokenizer, cfg, resources) if arm == "frozen" else None
    denominator = sum(len(record["decisions"]) for record in records)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
    while True:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        fields = loss = None
        try:
            for batch in chunks(records, resources.microbatch_states):
                resources.guard()
                fields = forward_records(model, tokenizer, batch, cfg, frozen_features)
                loss, _ = decision_loss(fields, batch)
                (loss / denominator).backward()
                total += float(loss.detach())
                fields = loss = None
        except torch.cuda.OutOfMemoryError:
            fields = loss = None
            opt.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            resources.reduce("smoke")
            continue
        break
    gradients = gradient_statistics(model)
    head_names = {}
    for endpoint, key in model.endpoint_keys.items():
        names = [name for name, stats in gradients.items()
                 if name.startswith("heads." + key + ".") and stats["nonzero"] > 0]
        if not names:
            raise RuntimeError(f"no actual active head gradient: {endpoint}")
        head_names[endpoint] = names[0]
    encoder_names = [name for name, stats in gradients.items()
                     if name.startswith("pool.encoder.") and stats["nonzero"] > 0]
    if arm == "full" and not encoder_names:
        raise RuntimeError("full-arm smoke has no nonzero encoder gradient")
    if arm == "frozen" and encoder_names:
        raise RuntimeError("frozen-arm smoke unexpectedly differentiated encoder")
    named = list(head_names.values())
    if encoder_names:
        preferred = [name for name in encoder_names if "encoder.layer.0.output.dense.weight" in name]
        named.append(preferred[0] if preferred else encoder_names[0])
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
            raise RuntimeError(f"smoke optimizer did not change active parameter: {name}")
    encoder_after = fingerprint(model.pool.encoder.state_dict())
    if (arm == "frozen" and encoder_after != encoder_before) or (arm == "full" and encoder_after == encoder_before):
        raise RuntimeError("smoke encoder liveness/frozen fingerprint violated")
    model.eval()
    saved_logits = probe_logits(model, tokenizer, records[0], cfg)
    with tempfile.TemporaryDirectory(prefix="smoke-weights-", dir=directory) as temporary:
        checkpoint = Path(temporary) / "smoke.safetensors"
        saved_hash = save_checkpoint(model, checkpoint)
        with torch.no_grad():
            parameters[named[0]].flatten()[0].add_(0.125)
        perturbed_hash = fingerprint(model.state_dict())
        restore = restore_checkpoint(model, checkpoint, saved_hash)
        restored_logits = probe_logits(model, tokenizer, records[0], cfg)
        restore.update({"saved_logits": saved_logits, "restored_logits": restored_logits,
                        "perturbed_fingerprint": perturbed_hash,
                        "max_abs_difference": numeric_difference(saved_logits, restored_logits)})
    temperatures = {endpoint: 1.0 for endpoint in catalogues}
    cache = StateFieldCache(model, tokenizer, temperatures, int(cfg["training"]["max_tokens"]), resources.guard)
    controls = cache_controls(cache, records, catalogues)
    proof = {"schema": "vey.native-field.smoke.v1", "arm": arm,
             "actual_fit_state_ids": [record["id"] for record in records],
             "actual_fit_decision_ids": [decision["id"] for record in records for decision in record["decisions"]],
             "valid_decisions": denominator, "loss": total / denominator,
             "gradient_statistics": gradients, "active_head_parameter_names": head_names,
             "active_encoder_parameter_name": named[-1] if encoder_names else None,
             "parameter_deltas": deltas, "encoder_before_sha256": encoder_before,
             "encoder_after_sha256": encoder_after, "initial_model_sha256": initial_model_hash,
             "strict_restore": restore, "cache_controls": controls,
             "encoder_calls": model.encoder_calls, "encoded_states": model.encoded_states,
             "smoke_weights_discarded": True, "custody": custody}
    del cache, frozen_features, opt, model, tokenizer, parameters, before
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return proof


def collect_logits(model, tokenizer, records, cfg, resources, frozen_features=None):
    model.eval()
    rows = []
    start = 0
    with torch.no_grad():
        while start < len(records):
            resources.guard()
            batch = records[start:start + resources.microbatch_states]
            try:
                fields = forward_records(model, tokenizer, batch, cfg, frozen_features)
            except torch.cuda.OutOfMemoryError:
                resources.reduce("eval")
                continue
            for index, record in enumerate(batch):
                for decision in record["decisions"]:
                    logits = fields[decision["endpoint"]][index].detach().cpu().tolist()
                    raw = probabilities(logits)
                    rows.append({**identity(record, decision), "logits": logits, "raw_probs": raw,
                                 "target_distribution": decision["target_distribution"],
                                 "gold": decision.get("gold"), "observed_raters": decision.get("observed_raters")})
            start += len(batch)
    return rows


def selection_metrics(rows):
    nll, accuracy = defaultdict(list), defaultdict(list)
    for row in rows:
        value = -math.fsum(target * math.log(max(mass, 1e-12))
                           for target, mass in zip(row["target_distribution"], row["raw_probs"]))
        nll[row["endpoint"]].append(value)
        if row["task"].lower() == "choice":
            maximum = max(row["raw_probs"])
            answer = min(key for key, mass in zip(row["candidate_ids"], row["raw_probs"]) if mass == maximum)
            accuracy[row["endpoint"]].append(int(answer == row["gold"]))
    if set(accuracy) != {"banking77.intent", "massive.intent"}:
        raise ValueError("selection must contain both native intent sources")
    source = {endpoint: math.fsum(values) / len(values) for endpoint, values in accuracy.items()}
    endpoint_nll = {endpoint: math.fsum(values) / len(values) for endpoint, values in nll.items()}
    return {"macro_source_intent_accuracy": math.fsum(source.values()) / len(source),
            "mean_endpoint_nll": math.fsum(endpoint_nll.values()) / len(endpoint_nll),
            "source_intent_accuracy": source, "endpoint_nll": endpoint_nll,
            "endpoint_counts": dict(Counter(row["endpoint"] for row in rows))}


def fit_epoch(model, tokenizer, records, cfg, resources, opt, epoch, frozen_features, rng):
    order = list(records)
    rng.shuffle(order)
    effective = int(cfg["training"]["effective_batch_states"])
    blocks = []
    total_loss, total_decisions = 0.0, 0
    model.train()
    for block_index, block in enumerate(chunks(order, effective)):
        denominator = sum(len(record["decisions"]) for record in block)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None
        while True:
            opt.zero_grad(set_to_none=True)
            raw_loss = 0.0
            fields = loss = None
            try:
                for batch in chunks(block, resources.microbatch_states):
                    resources.guard()
                    fields = forward_records(model, tokenizer, batch, cfg, frozen_features)
                    loss, _ = decision_loss(fields, batch)
                    (loss / denominator).backward()
                    raw_loss += float(loss.detach())
                    fields = loss = None
            except torch.cuda.OutOfMemoryError:
                fields = loss = None
                opt.zero_grad(set_to_none=True)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                resources.reduce(f"epoch{epoch}/block{block_index}")
                continue
            opt.step()
            break
        blocks.append({"states": len(block), "valid_decisions": denominator,
                       "normalized_loss": raw_loss / denominator,
                       "microbatch_states": resources.microbatch_states})
        total_loss += raw_loss
        total_decisions += denominator
    return {"valid_decisions": total_decisions, "states": len(records), "optimizer_updates": len(blocks),
            "mean_decision_loss": total_loss / total_decisions, "effective_batches": blocks}


def fit_temperatures(rows, catalogues):
    # Endpoints see only their disjoint calibration rows; raw predictions are retained.
    grid = [math.exp(math.log(0.5) + i * (math.log(10.0) - math.log(0.5)) / 100) for i in range(101)]
    grid[0], grid[-1] = 0.5, 10.0
    temperatures, evidence = {}, {}
    for endpoint in sorted(catalogues):
        selected = [row for row in rows if row["endpoint"] == endpoint]
        if not selected:
            raise ValueError(f"no calibration observations for {endpoint}")
        scored = []
        for temperature in grid:
            losses = []
            for row in selected:
                masses = probabilities(row["logits"], temperature)
                losses.append(-math.fsum(target * math.log(max(mass, 1e-12))
                                         for target, mass in zip(row["target_distribution"], masses)))
            scored.append({"temperature": temperature, "nll": math.fsum(losses) / len(losses)})
        best = min(scored, key=lambda item: (item["nll"], abs(item["temperature"] - 1.0), item["temperature"]))
        temperatures[endpoint] = best["temperature"]
        evidence[endpoint] = {"phase": "calibration", "decisions": len(selected), "grid": scored,
                              "selected_temperature": best["temperature"], "selected_nll": best["nll"]}
    return temperatures, evidence


def prediction_controls(directory, cache, dev, catalogues):
    endpoints = list(catalogues)
    plan = donor_plan(dev, endpoints)
    indexed = {decision["id"]: (record, decision) for record in dev for decision in record["decisions"]}
    counts = {endpoint: {"total": 0, "available": 0, "unavailable": 0, "unique_donors": 0}
              for endpoint in endpoints}
    donor_sets = defaultdict(set)
    repeated = cache_controls(cache, dev, catalogues)
    initial_entries = [{"model_identity": model_identity,
                        "state_sha256": hashlib.sha256(state_bytes).hexdigest()}
                       for model_identity, state_bytes in cache.entries]
    paths = {name: directory / (name + ".jsonl")
             for name in ("dev", "masked", "swapped", "swapped_unavailable", "coordinate_reversal")}
    path_counts = {}
    reversal_count = 0

    def counters():
        return {"encoder_calls": cache.encoder_calls, "cache_misses": cache.misses,
                "entries": len(cache.entries)}

    def receipt(name, before):
        after = counters()
        path_counts[name] = {key + "_before": value for key, value in before.items()}
        path_counts[name].update({key + "_after": value for key, value in after.items()})

    before = counters()
    with paths["dev"].open("x", encoding="utf-8") as original_stream, paths["coordinate_reversal"].open("x", encoding="utf-8") as reversal_stream:
        for record in dev:
            for decision in record["decisions"]:
                ids = catalogues[decision["endpoint"]]
                output = cache.read(record["state"], decision, ids)
                write_row(original_stream, {**identity(record, decision), **output})
                calls_before = cache.encoder_calls
                reversed_output = cache.read(record["state"], decision, ids, reverse=True)
                calls_after = cache.encoder_calls
                if calls_before != calls_after:
                    raise RuntimeError("coordinate reversal encoded state again")
                for mass_key in ("probs", "raw_probs", "logits"):
                    if output[mass_key] is not None and dict(zip(output["candidate_ids"], output[mass_key])) != dict(zip(reversed_output["candidate_ids"], reversed_output[mass_key])):
                        raise RuntimeError(f"coordinate reversal changed {mass_key} by native ID")
                for key in ("answer", "ties", "expected_level", "cdf", "raw_answer", "raw_ties", "raw_expected_level", "raw_cdf"):
                    if key in output and output[key] != reversed_output[key]:
                        raise RuntimeError(f"coordinate reversal changed {key}")
                write_row(reversal_stream, {**identity(record, decision),
                          "original": output, "reversed": reversed_output,
                          "encoder_calls_before": calls_before, "encoder_calls_after": calls_after})
                reversal_count += 1
    receipt("dev", before)
    before = counters()
    with paths["masked"].open("x", encoding="utf-8") as stream:
        for record in dev:
            for decision in record["decisions"]:
                output = cache.read(EMPTY_EVIDENCE, decision, catalogues[decision["endpoint"]])
                write_row(stream, {**identity(record, decision), **output})
    receipt("masked", before)
    before = counters()
    with paths["swapped"].open("x", encoding="utf-8") as swapped_stream, paths["swapped_unavailable"].open("x", encoding="utf-8") as unavailable_stream:
        for record in dev:
            for decision in record["decisions"]:
                endpoint = decision["endpoint"]
                counts[endpoint]["total"] += 1
                donor_id = plan[decision["id"]]
                if donor_id is None:
                    counts[endpoint]["unavailable"] += 1
                    write_row(unavailable_stream, {**identity(record, decision),
                              "reason": "no different-component different-native-target dev donor"})
                    continue
                donor_record, _ = indexed[donor_id]
                output = cache.read(donor_record["state"], decision, catalogues[endpoint])
                write_row(swapped_stream, {**identity(record, decision), **output, "donor_id": donor_id})
                counts[endpoint]["available"] += 1
                donor_sets[endpoint].add(donor_id)
    receipt("swapped", before)
    if path_counts["swapped"]["encoder_calls_before"] != path_counts["swapped"]["encoder_calls_after"]:
        raise RuntimeError("swapped donor read did not reuse the original learned state field")
    for endpoint in counts:
        counts[endpoint]["unique_donors"] = len(donor_sets[endpoint])
    return {"cache_controls": repeated, "coordinate_reversal_rows": reversal_count,
            "encoder_calls": cache.encoder_calls, "cache_misses": cache.misses,
            "model_identity": cache.identity, "swap_denominators": counts,
            "path_counts": path_counts, "entries_at_prediction_start": initial_entries,
            "donor_rule": "endpoint sorted-ID next cyclic different component and gold/mean; reuse allowed",
            "state_mask_text": EMPTY_EVIDENCE}


def metadata_base(cfg, manifest, root, arm, mode, resources):
    return {"schema": "vey.native-field.arm.v1", "arm": arm, "mode": mode,
            "protocol_sha256": sha256_file(data.PROTOCOL),
            "dataset_manifest_sha256": sha256_file(root / "dataset_manifest.json"),
            "dataset_root": str(root), "dataset_phases": manifest["phases"],
            "source_exports": manifest["source_exports"],
            "source_files": {str(HERE / name): sha256_file(HERE / name)
                             for name in ("native_field_model.py", "native_field_train.py", "native_field_data.py")},
            "eligible_endpoints": manifest["eligible_endpoints"],
            "ineligible_endpoints": manifest["ineligible_endpoints"],
            "catalogues": manifest["catalogues"], "recipe": cfg["training"],
            "state_mask_text": EMPTY_EVIDENCE,
            "semantic_scope": cfg["model"]["semantic_scope"],
            "execution_correction_sha256": (sha256_file(HERE / "native_field_execution_correction_v1.json")
                                             if data.execution_correction(manifest) else None),
            "no_dev_selection_or_calibration": True, "sealed_phases_accessed": False,
            "resource_events": resources.events}


def artifact_hashes(directory):
    return {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(directory.iterdir()) if path.is_file() and path.name != "metadata.json"}


def run(root: str | Path, arm: str, mode: str, device: str):
    cfg = data.protocol()
    root = Path(root).resolve()
    manifest, phases = data.load_dataset(root)
    endpoints = manifest["eligible_endpoints"]
    catalogues = {endpoint: manifest["catalogues"][endpoint] for endpoint in endpoints}
    phases = {phase: eligible_records(records, endpoints) for phase, records in phases.items()}
    correction = data.execution_correction(manifest)
    smoke_suffix = correction["smoke_directory_suffix"] if correction else "-smoke"
    directory = root / (arm if mode == "train" else arm + smoke_suffix)
    directory.mkdir(exist_ok=False)
    history = {"schema": "vey.native-field.history.v1", "arm": arm, "epochs": []}
    liveness = {"schema": "vey.native-field.liveness.v1", "arm": arm}
    with Resources(cfg, device) as resources:
        metadata = metadata_base(cfg, manifest, root, arm, mode, resources)
        try:
            if arm == "prior":
                prior, prior_counts = fit_prior(phases["fit"], catalogues)
                temperatures = {endpoint: 1.0 for endpoint in endpoints}
                cache = StateFieldCache(None, None, temperatures, int(cfg["training"]["max_tokens"]),
                                        resources.guard, prior=prior)
                liveness["smoke"] = {"kind": "state_blind_fit_prior", "fit_endpoint_counts": prior_counts,
                                     "encoder_gradient": "not applicable: no neural model",
                                     "prior": prior, "smoke_weights_discarded": True}
                history.update({"fit_endpoint_counts": prior_counts, "prior": prior,
                                "checkpoint_selection": "not applicable: empirical fit-only prior"})
                metadata.update({"prior": prior, "temperatures": temperatures,
                                 "temperature_evidence": "not applicable: exact empirical masses, T=1",
                                 "checkpoint": None, "custody": None})
                if mode == "smoke":
                    liveness["smoke"]["cache_controls"] = cache_controls(cache, smoke_records(phases["fit"], catalogues), catalogues)
                else:
                    liveness["controls"] = prediction_controls(directory, cache, phases["dev"], catalogues)
            else:
                liveness["smoke"] = neural_smoke(cfg, catalogues, phases["fit"], arm, device, directory, resources)
                if mode == "train":
                    seed(cfg)
                    tokenizer, model, custody = load_stock(cfg, catalogues, arm, device)
                    fresh_hash = fingerprint(model.state_dict())
                    if fresh_hash != liveness["smoke"]["initial_model_sha256"]:
                        raise RuntimeError("real arm was not freshly seeded after discarded smoke")
                    liveness["fresh_training_initial_sha256"] = fresh_hash
                    frozen_features = FrozenFeatures(model, tokenizer, cfg, resources) if arm == "frozen" else None
                    opt = optimizer(model)
                    rng = random.Random(int(cfg["training"]["seed"]))
                    best_rank, best_epoch, best_hash, selected_probe = None, None, None, None
                    checkpoint = directory / "selected.safetensors"
                    probe_record = phases["selection"][0]
                    with (directory / "selection.jsonl").open("x", encoding="utf-8") as selection_stream:
                        for epoch in range(int(cfg["training"]["epochs"]) + 1):
                            fit = None if epoch == 0 else fit_epoch(
                                model, tokenizer, phases["fit"], cfg, resources, opt, epoch, frozen_features, rng)
                            rows = collect_logits(model, tokenizer, phases["selection"], cfg, resources, frozen_features)
                            metrics = selection_metrics(rows)
                            for row in rows:
                                write_row(selection_stream, {"epoch": epoch, **row})
                            selection_stream.flush()
                            candidate_hash = fingerprint(model.state_dict())
                            rank = (-metrics["macro_source_intent_accuracy"], metrics["mean_endpoint_nll"], epoch)
                            if best_rank is None or rank < best_rank:
                                best_rank, best_epoch = rank, epoch
                                best_hash = save_checkpoint(model, checkpoint)
                                selected_probe = probe_logits(model, tokenizer, probe_record, cfg)
                            entry = {"epoch": epoch, "fit": fit, "selection": metrics,
                                     "state_dict_fingerprint": candidate_hash, "selected_epoch_so_far": best_epoch}
                            history["epochs"].append(entry)
                            history["selected_epoch"] = best_epoch
                            history["selected_fingerprint"] = best_hash
                            write_json(directory / "history.json", history)
                    before_restore = fingerprint(model.state_dict())
                    trained_encoder_hash = fingerprint(model.pool.encoder.state_dict())
                    final_logits = probe_logits(model, tokenizer, probe_record, cfg)
                    strict = restore_checkpoint(model, checkpoint, best_hash)
                    restored_logits = probe_logits(model, tokenizer, probe_record, cfg)
                    strict.update({"before_restore_fingerprint": before_restore, "selected_epoch": best_epoch,
                                   "probe_state_id": probe_record["id"], "selected_logits": selected_probe,
                                   "before_restore_logits": final_logits, "restored_logits": restored_logits,
                                   "max_abs_difference": numeric_difference(selected_probe, restored_logits)})
                    liveness["selected_restore"] = strict
                    selected_encoder_hash = fingerprint(model.pool.encoder.state_dict())
                    liveness["encoder_training_before_sha256"] = custody["encoder_initial_sha256"]
                    liveness["encoder_training_after_sha256"] = trained_encoder_hash
                    liveness["encoder_selected_sha256"] = selected_encoder_hash
                    if arm == "frozen" and (trained_encoder_hash != custody["encoder_initial_sha256"] or
                                            selected_encoder_hash != custody["encoder_initial_sha256"]):
                        raise RuntimeError("frozen encoder changed during head training")
                    if arm == "full" and trained_encoder_hash == custody["encoder_initial_sha256"]:
                        raise RuntimeError("full encoder unchanged after actual training")
                    model.zero_grad(set_to_none=True)
                    del opt
                    calibration = collect_logits(model, tokenizer, phases["calibration"], cfg, resources, frozen_features)
                    with (directory / "calibration.jsonl").open("x", encoding="utf-8") as stream:
                        for row in calibration:
                            write_row(stream, row)
                    temperatures, temperature_evidence = fit_temperatures(calibration, catalogues)
                    del frozen_features, calibration
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    cache = StateFieldCache(model, tokenizer, temperatures,
                                            int(cfg["training"]["max_tokens"]), resources.guard)
                    liveness["controls"] = prediction_controls(directory, cache, phases["dev"], catalogues)
                    metadata.update({"custody": custody, "temperatures": temperatures,
                                     "temperature_evidence": temperature_evidence,
                                     "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint),
                                                    "state_dict_fingerprint": best_hash, "selected_epoch": best_epoch}})
                else:
                    metadata.update({"custody": liveness["smoke"]["custody"], "checkpoint": None,
                                     "temperatures": None, "temperature_evidence": None})
            metadata["status"] = "smoke_complete" if mode == "smoke" else "train_complete"
            if "controls" in liveness:
                metadata["swap_denominators"] = liveness["controls"]["swap_denominators"]
            write_json(directory / "history.json", history)
            write_json(directory / "liveness.json", liveness)
            metadata["artifact_hashes"] = artifact_hashes(directory)
            write_json(directory / "metadata.json", metadata)
        except BaseException as exc:
            metadata.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            write_json(directory / "history.json", history)
            write_json(directory / "liveness.json", liveness)
            metadata["artifact_hashes"] = artifact_hashes(directory)
            write_json(directory / "metadata.json", metadata)
            raise
    return directory


def main():
    cfg = data.protocol()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=cfg["output_root"])
    parser.add_argument("--arm", required=True, choices=("prior", "frozen", "full"))
    parser.add_argument("--mode", choices=("smoke", "train"), default="train")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "cuda:0"))
    args = parser.parse_args()
    directory = run(args.root, args.arm, args.mode, args.device)
    print(json.dumps({"output": str(directory), "arm": args.arm, "mode": args.mode}, sort_keys=True))


if __name__ == "__main__":
    main()
