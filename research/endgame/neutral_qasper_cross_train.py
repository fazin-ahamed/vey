#!/usr/bin/env python3
"""Run the custody-bound QNATIVE-2 preflight, Modal numerical smoke and joint fit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import random
import re
import traceback

try:
    from . import neutral_qasper_cross_data as data
    from . import neutral_qasper_cross_model as neural
except ImportError:
    import neutral_qasper_cross_data as data
    import neutral_qasper_cross_model as neural

ENDPOINTS = (*data.INTENTS, "qasper.yes_no", "qasper.answerability", "qasper.evidence_retrieval", "qasper.extractive_answer")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(data.canonical(value) + "\n")


def write_row(stream, value):
    stream.write((data.canonical(value) + "\n").encode("utf-8"))


class Capture:
    def __init__(self, path, header):
        self.raw = Path(path).open("xb")
        self.stream = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)
        write_row(self.stream, header)

    def __enter__(self):
        return self.stream

    def __exit__(self, *args):
        self.stream.close()
        self.raw.close()


class Resources:
    def __init__(self, cuda):
        self.cuda = cuda
        self.events = []
        self.initial_swap = self.memory()[1]
        self.peak = 0

    @staticmethod
    def memory():
        fields = {line.split(":")[0]: int(line.split()[1]) * 1024
                  for line in Path("/proc/meminfo").read_text().splitlines()}
        return fields["MemAvailable"], fields["SwapTotal"] - fields["SwapFree"]

    def guard(self):
        available, swap = self.memory()
        data.require(available >= 4 * 2**30, "Available RAM below registered guard")
        data.require(swap - self.initial_swap <= 256 * 2**20, "Swap growth exceeds registered guard")
        if self.cuda:
            import torch
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            self.peak = max(self.peak, torch.cuda.max_memory_allocated())
            data.require(self.peak <= 12 * 2**30, "Peak allocated GPU memory exceeds 12GiB")
            data.require(free >= 2048 * 2**20, "Free GPU memory below registered guard")


def nll(distribution, probabilities):
    return -math.fsum(t * math.log(max(p, 1e-12)) for t, p in zip(distribution, probabilities))


def token_counter(texts):
    return Counter(word for text in texts for word in re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def annotation_f1(predicted, native):
    left, right = token_counter(predicted), token_counter(native)
    overlap = sum((left & right).values())
    return 2 * overlap / (sum(left.values()) + sum(right.values())) if left or right else 0.


def primary(row, prediction):
    target, task = row["target"], row["serving"]["task"]
    if task in ("bool", "choice"):
        distribution = target.get("distribution")
        if distribution is None:
            return None, None
        ids = prediction["candidate_ids"]
        gold = max(range(len(distribution)), key=lambda i: (distribution[i], -i))
        guess = min(range(len(ids)), key=lambda i: (-prediction["probabilities"][i], ids[i]))
        return float(gold == guess), nll(distribution, prediction["probabilities"])
    if task == "retrieval":
        items = [item for rating in target["ratings"] for item in rating["matches"]]
        if not items:
            return None, None
        selected = set(prediction["ranking"][:10])
        return sum(bool(selected.intersection(item["block_ids"])) for item in items) / len(items), None
    values = [annotation_f1([s["text"] for s in prediction["spans"]], [item["text"] for item in r["matches"]])
              for r in target["ratings"] if r["observed"] and r["matches"]]
    return math.fsum(values) / len(values) if values else None, None


def selection_summary(rows, predictions):
    values, losses = defaultdict(list), defaultdict(list)
    for row, prediction in zip(rows, predictions):
        value, loss = primary(row, prediction)
        task = row["serving"]["task"]
        if task == "retrieval":
            items = [item for rating in row["target"]["ratings"] for item in rating["matches"]]
            selected = set(prediction["ranking"][:10])
            values[row["endpoint"]].extend(float(bool(selected.intersection(item["block_ids"]))) for item in items)
        elif task == "extract":
            values[row["endpoint"]].extend(annotation_f1([s["text"] for s in prediction["spans"]],
                [item["text"] for item in rating["matches"]]) for rating in row["target"]["ratings"]
                if rating["observed"] and rating["matches"])
        elif value is not None:
            values[row["endpoint"]].append(value)
        if loss is not None:
            losses[row["endpoint"]].append(loss)
    data.require(all(values[e] for e in ENDPOINTS), "Missing selection primary endpoint")
    metrics = {e: math.fsum(values[e]) / len(values[e]) for e in ENDPOINTS}
    mean_nll = math.fsum(math.fsum(v) / len(v) for v in losses.values()) / len(losses)
    return {"primary": metrics, "macro_primary": math.fsum(metrics.values()) / 6,
            "mean_nll": mean_nll, "counts": {e: len(v) for e, v in values.items()}}


def fit_temperatures(rows, predictions):
    grid = [math.exp(math.log(.5) + i * (math.log(10) - math.log(.5)) / 100) for i in range(101)]
    result, evidence = {}, {}
    for endpoint in ENDPOINTS:
        if endpoint.endswith("extractive_answer"):
            continue
        samples = []
        for row, prediction in zip(rows, predictions):
            if row["endpoint"] != endpoint:
                continue
            target = data.retrieval_target(row["target"], prediction["candidate_ids"]) if row["serving"]["task"] == "retrieval" else row["target"].get("distribution")
            if target is not None:
                samples.append((target, prediction["raw_logits"]))
        data.require(bool(samples), "No categorical calibration targets: " + endpoint)
        candidates = [{"temperature": t, "nll": math.fsum(nll(target, neural.probabilities(logits, t))
                       for target, logits in samples) / len(samples)} for t in grid]
        best = min(candidates, key=lambda x: (x["nll"], abs(x["temperature"] - 1), x["temperature"]))
        result[endpoint] = best["temperature"]
        evidence[endpoint] = {"rows": len(samples), "grid": candidates, "selected": best}
    return result, evidence


class Baselines:
    def __init__(self, fit):
        self.priors = {}
        for endpoint in ("qasper.yes_no", "qasper.answerability"):
            targets = [r["target"]["distribution"] for r in fit if r["endpoint"] == endpoint and r["target"].get("distribution") is not None]
            data.require(bool(targets), "Missing FIT prior")
            self.priors[endpoint] = [math.fsum(t[i] for t in targets) / len(targets) for i in range(2)]
        texts = {b["text"] for r in fit for b in r["blocks"]}
        df = Counter(word for text in texts for word in set(token_counter([text])))
        self.idf = {word: math.log((len(texts) + 1) / (frequency + 1)) + 1 for word, frequency in df.items()}
        self.n = len(texts)
        self.vectors = {}

    def vector(self, text):
        if text in self.vectors:
            return self.vectors[text]
        values = {w: count * self.idf.get(w, math.log(self.n + 1) + 1) for w, count in token_counter([text]).items()}
        norm = math.sqrt(math.fsum(v * v for v in values.values()))
        result = {w: v / norm for w, v in values.items()} if norm else {}
        # Bound caches independently of corpus/question size.
        if len(self.vectors) >= 4096:
            self.vectors.clear()
        self.vectors[text] = result
        return result

    def predict(self, row):
        task = row["serving"]["task"]
        base = {"kind": "prediction", "id": row["id"], "endpoint": row["endpoint"], "component_id": row["component_id"],
                "phase": row["phase"], "condition": "baseline", "input_sha256": row["input_sha256"],
                "status": "OK", "error": None, "raw_logits": [], "temperature": 1., "spans": []}
        if task == "bool":
            probs = self.priors[row["endpoint"]]
            return {**base, "baseline": "FIT_prior", "candidate_ids": ["false", "true"],
                    "raw_probabilities": probs, "probabilities": probs,
                    "ranking": sorted(["false", "true"], key=lambda c: (-probs[int(c == "true")], c))}
        query = self.vector(row["serving"]["question"])
        scores = []
        for block in row["blocks"]:
            vector = self.vector(block["text"])
            scores.append(math.fsum(value * vector.get(word, 0.) for word, value in query.items()))
        ids = [b["id"] for b in row["blocks"]]
        ranking = sorted(ids, key=lambda cid: (-scores[ids.index(cid)], cid))
        spans = []
        if task == "extract" and ranking:
            block = next(b for b in row["blocks"] if b["id"] == ranking[0])
            spans = [{"block_id": block["id"], "start": 0, "end": len(block["text"]), "text": block["text"], "confidence": None}]
        return {**base, "baseline": "whole_TFIDF_block" if task == "extract" else "TFIDF",
                "candidate_ids": ids if task == "retrieval" else [], "raw_logits": scores if task == "retrieval" else [],
                "raw_probabilities": neural.probabilities(scores) if task == "retrieval" else [],
                "probabilities": neural.probabilities(scores) if task == "retrieval" else [], "ranking": ranking, "spans": spans}


def changed(row, condition, donor=None, subset=None):
    result = {**row, "serving": dict(row["serving"]), "blocks": row["blocks"]}
    if condition == "question_masked":
        result["serving"]["question"] = ""
    elif condition == "state_masked":
        result["serving"]["state"] = "[EMPTY EVIDENCE]"
        result["blocks"] = [{"id": b["id"], "text": "[EMPTY EVIDENCE]"} for b in row["blocks"]]
        if row["serving"]["task"] == "retrieval":
            result["serving"]["candidates"] = result["blocks"]
    elif condition == "question_swapped":
        result["serving"]["question"] = donor["serving"]["question"]
    elif condition == "reordered":
        result["blocks"] = list(reversed(row["blocks"]))
        if row["serving"]["task"] == "retrieval":
            result["serving"]["candidates"] = list(reversed(row["serving"]["candidates"]))
        result["original_blocks"] = row["blocks"]
    elif condition == "candidate_count":
        result["blocks"] = [b for b in row["blocks"] if b["id"] in subset]
        if row["serving"]["task"] == "retrieval":
            result["serving"]["candidates"] = [c for c in row["serving"]["candidates"] if c["id"] in subset]
    result["input_sha256"] = data.value_digest(result["serving"])
    return result


def donor_plan(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["blocks"]:
            grouped[(row["group_id"], row["paper_id"], row["endpoint"])].append(row)
    plan = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda r: r["id"])
        for i, row in enumerate(ordered):
            plan[row["id"]] = next((ordered[(i + step) % len(ordered)] for step in range(1, len(ordered))
                                    if ordered[(i + step) % len(ordered)]["serving"]["question"] != row["serving"]["question"]), None)
    return plan


def optimizer(model):
    import torch
    encoder = list(model.cross.transformer().parameters())
    ids = {id(p) for p in encoder}
    heads = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": encoder, "lr": 2e-5}, {"params": heads, "lr": 1e-3}], weight_decay=.01)


def train_epoch(model, rows, opt, resources, epoch):
    model.train()
    model.active_condition = "fit"
    order = list(rows)
    random.Random(7 + epoch).shuffle(order)
    # Determine eligibility from native ledgers/input-only support, before any gradient work.
    eligible = []
    counts, denominators, unavailable = Counter(), Counter(), Counter()
    for row in order:
        task = row["serving"]["task"]
        if row["endpoint"] in data.INTENTS:
            valid = row["target"].get("distribution") is not None
        elif task == "extract":
            windows, census = data.source_windows(model.tokenizer, row["serving"]["question"], row["blocks"])
            valid = data.bio_supervision(row["target"], windows, census)["supervised_source_tokens"] > 0
        elif task == "retrieval":
            valid = data.retrieval_target(row["target"], [b["id"] for b in row["blocks"]]) is not None
        if valid:
            eligible.append(row)
        if not valid:
            unavailable[row["endpoint"]] += 1
    total = 0.
    batches = 0
    for start in range(0, len(eligible), 32):
        block = eligible[start:start + 32]
        opt.zero_grad(set_to_none=True)
        for row in block:
            resources.guard()
            output = model.forward_row(row, resources.guard)
            loss, denominator = neural.decision_loss(row, output)
            data.require(loss is not None and bool(loss.isfinite()), "Nonfinite/missing eligible loss")
            (loss / len(block)).backward()
            total += float(loss.detach())
            counts[row["endpoint"]] += 1
            denominators[row["endpoint"]] += denominator
            del output, loss
        opt.step()
        batches += 1
    return {"supervised_counts": dict(counts), "loss_denominators": dict(denominators),
            "unavailable_counts": dict(unavailable), "effective_batches": batches,
            "mean_loss": total / len(eligible) if eligible else None}


def smoke(model, rows, resources, directory):
    import torch
    from safetensors.torch import save_file, load_file
    try:
        from .native_field_model import fingerprint
    except ImportError:
        from native_field_model import fingerprint
    model.active_condition = "smoke"
    selected = []
    for endpoint in ENDPOINTS:
        for row in rows:
            if row["endpoint"] != endpoint:
                continue
            model.eval()
            with torch.no_grad():
                output = model.forward_row(row, resources.guard)
                loss, _ = neural.decision_loss(row, output)
            if loss is not None:
                selected.append(row)
                break
        data.require(any(r["endpoint"] == endpoint for r in selected), "No supported smoke row: " + endpoint)
    initial = fingerprint(model.state_dict())
    before = {name: p.detach().clone() for name, p in model.named_parameters()
              if name in ("bool_head.weight", "bio_head.weight", "cross.scorer.head.0.weight")
              or name.endswith("embeddings.word_embeddings.weight")}
    opt = optimizer(model)
    opt.zero_grad(set_to_none=True)
    model.train()
    losses = {}
    for row in selected:
        output = model.forward_row(row, resources.guard)
        loss, _ = neural.decision_loss(row, output)
        data.require(bool(loss.isfinite()), "Smoke loss nonfinite")
        losses[row["endpoint"]] = float(loss.detach())
        (loss / len(selected)).backward()
        del output, loss
    gradients = {name: float(p.grad.detach().abs().sum()) for name, p in model.named_parameters()
                 if name in before and p.grad is not None}
    data.require(set(gradients) == set(before) and all(v > 0 and math.isfinite(v) for v in gradients.values()), "Inactive smoke gradients")
    opt.step()
    del opt
    deltas = {name: float((dict(model.named_parameters())[name].detach() - old).abs().sum()) for name, old in before.items()}
    data.require(all(v > 0 and math.isfinite(v) for v in deltas.values()), "Inactive smoke optimizer deltas")
    after = fingerprint(model.state_dict())
    checkpoint = directory / "smoke.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(checkpoint))
    model.load_state_dict(load_file(str(checkpoint)), strict=True)
    data.require(fingerprint(model.state_dict()) == after and after != initial, "Smoke strict restore failed")
    model.zero_grad(set_to_none=True)
    cache = neural.EvalCache(model, model.tokenizer, after, resources.guard)
    first = cache.predict(selected[0])
    repeat = cache.predict(selected[0])
    data.require(first["raw_logits"] == repeat["raw_logits"] and repeat["runtime_counters"]["encoded_sequences"] == 0, "Exact repeat failed")
    # Actual effective block and longest input-only joint source case are mandatory resource probes.
    opt = optimizer(model)
    order = list(rows)
    random.Random(8).shuffle(order)
    eligible, window_counts = [], {}
    for row in order:
        task = row["serving"]["task"]
        if row["endpoint"] in data.INTENTS:
            valid = row["target"].get("distribution") is not None
        elif not row["blocks"]:
            valid = False
        elif task == "extract":
            windows, census = data.source_windows(model.tokenizer, row["serving"]["question"], row["blocks"])
            valid = data.bio_supervision(row["target"], windows, census)["supervised_source_tokens"] > 0
        elif task == "retrieval":
            valid = data.retrieval_target(row["target"], [b["id"] for b in row["blocks"]]) is not None
        else:
            valid = row["target"].get("distribution") is not None
        if valid:
            eligible.append(row)
            if row["blocks"]:
                window_counts[row["id"]] = len(data.source_windows(model.tokenizer, row["serving"]["question"], row["blocks"])[0])
    block = eligible[:32]
    data.require(len(block) == 32, "Insufficient native supervised effective block")
    model.train()
    opt.zero_grad(set_to_none=True)
    for row in block:
        output = model.forward_row(row, resources.guard)
        loss, _ = neural.decision_loss(row, output)
        (loss / 32).backward()
        del loss, output
    opt.step()
    del opt
    longest = max((r for r in eligible if r["blocks"]), key=lambda r: window_counts[r["id"]])
    model.zero_grad(set_to_none=True)
    model.train()
    output = model.forward_row(longest, resources.guard)
    loss, _ = neural.decision_loss(longest, output)
    if loss is not None:
        loss.backward()
    resources.guard()
    return {"status": "PASS", "initial_model_sha256": initial, "after_step_sha256": after,
            "losses": losses, "named_gradients": gradients, "named_parameter_deltas": deltas,
            "strict_restore": {"strict": True, "fingerprint": after}, "effective_block_decisions": 32,
            "effective_block_ids": [r["id"] for r in block],
            "longest_input_id": longest["id"], "longest_input_windows": len(output["windows"]),
            "counters": dict(model.counters), "peak_allocated_gpu_bytes": resources.peak}


def run(out, mode, protocol_path=data.AMENDMENT):
    directory = Path(out) / mode
    directory.mkdir(parents=True, exist_ok=False)
    resources = Resources(mode != "preflight")
    metadata = {"schema": "vey.qnative.cross.metadata.v1", "mode": mode, "status": "running",
                "protocol_sha256": data.pin(protocol_path)["sha256"], "source_pins": {},
                "promotion": False, "Pareto_credit": False, "endgame_complete": False}
    history = {"epochs": [], "selected_epoch": None, "selected_fingerprint": None}
    liveness = {}
    try:
        if mode in ("smoke", "train"):
            data.require(bool(os.environ.get("QNATIVE_DATASET_ROOT")), "Neural jobs require independently verified derived inputs")
        cfg, pins = data.study_authority(protocol_path)
        metadata["source_pins"] = pins
        for name in ("selected_initial_cross_checkpoint", "selected_initial_cross_metadata"):
            entry = cfg["current_authority"][name]
            data.require(data.pin(entry["path"]) == entry, "Required initial lineage differs: " + name)
        entry = cfg["authority"]["native_field_protocol"]
        data.require(data.pin(entry["path"]) == entry, "Native encoder protocol identity differs")
        current_nice = os.getpriority(os.PRIO_PROCESS, 0)
        if current_nice < 10:
            os.nice(10 - current_nice)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        phases = {phase: data.load_phase(phase, protocol_path) for phase in data.PHASES}
        resources.guard()
        tokenizer = neural.load_tokenizer()
        phase_counts = {}
        for phase, rows in phases.items():
            counts, windows_total = Counter(), 0
            for row in rows:
                counts[row["endpoint"]] += 1
                if row["endpoint"] not in data.INTENTS:
                    windows, _ = data.source_windows(tokenizer, row["serving"]["question"], row["blocks"])
                    windows_total += len(windows)
            phase_counts[phase] = {"endpoint_counts": dict(counts), "rows": len(rows), "windows": windows_total}
        metadata["phase_counts"] = phase_counts
        metadata["tokenizer_sha256"] = __import__("hashlib").sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        if mode == "preflight":
            liveness = {"model_loads": 0, "model_forwards": 0, "sealed_access": False}
            metadata["status"] = "preflight_complete"
        else:
            import modal
            data.require(not modal.is_local(), "Smoke/train neural work is restricted to Modal")
            import torch
            from safetensors.torch import save_file, load_file
            try:
                from .native_field_model import fingerprint
            except ImportError:
                from native_field_model import fingerprint
            torch.set_num_threads(2)
            data.require(torch.cuda.is_available(), "CUDA required; no CPU neural fallback")
            catalogues = {endpoint: [c["id"] for c in next(r for r in phases["fit"] if r["endpoint"] == endpoint)["serving"]["candidates"]]
                          for endpoint in data.INTENTS}
            tokenizer, model, custody = neural.build_model(cfg, catalogues)
            model.tokenizer = tokenizer
            liveness["smoke"] = smoke(model, phases["fit"], resources, directory)
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            if mode == "smoke":
                metadata.update(status="smoke_complete", custody=custody)
            else:
                tokenizer, model, custody = neural.build_model(cfg, catalogues)
                model.tokenizer = tokenizer
                data.require(custody["initial_model_fingerprint"] == liveness["smoke"]["initial_model_sha256"], "Fresh training initialization differs")
                metadata["custody"] = custody
                original_meta = json.loads(Path(cfg["current_authority"]["selected_initial_cross_metadata"]["path"]).read_text())
                original_path = Path(cfg["current_authority"]["selected_initial_cross_metadata"]["path"]).with_name("dev.jsonl")
                original_pin = data.pin(original_path)
                data.require({k: original_pin[k] for k in ("bytes", "sha256")} == original_meta["artifact_hashes"]["dev.jsonl"],
                             "Fixed original cross DEV reference differs")
                indexed = {r["id"]: r for r in phases["dev"] if r["endpoint"] in data.INTENTS}
                original = {}
                with original_path.open() as stream:
                    for captured in map(json.loads, stream):
                        rid = captured["id"]
                        data.require(rid in indexed and rid not in original, "Original intent DEV membership differs")
                        row = indexed[rid]
                        ids = captured["candidate_ids"]
                        data.require(ids == [c["id"] for c in row["serving"]["candidates"]]
                                     and captured["component_id"] == row["component_id"]
                                     and captured["endpoint"] == row["endpoint"], "Original cross DEV coordinates differ")
                        original[rid] = {"kind": "prediction", "id": rid, "endpoint": row["endpoint"],
                                         "component_id": row["component_id"], "phase": "dev", "condition": "baseline",
                                         "baseline": "original_cross", "input_sha256": row["input_sha256"], "status": "OK", "error": None,
                                         "candidate_ids": ids, "raw_logits": captured["logits"], "raw_probabilities": captured["raw_probs"],
                                         "probabilities": captured["probs"], "temperature": original_meta["temperatures"][row["endpoint"]],
                                         "ranking": sorted(ids, key=lambda cid: (-captured["probs"][ids.index(cid)], cid)), "spans": []}
                data.require(set(original) == set(indexed), "Original cross DEV reference incomplete")
                metadata["original_intent_reference"] = original_pin
                opt = optimizer(model)
                best, best_epoch, best_fingerprint = None, None, None
                best_probes = None
                candidates = []
                def header(phase, condition, checkpoint_sha256=None):
                    return {"kind": "header", "schema": "vey.qnative.cross.capture.v1", "phase": phase,
                            "condition": condition, "protocol_sha256": metadata["protocol_sha256"],
                            "source_pins": pins, "checkpoint_sha256": checkpoint_sha256}
                with Capture(directory / "support.jsonl.gz", header("all", "support")) as stream:
                    for phase, rows in phases.items():
                        for row in rows:
                            windows, coverage = data.source_windows(tokenizer, row["serving"]["question"], row["blocks"]) if row["blocks"] else ([], [])
                            support = data.bio_supervision(row["target"], windows, coverage) if row["serving"]["task"] == "extract" else None
                            write_row(stream, {"kind": "support", "id": row["id"], "endpoint": row["endpoint"],
                                               "component_id": row["component_id"], "phase": phase,
                                               "target_sha256": data.value_digest(row["target"]), "target": row["target"],
                                               "coverage": coverage, "windows": windows, "support": support,
                                               "source_token_offsets": {b["block_id"]: b["source_token_offsets"] for b in coverage}})
                with Capture(directory / "selection.jsonl.gz", header("selection", "selection")) as stream:
                    for epoch in range(11):
                        fit = {"supervised_counts": {}, "loss_denominators": {}} if epoch == 0 else train_epoch(model, phases["fit"], opt, resources, epoch)
                        identity = fingerprint(model.state_dict())
                        cache = neural.EvalCache(model, tokenizer, identity, resources.guard)
                        predictions = []
                        for row in phases["selection"]:
                            prediction = cache.predict(row, condition="selection")
                            write_row(stream, {**prediction, "epoch": epoch, "checkpoint_fingerprint": identity})
                            predictions.append({k: v for k, v in prediction.items() if k not in ("windows", "coverage", "source_token_offsets", "support")})
                        summary = selection_summary(phases["selection"], predictions)
                        path = directory / ("epoch%02d.safetensors" % epoch)
                        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))
                        entry = {**data.pin(path), "epoch": epoch, "state_dict_fingerprint": identity}
                        candidates.append(entry)
                        rank = (-summary["macro_primary"], summary["mean_nll"], epoch)
                        if best is None or rank < best:
                            best, best_epoch, best_fingerprint = rank, epoch, identity
                            best_probes = [next(p for p in predictions if p["endpoint"] == endpoint) for endpoint in ENDPOINTS]
                        history["epochs"].append({"epoch": epoch, "selection": summary, "mean_nll": summary["mean_nll"],
                                                  "state_dict_fingerprint": identity, "checkpoint": entry, **fit})
                        del cache, predictions
                del opt
                before_restore = fingerprint(model.state_dict())
                liveness["encoder_training_before_sha256"] = custody["encoder_initial_sha256"]
                liveness["encoder_training_after_sha256"] = fingerprint(model.cross.transformer().state_dict())
                data.require(liveness["encoder_training_after_sha256"] != liveness["encoder_training_before_sha256"],
                             "Shared encoder unchanged after actual ten-epoch adaptation")
                model.load_state_dict(load_file(str(directory / ("epoch%02d.safetensors" % best_epoch))), strict=True)
                data.require(fingerprint(model.state_dict()) == best_fingerprint, "Selected strict restore fingerprint differs")
                selected_path = directory / "selected.safetensors"
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(selected_path))
                history.update(selected_epoch=best_epoch, selected_fingerprint=best_fingerprint)
                restored_cache = neural.EvalCache(model, tokenizer, best_fingerprint, resources.guard)
                indexed_selection = {r["id"]: r for r in phases["selection"]}
                restored_probes = [restored_cache.predict(indexed_selection[p["id"]], condition="selection") for p in best_probes]
                probe_fields = ("raw_logits", "raw_probabilities", "spans")
                data.require(all(all(old[k] == new[k] for k in probe_fields) for old, new in zip(best_probes, restored_probes)),
                             "Selected restore neural probe disagreement")
                liveness["selected_restore"] = {"strict": True, "selected_epoch": best_epoch,
                                                "before_restore_fingerprint": before_restore, "fingerprint": best_fingerprint,
                                                "expected_fingerprint": best_fingerprint, "restored_fingerprint": best_fingerprint,
                                                "missing_keys": [], "unexpected_keys": [],
                                                "checkpoint_sha256": data.pin(selected_path)["sha256"],
                                                "selected_rows": best_probes,
                                                "restored_rows": [{k: v for k, v in p.items() if k not in ("windows", "coverage", "source_token_offsets", "support")}
                                                                  for p in restored_probes], "max_abs_difference": 0.}
                metadata["candidate_checkpoints"] = candidates
                metadata["checkpoint"] = {**data.pin(selected_path), "state_dict_fingerprint": best_fingerprint,
                                          "selected_epoch": best_epoch, "encoder_prefix": "cross.scorer.pool.encoder.",
                                          "encoder_state_dict_fingerprint": fingerprint(model.cross.transformer().state_dict())}
                liveness["encoder_selected_sha256"] = metadata["checkpoint"]["encoder_state_dict_fingerprint"]
                cache = neural.EvalCache(model, tokenizer, best_fingerprint, resources.guard)
                calibration = []
                with Capture(directory / "calibration.jsonl.gz", header("calibration", "calibration", metadata["checkpoint"]["sha256"])) as stream:
                    for row in phases["calibration"]:
                        prediction = cache.predict(row, condition="calibration")
                        write_row(stream, prediction)
                        calibration.append({k: v for k, v in prediction.items() if k not in ("windows", "coverage", "source_token_offsets", "support")})
                temperatures, evidence = fit_temperatures(phases["calibration"], calibration)
                metadata.update(temperatures=temperatures, temperature_evidence=evidence)
                del calibration
                baseline = Baselines(phases["fit"])
                metadata["baseline"] = {"priors": baseline.priors, "deduplicated_fit_texts": baseline.n,
                                        "idf": baseline.idf, "idf_sha256": data.value_digest(baseline.idf),
                                        "original_cross_checkpoint": cfg["current_authority"]["selected_initial_cross_checkpoint"]}
                donors = donor_plan(phases["dev"])
                captures = {}
                import contextlib
                with contextlib.ExitStack() as stack:
                    for condition in ("dev", "baselines", "question_masked", "state_masked", "question_swapped", "reordered", "candidate_count"):
                        captures[condition] = stack.enter_context(Capture(directory / (condition + ".jsonl.gz"),
                                                                           header("dev", condition, metadata["checkpoint"]["sha256"])))
                    for row in phases["dev"]:
                        temperature = temperatures.get(row["endpoint"], 1.)
                        prediction = cache.predict(row, temperature, "dev")
                        write_row(captures["dev"], prediction)
                        write_row(captures["baselines"], original[row["id"]] if row["endpoint"] in data.INTENTS else baseline.predict(row))
                        if row["endpoint"] in data.INTENTS:
                            continue
                        for condition in ("question_masked", "state_masked", "reordered"):
                            altered = changed(row, condition)
                            # Controls cannot use original extraction targets against changed source offsets.
                            if condition == "state_masked" and altered["serving"]["task"] == "extract":
                                altered = {**altered, "target": {"ratings": []}}
                            control = cache.predict(altered, temperature, condition)
                            write_row(captures[condition], {**control, "original_id": row["id"]})
                        donor = donors.get(row["id"])
                        if donor is None:
                            write_row(captures["question_swapped"], {**prediction, "condition": "question_swapped", "donor_id": None,
                                                                    "runtime_counters": {k: 0 for k in model.counters},
                                                                    "cumulative_counters": dict(model.counters), "cache_key": None,
                                                                    "status": "UNSUPPORTED", "error": "no_distinct_same_paper_endpoint_question",
                                                                    "raw_logits": [], "raw_probabilities": [], "probabilities": [], "ranking": [], "spans": [],
                                                                    "windows": [], "coverage": [], "source_token_offsets": {}, "support": None,
                                                                    "paired": False, "reason": "no_distinct_same_paper_endpoint_question"})
                        else:
                            altered = changed(row, "question_swapped", donor)
                            control = cache.predict(altered, temperature, "question_swapped")
                            write_row(captures["question_swapped"], {**control, "original_id": row["id"], "donor_id": donor["id"], "paired": True})
                        ids = [b["id"] for b in row["blocks"]] if row["blocks"] else [c["id"] for c in row["serving"]["candidates"]]
                        ordered = sorted(ids, key=lambda cid: (__import__("hashlib").sha256(
                            ("qnative-cross-v1|7|" + row["component_id"] + "|" + cid).encode()).hexdigest(), cid))
                        sizes = sorted({min(k, len(ids)) for k in (2, 4, 8, 16, 32, 64, 128, 255, 512, 1000, len(ids)) if ids})
                        for size in sizes:
                            subset = ordered[:size]
                            altered = changed(row, "candidate_count", subset=subset)
                            if altered["serving"]["task"] == "extract":
                                altered = {**altered, "target": {"ratings": []}}
                            control = cache.predict(altered, temperature, "candidate_count")
                            write_row(captures["candidate_count"], {**control, "kind": "candidate_count",
                                                                  "subset_ids": subset, "full_candidate_count": len(ids),
                                                                  "K": size, "input_only_prefix": True})
                # Three rounds over two actual distinct questions, fresh bounded exact cache.
                native_questions = {(r["paper_id"], r["question_ordinal"]): r["serving"]["question"]
                                    for r in phases["dev"] if r["endpoint"] == "qasper.extractive_answer"}
                questions = {}
                for row in sorted((r for r in phases["dev"] if r["blocks"]), key=lambda r: (r["paper_id"], r["id"])):
                    questions.setdefault(row["paper_id"], [])
                    if all(r["question_ordinal"] != row["question_ordinal"] and
                           native_questions[(r["paper_id"], r["question_ordinal"])] !=
                           native_questions[(row["paper_id"], row["question_ordinal"])] for r in questions[row["paper_id"]]):
                        questions[row["paper_id"]].append(row)
                first_paper = min(questions)
                repeat_rows = questions[first_paper][:2]
                repeat_cache = neural.EvalCache(model, tokenizer, best_fingerprint, resources.guard)
                with (directory / "repeated_state.jsonl").open("xb") as stream:
                    write_row(stream, header("dev", "repeat", metadata["checkpoint"]["sha256"]))
                    if len(repeat_rows) < 2:
                        write_row(stream, {"kind": "missing_diagnostic", "reason": "second_distinct_question_unavailable"})
                    for round_index in range(3):
                        for row in repeat_rows:
                            prediction = repeat_cache.predict(row, temperatures.get(row["endpoint"], 1.), "repeat")
                            write_row(stream, {**prediction, "round": round_index, "question_ids": [r["id"] for r in repeat_rows]})
                resources.guard()
                metadata["status"] = "train_complete"
                liveness["counters"] = dict(model.counters)
        write_json(directory / "history.json", history)
        write_json(directory / "liveness.json", liveness)
        write_json(directory / "runtime.json", {"peak_allocated_gpu_bytes": resources.peak,
                                              "resource_events": resources.events, "counters": liveness.get("counters", {})})
        metadata["artifact_hashes"] = {p.name: data.pin(p) for p in sorted(directory.iterdir()) if p.is_file()}
        write_json(directory / "metadata.json", metadata)
    except BaseException as exc:
        failure = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "protocol_sha256": metadata["protocol_sha256"],
                   "artifacts": {p.name: data.pin(p) for p in directory.iterdir() if p.is_file()},
                   "history": history, "liveness": liveness, "peak_allocated_gpu_bytes": resources.peak}
        active_model = locals().get("model")
        context = getattr(active_model, "active_decision", None)
        if context:
            failure["prediction_error"] = {**context, "kind": "prediction", "status": "ERROR", "error": str(exc),
                                            "condition": getattr(active_model, "active_condition", mode),
                                            "raw_logits": [], "raw_probabilities": [], "probabilities": [],
                                            "ranking": [], "spans": [], "temperature": None}
        write_json(directory / "failure.json", failure)
        metadata.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        for filename, value in (("history.json", history), ("liveness.json", liveness),
                                ("runtime.json", {"peak_allocated_gpu_bytes": resources.peak, "resource_events": resources.events})):
            if not (directory / filename).exists():
                write_json(directory / filename, value)
        metadata["artifact_hashes"] = {p.name: data.pin(p) for p in directory.iterdir() if p.is_file()}
        if not (directory / "metadata.json").exists():
            write_json(directory / "metadata.json", metadata)
        raise
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args(argv)
    data.require(not (args.out and args.root), "Use one output override only")
    cfg = json.loads(data.AMENDMENT.read_text())
    print(json.dumps({"output": str(run(args.out or args.root or cfg["output_root"], args.mode))}))


if __name__ == "__main__":
    main()
