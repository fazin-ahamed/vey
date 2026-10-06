"""One strict-initialized cross encoder with bounded joint source-window readouts."""
from __future__ import annotations

import math
from collections import OrderedDict

try:
    from .neutral_qasper_cross_data import source_windows, value_digest, retrieval_target, bio_supervision, require
except ImportError:
    from neutral_qasper_cross_data import source_windows, value_digest, retrieval_target, bio_supervision, require


def probabilities(logits, temperature=1.):
    require(temperature > 0 and all(math.isfinite(x) for x in logits), "Invalid categorical logits")
    if not logits:
        return []
    maximum = max(logits)
    masses = [math.exp((v - maximum) / temperature) for v in logits]
    return [v / math.fsum(masses) for v in masses]


def decode_bio(blocks, windows):
    text = {b["id"]: b["text"] for b in blocks}
    intervals = {}
    for window in windows:
        active = []
        def flush():
            if active:
                start, end = active[0][0], active[-1][1]
                if end > start:
                    intervals.setdefault(window["block_id"], []).append((start, end, list(active)))
                active.clear()
        for offset, logits in zip(window["offsets"], window["bio_logits"]):
            if offset[1] <= offset[0]:
                continue
            probs = probabilities(logits)
            tag = max(range(3), key=lambda i: (probs[i], -i))
            if tag == 0 or tag == 1:
                flush()
            if tag and offset[1] > offset[0]:
                active.append((offset[0], offset[1], probs[tag]))
        flush()
    output = []
    for block in blocks:
        unique = {}
        for start, end, tokens in intervals.get(block["id"], []):
            scores = unique.setdefault((start, end), {})
            for a, b, confidence in tokens:
                scores[(a, b)] = max(scores.get((a, b), 0.), confidence)
        merged = []
        for (start, end), scores in sorted(unique.items()):
            if merged and start < merged[-1][1]:
                a, b, previous = merged[-1]
                for offset, confidence in scores.items():
                    previous[offset] = max(previous.get(offset, 0.), confidence)
                merged[-1] = (a, max(b, end), previous)
            else:
                merged.append((start, end, dict(scores)))
        for start, end, scores in merged:
            output.append({"block_id": block["id"], "start": start, "end": end,
                           "text": text[block["id"]][start:end], "confidence": math.fsum(scores.values()) / len(scores)})
    return output


def load_tokenizer():
    import os
    from transformers import AutoTokenizer
    try:
        from . import native_field_data
    except ImportError:
        import native_field_data
    cfg = native_field_data.protocol()
    spec = cfg["model"]
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        cache = os.path.join(os.environ.get("HF_HOME", "/home/fazinahamed/Documents/vey-data/decisionmix/hf-home"), "hub")
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"], cache_dir=cache,
                                            local_files_only=True, use_fast=True, trust_remote_code=False)
    require(tokenizer.is_fast and tokenizer.pad_token_id is not None, "Fast padded tokenizer required")
    return tokenizer


def build_model(cfg, catalogues):
    import modal
    require(not modal.is_local(), "Actual neural work is restricted to isolated Modal jobs")
    import torch
    from torch import nn
    from safetensors.torch import load_file
    try:
        from .native_arch_model import load_arch_encoder, CrossArchitecture
        from .native_field_model import masked_mean, fingerprint
        from . import native_field_data
    except ImportError:
        from native_arch_model import load_arch_encoder, CrossArchitecture
        from native_field_model import masked_mean, fingerprint
        import native_field_data
    require(torch.cuda.is_available(), "Actual neural work requires CUDA")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    tokenizer, encoder, custody = load_arch_encoder(native_field_data.protocol(), catalogues, "cuda")
    cross = CrossArchitecture(encoder).to(device="cuda", dtype=torch.float32)
    entry = cfg["current_authority"]["selected_initial_cross_checkpoint"]
    try:
        from .neutral_qasper_cross_data import pin
    except ImportError:
        from neutral_qasper_cross_data import pin
    require(pin(entry["path"]) == entry, "Initial cross checkpoint changed")
    state = load_file(entry["path"], device="cpu")
    cross.load_state_dict(state, strict=True)
    initial_cross = fingerprint(cross.state_dict())
    import json
    metadata_entry = cfg["current_authority"]["selected_initial_cross_metadata"]
    require(pin(metadata_entry["path"]) == metadata_entry, "Initial cross metadata changed")
    metadata = json.loads(open(cfg["current_authority"]["selected_initial_cross_metadata"]["path"]).read())
    require(metadata["checkpoint"]["state_dict_fingerprint"] == initial_cross
            and metadata["checkpoint"]["selected_epoch"] == 10, "Cross selected identity differs")
    del state
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    class SharedCross(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = cross
            self.bool_head = nn.Linear(int(encoder.config.hidden_size), 2)
            self.bio_head = nn.Linear(int(encoder.config.hidden_size), 3)
            self.counters = {"encoder_forward_calls": 0, "encoded_sequences": 0,
                             "encoded_windows": 0, "state_containing_sequences": 0, "cache_hits": 0}

        def heads(self, ids, mask):
            hidden = self.cross.transformer()(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = masked_mean(hidden, mask)
            return self.cross.scorer.head(pooled).squeeze(-1), self.bool_head(pooled), self.bio_head(hidden)

        def encode(self, ids, mask, question_path=True):
            from torch.utils.checkpoint import checkpoint
            if self.training and torch.is_grad_enabled():
                outputs = checkpoint(self.heads, ids, mask, use_reentrant=False, preserve_rng_state=True)
            else:
                outputs = self.heads(ids, mask)
            self.counters["encoder_forward_calls"] += 1
            self.counters["encoded_sequences"] += len(ids)
            self.counters["state_containing_sequences"] += len(ids)
            self.counters["encoded_windows"] += len(ids) if question_path else 0
            return outputs

        def forward_row(self, row, guard=lambda: None):
            view = row["serving"]
            self.active_decision = {k: row[k] for k in ("id", "endpoint", "component_id", "phase")}
            self.active_decision["input_sha256"] = value_digest(view)
            self.active_decision["candidate_ids"] = ([b["id"] for b in row["blocks"]] if view["task"] == "retrieval"
                                                      else [c["id"] for c in view["candidates"]])
            if row["endpoint"] in ("banking77.intent", "massive.intent"):
                texts = [view["state"] + self.cross.separator + c["text"] for c in view["candidates"]]
                unique = list(dict.fromkeys(texts))
                score_parts = []
                for start in range(0, len(unique), 8):
                    guard()
                    batch = tokenizer(unique[start:start + 8], padding=True, truncation=True,
                                      max_length=512, return_tensors="pt")
                    relevance, _, _ = self.encode(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), False)
                    score_parts.append(relevance)
                values = torch.cat(score_parts)
                positions = {v: i for i, v in enumerate(unique)}
                gather = torch.tensor([positions[t] for t in texts], device="cuda")
                return {"logits": values.index_select(0, gather), "windows": [], "coverage": [], "support": None}
            windows, census = source_windows(tokenizer, view["question"], row["blocks"])
            require(bool(windows), "No source blocks for joint encoding")
            windows.sort(key=lambda w: (w["block_id"], w["window_index"]))
            relevance_parts, bool_parts, bio_parts = [], [], []
            for start in range(0, len(windows), 8):
                guard()
                chunk = windows[start:start + 8]
                batch = tokenizer.pad([{k: w[k] for k in ("input_ids", "attention_mask")} for w in chunk],
                                      padding=True, return_tensors="pt")
                relevance, boolean, bio = self.encode(batch["input_ids"].cuda(), batch["attention_mask"].cuda())
                relevance_parts.append(relevance)
                bool_parts.append(boolean)
                bio_parts.extend(bio[i, w["source_positions"]] for i, w in enumerate(chunk))
            relevance = torch.cat(relevance_parts)
            boolean = torch.cat(bool_parts)
            block_scores = torch.stack([relevance[[i for i, w in enumerate(windows) if w["block_id"] == b["id"]]].max()
                                        for b in row["blocks"]])
            logits = (relevance.softmax(0).unsqueeze(1) * boolean).sum(0) if view["task"] == "bool" else block_scores
            support = bio_supervision(row["target"], windows, census) if view["task"] == "extract" else None
            return {"logits": logits, "windows": windows, "coverage": census, "support": support,
                    "relevance": relevance, "bool_logits": boolean, "bio_logits": bio_parts}

    model = SharedCross().to(device="cuda", dtype=torch.float32)
    custody["stock_encoder_sha256"] = custody["encoder_initial_sha256"]
    custody["encoder_initial_sha256"] = fingerprint(model.cross.transformer().state_dict())
    custody.update(initial_cross_fingerprint=initial_cross, initial_model_fingerprint=fingerprint(model.state_dict()))
    return tokenizer, model, custody


def decision_loss(row, output):
    import torch
    task = row["serving"]["task"]
    if task == "extract":
        support = output["support"]
        if not support["supervised_source_tokens"]:
            return None, 0
        # Each unique source token contributes once despite overlap windows.
        multiplicity = {}
        for w in output["windows"]:
            for i in w["source_token_indices"]:
                key = (w["block_id"], i)
                multiplicity[key] = multiplicity.get(key, 0) + 1
        numerator = output["bio_logits"][0].sum() * 0
        denominator = 0.
        for window, logits in zip(output["windows"], output["bio_logits"]):
            for j, index in enumerate(window["source_token_indices"]):
                distribution = support["token_distributions"][window["block_id"]][index]
                if distribution is not None:
                    weight = torch.tensor([distribution[0], 10 * distribution[1], 10 * distribution[2]], device=logits.device)
                    scale = 1 / multiplicity[(window["block_id"], index)]
                    numerator = numerator - (weight * logits[j].log_softmax(-1)).sum() * scale
                    denominator += sum([distribution[0], 10 * distribution[1], 10 * distribution[2]]) * scale
        return numerator / denominator, denominator
    ids = [b["id"] for b in row["blocks"]] if task == "retrieval" else [c["id"] for c in row["serving"]["candidates"]]
    distribution = retrieval_target(row["target"], ids) if task == "retrieval" else row["target"].get("distribution")
    if distribution is None:
        return None, 0
    target = torch.tensor(distribution, device=output["logits"].device)
    return -(target * output["logits"].log_softmax(0)).sum(), 1


class EvalCache:
    """Bounded exact-input/weight memoization, never question-independent state reuse."""
    def __init__(self, model, tokenizer, identity, guard=lambda: None, capacity=2):
        self.model, self.tokenizer, self.identity, self.guard = model, tokenizer, identity, guard
        self.capacity = capacity
        self.entries = OrderedDict()

    def predict(self, row, temperature=1., condition="real"):
        import torch
        self.model.active_condition = condition
        if row["endpoint"] not in ("banking77.intent", "massive.intent") and not row["blocks"]:
            return {"kind": "prediction", "id": row["id"], "endpoint": row["endpoint"],
                    "component_id": row["component_id"], "phase": row["phase"], "condition": condition,
                    "input_sha256": value_digest(row["serving"]), "status": "UNSUPPORTED", "error": "empty_source_catalogue",
                    "candidate_ids": [c["id"] for c in row["serving"]["candidates"]], "raw_logits": [],
                    "raw_probabilities": [], "probabilities": [], "temperature": temperature, "ranking": [], "spans": [],
                    "windows": [], "coverage": [], "source_token_offsets": {}, "support": None,
                    "cache_key": None, "weight_identity": self.identity,
                    "runtime_counters": {k: 0 for k in self.model.counters}, "cumulative_counters": dict(self.model.counters)}
        key = value_digest([self.identity, row["serving"], row["blocks"]])
        before = dict(self.model.counters)
        if key in self.entries:
            values = self.entries.pop(key)
            self.entries[key] = values
            self.model.counters["cache_hits"] += 1
        else:
            self.model.eval()
            with torch.no_grad():
                output = self.model.forward_row(row, self.guard)
            task = row["serving"]["task"]
            windows = []
            for i, w in enumerate(output["windows"]):
                windows.append({**w,
                                "bio_logits": output["bio_logits"][i].cpu().tolist() if task == "extract" else None,
                                "relevance_logit": float(output["relevance"][i]),
                                "bool_logits": output["bool_logits"][i].cpu().tolist()})
            values = {"raw_logits": output["logits"].cpu().tolist() if task != "extract" else [],
                      "windows": windows, "coverage": output["coverage"], "support": output["support"]}
            self.entries[key] = values
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        task = row["serving"]["task"]
        ids = ([b["id"] for b in row["blocks"]] if task == "retrieval" else
               [c["id"] for c in row["serving"]["candidates"]] if task != "extract" else [])
        raw = probabilities(values["raw_logits"])
        calibrated = probabilities(values["raw_logits"], temperature)
        ranking = sorted(ids, key=lambda cid: (-calibrated[ids.index(cid)], cid))
        spans = decode_bio(row.get("original_blocks", row["blocks"]), values["windows"]) if task == "extract" else []
        after = dict(self.model.counters)
        return {"kind": "prediction", "id": row["id"], "endpoint": row["endpoint"],
                "component_id": row["component_id"], "phase": row["phase"], "condition": condition,
                "input_sha256": value_digest(row["serving"]), "status": "OK", "error": None,
                "candidate_ids": ids, "raw_logits": values["raw_logits"], "raw_probabilities": raw,
                "probabilities": calibrated, "temperature": temperature, "ranking": ranking, "spans": spans,
                "windows": values["windows"], "coverage": values["coverage"],
                "source_token_offsets": {b["block_id"]: b["source_token_offsets"] for b in values["coverage"]},
                "support": values["support"], "cache_key": key, "weight_identity": self.identity,
                "runtime_counters": {k: after[k] - before[k] for k in after}, "cumulative_counters": after}
