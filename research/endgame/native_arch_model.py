"""NATIVE-2 cross / dual / Evidence-Pages architectures over the shared native field.

Only the two eligible English intent endpoints are trained. Every architecture
consumes canonical serving state text and native candidate descriptions; no
free-form question, alias or target text ever reaches the encoder.
"""
from __future__ import annotations

import hashlib
import importlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

try:
    from .native_field_model import (
        fingerprint, load_stock, masked_mean, native_readout, probabilities, sha256_file,
    )
except ImportError:
    from native_field_model import (
        fingerprint, load_stock, masked_mean, native_readout, probabilities, sha256_file,
    )


CANONICAL_ROOT = Path("/home/fazinahamed/Documents/vey")
CANONICAL_DEPENDENCIES = {
    "vey_u/semantic/models.py": "421f8d20aab9f1b545d18ad74eee6cf02c00fd13f0a7feac841e848f366dbc94",
    "vey_u/evidence/pages.py": "1287a2c3b62ec3d8cbf969da8f7707f6827d02512633d85d5af531e271140b96",
}
ARMS = ("cross", "dual", "pages")
ENCODER_PREFIXES = {"cross": "scorer.pool.encoder.", "dual": "scorer.pool.encoder.",
                    "pages": "encoder."}
SEPARATOR = "\n"
HEAD_HIDDEN = 128
EMBED_DIM = 256
CANDIDATE_COUNT_GRID = (2, 4, 8, 16, 32)
REPEATED_STATE_REPEATS = 3
_CANONICAL = None


def import_canonical(root: str | Path = CANONICAL_ROOT):
    """Pin and import the canonical cross/dual/pages components before any use."""
    root = Path(root).resolve()
    files = {}
    for relative, pinned in sorted(CANONICAL_DEPENDENCIES.items()):
        digest = sha256_file(root / relative)
        if digest != pinned:
            raise RuntimeError("canonical dependency hash differs: " + relative)
        files[relative] = digest
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    semantic = importlib.import_module("vey_u.semantic.models")
    pages = importlib.import_module("vey_u.evidence.pages")
    for module, relative in ((semantic, "vey_u/semantic/models.py"),
                             (pages, "vey_u/evidence/pages.py")):
        if Path(module.__file__).resolve() != (root / relative).resolve():
            raise RuntimeError("canonical module resolved elsewhere: " + relative)
    return semantic, pages, {"root": str(root), "files": files}


def canonical_modules():
    global _CANONICAL
    if _CANONICAL is None:
        _CANONICAL = import_canonical()
    return _CANONICAL


def load_arch_encoder(config, catalogues, device):
    """Reuse the NATIVE-1 custody loader, then detach its pinned encoder."""
    tokenizer, native, custody = load_stock(config, catalogues, "full", device)
    encoder = native.pool.encoder
    del native
    encoder.requires_grad_(True)
    checkpointing = {"use_reentrant": False, "preserve_rng_state": True}
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing)
    if not encoder.is_gradient_checkpointing:
        raise RuntimeError("NATIVE-2 activation checkpointing did not enable")
    custody["activation_checkpointing"] = checkpointing
    return tokenizer, encoder, custody


def tokenize(tokenizer, texts, max_tokens, device):
    if not texts:
        raise ValueError("cannot tokenize an empty text batch")
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_tokens,
                      return_tensors="pt")
    return batch["input_ids"].to(device), batch["attention_mask"].to(device)


class _Counters:
    def reset_counters(self):
        self.state_encode_calls = 0
        self.candidate_encode_calls = 0
        self.encoder_forward_calls = 0
        self.encoded_states = 0
        self.encoded_candidates = 0

    def counters(self):
        return {"state_encode_calls": self.state_encode_calls,
                "candidate_encode_calls": self.candidate_encode_calls,
                "encoder_forward_calls": self.encoder_forward_calls,
                "encoded_states": self.encoded_states,
                "encoded_candidates": self.encoded_candidates}


class CrossArchitecture(nn.Module, _Counters):
    """Reread state+candidate jointly for every catalogue candidate."""

    arm = "cross"

    def __init__(self, encoder, hidden: int = HEAD_HIDDEN, separator: str = SEPARATOR):
        super().__init__()
        semantic, _, _ = canonical_modules()
        self.scorer = semantic.CrossEncoderScorer(encoder, hidden=hidden)
        self.separator = separator
        self.reset_counters()

    def transformer(self):
        return self.scorer.pool.encoder

    def score(self, tokenizer, records, max_tokens):
        device = next(self.parameters()).device
        output = [dict() for _ in records]
        for index, record in enumerate(records):
            state_bytes = record["state"].encode("utf-8")
            for decision in record["decisions"]:
                keys = [(state_bytes, decision["candidates"][cid].encode("utf-8"))
                        for cid in decision["candidate_ids"]]
                unique = list(dict.fromkeys(keys))
                position = {key: slot for slot, key in enumerate(unique)}
                texts = [record["state"] + self.separator
                         + key[1].decode("utf-8") for key in unique]
                input_ids, attention_mask = tokenize(tokenizer, texts, max_tokens, device)
                values = self.scorer(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))[0]
                gather = torch.tensor([position[key] for key in keys], dtype=torch.long,
                                      device=values.device)
                output[index][decision["endpoint"]] = values.index_select(0, gather)
                self.encoder_forward_calls += 1
                self.state_encode_calls += 1
                self.candidate_encode_calls += 1
                self.encoded_states += len(unique)
                self.encoded_candidates += len(unique)
        return output


class DualArchitecture(nn.Module, _Counters):
    """One pooled state embedding and one pooled candidate embedding, scaled dot product."""

    arm = "dual"

    def __init__(self, encoder, embed_dim: int = EMBED_DIM):
        super().__init__()
        semantic, _, _ = canonical_modules()
        self.scorer = semantic.DualEncoderScorer(encoder, embed_dim=embed_dim)
        self.reset_counters()

    def transformer(self):
        return self.scorer.pool.encoder

    def _encode_states(self, tokenizer, states, max_tokens):
        device = next(self.parameters()).device
        input_ids, attention_mask = tokenize(tokenizer, states, max_tokens, device)
        self.encoder_forward_calls += 1
        self.state_encode_calls += 1
        self.encoded_states += len(states)
        return self.scorer.encode_state(input_ids, attention_mask)

    def _encode_candidates(self, tokenizer, texts, max_tokens):
        device = next(self.parameters()).device
        input_ids, attention_mask = tokenize(tokenizer, texts, max_tokens, device)
        self.encoder_forward_calls += 1
        self.candidate_encode_calls += 1
        self.encoded_candidates += len(texts)
        return self.scorer.encode_candidates(input_ids.unsqueeze(0),
                                             attention_mask.unsqueeze(0))[0]

    def score(self, tokenizer, records, max_tokens):
        states = [record["state"] for record in records]
        unique = list(dict.fromkeys(states))
        position = {state: index for index, state in enumerate(unique)}
        state_embedding = self._encode_states(tokenizer, unique, max_tokens)
        gather = torch.tensor([position[state] for state in states], dtype=torch.long,
                              device=state_embedding.device)
        state_embedding = state_embedding.index_select(0, gather)
        groups = defaultdict(list)
        for index, record in enumerate(records):
            for decision in record["decisions"]:
                groups[decision["endpoint"]].append((index, decision))
        output = [dict() for _ in records]
        scale = self.scorer.log_scale.exp().clamp(max=100.0)
        for endpoint, items in groups.items():
            first = items[0][1]
            texts = [first["candidates"][cid] for cid in first["candidate_ids"]]
            candidate = self._encode_candidates(tokenizer, texts, max_tokens)
            rows = state_embedding.index_select(
                0, torch.tensor([index for index, _ in items], dtype=torch.long,
                                device=state_embedding.device))
            logits = scale * torch.matmul(rows, candidate.transpose(0, 1))
            for offset, (index, _decision) in enumerate(items):
                output[index][endpoint] = logits[offset]
        return output


class PagesArchitecture(nn.Module, _Counters):
    """Page-pool the state once, retrieve top-k pages by query+candidate, late-interact."""

    arm = "pages"

    def __init__(self, encoder, page_tokens: int = 64, top_k: int = 4,
                 hidden: int = HEAD_HIDDEN):
        super().__init__()
        _, pages, _ = canonical_modules()
        self.encoder = encoder
        dim = int(encoder.config.hidden_size)
        self.cand_proj = nn.Linear(dim, dim)
        self.query_head = nn.Linear(dim, dim)
        self.decision = pages.EvidencePageDecision(dim, top_k=top_k, hidden=hidden)
        self.page_tokens = int(page_tokens)
        self.top_k = int(top_k)
        self.reset_counters()

    def transformer(self):
        return self.encoder

    def _state_representation(self, tokenizer, states, max_tokens):
        _, pages, _ = canonical_modules()
        device = next(self.parameters()).device
        input_ids, attention_mask = tokenize(tokenizer, states, max_tokens, device)
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        self.encoder_forward_calls += 1
        self.state_encode_calls += 1
        self.encoded_states += len(states)
        cached = pages.mean_page_pool(hidden, attention_mask, self.page_tokens)
        query = self.query_head(masked_mean(hidden, attention_mask))
        return cached, query

    def _candidate_embedding(self, tokenizer, texts, max_tokens):
        device = next(self.parameters()).device
        input_ids, attention_mask = tokenize(tokenizer, texts, max_tokens, device)
        hidden = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        self.encoder_forward_calls += 1
        self.candidate_encode_calls += 1
        self.encoded_candidates += len(texts)
        return self.cand_proj(masked_mean(hidden, attention_mask))

    def score(self, tokenizer, records, max_tokens):
        _, pages, _ = canonical_modules()
        states = [record["state"] for record in records]
        unique = list(dict.fromkeys(states))
        position = {state: index for index, state in enumerate(unique)}
        cached, query = self._state_representation(tokenizer, unique, max_tokens)
        groups = defaultdict(list)
        for index, record in enumerate(records):
            for decision in record["decisions"]:
                groups[decision["endpoint"]].append((index, decision))
        output = [dict() for _ in records]
        for endpoint, items in groups.items():
            first = items[0][1]
            texts = [first["candidates"][cid] for cid in first["candidate_ids"]]
            candidate = self._candidate_embedding(tokenizer, texts, max_tokens)
            for index, decision in items:
                row = position[records[index]["state"]]
                single = pages.CachedPages(cached.pages[row:row + 1],
                                           cached.page_mask[row:row + 1],
                                           cached.token_ranges)
                query_row = query[row:row + 1]
                logits = []
                for slot in range(candidate.shape[0]):
                    value, _, _ = self.decision(single, query_row, candidate[slot:slot + 1])
                    logits.append(value[0])
                output[index][endpoint] = torch.stack(logits)
        return output


def build_arch(encoder, catalogues, config, arm):
    if arm not in ARMS:
        raise ValueError("unknown architecture: " + str(arm))
    training = config["training"]
    if arm == "cross":
        return CrossArchitecture(encoder)
    if arm == "dual":
        return DualArchitecture(encoder)
    return PagesArchitecture(encoder, page_tokens=int(training["page_tokens"]),
                             top_k=int(training["top_k"]))


def nested_subsets(candidate_ids, component_id, seed: int = 7, grid=CANDIDATE_COUNT_GRID):
    """Input-only nested prefixes of one component-seeded whole-catalogue shuffle."""
    count = len(candidate_ids)
    sizes = sorted({size for size in (*grid, count) if 2 <= size <= count})
    if not sizes:
        return {}
    rng = random.Random(int.from_bytes(
        hashlib.sha256(("native-arch-candidate-count|%d|%s" % (seed, component_id)).encode()).digest()[:8],
        "big"))
    ordered = list(candidate_ids)
    rng.shuffle(ordered)
    return {size: list(ordered[:size]) for size in sizes}


def gold_rank(subset_ids, masses, gold):
    by_id = dict(zip(subset_ids, masses))
    if gold not in by_id:
        return None
    reference = by_id[gold]
    rank = 1
    for candidate, mass in by_id.items():
        if candidate == gold:
            continue
        if mass > reference or (mass == reference and candidate < gold):
            rank += 1
    return rank


class ArchEvalCache:
    """Eval-only memoization; cross caches joint logits, dual/pages reuse state features."""

    def __init__(self, model, tokenizer, temperatures, max_tokens, guard=None):
        self.model = model
        self.tokenizer = tokenizer
        self.temperatures = dict(temperatures)
        self.max_tokens = int(max_tokens)
        self.guard = guard
        model.eval()
        self.identity = fingerprint(model.state_dict())
        self.pair_logits = {}
        self.state_entries = {}
        self.candidate_entries = {}
        self.misses = 0
        self.reset_counters()

    def reset_counters(self):
        self.state_encode_calls = 0
        self.candidate_encode_calls = 0
        self.encoder_forward_calls = 0
        self.encoded_states = 0
        self.encoded_candidates = 0

    def counters(self):
        return {"state_encode_calls": self.state_encode_calls,
                "candidate_encode_calls": self.candidate_encode_calls,
                "encoder_forward_calls": self.encoder_forward_calls,
                "encoded_states": self.encoded_states,
                "encoded_candidates": self.encoded_candidates,
                "cache_misses": self.misses}

    def _guard(self):
        if self.guard is not None:
            self.guard()

    @property
    def device(self):
        return next(self.model.parameters()).device

    def field_logits(self, state, decision):
        arm = self.model.arm
        if arm == "cross":
            return self._cross(state, decision)
        if arm == "dual":
            return self._dual(state, decision)
        return self._pages(state, decision)

    def _cross(self, state, decision):
        state_bytes = state.encode("utf-8")
        texts = [state + SEPARATOR + decision["candidates"][cid]
                 for cid in decision["candidate_ids"]]
        keys = [(state_bytes, text.encode("utf-8")) for text in texts]
        missing = [index for index, key in enumerate(keys) if key not in self.pair_logits]
        if missing:
            self._guard()
            self.misses += 1
            input_ids, attention_mask = tokenize(self.tokenizer, [texts[i] for i in missing],
                                                 self.max_tokens, self.device)
            with torch.no_grad():
                values = self.model.scorer(input_ids.unsqueeze(0),
                                           attention_mask.unsqueeze(0))[0]
            values = values.detach().cpu().tolist()
            for index, value in zip(missing, values):
                self.pair_logits[keys[index]] = value
            self.encoder_forward_calls += 1
            self.state_encode_calls += 1
            self.candidate_encode_calls += 1
            self.encoded_states += len(missing)
            self.encoded_candidates += len(missing)
        return [self.pair_logits[key] for key in keys]

    def _candidate_keys(self, decision):
        return [(decision["endpoint"], cid,
                 decision["candidates"][cid].encode("utf-8"))
                for cid in decision["candidate_ids"]]

    def _missing_candidates(self, keys):
        return [key for key in keys if key not in self.candidate_entries]

    def _dual_state(self, state):
        state_bytes = state.encode("utf-8")
        if state_bytes not in self.state_entries:
            self._guard()
            self.misses += 1
            input_ids, attention_mask = tokenize(self.tokenizer, [state], self.max_tokens,
                                                 self.device)
            with torch.no_grad():
                embedding = self.model.scorer.encode_state(input_ids, attention_mask)[0]
            self.state_entries[state_bytes] = embedding.detach().cpu()
            self.encoder_forward_calls += 1
            self.state_encode_calls += 1
            self.encoded_states += 1
        return self.state_entries[state_bytes]

    def _dual_candidates(self, decision):
        keys = self._candidate_keys(decision)
        missing = self._missing_candidates(keys)
        if missing:
            self._guard()
            self.misses += 1
            texts = [key[2].decode("utf-8") for key in missing]
            input_ids, attention_mask = tokenize(self.tokenizer, texts, self.max_tokens,
                                                 self.device)
            with torch.no_grad():
                embedding = self.model.scorer.encode_candidates(
                    input_ids.unsqueeze(0), attention_mask.unsqueeze(0))[0]
            embedding = embedding.detach().cpu()
            for offset, key in enumerate(missing):
                self.candidate_entries[key] = embedding[offset]
            self.encoder_forward_calls += 1
            self.candidate_encode_calls += 1
            self.encoded_candidates += len(missing)
        return torch.stack([self.candidate_entries[key] for key in keys])

    def _dual(self, state, decision):
        state_embedding = self._dual_state(state)
        candidate = self._dual_candidates(decision)
        scale = self.model.scorer.log_scale.exp().clamp(max=100.0).detach().cpu()
        return (scale * torch.matmul(state_embedding, candidate.transpose(0, 1))).tolist()

    def _pages_state(self, state):
        state_bytes = state.encode("utf-8")
        if state_bytes not in self.state_entries:
            self._guard()
            self.misses += 1
            _, pages, _ = canonical_modules()
            input_ids, attention_mask = tokenize(self.tokenizer, [state], self.max_tokens,
                                                 self.device)
            with torch.no_grad():
                hidden = self.model.encoder(input_ids=input_ids,
                                            attention_mask=attention_mask).last_hidden_state
                cached = pages.mean_page_pool(hidden, attention_mask, self.model.page_tokens)
                query = self.model.query_head(masked_mean(hidden, attention_mask))
            self.state_entries[state_bytes] = (cached.pages[0].detach().cpu(),
                                               cached.page_mask[0].detach().cpu(),
                                               query[0].detach().cpu())
            self.encoder_forward_calls += 1
            self.state_encode_calls += 1
            self.encoded_states += 1
        return self.state_entries[state_bytes]

    def _pages_candidates(self, decision):
        keys = self._candidate_keys(decision)
        missing = self._missing_candidates(keys)
        if missing:
            self._guard()
            self.misses += 1
            texts = [key[2].decode("utf-8") for key in missing]
            input_ids, attention_mask = tokenize(self.tokenizer, texts, self.max_tokens,
                                                 self.device)
            with torch.no_grad():
                hidden = self.model.encoder(input_ids=input_ids,
                                            attention_mask=attention_mask).last_hidden_state
                embedding = self.model.cand_proj(masked_mean(hidden, attention_mask))
            embedding = embedding.detach().cpu()
            for offset, key in enumerate(missing):
                self.candidate_entries[key] = embedding[offset]
            self.encoder_forward_calls += 1
            self.candidate_encode_calls += 1
            self.encoded_candidates += len(missing)
        return torch.stack([self.candidate_entries[key] for key in keys])

    def _pages(self, state, decision):
        _, pages, _ = canonical_modules()
        page_tensor, page_mask, query = self._pages_state(state)
        candidate = self._pages_candidates(decision)
        device = self.device
        cached = pages.CachedPages(page_tensor.unsqueeze(0).to(device),
                                   page_mask.unsqueeze(0).to(device), ())
        query_row = query.unsqueeze(0).to(device)
        candidate = candidate.to(device)
        logits = []
        with torch.no_grad():
            for slot in range(candidate.shape[0]):
                value, _, _ = self.model.decision(cached, query_row, candidate[slot:slot + 1])
                logits.append(float(value[0]))
        return logits

    def read(self, state, decision, *, reverse=False):
        logits = self.field_logits(state, decision)
        identifiers = list(decision["candidate_ids"])
        if reverse:
            identifiers = identifiers[::-1]
            logits = logits[::-1]
        raw = probabilities(logits)
        calibrated = probabilities(logits, self.temperatures[decision["endpoint"]])
        output = {"candidate_ids": identifiers, "logits": list(logits),
                  "raw_probs": raw, "probs": calibrated}
        for prefix, key in (("", "probs"), ("raw_", "raw_probs")):
            result = native_readout(identifiers, output[key], decision["task"],
                                    list(decision["candidate_ids"]), decision.get("level_max"))
            output.update({prefix + name: value for name, value in result.items()})
        return output
