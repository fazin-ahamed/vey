"""Pinned stock state encoder and canonical native categorical/ordinal fields."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch
from torch import nn


EMPTY_EVIDENCE = "[EMPTY EVIDENCE]"
EXPECTED_TASK_KEYS = frozenset({
    "deberta.embeddings.word_embeddings._weight",
    "lm_predictions.lm_head.LayerNorm.bias", "lm_predictions.lm_head.LayerNorm.weight",
    "lm_predictions.lm_head.bias", "lm_predictions.lm_head.dense.bias",
    "lm_predictions.lm_head.dense.weight", "mask_predictions.LayerNorm.bias",
    "mask_predictions.LayerNorm.weight", "mask_predictions.classifier.bias",
    "mask_predictions.classifier.weight", "mask_predictions.dense.bias",
    "mask_predictions.dense.weight",
})


def sha256_file(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(state_dict) -> str:
    """Hash actual tensor bytes with names/shapes/dtypes, not a pickle encoding."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        header = canonical_bytes([name, str(value.dtype), list(value.shape)])
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Same pooling as vey_u.semantic.models; no legacy trainer dependency.
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("expected hidden [B,L,D] and attention_mask [B,L]")
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denominator


class EncoderPool(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean(output.last_hidden_state, attention_mask)


class NativeFieldModel(nn.Module):
    """One pooled state produces all declared heads; questions are not encoded."""

    def __init__(self, encoder, catalogues: dict[str, list[str]], arm: str):
        super().__init__()
        if arm not in {"frozen", "full"}:
            raise ValueError("neural arm must be frozen or full")
        if int(encoder.config.hidden_size) != 384:
            raise ValueError("NATIVE-1 requires the pinned 384-wide stock encoder")
        self.pool = EncoderPool(encoder)
        self.catalogues = {key: tuple(value) for key, value in sorted(catalogues.items())}
        self.endpoint_keys = {endpoint: f"field_{i}" for i, endpoint in enumerate(self.catalogues)}
        self.heads = nn.ModuleDict({
            self.endpoint_keys[endpoint]: nn.Sequential(
                nn.Linear(384, 256), nn.GELU(), nn.Linear(256, len(ids)))
            for endpoint, ids in self.catalogues.items()
        })
        self.arm = arm
        self.encoder_calls = 0
        self.encoded_states = 0
        self.pool.encoder.requires_grad_(arm == "full")
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.arm == "frozen":
            self.pool.eval()
        return self

    def encode(self, input_ids, attention_mask):
        self.encoder_calls += 1
        self.encoded_states += int(input_ids.shape[0])
        if self.arm == "frozen":
            self.pool.eval()
            with torch.no_grad():
                return self.pool(input_ids, attention_mask).detach()
        return self.pool(input_ids, attention_mask)

    def fields(self, features):
        return {endpoint: self.heads[key](features)
                for endpoint, key in self.endpoint_keys.items()}

    def forward(self, input_ids, attention_mask):
        return self.fields(self.encode(input_ids, attention_mask))


def load_stock(config: dict, catalogues: dict[str, list[str]], arm: str, device: str):
    """Load only existing, byte-pinned stock artifacts and audit every exclusion."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoModel, AutoTokenizer
    import transformers

    spec = config["model"]
    cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        home = os.environ.get("HF_HOME", "/home/fazinahamed/Documents/vey-data/decisionmix/hf-home")
        cache = str(Path(home) / "hub")
    snapshot = Path(snapshot_download(spec["repo"], revision=spec["revision"],
                                      cache_dir=cache, local_files_only=True))
    if snapshot.name != spec["revision"]:
        raise RuntimeError("HF cache returned a different stock revision")
    weights = snapshot / "model.safetensors"
    if sha256_file(weights) != spec["weights_sha256"]:
        raise RuntimeError("stock safetensors bytes differ from protocol")
    source_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        spec["repo"], revision=spec["revision"], cache_dir=cache, use_fast=True,
        local_files_only=True, trust_remote_code=False,
    )
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise RuntimeError("cached stock fast tokenizer with padding is required")
    encoder, loading = AutoModel.from_pretrained(
        spec["repo"], revision=spec["revision"], cache_dir=cache, use_safetensors=True,
        dtype=torch.float32, output_loading_info=True, local_files_only=True,
        trust_remote_code=False,
    )
    for name in ("missing_keys", "mismatched_keys", "error_msgs"):
        if loading.get(name):
            raise RuntimeError(f"incomplete stock encoder load: {name}={loading[name]}")
    inactive_position_keys = set()
    if not source_config.get("position_biased_input", True):
        inactive_position_keys = {
            "embeddings.position_embeddings.weight", "deberta.embeddings.position_embeddings.weight",
            "deberta.embeddings.position_embeddings._weight",
        }
    unexpected = set(loading.get("unexpected_keys", []))
    unexplained = unexpected - EXPECTED_TASK_KEYS - inactive_position_keys
    if unexplained:
        raise RuntimeError(f"unexpected base encoder keys: {sorted(unexplained)}")
    if getattr(encoder.config, "_commit_hash", None) != spec["revision"]:
        raise RuntimeError("loaded stock encoder commit differs from protocol")
    if any(parameter.dtype != torch.float32 for parameter in encoder.parameters()):
        raise RuntimeError("stock encoder must remain FP32")
    live_keys = set(encoder.state_dict())
    ledger = {}
    with safe_open(str(weights), framework="np", device="cpu") as tensors:
        for name in tensors.keys():
            value = tensors.get_slice(name)
            base_name = name.removeprefix("deberta.")
            if name in EXPECTED_TASK_KEYS:
                reason = "expected stock MLM/task or embedding alias exclusion"
            elif name in inactive_position_keys:
                reason = "config disables absolute position embeddings"
                if list(value.get_shape()) != [source_config["max_position_embeddings"], source_config["hidden_size"]]:
                    raise RuntimeError(f"inactive position table shape differs: {name}")
            elif base_name in live_keys:
                reason = "loaded encoder state"
            else:
                raise RuntimeError(f"unexplained stored stock key: {name}")
            ledger[name] = {"shape": list(value.get_shape()), "dtype": value.get_dtype(),
                            "reason": reason, "loaded_key": base_name if base_name in live_keys else None}
    model = NativeFieldModel(encoder, catalogues, arm).to(device=device, dtype=torch.float32).eval()
    artifacts = {path.name: {"path": str(path), "resolved_path": str(path.resolve()),
                             "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(snapshot.iterdir()) if path.is_file()}
    if "spm.model" not in artifacts or "config.json" not in artifacts:
        raise RuntimeError("stock tokenizer/config custody incomplete")
    custody = {
        "repo": spec["repo"], "revision": spec["revision"], "snapshot": str(snapshot),
        "weights_sha256": artifacts["model.safetensors"]["sha256"],
        "artifacts": artifacts, "config": encoder.config.to_dict(),
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_is_fast": tokenizer.is_fast,
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "tokenizer_special_tokens": tokenizer.special_tokens_map,
        "loading": {key: sorted(value, key=repr) if isinstance(value, (set, tuple, list)) else value
                    for key, value in sorted(loading.items())},
        "expected_task_exclusions": sorted(unexpected & EXPECTED_TASK_KEYS),
        "unexpected_base_keys": sorted(unexplained), "stored_key_ledger": ledger,
        "encoder_parameters": sum(value.numel() for value in encoder.parameters()),
        "encoder_initial_sha256": fingerprint(encoder.state_dict()),
        "dtype": "float32", "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "pooling": "EncoderPool/masked_mean", "head": "Linear384->256/GELU/Linear256->C",
        "input_scope": "canonical serving state only; no question/candidate text or targets",
    }
    return tokenizer, model, json.loads(json.dumps(custody, default=str, allow_nan=False))


def encode_texts(model: NativeFieldModel, tokenizer, texts: list[str], max_tokens: int):
    """Deduplicate exact serving bytes within a grouped-state microbatch."""
    if not texts:
        raise ValueError("cannot encode an empty state batch")
    unique = list(dict.fromkeys(texts))
    indices = {text: i for i, text in enumerate(unique)}
    device = next(model.parameters()).device
    tokens = tokenizer(unique, padding=True, truncation=True,
                       max_length=max_tokens, return_tensors="pt")
    features = model.encode(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
    gather = torch.tensor([indices[text] for text in texts], dtype=torch.long, device=device)
    return features.index_select(0, gather)


def probabilities(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not 0.5 <= temperature <= 10.0:
        raise ValueError("temperature outside registered bounds")
    if not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("finite nonempty logits required")
    values = [value / temperature for value in logits]
    maximum = max(values)
    masses = [math.exp(value - maximum) for value in values]
    total = math.fsum(masses)
    return [value / total for value in masses]


def native_readout(candidate_ids: list[str], masses: list[float], task: str,
                   canonical_ids: list[str], level_max: int | None = None) -> dict:
    if len(candidate_ids) != len(masses) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("invalid native coordinates")
    if set(candidate_ids) != set(canonical_ids):
        raise ValueError("unsupported native candidate IDs")
    if any(not math.isfinite(value) or value < 0 for value in masses):
        raise ValueError("invalid probability masses")
    if not math.isclose(math.fsum(masses), 1.0, abs_tol=1e-10):
        raise ValueError("probability masses must sum to one")
    by_id = dict(zip(candidate_ids, masses))
    if task.lower() == "choice":
        maximum = max(masses)
        ties = sorted(key for key, value in by_id.items() if value == maximum)
        return {"answer": ties[0], "ties": ties}
    if task.lower() != "score" or level_max is None or len(canonical_ids) != level_max + 1:
        raise ValueError("unsupported ordinal field")
    ordered = [by_id[key] for key in canonical_ids]
    return {"expected_level": math.fsum(level * value for level, value in enumerate(ordered)),
            "cdf": [math.fsum(ordered[:level + 1]) for level in range(level_max + 1)]}


def membership(candidate_ids: list[str], masses: list[float], members: list[str]) -> dict:
    """Canonical membership truth is the learned mass over an explicit ID subset."""
    if len(set(members)) != len(members) or not set(members) <= set(candidate_ids):
        raise ValueError("membership supports unique canonical intent IDs only")
    by_id = dict(zip(candidate_ids, masses))
    mass = math.fsum(by_id[key] for key in sorted(members))
    return {"members": sorted(members), "true_mass": mass, "false_mass": 1.0 - mass,
            "answer": mass >= 0.5}


class StateFieldCache:
    """Eval-only model-identity/exact-state cache, retaining pooled and all fields."""

    def __init__(self, model, tokenizer, temperatures, max_tokens,
                 guard: Callable[[], object] | None = None, prior=None):
        self.model = model
        self.tokenizer = tokenizer
        self.temperatures = dict(temperatures)
        self.max_tokens = max_tokens
        self.guard = guard
        self.prior = prior
        if model is None and prior is None:
            raise ValueError("a neural model or fit prior is required")
        if model is not None:
            model.eval()
            self.identity = fingerprint(model.state_dict())
        else:
            self.identity = hashlib.sha256(canonical_bytes(prior)).hexdigest()
        self.entries = {}
        self.misses = 0

    @property
    def encoder_calls(self):
        return 0 if self.model is None else self.model.encoder_calls

    def field(self, state: str, endpoint: str, *, cached=True):
        key = (self.identity, state.encode("utf-8"))
        if cached and key in self.entries:
            entry = self.entries[key]
        else:
            if self.guard is not None:
                self.guard()
            if self.model is None:
                entry = {name: {"logits": None, "raw_probs": list(mass), "probs": list(mass)}
                         for name, mass in self.prior.items()}
                pooled = None
            else:
                if self.model.training:
                    raise RuntimeError("state cache cannot encode a train-mode model")
                with torch.no_grad():
                    pooled = encode_texts(self.model, self.tokenizer, [state], self.max_tokens)
                    fields = self.model.fields(pooled)
                entry = {}
                for name, tensor in fields.items():
                    logits = tensor[0].detach().cpu().tolist()
                    entry[name] = {"logits": logits, "raw_probs": probabilities(logits),
                                   "probs": probabilities(logits, self.temperatures[name])}
                pooled = pooled[0].detach().cpu()
            self.misses += 1
            if cached:
                self.entries[key] = (pooled, entry)
        if isinstance(entry, tuple):
            entry = entry[1]
        if endpoint not in entry:
            raise ValueError(f"unsupported/untrained field: {endpoint}")
        # Do not expose mutable cached lists to coordinate gathers.
        return {name: None if values is None else list(values) for name, values in entry[endpoint].items()}

    def read(self, state: str, decision: dict, canonical_ids: list[str], *, cached=True,
             reverse=False):
        field = self.field(state, decision["endpoint"], cached=cached)
        requested = list(reversed(decision["candidate_ids"])) if reverse else list(decision["candidate_ids"])
        coordinates = {key: index for index, key in enumerate(canonical_ids)}
        gather = [coordinates[key] for key in requested]
        output = {"candidate_ids": requested,
                  "logits": None if field["logits"] is None else [field["logits"][i] for i in gather],
                  "raw_probs": [field["raw_probs"][i] for i in gather],
                  "probs": [field["probs"][i] for i in gather]}
        for prefix, key in (("", "probs"), ("raw_", "raw_probs")):
            result = native_readout(requested, output[key], decision["task"], canonical_ids,
                                    decision.get("level_max"))
            output.update({prefix + name: value for name, value in result.items()})
        return output


def donor_plan(records: list[dict], eligible_endpoints: list[str]) -> dict[str, str | None]:
    """Next cyclic endpoint ID with a different component and native target."""
    plan = {}
    for endpoint in sorted(eligible_endpoints):
        rows = sorted([(decision["id"], record["component_id"], decision)
                       for record in records for decision in record["decisions"]
                       if decision["endpoint"] == endpoint], key=lambda value: value[0])
        for index, (identifier, component, decision) in enumerate(rows):
            original = decision["gold"] if decision["task"].lower() == "choice" else decision["mean_level"]
            for offset in range(1, len(rows)):
                donor_id, donor_component, donor = rows[(index + offset) % len(rows)]
                target = donor["gold"] if donor["task"].lower() == "choice" else donor["mean_level"]
                if donor_component != component and target != original:
                    plan[identifier] = donor_id
                    break
            else:
                plan[identifier] = None
    return plan
