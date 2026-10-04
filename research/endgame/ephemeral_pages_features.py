"""ECA-1 frozen feature capture, canonical IR projection, and cached runtime.

Only StateBlock.text and term.question are tokenized. Structural metadata is
used to group pages and form separate supervision arrays; it never enters the
encoder or lexical text serializer.
"""
from __future__ import annotations

import os

# Set process policy before NumPy, Transformers, or BLAS libraries are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import fcntl
import hashlib
import importlib
import json
import math
import os.path
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "ephemeral_pages_protocol.json"
PROTOCOL_SHA256 = "4c0c1efe8ad8ffdde79004536a44fc6d034b7701cdedde8e2034dc3f4e52b489"
CANONICAL_ROOT = Path("/home/fazinahamed/Documents/vey")
CANONICAL_FILE_HASHES = {
    "vey_u/ir.py": "0184e25e05c17638117e776fc660fd6e4896c3590da0fcf94fc753509869ea2d",
    "vey_u/semantic/format.py": "3e7ad63a26b68e7bb74142d8c69306bd4c0473f42eb5a69680e0474422c0e1ab",
    "vey_u/semantic/models.py": "421f8d20aab9f1b545d18ad74eee6cf02c00fd13f0a7feac841e848f366dbc94",
}
WIDTH = 384
MAX_TOKENS = 128
BATCH_SIZE = 32
GPU_LOCK = Path("/tmp/vey-gpu.lock")
CACHE_SCHEMA = "eca1-features-v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _protocol(path: str | Path = PROTOCOL_PATH) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    digest = sha256_bytes(raw)
    if digest != PROTOCOL_SHA256:
        raise RuntimeError(f"ECA protocol hash changed: {digest}")
    cfg = json.loads(raw)
    enc = cfg["encoder"]
    if (enc["repo"] != "cross-encoder/nli-deberta-v3-xsmall" or
            enc["revision"] != "a150876415327c80daeff35ca6f68f5ed8cf5c24" or
            enc["weight_sha256"] != "4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9" or
            enc["max_tokens"] != MAX_TOKENS or enc["truncate"] is not False):
        raise RuntimeError("ECA encoder/tokenization contract changed")
    return cfg, digest

CONTROL_IDS = {"pages", "cross", "cosine", "lexical", "query_blind"}
CAPTURE_PHASES = {"train", "validation", "calibration", "development", "final"}


def validate_final_receipt(path: str | Path, protocol_path: str | Path = PROTOCOL_PATH) -> dict[str, Any]:
    """Verify the immutable selection/calibration artifacts before final IR loading."""
    _, protocol_hash = _protocol(protocol_path)
    receipt_path = Path(path).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "vey.eca.selection-calibration.v1":
        raise RuntimeError("final capture requires the frozen ECA selection/calibration receipt")
    if receipt.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("selection/calibration receipt was made for a different ECA protocol")
    if receipt.get("eligible_arm") != "pages" or receipt.get("final_outcomes_used") is not False:
        raise RuntimeError("final receipt must keep pages eligible and exclude final outcomes from selection")

    def checked_artifact(entry: Any, label: str) -> dict[str, str]:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise RuntimeError(f"receipt {label} must contain path and sha256")
        artifact_path = Path(entry["path"])
        if not artifact_path.is_absolute():
            artifact_path = receipt_path.parent / artifact_path
        artifact_path = artifact_path.resolve()
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64 or not artifact_path.is_file():
            raise RuntimeError(f"receipt {label} artifact is unavailable")
        actual = sha256_file(artifact_path)
        if actual != expected:
            raise RuntimeError(f"receipt {label} hash mismatch: {actual}")
        return {"path": str(artifact_path), "sha256": actual}

    checkpoints = receipt.get("checkpoint_files")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != CONTROL_IDS:
        raise RuntimeError("final receipt must hash checkpoints for all five fixed controls")
    verified_checkpoints = {
        control: checked_artifact(checkpoints[control], f"checkpoint_files.{control}")
        for control in sorted(CONTROL_IDS)
    }
    selection = checked_artifact(receipt.get("selection_file"), "selection_file")
    calibration = checked_artifact(receipt.get("calibration_file"), "calibration_file")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "protocol_sha256": protocol_hash,
        "eligible_arm": "pages",
        "final_outcomes_used": False,
        "checkpoint_files": verified_checkpoints,
        "selection_file": selection,
        "calibration_file": calibration,
    }


def _check_canonical_dependencies(cfg: Mapping[str, Any]) -> dict[str, str]:
    root = Path(cfg["canonical_dependency"]["root"]).resolve()
    if root != CANONICAL_ROOT.resolve():
        raise RuntimeError(f"unexpected canonical dependency root: {root}")
    expected = cfg["canonical_dependency"]["files_sha256"]
    if expected != CANONICAL_FILE_HASHES:
        raise RuntimeError("protocol canonical dependency hashes differ from implementation pins")
    actual: dict[str, str] = {}
    for relpath, pinned in CANONICAL_FILE_HASHES.items():
        path = root / relpath
        digest = sha256_file(path)
        if digest != pinned:
            raise RuntimeError(f"canonical source hash mismatch for {relpath}: {digest}")
        actual[relpath] = digest
    if str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    existing = sys.modules.get("vey_u.semantic.models")
    if existing is not None and Path(existing.__file__).resolve() != (root / "vey_u/semantic/models.py").resolve():
        raise RuntimeError("a different vey_u semantic model module is already imported")
    module = importlib.import_module("vey_u.semantic.models")
    if Path(module.__file__).resolve() != (root / "vey_u/semantic/models.py").resolve():
        raise RuntimeError("canonical EncoderPool/masked_mean imported from a different source")
    return actual

def _parse_meminfo() -> tuple[int, int]:
    values: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="ascii") as stream:
        for line in stream:
            name, rest = line.split(":", 1)
            if name in {"MemAvailable", "SwapTotal", "SwapFree"}:
                values[name] = int(rest.split()[0])
    if set(values) != {"MemAvailable", "SwapTotal", "SwapFree"}:
        raise RuntimeError("could not read RAM/swap resource guard inputs")
    available_mib = values["MemAvailable"] // 1024
    swap_used_mib = (values["SwapTotal"] - values["SwapFree"]) // 1024
    return int(available_mib), int(swap_used_mib)


def _source_lineage(cfg: Mapping[str, Any], protocol_hash: str,
                    dependency_hashes: Mapping[str, str], tokenizer_hash: str,
                    weight_hash: str) -> dict[str, Any]:
    model_source = HERE / "ephemeral_pages_model.py"
    return {
        "protocol_sha256": protocol_hash,
        "canonical_root": str(CANONICAL_ROOT),
        "canonical_files_sha256": dict(dependency_hashes),
        "feature_code_sha256": sha256_file(__file__),
        "reader_code_sha256": sha256_file(model_source),
        "encoder_repo": cfg["encoder"]["repo"],
        "encoder_revision": cfg["encoder"]["revision"],
        "encoder_weight_sha256": weight_hash,
        "tokenizer_sha256": tokenizer_hash,
        "serialization": "single text; cross uses tokenizer(question,text_pair=page_text); masked_mean(last_hidden_state)",
        "pooling": "canonical EncoderPool Identity projection; FP32 final-layer masked mean",
        "native_pooler_unused": True,
        "classifier_unused": True,
        "token_limit": MAX_TOKENS,
        "truncation": False,
        "batch_size": BATCH_SIZE,
    }




def _apply_process_priority() -> dict[str, Any]:
    receipt: dict[str, Any] = {"nice_target": 10, "ionice_target": "2/7", "threads": 4,
                               "blas_threads": 1}
    try:
        current = os.nice(0)
        if current < 10:
            os.nice(10 - current)
        receipt["nice_result"] = os.nice(0)
    except (AttributeError, OSError) as exc:
        receipt["nice_error"] = f"{type(exc).__name__}: {exc}"
    ionice = None
    for executable in ("/usr/bin/ionice", "/bin/ionice"):
        if Path(executable).exists():
            ionice = executable
            break
    if ionice:
        proc = subprocess.run([ionice, "-c", "2", "-n", "7", "-p", str(os.getpid())],
                              check=False, capture_output=True, text=True)
        receipt["ionice_returncode"] = proc.returncode
        if proc.stderr.strip():
            receipt["ionice_stderr"] = proc.stderr.strip()
    else:
        receipt["ionice_error"] = "ionice executable not found"
    return receipt


def _resource_guard(torch, initial_swap_mib: int) -> dict[str, int]:
    paused = False
    while True:
        available_mib, swap_used_mib = _parse_meminfo()
        if available_mib >= 4096 and swap_used_mib <= initial_swap_mib + 256:
            if paused:
                torch.set_num_threads(4)
            return {"available_RAM_MiB": available_mib, "swap_used_MiB": swap_used_mib}
        if not paused:
            torch.set_num_threads(2)
            paused = True
        time.sleep(5)


class FeatureEncoder:
    """Pinned frozen encoder; CUDA lock is held until close/context exit."""

    def __init__(self, device: str = "cuda", protocol_path: str | Path = PROTOCOL_PATH):
        self.device_name = str(device)
        self._lock_stream = None
        self._closed = False
        cfg, protocol_hash = _protocol(protocol_path)
        dependency_hashes = _check_canonical_dependencies(cfg)
        if self.device_name.startswith("cuda"):
            self._lock_stream = GPU_LOCK.open("a+")
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            import torch
            torch.set_num_threads(4)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            initial_swap_mib = _parse_meminfo()[1]
            startup_resources = _resource_guard(torch, initial_swap_mib)
            priority = _apply_process_priority()
            if self.device_name.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested by ECA but unavailable")
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel, AutoTokenizer

            spec = cfg["encoder"]
            weight_path = Path(hf_hub_download(spec["repo"], "model.safetensors",
                                               revision=spec["revision"], local_files_only=True))
            weight_hash = sha256_file(weight_path)
            if weight_hash != spec["weight_sha256"]:
                raise RuntimeError(f"encoder weight hash mismatch: {weight_hash}")
            tokenizer = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"],
                                                      use_fast=True, local_files_only=True,
                                                      trust_remote_code=False)
            if not tokenizer.is_fast or tokenizer.pad_token_id is None:
                raise RuntimeError("pinned ECA tokenizer must be fast and have a pad token")
            tokenizer_hash = sha256_bytes(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
            model, loading = AutoModel.from_pretrained(
                spec["repo"], revision=spec["revision"], use_safetensors=True,
                dtype=torch.float32, output_loading_info=True, trust_remote_code=False,
                local_files_only=True,
            )
            if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise RuntimeError(f"pinned base encoder load is incomplete: {loading}")
            if getattr(model.config, "_commit_hash", None) != spec["revision"]:
                raise RuntimeError("loaded encoder commit does not match pinned revision")
            if int(model.config.hidden_size) != WIDTH:
                raise RuntimeError(f"encoder width must be {WIDTH}")
            if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
                raise RuntimeError("encoder must remain FP32")
            model.requires_grad_(False)
            model.to(self.device_name).eval()
            from vey_u.semantic.models import EncoderPool
            pool = EncoderPool(model).to(self.device_name).eval()
            pool.requires_grad_(False)
            self._torch = torch
            self.tokenizer = tokenizer
            self.model = model
            self.pool = pool
            self.protocol = cfg
            self.protocol_hash = protocol_hash
            self.dependency_hashes = dependency_hashes
            self.tokenizer_hash = tokenizer_hash
            self.weight_hash = weight_hash
            self.initial_swap_mib = initial_swap_mib
            self.priority = priority
            self.resource_startup = startup_resources
            self.lineage = _source_lineage(cfg, protocol_hash, dependency_hashes,
                                           tokenizer_hash, weight_hash)
            self.lineage["loading"] = {
                key: sorted(value, key=repr) if isinstance(value, (set, list, tuple)) else value
                for key, value in sorted(loading.items())
            }
            self.lineage["resource_startup"] = dict(startup_resources)
            self.lineage["encoder_parameters_sha256_before"] = self.parameter_hash()
            self.forward_counts = {"page": 0, "query": 0, "cross": 0}
            self.encoded_counts = {"page": 0, "query": 0, "cross": 0}
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "FeatureEncoder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._lock_stream is not None:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()
            self._lock_stream = None

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for name, parameter in sorted(self.model.named_parameters()):
            digest.update(name.encode("utf-8"))
            digest.update(str(parameter.dtype).encode("ascii"))
            digest.update(_tensor_bytes(parameter))
        return digest.hexdigest()

    def finalize_lineage(self) -> dict[str, Any]:
        after = self.parameter_hash()
        if after != self.lineage["encoder_parameters_sha256_before"]:
            raise RuntimeError("frozen encoder parameter bytes changed during feature capture")
        result = dict(self.lineage)
        result["encoder_parameters_sha256_after"] = after
        result["encoder_parameters_unchanged"] = True
        return result

    def tokenize(self, texts: Sequence[str], pairs: Sequence[str] | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
        if pairs is not None and len(texts) != len(pairs):
            raise ValueError("text pair arrays must have equal lengths")
        ids = np.full((len(texts), MAX_TOKENS), int(self.tokenizer.pad_token_id), dtype=np.int32)
        masks = np.zeros((len(texts), MAX_TOKENS), dtype=np.uint8)
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                raise ValueError("encoder text must be a nonempty semantic string")
            kwargs = {"add_special_tokens": True, "truncation": False,
                      "return_attention_mask": True}
            if pairs is None:
                encoded = self.tokenizer(text, **kwargs)
            else:
                if not isinstance(pairs[index], str) or not pairs[index]:
                    raise ValueError("cross page text must be a nonempty semantic string")
                encoded = self.tokenizer(text, text_pair=pairs[index], **kwargs)
            token_ids = encoded["input_ids"]
            attention = encoded["attention_mask"]
            if len(token_ids) > MAX_TOKENS:
                kind = "cross question/page pair" if pairs is not None else "semantic text"
                raise ValueError(f"{kind} has {len(token_ids)} tokens; truncation is forbidden at {MAX_TOKENS}")
            ids[index, :len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
            masks[index, :len(attention)] = np.asarray(attention, dtype=np.uint8)
        return ids, masks

    def encode_batch(self, input_ids: np.ndarray, attention_mask: np.ndarray,
                     modality: str) -> np.ndarray:
        if modality not in self.forward_counts:
            raise ValueError(f"unknown encoder modality {modality!r}")
        if input_ids.ndim != 2 or input_ids.shape[1] != MAX_TOKENS or attention_mask.shape != input_ids.shape:
            raise ValueError("token arrays must have shape [N,128]")
        _resource_guard(self._torch, self.initial_swap_mib)
        inputs = self._torch.as_tensor(input_ids, dtype=self._torch.long, device=self.device_name)
        mask = self._torch.as_tensor(attention_mask, dtype=self._torch.long, device=self.device_name)
        with self._torch.inference_mode():
            values = self.pool(input_ids=inputs, attention_mask=mask)
        values = values.to(dtype=self._torch.float32)
        if values.ndim != 2 or values.shape != (input_ids.shape[0], WIDTH):
            raise RuntimeError("canonical EncoderPool returned an unexpected feature shape")
        result = values.detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("encoder produced nonfinite FP32 features")
        self.forward_counts[modality] += 1
        self.encoded_counts[modality] += int(input_ids.shape[0])
        return result


def _tensor_bytes(value) -> bytes:
    array = value.detach().cpu().contiguous().numpy()
    return memoryview(array).cast("B").tobytes()


def _block_value(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, Mapping):
        return block.get(name, default)
    return getattr(block, name, default)


def rows_to_examples(rows: Iterable[Mapping[str, Any] | Any]) -> list[dict[str, Any]]:
    """Project canonical DecisionIR rows to one term/candidate training record.

    Page ownership, opaque field keys, grades, orientation and knownness become
    targets/structural associations only. Exact blocks and unowned blocks are
    excluded before any feature extraction. Metadata and identifiers are never
    placed in a token string.
    """
    output: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "to_json"):
            row = json.loads(row.to_json())
        if not isinstance(row, Mapping):
            raise TypeError("rows_to_examples expects canonical DecisionIR mappings/objects")
        meta = row.get("metadata") or {}
        if not isinstance(meta, Mapping):
            raise TypeError("DecisionIR metadata must be a mapping")
        state_blocks = row.get("state_blocks", ())
        candidates = row.get("candidates", ())
        owners = meta.get("page_owners", {})
        grades = meta.get("page_grades", {})
        fields = meta.get("page_fields", {})
        known_map = meta.get("known", {})
        if not isinstance(known_map, Mapping):
            raise TypeError("metadata.known must be a candidate-to-bool mapping")
        terms = meta.get("terms", ())
        if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)) or not terms:
            raise ValueError(f"DecisionIR row {row.get('id')!r} has no explicit semantic terms")
        if not isinstance(owners, Mapping) or not isinstance(grades, Mapping) or not isinstance(fields, Mapping):
            raise TypeError("page provenance maps must be mappings")
        all_candidate_ids = [candidate.get("id") if isinstance(candidate, Mapping) else getattr(candidate, "id")
                             for candidate in candidates]
        if len(all_candidate_ids) != len(set(all_candidate_ids)):
            raise ValueError("candidate IDs must be unique within DecisionIR")
        # The explicit UNKNOWN sentinel is a decision outcome, not an evidence
        # candidate; it never receives page features or a synthetic score.
        candidate_ids = [candidate_id for candidate_id in all_candidate_ids
                         if candidate_id != "__unknown__"]
        pages_by_candidate: dict[str, list[dict[str, Any]]] = {cid: [] for cid in candidate_ids}
        seen_block_ids: set[str] = set()
        for block in state_blocks:
            block_id = _block_value(block, "id")
            if block_id in seen_block_ids:
                raise ValueError(f"duplicate StateBlock id {block_id!r}")
            seen_block_ids.add(block_id)
            text = _block_value(block, "text")
            exact = bool(_block_value(block, "exact", False))
            if exact or block_id not in owners:
                continue
            owner = owners[block_id]
            if owner not in pages_by_candidate:
                raise ValueError(f"page owner {owner!r} is not an evidence candidate in row {row.get('id')!r}")
            if not isinstance(text, str) or not text:
                raise ValueError(f"semantic page {block_id!r} has no text")
            grade = grades.get(block_id)
            if grade is not None and (isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4):
                raise ValueError(f"page grade for {block_id!r} must be integer 0..4 or null")
            field_key = fields.get(block_id)
            pages_by_candidate[owner].append({
                "block_id": block_id,
                "text": text,
                "grade_target": None if grade is None else float(grade) / 4.0,
                "field_key": field_key,
            })
        for page_list in pages_by_candidate.values():
            page_list.sort(key=lambda item: str(item["block_id"]))
        row_id = row.get("id")
        split = row.get("split")
        world_id = meta.get("world_id")
        for term_index, term in enumerate(terms):
            if not isinstance(term, Mapping):
                raise TypeError("metadata.terms entries must be mappings")
            question = term.get("question")
            if term_index == 0 and len(terms) == 1:
                # Atomic core term is deliberately the DecisionIR question.
                question = row.get("question")
            if not isinstance(question, str) or not question:
                raise ValueError(f"term {term_index} in row {row_id!r} has no question")
            field_key = term.get("field_key")
            orientation = term.get("orientation")
            if orientation is None:
                orientation_value = None
            elif isinstance(orientation, (int, float)) and not isinstance(orientation, bool) and float(orientation) in (-1.0, 1.0):
                orientation_value = float(orientation)
            else:
                raise ValueError(f"term orientation must be numeric -1 or +1, got {orientation!r}")
            weight = term.get("weight", 1)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                raise ValueError("term weight must be a finite numeric factor")
            relevant = {}
            for candidate_id in candidate_ids:
                pages = pages_by_candidate[candidate_id]
                rel = [page["field_key"] == field_key for page in pages]
                count = sum(rel)
                relevance_target = [1.0 / count if flag and count else 0.0 for flag in rel]
                if candidate_id not in known_map:
                    raise ValueError(f"metadata.known omits evidence candidate {candidate_id!r}")
                known_value = known_map[candidate_id]
                if not isinstance(known_value, bool):
                    raise ValueError(f"known[{candidate_id!r}] must be boolean")
                candidate_known = known_value
                grade_targets = [page["grade_target"] for page in pages]
                grade_mask = [bool(flag and candidate_known and value is not None)
                              for flag, value in zip(rel, grade_targets)]
                orientation_targets = [orientation_value if flag and candidate_known else None for flag in rel]
                orientation_mask = [value is not None for value in orientation_targets]
                directed_grade_targets = [
                    (value if orientation_value == 1.0 else 1.0 - value)
                    if flag and candidate_known and value is not None and orientation_value is not None
                    else None
                    for flag, value in zip(rel, grade_targets)
                ]
                relevant[candidate_id] = {
                    "pages": pages,
                    "relevance_target": relevance_target,
                    "grade_target": grade_targets,
                    "grade_mask": grade_mask,
                    "directed_grade_target": directed_grade_targets,
                    "orientation_target": orientation_targets,
                    "orientation_mask": orientation_mask,
                    "known_target": candidate_known,
                }
            for candidate_id in candidate_ids:
                labels = relevant[candidate_id]
                output.append({
                    "row_id": row_id,
                    "split": split,
                    "world_id": world_id,
                    "term_index": term_index,
                    "candidate_id": candidate_id,
                    "question": question,
                    "term_weight": float(weight),
                    "pages": labels["pages"],
                    "relevance_target": labels["relevance_target"],
                    "grade_target": labels["grade_target"],
                    "directed_grade_target": labels["directed_grade_target"],
                    "grade_mask": labels["grade_mask"],
                    "orientation_target": labels["orientation_target"],
                    "orientation_mask": labels["orientation_mask"],
                    "known_target": labels["known_target"],
                    "teacher_score": (meta.get("teacher_scores") or {}).get(candidate_id),
                })
    return output


def _text_hash(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def _item_cache_key(modality: str, item_hashes: Sequence[str], lineage: Mapping[str, Any]) -> str:
    return sha256_bytes(_canonical_json({"schema": CACHE_SCHEMA, "modality": modality,
                                         "items": list(item_hashes), "lineage": lineage}))


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _array_sha(path: Path) -> str:
    return sha256_file(path)


class FeatureCorpus:
    """Captured unique features plus packed mmap arrays for a set of term/candidate rows."""

    def __init__(self, records: list[dict[str, Any]], query_features: np.ndarray,
                 page_features: np.ndarray, cross_features: np.ndarray,
                 query_index: Mapping[str, int], page_index: Mapping[str, int],
                 cross_index: Mapping[tuple[str, str], int], cache_dir: Path,
                 lineage: Mapping[str, Any], counters: Mapping[str, Any],
                 token_receipts: Mapping[str, Any]):
        self.records = records
        self.query_features = query_features
        self.page_features = page_features
        self.cross_features = cross_features
        self.query_index = dict(query_index)
        self.page_index = dict(page_index)
        self.cross_index = dict(cross_index)
        self.cache_dir = cache_dir
        self.lineage = dict(lineage)
        self.counters = dict(counters)
        self.token_receipts = dict(token_receipts)

    def arrays(self) -> dict[str, Any]:
        """Return padded mmap feature/target arrays and records in stable record order."""
        signature = [{
            "row_id": r["row_id"], "split": r["split"], "term_index": r["term_index"],
            "candidate_id": r["candidate_id"], "question_hash": _text_hash(r["question"]),
            "page_hashes": [_text_hash(page["text"]) for page in r["pages"]],
            "page_block_ids": [page["block_id"] for page in r["pages"]],
            "targets": {key: r[key] for key in ("relevance_target", "grade_target",
                                                    "directed_grade_target", "grade_mask",
                                                    "orientation_target", "orientation_mask", "known_target")},
        } for r in self.records]
        packed_key = sha256_bytes(_canonical_json({"lineage": self.lineage, "records": signature}))
        names = ("q", "pages", "cross", "page_mask", "relevance_target", "grade_target",
                 "directed_grade_target", "grade_mask", "orientation_target", "orientation_mask", "known_target")
        paths = {name: self.cache_dir / f"packed_{packed_key}_{name}.npy" for name in names}
        manifest_path = self.cache_dir / f"packed_{packed_key}.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("packed_key") != packed_key or manifest.get("records") != len(self.records):
                raise RuntimeError("packed feature cache manifest mismatch")
            for name, path in paths.items():
                if not path.exists() or sha256_file(path) != manifest["files"][path.name]["sha256"]:
                    raise RuntimeError(f"packed mmap cache is corrupt: {path.name}")
            arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
            return {**arrays, "raw_q": arrays["q"], "raw_pages": arrays["pages"],
                    "records": self.records, "lineage": self.lineage,
                    "counters": self.counters, "token_receipts": self.token_receipts}

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        n = len(self.records)
        pmax = max(1, max((len(record["pages"]) for record in self.records), default=0))
        shapes = {
            "q": (n, WIDTH), "pages": (n, pmax, WIDTH), "cross": (n, pmax, WIDTH),
            "page_mask": (n, pmax), "relevance_target": (n, pmax),
            "grade_target": (n, pmax), "directed_grade_target": (n, pmax),
            "grade_mask": (n, pmax), "orientation_target": (n, pmax),
            "orientation_mask": (n, pmax), "known_target": (n,),
        }
        dtypes = {"q": np.float32, "pages": np.float32, "cross": np.float32,
                  "page_mask": np.bool_, "relevance_target": np.float32,
                  "grade_target": np.float32, "directed_grade_target": np.float32,
                  "grade_mask": np.bool_, "orientation_target": np.float32,
                  "orientation_mask": np.bool_, "known_target": np.float32}
        temps = {name: path.with_suffix(path.suffix + ".partial") for name, path in paths.items()}
        mmap = {name: np.lib.format.open_memmap(temps[name], mode="w+", dtype=dtypes[name], shape=shapes[name])
                for name in names}
        for name, array in mmap.items():
            if name in {"grade_target", "directed_grade_target", "orientation_target"}:
                array.fill(np.nan)
            elif name == "known_target":
                array.fill(0.0)
            else:
                array.fill(False if dtypes[name] is np.bool_ else 0.0)
        for index, record in enumerate(self.records):
            question_hash = _text_hash(record["question"])
            mmap["q"][index] = self.query_features[self.query_index[question_hash]]
            mmap["known_target"][index] = float(record["known_target"])
            for page_pos, page in enumerate(record["pages"]):
                page_hash = _text_hash(page["text"])
                pair_key = (question_hash, page_hash)
                mmap["pages"][index, page_pos] = self.page_features[self.page_index[page_hash]]
                mmap["cross"][index, page_pos] = self.cross_features[self.cross_index[pair_key]]
                mmap["page_mask"][index, page_pos] = True
                mmap["relevance_target"][index, page_pos] = record["relevance_target"][page_pos]
                for target_name in ("grade_target", "directed_grade_target", "orientation_target"):
                    target_value = record[target_name][page_pos]
                    mmap[target_name][index, page_pos] = np.nan if target_value is None else target_value
                mmap["grade_mask"][index, page_pos] = record["grade_mask"][page_pos]
                mmap["orientation_mask"][index, page_pos] = record["orientation_mask"][page_pos]
        for array in mmap.values():
            array.flush()
        del mmap
        for name in names:
            os.replace(temps[name], paths[name])
        file_info = {path.name: {"sha256": sha256_file(path), "shape": list(shapes[name]),
                                 "dtype": np.dtype(dtypes[name]).name}
                     for name, path in paths.items()}
        manifest = {"packed_key": packed_key, "records": n, "page_slots": pmax,
                    "files": file_info, "lineage": self.lineage}
        manifest_tmp = manifest_path.with_suffix(".json.partial")
        manifest_tmp.write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(manifest_tmp, manifest_path)
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
        return {**arrays, "raw_q": arrays["q"], "raw_pages": arrays["pages"],
                "records": self.records, "lineage": self.lineage,
                "counters": self.counters, "token_receipts": self.token_receipts}


class FeatureStore:
    """Immutable content-keyed mmap caches for single text and question/page-pair features."""

    def __init__(self, root: str | Path, protocol_path: str | Path = PROTOCOL_PATH):
        self.protocol_path = Path(protocol_path)
        cfg, _ = _protocol(self.protocol_path)
        self.output_root = Path(cfg["output_root"]).resolve()
        self.root = Path(root).resolve()
        if self.root != self.output_root and self.output_root not in self.root.parents:
            raise ValueError("ECA caches must stay under the protocol output_root outside Git")
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(self, rows: Iterable[Mapping[str, Any] | Any], *, phase: str,
                device: str = "cuda",
                selection_calibration_receipt: str | Path | None = None) -> FeatureCorpus:
        if phase not in CAPTURE_PHASES:
            raise ValueError(f"capture phase must be one of {sorted(CAPTURE_PHASES)}")
        cfg, protocol_hash = _protocol(self.protocol_path)
        final_receipt = None
        if phase == "final":
            if selection_calibration_receipt is None:
                raise RuntimeError("final IR features are locked until selection/calibration receipt exists")
            final_receipt = validate_final_receipt(selection_calibration_receipt, self.protocol_path)
        elif selection_calibration_receipt is not None:
            raise ValueError("selection/calibration final receipt is only valid for phase='final'")
        # The receipt check is deliberately before iterating rows; drivers can
        # call validate_final_receipt before opening the final IR file itself.
        records = rows_to_examples(rows)
        if not records:
            raise ValueError("cannot capture an empty ECA corpus")
        wrong_splits = sorted({record["split"] for record in records if record["split"] != phase})
        if wrong_splits:
            raise ValueError(f"phase {phase!r} cannot capture rows from splits {wrong_splits}")
        with FeatureEncoder(device, self.protocol_path) as encoder:
            lineage_base = dict(encoder.lineage)
            cache_dir = self.root / "mmap"
            cache_dir.mkdir(parents=True, exist_ok=True)
            query_texts = sorted({record["question"] for record in records}, key=_text_hash)
            page_texts = sorted({page["text"] for record in records for page in record["pages"]}, key=_text_hash)
            pair_map: dict[tuple[str, str], tuple[str, str]] = {}
            for record in records:
                for page in record["pages"]:
                    key = (_text_hash(record["question"]), _text_hash(page["text"]))
                    old = pair_map.get(key)
                    value = (record["question"], page["text"])
                    if old is not None and old != value:
                        raise RuntimeError("SHA collision in question/page pair cache")
                    pair_map[key] = value
            pairs = sorted(pair_map.items(), key=lambda item: item[0])
            query_features, _, query_receipt = self._cache_modality(
                encoder, cache_dir, "query", [(text, None) for text in query_texts], lineage_base)
            page_features, _, page_receipt = self._cache_modality(
                encoder, cache_dir, "page", [(text, None) for text in page_texts], lineage_base)
            cross_features, _, cross_receipt = self._cache_modality(
                encoder, cache_dir, "cross", [value for _, value in pairs], lineage_base)
            lineage = encoder.finalize_lineage()
            lineage["capture_phase"] = phase
            if final_receipt is not None:
                lineage["selection_calibration_receipt"] = final_receipt
            counters = {
                "phase": phase,
                "page": page_receipt,
                "query": query_receipt,
                "cross": cross_receipt,
                "total_encoder_forward_calls_this_capture": sum(
                    receipt["forward_calls_this_capture"] for receipt in
                    (page_receipt, query_receipt, cross_receipt)),
                "total_encoded_examples_this_capture": sum(
                    receipt["encoded_examples_this_capture"] for receipt in
                    (page_receipt, query_receipt, cross_receipt)),
                "batch_size": BATCH_SIZE,
                "lock_path": str(GPU_LOCK) if str(device).startswith("cuda") else None,
                "resources": encoder.priority,
                "selection_calibration_receipt_sha256": (
                    final_receipt["receipt_sha256"] if final_receipt is not None else None),
            }
            token_receipts = {"page": page_receipt["files"], "query": query_receipt["files"],
                              "cross": cross_receipt["files"]}
        query_index = {_text_hash(text): index for index, text in enumerate(query_texts)}
        page_index = {_text_hash(text): index for index, text in enumerate(page_texts)}
        cross_index = {key: index for index, (key, _) in enumerate(pairs)}
        return FeatureCorpus(records, query_features, page_features, cross_features,
                             query_index, page_index, cross_index, cache_dir,
                             lineage, counters, token_receipts)

    def _cache_modality(self, encoder: FeatureEncoder, cache_dir: Path, modality: str,
                        items: Sequence[tuple[str, str | None]], lineage: Mapping[str, Any]
                        ) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
        if not items:
            cache_key = _item_cache_key(modality, (), lineage)
            receipt = {
                "modality": modality, "cache_key": cache_key, "unique_inputs": 0,
                "cache_hits": 0, "cache_misses": 0, "forward_calls_this_capture": 0,
                "encoded_examples_this_capture": 0, "original_forward_calls": 0,
                "original_encoded_examples": 0, "files": {},
            }
            return np.empty((0, WIDTH), dtype=np.float32), {}, receipt
        if modality == "cross":
            item_hashes = [sha256_bytes(_canonical_json([_text_hash(a), _text_hash(b or "")]))
                           for a, b in items]
        else:
            item_hashes = [_text_hash(a) for a, _ in items]
        if len(item_hashes) != len(set(item_hashes)):
            raise RuntimeError(f"duplicate {modality} cache items were not deduplicated")
        cache_key = _item_cache_key(modality, item_hashes, lineage)
        stem = f"{modality}_{cache_key}"
        paths = {name: cache_dir / f"{stem}_{name}.npy"
                 for name in ("input_ids", "attention_mask", "features")}
        manifest_path = cache_dir / f"{stem}.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("cache_key") != cache_key or manifest.get("modality") != modality or
                    manifest.get("item_hashes") != item_hashes or manifest.get("lineage") != dict(lineage)):
                raise RuntimeError(f"{modality} cache manifest does not match frozen inputs/lineage")
            for name, path in paths.items():
                if not path.exists() or sha256_file(path) != manifest["files"][path.name]["sha256"]:
                    raise RuntimeError(f"{modality} mmap cache is corrupt: {path.name}")
            arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
            receipt = {"modality": modality, "cache_key": cache_key,
                       "unique_inputs": len(items), "cache_hits": len(items), "cache_misses": 0,
                       "forward_calls_this_capture": 0, "encoded_examples_this_capture": 0,
                       "original_forward_calls": manifest["forward_calls"],
                       "original_encoded_examples": manifest["encoded_examples"],
                       "files": {name: {"path": str(path), "sha256": manifest["files"][path.name]["sha256"]}
                                 for name, path in paths.items()}}
            return arrays["features"], {key: index for index, key in enumerate(item_hashes)}, receipt

        n = len(items)
        temps = {name: path.with_suffix(path.suffix + ".partial") for name, path in paths.items()}
        input_ids = np.lib.format.open_memmap(temps["input_ids"], mode="w+", dtype=np.int32,
                                              shape=(n, MAX_TOKENS))
        attention = np.lib.format.open_memmap(temps["attention_mask"], mode="w+", dtype=np.uint8,
                                              shape=(n, MAX_TOKENS))
        features = np.lib.format.open_memmap(temps["features"], mode="w+", dtype=np.float32,
                                             shape=(n, WIDTH))
        tokenized_hash = hashlib.sha256()
        forward_start = encoder.forward_counts[modality]
        encoded_start = encoder.encoded_counts[modality]
        try:
            # Persist the literal inputs actually handed to the frozen encoder.
            for start in range(0, n, BATCH_SIZE):
                stop = min(n, start + BATCH_SIZE)
                batch = items[start:stop]
                texts = [item[0] for item in batch]
                second = [item[1] for item in batch] if modality == "cross" else None
                ids, masks = encoder.tokenize(texts, second)
                input_ids[start:stop] = ids
                attention[start:stop] = masks
                tokenized_hash.update(memoryview(np.ascontiguousarray(ids)).cast("B"))
                tokenized_hash.update(memoryview(np.ascontiguousarray(masks)).cast("B"))
            input_ids.flush()
            attention.flush()
            for start in range(0, n, BATCH_SIZE):
                stop = min(n, start + BATCH_SIZE)
                features[start:stop] = encoder.encode_batch(input_ids[start:stop], attention[start:stop], modality)
            features.flush()
            del input_ids, attention, features
            for name in paths:
                os.replace(temps[name], paths[name])
        except Exception:
            del input_ids, attention, features
            for temp in temps.values():
                if temp.exists():
                    temp.unlink()
            raise
        files = {path.name: {"sha256": sha256_file(path), "shape": list(np.load(path, mmap_mode="r").shape),
                             "dtype": np.load(path, mmap_mode="r").dtype.name}
                 for path in paths.values()}
        calls = encoder.forward_counts[modality] - forward_start
        encoded = encoder.encoded_counts[modality] - encoded_start
        manifest = {
            "schema": CACHE_SCHEMA,
            "modality": modality,
            "cache_key": cache_key,
            "item_hashes": list(item_hashes),
            "lineage": dict(lineage),
            "token_ids_and_masks_sha256": tokenized_hash.hexdigest(),
            "forward_calls": calls,
            "encoded_examples": encoded,
            "files": files,
        }
        temp_manifest = manifest_path.with_suffix(".json.partial")
        temp_manifest.write_bytes(_canonical_json(manifest) + b"\n")
        os.replace(temp_manifest, manifest_path)
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in paths.items()}
        receipt = {"modality": modality, "cache_key": cache_key,
                   "unique_inputs": n, "cache_hits": 0, "cache_misses": n,
                   "forward_calls_this_capture": calls, "encoded_examples_this_capture": encoded,
                   "original_forward_calls": calls, "original_encoded_examples": encoded,
                   "token_ids_and_masks_sha256": tokenized_hash.hexdigest(),
                   "files": {name: {"path": str(path), "sha256": files[path.name]["sha256"]}
                             for name, path in paths.items()}}
        return arrays["features"], {key: index for index, key in enumerate(item_hashes)}, receipt


@dataclass
class RuntimeState:
    """One encoded state, with candidate-owned raw page features retained by text."""
    pages_by_candidate: dict[str, list[dict[str, Any]]]
    state_token_hashes: tuple[str, ...]


class CachedRuntime:
    """Runtime cache: encode state pages once; each distinct question once; cross pairs jointly."""

    def __init__(self, normalizer: "FeatureNormalizer | None" = None,
                 device: str = "cuda", protocol_path: str | Path = PROTOCOL_PATH):
        self.encoder = FeatureEncoder(device, protocol_path)
        self.normalizer = normalizer
        self.page_cache: dict[str, tuple[str, np.ndarray]] = {}
        self.query_cache: dict[str, tuple[str, np.ndarray]] = {}
        self.counters = {
            "page_forward_calls": 0, "page_encoded_examples": 0,
            "query_forward_calls": 0, "query_encoded_examples": 0,
            "cross_forward_calls": 0, "cross_encoded_examples": 0,
            "page_cache_hits": 0, "query_cache_hits": 0, "cross_cache_hits": 0,
        }
        self._cross_cache: dict[str, tuple[tuple[str, str], np.ndarray]] = {}

    def __enter__(self) -> "CachedRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self.encoder.close()

    def _encode_items(self, modality: str, items: Sequence[tuple[str, str | None]]) -> list[np.ndarray]:
        output: list[np.ndarray] = []
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start:start + BATCH_SIZE]
            ids, masks = self.encoder.tokenize([item[0] for item in batch],
                                               [item[1] for item in batch] if modality == "cross" else None)
            before_calls = self.encoder.forward_counts[modality]
            before_examples = self.encoder.encoded_counts[modality]
            vectors = self.encoder.encode_batch(ids, masks, modality)
            self.counters[f"{modality}_forward_calls"] += self.encoder.forward_counts[modality] - before_calls
            self.counters[f"{modality}_encoded_examples"] += self.encoder.encoded_counts[modality] - before_examples
            output.extend(vectors[index] for index in range(len(vectors)))
        return output

    def encode_state(self, state_blocks: Iterable[Any], page_owners: Mapping[str, str]) -> RuntimeState:
        if not isinstance(page_owners, Mapping):
            raise TypeError("page_owners must map block IDs to candidate IDs")
        grouped: dict[str, list[tuple[str, str]]] = {}
        page_texts: dict[str, str] = {}
        for block in state_blocks:
            block_id = _block_value(block, "id")
            if bool(_block_value(block, "exact", False)) or block_id not in page_owners:
                continue
            owner = page_owners[block_id]
            text = _block_value(block, "text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"page {block_id!r} has no semantic text")
            digest = _text_hash(text)
            old = page_texts.get(digest)
            if old is not None and old != text:
                raise RuntimeError("SHA collision for state page")
            page_texts[digest] = text
            grouped.setdefault(owner, []).append((block_id, text))
        missing = [(digest, text) for digest, text in page_texts.items() if digest not in self.page_cache]
        vectors = self._encode_items("page", [(text, None) for _, text in missing])
        for (digest, text), vector in zip(missing, vectors):
            self.page_cache[digest] = (text, vector)
        self.counters["page_cache_hits"] += len(page_texts) - len(missing)
        pages_by_candidate: dict[str, list[dict[str, Any]]] = {}
        for candidate_id, entries in grouped.items():
            values = []
            for block_id, text in sorted(entries, key=lambda pair: str(pair[0])):
                digest = _text_hash(text)
                cached_text, feature = self.page_cache[digest]
                if cached_text != text:
                    raise RuntimeError("state cache hash collision")
                values.append({"block_id": block_id, "text": text, "raw_feature": feature})
            pages_by_candidate[candidate_id] = values
        # Keep candidates with no owned pages explicit; callers can pass their IDs
        # through candidate_ids to score as UNKNOWN without synthesizing page values.
        return RuntimeState(pages_by_candidate, tuple(sorted(page_texts)))

    def _question(self, question: str) -> np.ndarray:
        if not isinstance(question, str) or not question:
            raise ValueError("question must be nonempty semantic text")
        digest = _text_hash(question)
        old = self.query_cache.get(digest)
        if old is not None:
            if old[0] != question:
                raise RuntimeError("question cache hash collision")
            self.counters["query_cache_hits"] += 1
            return old[1]
        (vector,) = self._encode_items("query", [(question, None)])
        self.query_cache[digest] = (question, vector)
        return vector

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            return np.asarray(values, dtype=np.float32)
        if self.normalizer.mode != "pages":
            raise ValueError("score_question requires the train-only pages normalizer")
        return self.normalizer.transform(values)

    def score_question(self, state: RuntimeState, question: str, reader,
                       candidate_ids: Sequence[str] | None = None,
                       *, uniform_attention: bool = False,
                       zero_question: bool = False,
                       zero_pages: bool = False) -> dict[str, Any]:
        """Score all candidates using cached state pages and one cached query vector."""
        import torch
        try:
            from .ephemeral_pages_model import intervene_features
        except ImportError:
            from ephemeral_pages_model import intervene_features

        query_raw = self._question(question)
        candidates = list(candidate_ids) if candidate_ids is not None else list(state.pages_by_candidate)
        pmax = max(1, max((len(state.pages_by_candidate.get(cid, ())) for cid in candidates), default=0))
        page_raw = np.zeros((len(candidates), pmax, WIDTH), dtype=np.float32)
        mask = np.zeros((len(candidates), pmax), dtype=np.bool_)
        for candidate_index, candidate_id in enumerate(candidates):
            for page_index, page in enumerate(state.pages_by_candidate.get(candidate_id, ())):
                page_raw[candidate_index, page_index] = page["raw_feature"]
                mask[candidate_index, page_index] = True
        query_raw_batch = np.repeat(query_raw[None, :], len(candidates), axis=0)
        q = self._normalize(query_raw_batch)
        pages = self._normalize(page_raw)
        q_tensor = torch.as_tensor(q, dtype=torch.float32, device=self.encoder.device_name)
        page_tensor = torch.as_tensor(pages, dtype=torch.float32, device=self.encoder.device_name)
        raw_q_tensor = torch.as_tensor(query_raw_batch, dtype=torch.float32, device=self.encoder.device_name)
        raw_page_tensor = torch.as_tensor(page_raw, dtype=torch.float32, device=self.encoder.device_name)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.encoder.device_name)
        q_tensor, page_tensor, mask_tensor, raw_q_tensor, raw_page_tensor, uniform_attention = intervene_features(
            q_tensor, page_tensor, mask_tensor, raw_q_tensor, raw_page_tensor,
            zero_question=zero_question, zero_pages=zero_pages,
            uniform_attention=uniform_attention,
        )
        reader = reader.to(self.encoder.device_name).eval()
        with torch.inference_mode():
            output = reader(q_tensor, page_tensor, mask_tensor, raw_q_tensor, raw_page_tensor,
                            uniform_attention=uniform_attention)
        return {
            "candidate_ids": candidates,
            "output": output,
            "raw_query_forward_count": self.counters["query_forward_calls"],
            "raw_page_forward_count": self.counters["page_forward_calls"],
            "counters": dict(self.counters),
        }

    def score_cross(self, state: RuntimeState, question: str, reader,
                    candidate_ids: Sequence[str] | None = None,
                    *, uniform_attention: bool = False) -> dict[str, Any]:
        """Jointly encode every current question/page pair; never substitute cached singles."""
        import torch

        candidates = list(candidate_ids) if candidate_ids is not None else list(state.pages_by_candidate)
        query_hash = _text_hash(question)
        page_rows = [state.pages_by_candidate.get(cid, ()) for cid in candidates]
        pmax = max(1, max((len(pages) for pages in page_rows), default=0))
        pair_raw = np.zeros((len(candidates), pmax, WIDTH), dtype=np.float32)
        mask = np.zeros((len(candidates), pmax), dtype=np.bool_)
        misses: list[tuple[int, int, tuple[str, str], str, str]] = []
        for candidate_index, pages in enumerate(page_rows):
            for page_index, page in enumerate(pages):
                page_hash = _text_hash(page["text"])
                key = sha256_bytes(_canonical_json([query_hash, page_hash]))
                cached = self._cross_cache.get(key)
                if cached is not None:
                    if cached[0] != (question, page["text"]):
                        raise RuntimeError("cross cache hash collision")
                    pair_raw[candidate_index, page_index] = cached[1]
                    self.counters["cross_cache_hits"] += 1
                else:
                    misses.append((candidate_index, page_index, (question, page["text"]),
                                   key, page["text"]))
                mask[candidate_index, page_index] = True
        encoded = self._encode_items("cross", [item[2] for item in misses])
        for item, vector in zip(misses, encoded):
            candidate_index, page_index, pair, key, page_text = item
            self._cross_cache[key] = (pair, vector)
            pair_raw[candidate_index, page_index] = vector
        tensor = torch.as_tensor(pair_raw, dtype=torch.float32, device=self.encoder.device_name)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.encoder.device_name)
        if self.normalizer is not None:
            if self.normalizer.mode != "cross":
                raise ValueError("score_cross requires the train-only cross normalizer")
            tensor = torch.as_tensor(self.normalizer.transform(pair_raw), dtype=torch.float32,
                                     device=self.encoder.device_name)
        reader = reader.to(self.encoder.device_name).eval()
        with torch.inference_mode():
            output = reader(tensor, mask_tensor, uniform_attention=uniform_attention)
        return {"candidate_ids": candidates, "output": output, "counters": dict(self.counters)}


def capture_features(rows: Iterable[Mapping[str, Any] | Any], cache_root: str | Path, *,
                     phase: str, device: str = "cuda",
                     protocol_path: str | Path = PROTOCOL_PATH,
                     selection_calibration_receipt: str | Path | None = None) -> FeatureCorpus:
    """Capture one explicitly scoped phase; final requires a verified receipt."""
    return FeatureStore(cache_root, protocol_path).capture(
        rows, phase=phase, device=device,
        selection_calibration_receipt=selection_calibration_receipt)


class FeatureNormalizer:
    """Train-only per-coordinate feature normalization with the pinned 0.01 floor."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 mode: str = "pages"):
        if mode not in {"pages", "cross"}:
            raise ValueError("normalizer mode must be 'pages' or 'cross'")
        self.mode = mode
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        if self.mean is not None and (self.mean.shape != (WIDTH,) or self.std.shape != (WIDTH,)):
            raise ValueError(f"normalizer arrays must each have shape [{WIDTH}]")
        if self.std is not None and (not np.isfinite(self.std).all() or (self.std < 0.01).any()):
            raise ValueError("normalizer std must be finite and floored at 0.01")

    def fit(self, arrays: Mapping[str, Any], mode: str | None = None) -> "FeatureNormalizer":
        if mode is not None:
            if mode not in {"pages", "cross"}:
                raise ValueError("normalizer mode must be 'pages' or 'cross'")
            self.mode = mode
        required = ("page_mask", "records", "cross") if self.mode == "cross" else (
            "q", "pages", "page_mask", "records")
        missing = [key for key in required if key not in arrays]
        if missing:
            raise KeyError(f"normalizer input missing: {missing}")
        records = arrays["records"]
        train_rows = np.asarray([record.get("split") == "train" for record in records], dtype=np.bool_)
        if not train_rows.any():
            raise ValueError("normalizer requires at least one train-only example")
        if any(record.get("split") not in {"train", "validation", "calibration", "development", "final"}
               for record in records):
            raise ValueError("normalizer split labels are not canonical")
        mask = np.asarray(arrays["page_mask"][train_rows], dtype=np.bool_)
        if self.mode == "cross":
            features = np.asarray(arrays["cross"][train_rows], dtype=np.float64)
            merged = features[mask]
            fit_policy = "train examples only; cross features for valid question/page pairs"
        else:
            q = np.asarray(arrays["q"][train_rows], dtype=np.float64)
            pages = np.asarray(arrays["pages"][train_rows], dtype=np.float64)
            merged = np.concatenate((q, pages[mask]), axis=0)
            fit_policy = "train examples only; concatenated q and valid page feature rows"
        if not len(merged):
            raise ValueError("no train features available for normalization")
        if not np.isfinite(merged).all():
            raise FloatingPointError("train-only features contain nonfinite values")
        self.mean = merged.mean(axis=0).astype(np.float32)
        self.std = np.maximum(merged.std(axis=0, ddof=0), 0.01).astype(np.float32)
        self.fit_policy = fit_policy
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("fit or load the train-only normalizer before transforming")
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != WIDTH:
            raise ValueError(f"feature width must be {WIDTH}")
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def state_dict(self) -> dict[str, Any]:
        if self.mean is None or self.std is None:
            raise RuntimeError("normalizer has not been fitted")
        return {"width": WIDTH, "std_floor": 0.01, "mode": self.mode,
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "fit_policy": self.fit_policy}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".partial")
        temp.write_bytes(_canonical_json(self.state_dict()) + b"\n")
        os.replace(temp, target)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("width") != WIDTH or data.get("std_floor") != 0.01:
            raise ValueError("persisted ECA normalizer does not match the pinned feature contract")
        instance = cls(np.asarray(data["mean"], dtype=np.float32),
                       np.asarray(data["std"], dtype=np.float32),
                       mode=data.get("mode", "pages"))
        instance.fit_policy = data.get("fit_policy")
        return instance
