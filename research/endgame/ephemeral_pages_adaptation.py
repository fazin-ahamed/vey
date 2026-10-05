#!/usr/bin/env python3
"""ECA-2 minimal final-layer encoder adaptation driver: development only.

Implements research/endgame/ephemeral_pages_adaptation_preregistration.json.
One arm, ``pages_adapted_final_layer``: the byte-identical frozen rank-64
PageReader reading page features produced by an encoder whose final layer and
whose final-layer diagonal adapter are trainable. Everything else -- corpus,
splits, authored text, exact compiler, calibration grids, page mask, loss
components, optimizer membership, epoch count, selection rule -- is inherited
from the frozen recipe in ephemeral_pages_train and ephemeral_pages_evaluate.

Scope limits that are structural, not advisory:
  * the ECA-2 final pool is never read, scored, encoded or inspected;
  * calibration fits only the original corrected calibration worlds, and the
    only evaluation phase is development;
  * the five frozen controls are never refitted, reselected or recalibrated --
    their development ledgers are reconstructed from pinned artifacts;
  * every encoder parameter outside deberta.encoder.layer.11 and outside the
    adapter is hash-verified byte-identical to the pinned weight file, after
    every optimizer epoch and again at checkpoint load;
  * stdout carries metadata only: no authored text, question, page, rubric or
    per-example value leaves this process.

Screens A, B, C0 and the gradient-prerequisite CPU C1 ceiling must have passed
with receipts on disk before this driver writes a launch receipt or takes an
optimizer step. This file never weakens a screen or re-authors inputs.
"""
from __future__ import annotations

import os

# Process policy is fixed before NumPy, Torch, Transformers or BLAS load.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from contextlib import nullcontext as _nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ephemeral_pages_adaptation_screens as screens
import ephemeral_pages_capture as capture
import ephemeral_pages_conditional as conditional
import ephemeral_pages_evaluate as evaluator
import ephemeral_pages_model as readers
import ephemeral_pages_train as trainer
from ephemeral_pages_features import FeatureEncoder, WIDTH, sha256_bytes

HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "ephemeral_pages_adaptation_preregistration.json"
PREREG_SHA256 = "2aa64bc77c236192681368a8c83751aeac185414d0cd0660578b2ada9cd7731c"
EXPERIMENT = capture.ATOMIC_EXPERIMENT
DEFAULT_RUN_ROOT = capture.ATOMIC_ROOT / "runs" / "interface-audit-v1" / "adaptation-v1"

ARM = "pages_adapted_final_layer"
ADAPTER_PREFIX = "adapter."          # canonical preregistered name prefix
ADAPTER_STATE_PREFIX = "adapter."     # state_dict keys are adapter.0 .. adapter.5
LAYER11_PREFIX = screens.LAYER11_PREFIX

SEED = 7
EPOCHS = 400
CHUNK_SIZE = 32
WEIGHT_DECAY = 0.0001
READER_LR = screens.READER_LR
ENCODER_LR = screens.ENCODER_LR
PRIMARY_METRIC = "atomic_choice_macro"
CLAIMS_ALPHA = 0.05 / 4
CLAIM_NAMES = ("pages_frozen_baseline", "adapter_blind", "zero_pages", "lexical")
BASELINE_CONTROLS = ("pages", "lexical")
INTERVENTION_ABLATIONS = ("zero_pages", "zero_question", "uniform_attention")
ENCODER_ABLATIONS = ("adapter_blind", "layer11_frozen")
ABLATIONS = ENCODER_ABLATIONS + INTERVENTION_ABLATIONS
ALLOWED_PHASES = ("train", "validation", "calibration", "development")
SCREEN_RECEIPTS = ("screen_a_smoke_receipt.json", "screen_b_immutability_receipt.json",
                   "screen_c0_liveness_restore_receipt.json", "screen_c1_gradient_ceiling_receipt.json")
EVALUATION_PHASES = ("calibration", "development")

SCOPE = {
    "schema": "vey.eca2.final-layer-encoder-adaptation-development.v1",
    "arm": ARM, "phase_scope": "development only", "evidence_class": "HYPOTHESIS",
    "promotion": False, "B_STEF_allowed": False, "endgame_complete": False,
    "final_pool_access": False, "final_outcomes_used_for_selection": False,
    "certificate": "unavailable", "competitor_or_neutral_credit": False,
    "adaptation_surface": "deberta.encoder.layer.11 plus its six rank>=2 diagonal adapters",
    "reader_equations_byte_identical": True,
    "gate_status": "development_screen",
    "mechanism_role": "binding development mechanism decomposition; earns no promotion credit",
}

SOURCE_FILES = ("ephemeral_pages_adaptation.py", "ephemeral_pages_adaptation_screens.py",
                "ephemeral_pages_train.py", "ephemeral_pages_model.py",
                "ephemeral_pages_evaluate.py", "ephemeral_pages_capture.py",
                "ephemeral_pages_features.py", "ephemeral_pages_components.py",
                "ephemeral_pages_conditional.py", "ephemeral_pages_verify.py",
                "ephemeral_pages_protocol.json", "ephemeral_pages_atomic_protocol.json",
                "ephemeral_pages_adaptation_preregistration.json",
                "ephemeral_pages_adaptation_c0_amendment.json",
                "ephemeral_pages_adaptation_prerequisite.py",
                "ephemeral_pages_adaptation_c1_gradient_preregistration.json")


# --------------------------------------------------------------------------- #
# custody                                                                      #
# --------------------------------------------------------------------------- #

def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(evaluator.plain(value), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _preregistration() -> dict:
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _source_hashes() -> dict:
    return {name: trainer._hash_file(HERE / name) for name in SOURCE_FILES}


def _environment() -> dict:
    repo = Path(__file__).resolve().parents[2]
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "numpy": np.__version__, "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                               cwd=repo, text=True).strip(),
        "variables": {key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HOME")},
    }


def _verify_preregistration() -> dict:
    """Fail closed unless this driver runs against the pinned preregistration."""
    if trainer._hash_file(PREREG_PATH) != PREREG_SHA256:
        raise RuntimeError("adaptation preregistration changed after the implementation pin")
    screens.verify_gradient_protocol()
    prereg = _preregistration()
    for name, entry in sorted(prereg["pinned_artifacts"]["implementation"].items()):
        path = Path(entry["path"])
        if not path.exists():
            raise RuntimeError(f"preregistered implementation is missing: {name}")
        if trainer._hash_file(path) != entry["sha256"]:
            raise RuntimeError(f"preregistered implementation changed: {name}")
    encoder = prereg["pinned_artifacts"]["encoder"]
    weight_path = Path(encoder["model_safetensors"]["path"])
    if trainer._hash_file(weight_path) != encoder["model_safetensors"]["sha256"]:
        raise RuntimeError("pinned encoder weight file changed")
    if Path(screens.pinned_weight_path()).resolve() != weight_path.resolve():
        raise RuntimeError("screens module resolves a different pinned weight file")
    # ratio_encoder_to_reader is encoder_lr / reader_lr, not its reciprocal.
    learning = prereg["learning_rates"]
    if (ENCODER_LR != float(learning["encoder_lr"]) or
            READER_LR != float(learning["reader_lr"]) or
            ENCODER_LR != float(learning["adapter_lr"]) or
            ENCODER_LR / READER_LR != float(learning["ratio_encoder_to_reader"])):
        raise RuntimeError("learning rates differ from the preregistered frozen ratio")
    return prereg


def _screen_receipts(root: Path) -> dict:
    """Load the four mandatory screen receipts; every one must be a recorded pass."""
    receipts = {}
    for name in SCREEN_RECEIPTS:
        path = Path(root) / name
        if not path.exists():
            raise RuntimeError(f"mandatory screen receipt is missing: {name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("verdict") != "pass":
            raise RuntimeError(f"mandatory screen did not pass: {name}")
        if name == screens.RECEIPT_NAMES["c0_liveness"]:
            if payload.get("c0_amendment", {}).get("sha256") != screens.C0_AMENDMENT_SHA256:
                raise RuntimeError("C0 receipt does not match the approved amendment")
        if name == screens.RECEIPT_NAMES["c1_adapted_ceiling"]:
            prerequisite, _ = screens.verify_gradient_prerequisite(Path(root))
            if payload.get("gradient_protocol", {}).get("sha256") != screens.C1_GRADIENT_PROTOCOL_SHA256:
                raise RuntimeError("C1 gradient protocol mismatch")
            expected_receipt = evaluator.artifact(Path(root) / "c1-gradient-v1" / "prerequisite_receipt.json")
            if payload.get("prerequisite_receipt") != expected_receipt:
                raise RuntimeError("C1 prerequisite receipt lineage mismatch")
            if payload.get("prerequisite_checkpoint") != prerequisite["checkpoint"]:
                raise RuntimeError("C1 prerequisite checkpoint lineage mismatch")
            if payload.get("population_sha256") != prerequisite["population_sha256"]:
                raise RuntimeError("C1 prerequisite population mismatch")
        receipts[name] = {**evaluator.artifact(path), "verdict": "pass"}
    return receipts


def _inside(path, root) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def _safe_root(run_root, *, smoke: bool = False, experiment=EXPERIMENT) -> Path:
    """Refuse any root inside the sealed final pool, the corpus or the feature caches."""
    root = Path(run_root).resolve()
    if smoke:
        root = root / "smoke"
    protected = (conditional.baseline_root(experiment), experiment.corpus_root,
                 experiment.cache_root)
    for path in protected:
        if _inside(root, path):
            raise ValueError("adaptation output cannot modify baseline, corpus or captured features")
    # The sealed final pool is refused whether or not it exists yet: a run root
    # whose path names it must never be creatable under the ECA-2 data root.
    if "final" in root.parts:
        raise ValueError("adaptation output cannot live inside the sealed final pool")
    for entry in capture.ATOMIC_ROOT.rglob("*"):
        if entry.name in {"final", "final.jsonl"} and (_inside(root, entry) or _inside(entry, root)):
            raise ValueError("adaptation output cannot live inside the sealed final pool")
    return root


# --------------------------------------------------------------------------- #
# phase loading and row evidence                                              #
# --------------------------------------------------------------------------- #

def load_phase(phase: str, experiment=EXPERIMENT) -> dict:
    """Load one non-final phase and attach its unique-text index tables."""
    if phase not in ALLOWED_PHASES:
        raise ValueError("adaptation permits only train/validation/calibration/development")
    data = capture.load_phase(phase, experiment)
    if any(record["split"] != phase for record in data["records"]):
        raise RuntimeError("captured record split mismatch")
    return _attach_index_tables(data)


def _phase_evidence(phase, data, indices, experiment=EXPERIMENT) -> dict:
    return conditional._phase_evidence(phase, data, indices, experiment)


def _unique_text_tables(data) -> dict:
    """Deterministic unique question/page text tables plus per-record row indices."""
    page_text, query_text = {}, {}
    for record in data["records"]:
        for page in record["pages"]:
            page_text.setdefault(sha256_bytes(page["text"].encode("utf-8")), page["text"])
        query_text.setdefault(sha256_bytes(record["question"].encode("utf-8")), record["question"])
    pages = [page_text[key] for key in sorted(page_text)]
    queries = [query_text[key] for key in sorted(query_text)]
    page_order = {sha256_bytes(text.encode("utf-8")): index for index, text in enumerate(pages)}
    query_order = {sha256_bytes(text.encode("utf-8")): index for index, text in enumerate(queries)}
    n, pmax = data["page_mask"].shape
    page_ix = np.zeros((n, pmax), dtype=np.int64)
    query_ix = np.zeros(n, dtype=np.int64)
    for i, record in enumerate(data["records"]):
        query_ix[i] = query_order[sha256_bytes(record["question"].encode("utf-8"))]
        for j, page in enumerate(record["pages"]):
            page_ix[i, j] = page_order[sha256_bytes(page["text"].encode("utf-8"))]
    return {"q_texts": queries, "page_texts": pages, "query_ix": query_ix, "page_ix": page_ix,
            "unique_questions": len(queries), "unique_pages": len(pages)}

def _attach_index_tables(data) -> dict:
    """Attach the unique-text tables and per-record row indices used by the reader."""
    tables = _unique_text_tables(data)
    data["_index_tables"] = tables
    data["_page_ix"] = tables["page_ix"]
    data["_query_ix"] = tables["query_ix"]
    return data


# --------------------------------------------------------------------------- #
# parameter hashing, bounds and the pinned-weight comparison                  #
# --------------------------------------------------------------------------- #

def _parameter_digest(name: str, tensor) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B").tobytes())
    return digest.hexdigest()


def _parameter_digests(module, adapters=None) -> dict:
    """Per-name digests over the adapted encoder in the preregistered namespace.

    Delegated to the screens module so one hashing convention governs the
    launch receipt, the per-epoch screen and the checkpoint. Adapter names are
    the canonical dotted preregistered names, index-aligned with adapter.0..5.
    """
    return screens.parameter_hashes(module, adapters)["per_name_sha256"]


def _is_frozen_name(name: str) -> bool:
    """Only layer 11 and the adapter are trainable; everything else is frozen."""
    return not (name.startswith(LAYER11_PREFIX) or name.startswith(ADAPTER_PREFIX))


def _frozen_subset_hash(per_name: dict) -> str:
    frozen = {name: value for name, value in per_name.items() if _is_frozen_name(name)}
    if not frozen:
        raise RuntimeError("no frozen encoder parameter was found to guard")
    digest = hashlib.sha256()
    for name in sorted(frozen):
        digest.update(f"{name}={frozen[name]}\n".encode("ascii"))
    return digest.hexdigest()



def _trainable_surface_report(model, adapters, reader, prereg: dict) -> dict:
    """Reconcile the live surface against the preregistration and the pinned ledger.

    screens.trainable_report enforces the live counts, both denominators and both
    bounds and fails closed on any escape from layer 11 plus adapter. This adds
    the preregistration cross-check and the metadata-only safetensors header
    comparison, so the bound rests on header metadata and not on a model load.
    """
    report = screens.trainable_report(model, reader)
    surface = prereg["trainable_surface"]
    bound = prereg["trainable_parameter_fraction_bound"]
    if report["layer11_trainable_scalars"] != int(surface["unfrozen_encoder_parameters"]["scalar_count"]):
        raise RuntimeError("live layer-11 scalar count differs from the preregistration")
    if report["adapter_trainable_scalars"] != int(surface["diagonal_adapter"]["scalar_count"]):
        raise RuntimeError("live adapter scalar count differs from the preregistration")
    if report["reader_trainable_scalars"] != int(surface["reader_parameters"]["trainable_scalars"]):
        raise RuntimeError("live reader scalar count differs from the preregistration")
    if (report["encoder_side_trainable_scalars"] != int(surface["encoder_side_trainable_scalars"]) or
            report["combined_trainable_scalars"] != int(surface["combined_trainable_scalars"])):
        raise RuntimeError("live trainable scalar counts differ from the preregistration")
    if (report["denominator_encoder"] != int(bound["denominator_encoder"]) or
            report["denominator_combined"] != int(bound["denominator_combined"])):
        raise RuntimeError("live trainable denominators differ from the preregistration")
    if sorted(adapters) != sorted(screens.ADAPTER_NAMES):
        raise RuntimeError("live adapter names differ from the preregistration")
    for name, tensor in adapters.items():
        if not bool(torch.all(tensor == 1.0)):
            raise RuntimeError("the diagonal adapter is not at its pinned all-ones initialization")
    header = screens.safetensors_ledger()
    ledger = prereg["pinned_artifacts"]["encoder"]["parameter_ledger"]
    if (header.get("stored_f32_scalars") != int(ledger["stored_f32_scalars"]) or
            header.get("layer_11_scalars") != int(ledger["layer_11_scalars"])):
        raise RuntimeError("pinned weight header ledger differs from the preregistration")
    return {**report,
            "layer_11_names": sorted(name for name in report["trainable_names"]
                                     if name.startswith(LAYER11_PREFIX)),
            "adapter_names": sorted(adapters),
            "safetensors_header_ledger": header}


def _verify_against_pinned_weight_file(model) -> dict:
    """At launch every loaded encoder parameter must equal the pinned weight file.

    Layer 11 is included: it is the trainable surface, so its launch bytes are the
    baseline the per-epoch immutability check measures movement against. The
    adapter lives in state_dict under adapter.<index>; the canonical dotted names
    come from the returned mapping, which screens.parameter_hashes supplies.
    """
    from safetensors import safe_open
    digests = {}
    with safe_open(str(screens.pinned_weight_path()), framework="pt") as handle:
        for name, tensor in model.named_parameters():
            if name.startswith(ADAPTER_STATE_PREFIX):
                continue
            try:
                reference = handle.get_tensor(name)
            except Exception as exc:  # any store failure is a custody failure
                raise RuntimeError(f"pinned weight file has no entry for {name}") from exc
            if list(reference.shape) != list(tensor.shape) or reference.dtype != tensor.dtype:
                raise RuntimeError(f"pinned weight shape or dtype differs for {name}")
            if not np.array_equal(reference.numpy(), tensor.detach().cpu().numpy()):
                raise RuntimeError(f"loaded parameter differs from the pinned weight file: {name}")
            digests[name] = _parameter_digest(name, tensor)
    if not any(name.startswith(LAYER11_PREFIX) for name in digests):
        raise RuntimeError("the pinned weight comparison covered no final-layer parameter")
    if not [name for name in digests if _is_frozen_name(name)]:
        raise RuntimeError("the pinned weight comparison covered no frozen parameter")
    return digests




def _epoch_immutability(model, adapters, epoch, launch_digests) -> dict:
    """Screen B at every epoch: frozen bytes unchanged, changes confined to layer 11 plus adapter."""
    digests = _parameter_digests(model, adapters)
    frozen_now = {name: value for name, value in digests.items() if _is_frozen_name(name)}
    frozen_before = {name: value for name, value in launch_digests.items() if _is_frozen_name(name)}
    if frozen_now != frozen_before:
        changed = sorted(name for name in set(frozen_now) | set(frozen_before)
                         if frozen_now.get(name) != frozen_before.get(name))
        raise RuntimeError(f"frozen encoder parameter changed at epoch {epoch}: {changed}")
    changed = sorted(name for name in set(digests) | set(launch_digests)
                     if digests.get(name) != launch_digests.get(name))
    allowed = {name for name in set(launch_digests) | set(adapters)
               if name.startswith(LAYER11_PREFIX) or name.startswith(ADAPTER_PREFIX)}
    if not set(changed) <= allowed:
        raise RuntimeError(f"parameter changed outside the final layer plus adapter at "
                           f"epoch {epoch}: {sorted(set(changed) - allowed)}")
    return {"epoch": epoch, "frozen_parameter_sha256": _frozen_subset_hash(digests),
            "changed_final_layer_names": sorted(name for name in changed
                                                if name.startswith(LAYER11_PREFIX)),
            "changed_adapter_names": sorted(name for name in changed
                                             if name.startswith(ADAPTER_PREFIX)),
            "adapter_names": sorted(adapters)}




# --------------------------------------------------------------------------- #
# adapted forward, feature tables and the objective                            #
# --------------------------------------------------------------------------- #

def _encode_table(model, encoder, texts, device, *, grad: bool, counters: dict) -> torch.Tensor:
    """Encode unique texts through the adapted surface with the frozen pooling policy."""
    if not texts:
        return torch.zeros((0, WIDTH), dtype=torch.float32, device=device)
    ids, masks = encoder.tokenize(texts)
    rows = []
    with (torch.enable_grad() if grad else torch.no_grad()):
        for start in range(0, len(texts), CHUNK_SIZE):
            stop = min(len(texts), start + CHUNK_SIZE)
            tokens = torch.as_tensor(np.array(ids[start:stop], copy=True), dtype=torch.long,
                                    device=device)
            attention = torch.as_tensor(np.array(masks[start:stop], copy=True), dtype=torch.long,
                                        device=device)
            values = model(input_ids=tokens, attention_mask=attention).to(dtype=torch.float32)
            if values.ndim != 2 or values.shape != (stop - start, WIDTH):
                raise RuntimeError("the adapted encoder returned an unexpected feature shape")
            counters["forward_calls"] += 1
            counters["encoded_examples"] += int(stop - start)
            rows.append(values)
    table = torch.cat(rows, dim=0)
    if not bool(torch.isfinite(table).all()):
        raise FloatingPointError("the adapted encoder produced nonfinite FP32 features")
    return table


def _stage_tables(data, model, encoder, device, *, grad: bool, counters: dict,
                  index_tables: dict) -> dict:
    return {"q": _encode_table(model, encoder, index_tables["q_texts"], device, grad=grad,
                               counters=counters),
            "pages": _encode_table(model, encoder, index_tables["page_texts"], device, grad=grad,
                                   counters=counters)}


def _reader_forward(reader, tables, arrays, indices, normalizer, device, *, intervention=None):
    """The frozen reader equations over adapted features.

    Normalization and the zero_question/zero_pages interventions are applied in
    the same order as ephemeral_pages_train._forward: features are normalized
    first, then the frozen intervene_features zeroes reader space and the raw
    cosine features together. Deviating would silently redefine the ablations.
    """
    if intervention not in {None, *INTERVENTION_ABLATIONS}:
        raise ValueError(f"unknown intervention {intervention!r}")
    mask = torch.as_tensor(np.array(arrays["page_mask"][indices], copy=True), device=device)
    page_ix = torch.as_tensor(np.asarray(arrays["_page_ix"][indices], dtype=np.int64), device=device)
    query_ix = torch.as_tensor(np.asarray(arrays["_query_ix"][indices], dtype=np.int64), device=device)
    # Page slots repeat one text across candidates, so gather with advanced
    # indexing; index_select only accepts a 1-D index vector.
    raw_pages = tables["pages"][page_ix]
    raw_q = tables["q"][query_ix]
    mean_p = torch.as_tensor(normalizer["pages"]["mean"], device=device)
    std_p = torch.as_tensor(normalizer["pages"]["std"], device=device)
    mean_q = torch.as_tensor(normalizer["q"]["mean"], device=device)
    std_q = torch.as_tensor(normalizer["q"]["std"], device=device)
    pages = (raw_pages - mean_p) / std_p
    q = (raw_q - mean_q) / std_q
    q, pages, mask, raw_q, raw_pages, uniform = readers.intervene_features(
        q, pages, mask, raw_q, raw_pages,
        zero_question=intervention == "zero_question",
        zero_pages=intervention == "zero_pages",
        uniform_attention=intervention == "uniform_attention")
    return reader(q, pages, mask, raw_q, raw_pages, uniform_attention=uniform)


def _objective(reader, tables, arrays, indices, denominators, chunk_size, swap_baseline,
               normalizer, *, backward: bool, counters: dict) -> dict:
    """Exact full-batch component means accumulated before one optimizer step."""
    totals = dict.fromkeys(trainer.COMPONENTS, 0.0)
    device = reader.bk.device
    for ix in trainer._chunks(indices, chunk_size):
        trainer._resource_guard(torch, swap_baseline)
        targets = trainer._batch(arrays, ix, device)
        output = _reader_forward(reader, tables, arrays, ix, normalizer, device)
        losses = readers.eca_loss(output, targets, directed_grade=False)
        counts = trainer._counts(targets, "pages")
        weighted = []
        for key in trainer.COMPONENTS:
            if counts[key] and denominators[key]:
                term = losses[key] * (counts[key] / denominators[key])
                if not torch.isfinite(term):
                    raise FloatingPointError(f"nonfinite {ARM} {key} objective")
                totals[key] += float(term.detach())
                weighted.append(term)
        if backward and weighted:
            sum(weighted).backward()
            counters["backward_calls"] += 1
    totals["total"] = sum(totals.values())
    return totals



# --------------------------------------------------------------------------- #
# reader initialization and gradient checks                                    #
# --------------------------------------------------------------------------- #

def _baseline_files(prereg: dict) -> dict:
    files = {}
    for name, entry in sorted(prereg["pinned_artifacts"]["checkpoints"].items()):
        path = Path(entry["path"])
        if not path.exists() or trainer._hash_file(path) != entry["sha256"]:
            raise RuntimeError(f"pinned baseline checkpoint changed: {name}")
        files[name] = evaluator.artifact(path)
    return files


def _initial_reader(prereg: dict, device):
    """Reader initialization is pinned to the corrected ECA-2 pages checkpoint."""
    checkpoint = torch.load(Path(prereg["pinned_artifacts"]["checkpoints"]["pages"]["path"]),
                            map_location="cpu", weights_only=False)
    if (checkpoint.get("protocol_sha256") != EXPERIMENT.protocol()[1] or
            checkpoint.get("experiment_context") != EXPERIMENT.context()):
        raise RuntimeError("the pinned pages checkpoint does not belong to this experiment")
    reader = readers.PageReader()
    reader.load_state_dict(checkpoint["state_dict"], strict=True)
    return reader.to(device).train(), checkpoint["normalizer"]


def _gradient_report(model, adapters, reader) -> dict:
    named = {f"encoder.{name}": tensor for name, tensor in model.named_parameters()
             if tensor.requires_grad}
    named.update(adapters)
    named.update({f"reader.{name}": tensor for name, tensor in reader.named_parameters()
                  if tensor.requires_grad})
    missing, nonfinite = [], []
    total = 0.0
    for name in sorted(named):
        grad = named[name].grad
        if grad is None:
            missing.append(name)
            continue
        if not bool(torch.isfinite(grad).all()):
            nonfinite.append(name)
            continue
        total += float(torch.sum(grad.detach().to(torch.float64) ** 2))
    return {"trainable_tensors": len(named), "missing_gradients": missing,
            "nonfinite_gradients": nonfinite, "gradient_norm": math.sqrt(total)}


def _require_finite_gradients(model, adapters, reader, epoch) -> dict:
    """Every trainable tensor must carry a finite gradient before the step is taken."""
    report = _gradient_report(model, adapters, reader)
    if report["missing_gradients"] or report["nonfinite_gradients"] or \
            not math.isfinite(report["gradient_norm"]):
        raise RuntimeError(f"gradient screen failed closed at epoch {epoch}: {report}")
    return {"epoch": epoch, "gradient_norm": report["gradient_norm"],
            "trainable_tensors": report["trainable_tensors"]}


# --------------------------------------------------------------------------- #
# training                                                                    #
# --------------------------------------------------------------------------- #

def _fit_surface(root: Path, encoder, device, chunk_size, epochs, *, normalizer, prereg,
                 screen_receipts, evidence, sources, environment, launcher) -> dict:
    """Joint reader plus final-layer adaptation fit under the frozen recipe."""
    trainer._seed()
    model, adapters = screens.attach_trainable_surface(torch, encoder, device)
    reader, pinned_normalizer = _initial_reader(prereg, device)
    for key in ("q", "pages"):
        expected, fitted = pinned_normalizer[key], normalizer[key]
        if (fitted["count"] != expected["count"] or
                not np.array_equal(fitted["mean"], expected["mean"]) or
                not np.array_equal(fitted["std"], expected["std"])):
            raise RuntimeError("the adaptation normalizer differs from the frozen train normalizer")
    surface = _trainable_surface_report(model, adapters, reader, prereg)
    launch_digests = _verify_against_pinned_weight_file(model)
    train, validation = evidence.pop("arrays")
    train_ix, val_ix = evidence.pop("indices")
    train_den = trainer._denominators(train, train_ix, "pages", chunk_size)
    val_den = trainer._denominators(validation, val_ix, "pages", chunk_size)
    # The adapters already appear in model.named_parameters() as adapter.0..5
    # with requires_grad True; adding them again would put duplicate tensors in
    # one parameter group and step them twice per epoch.
    encoder_side = [tensor for tensor in model.parameters() if tensor.requires_grad]
    expected_tensors = len(surface["layer_11_names"]) + len(adapters)
    if len(encoder_side) != expected_tensors:
        raise RuntimeError(f"encoder-side optimizer group holds {len(encoder_side)} tensors, "
                           f"expected {expected_tensors}")
    if len({id(tensor) for tensor in encoder_side}) != len(encoder_side):
        raise RuntimeError("a trainable tensor appears twice in the encoder-side group")
    optimizer = torch.optim.AdamW(
        [{"params": encoder_side, "lr": ENCODER_LR},
         {"params": [tensor for tensor in reader.parameters() if tensor.requires_grad],
          "lr": READER_LR}],
        weight_decay=WEIGHT_DECAY)
    recipe = {"seed": SEED, "epochs": epochs, "optimizer": "AdamW",
              "reader_lr": READER_LR, "encoder_lr": ENCODER_LR, "adapter_lr": ENCODER_LR,
              "weight_decay": WEIGHT_DECAY, "lr_ratio_encoder_to_reader": ENCODER_LR / READER_LR,
              "lr_ratio_search": False, "steps_per_epoch": 1, "chunk_size": chunk_size,
              "selection": "earliest minimum held-world validation objective",
              "components": list(trainer.COMPONENTS),
              "normalizer": "train-only optimizer rows; pinned 0.01 floor"}
    receipt = {
        "schema": "vey.eca2.adaptation-launch.v1", **SCOPE,
        "launched_by": launcher, "device": str(device),
        "preregistration_sha256": PREREG_SHA256, "protocol_sha256": EXPERIMENT.protocol()[1],
        "experiment_context": EXPERIMENT.context(),
        "git_commit": environment["git_commit"], "git_branch": environment["git_branch"],
        "source_sha256": sources, "environment": environment,
        "resources": encoder.priority, "screen_receipts": screen_receipts,
        "encoder": {key: prereg["pinned_artifacts"]["encoder"][key] for key in
                    ("repo", "revision", "pinned_encoder_parameters_sha256",
                     "pinned_tokenizer_sha256", "pinned_weight_sha256")},
        "encoder_forwards_at_launch": 0,
        "pooling": "FP32 final-layer masked mean over non-padding tokens; width 384",
        "native_pooler_unused": True, "classifier_unused": True,
        "reader": {"equations": "frozen rank-64 PageReader; byte-identical",
                   "initialization": "pinned corrected ECA-2 pages checkpoint"},
        "baseline_checkpoint_files": _baseline_files(prereg),
        "recipe": recipe,
        "normalizer_sha256": _hash_json(normalizer),
        "component_denominators": {"train": train_den, "validation": val_den},
        "trainable_surface": surface,
        "frozen_parameter_sha256": _frozen_subset_hash(launch_digests),
        "row_evidence": evidence,
        "selection_uses_development": False, "calibration_uses_development": False,
        "final_pool_access": False,
    }
    evaluator.write_json(root / "launch_receipt.json", receipt)
    counters = {"forward_calls": 0, "encoded_examples": 0, "backward_calls": 0}
    history, immutability = [], []
    best = float("inf")
    selected = None
    initial_reader_sha256 = trainer._parameter_hash(reader)
    started = time.time_ns()
    for epoch in range(1, epochs + 1):
        model.train()
        reader.train()
        optimizer.zero_grad(set_to_none=True)
        tables = _stage_tables(train, model, encoder, device, grad=True, counters=counters,
                              index_tables=train["_index_tables"])
        train_loss = _objective(reader, tables, train, train_ix, train_den, chunk_size,
                                encoder.initial_swap_mib, normalizer, backward=True,
                                counters=counters)
        if epoch == 1:
            gradient_screen = _require_finite_gradients(model, adapters, reader, epoch)
        optimizer.step()
        del tables
        immutability.append(_epoch_immutability(model, adapters, epoch, launch_digests))
        model.eval()
        reader.eval()
        with torch.no_grad():
            tables = _stage_tables(validation, model, encoder, device, grad=False,
                                  counters=counters, index_tables=validation["_index_tables"])
            val_loss = _objective(reader, tables, validation, val_ix, val_den, chunk_size,
                                  encoder.initial_swap_mib, normalizer, backward=False,
                                  counters=counters)
            del tables
        history.append({"epoch": epoch, "train": train_loss, "validation": val_loss})
        if val_loss["total"] < best:
            best = val_loss["total"]
            selected = {"epoch": epoch, "validation_objective": best,
                        "model_state": {key: value.detach().cpu().clone()
                                        for key, value in model.state_dict().items()},
                        "adapters": {name: tensor.detach().cpu().clone()
                                     for name, tensor in adapters.items()},
                        # Canonical dotted names, index-aligned with adapter.0..5
                        # in model_state, so a reader can address either spelling.
                        "adapter_names": tuple(screens.ADAPTER_NAMES),
                        "reader_state": {key: value.detach().cpu().clone()
                                         for key, value in reader.state_dict().items()}}
    if selected is None:
        raise RuntimeError("no finite adaptation checkpoint was selected")
    result = {"selected": selected, "history": history, "immutability": immutability,
              "gradient_screen": gradient_screen,
              "counters": counters, "surface": surface, "launch_digests": launch_digests,
              "initial_reader_sha256": initial_reader_sha256, "receipt": receipt,
              "train_den": train_den, "val_den": val_den,
              "seconds": time.time_ns() - started, "normalizer": normalizer}
    del optimizer, model, adapters, reader
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def train_adapted(run_root=DEFAULT_RUN_ROOT, device: str = "cuda", *,
                  launcher: str = "train_adapted") -> Path:
    """Write the launch receipt before the first optimizer step, then fit 400 epochs."""
    prereg = _verify_preregistration()
    root = _safe_root(run_root)
    if any((root / name).exists() for name in ("launch_receipt.json", "calibration.json", "evaluation")):
        raise FileExistsError("refusing to replace adaptation artifacts")
    screen_receipts = _screen_receipts(root)
    sources = _source_hashes()
    environment = _environment()
    with FeatureEncoder(device, EXPERIMENT.protocol_path) as encoder:
        train = load_phase("train")
        validation = load_phase("validation")
        conditional.leaf_truth_guard(train, "train", EXPERIMENT)
        conditional.leaf_truth_guard(validation, "validation", EXPERIMENT)
        train_ix = trainer.optimizer_indices(train, "train", EXPERIMENT)
        val_ix = trainer.optimizer_indices(validation, "validation", EXPERIMENT)
        normalizer = trainer.fit_normalizer(train, train_ix, CHUNK_SIZE)
        # Row evidence covers all four child-local phases: train and validation
        # selection, plus the calibration and development pools the fit never
        # selects on. Selection uses no calibration or development index.
        calibration = load_phase("calibration")
        development = load_phase("development")
        evidence = {
            "arrays": (train, validation), "indices": (train_ix, val_ix),
            "train": _phase_evidence("train", train, train_ix),
            "validation": _phase_evidence("validation", validation, val_ix),
            "calibration": _phase_evidence("calibration", calibration,
                                           np.arange(len(calibration["records"]))),
            "development": _phase_evidence("development", development,
                                           np.arange(len(development["records"]))),
            "normalizer_train_indices_sha256": _hash_json(train_ix.tolist()),
        }
        del calibration, development
        # The run root already holds the three mandatory screen receipts.
        root.mkdir(parents=True, exist_ok=True)
        result = _fit_surface(root, encoder, device, CHUNK_SIZE, EPOCHS, normalizer=normalizer,
                              prereg=prereg, screen_receipts=screen_receipts, evidence=evidence,
                              sources=sources, environment=environment, launcher=launcher)
    _persist_checkpoint(root, result, prereg, sources, environment)
    return root


# --------------------------------------------------------------------------- #
# checkpoint custody                                                          #
# --------------------------------------------------------------------------- #

def _persist_checkpoint(root: Path, result: dict, prereg: dict, sources: dict,
                        environment: dict) -> dict:
    selected = result["selected"]
    model_parameters = {name: tensor for name, tensor in selected["model_state"].items()
                        if not name.startswith(ADAPTER_PREFIX)}
    checkpoint = {
        "schema": "vey.eca2.adaptation-checkpoint.v1", **SCOPE,
        "eligible_for_calibration_or_development": True,
        "arm": ARM, "epoch": int(selected["epoch"]),
        "validation_objective": float(selected["validation_objective"]),
        "protocol_sha256": EXPERIMENT.protocol()[1], "preregistration_sha256": PREREG_SHA256,
        "experiment_context": EXPERIMENT.context(),
        "screen_receipts": result["receipt"]["screen_receipts"],
        "source_sha256": sources, "environment": environment,
        "launch_receipt": evaluator.artifact(root / "launch_receipt.json"),
        "baseline_checkpoint_files": _baseline_files(prereg),
        "trainable_surface": result["surface"],
        "frozen_parameter_sha256": _frozen_subset_hash(result["launch_digests"]),
        "final_layer_parameter_sha256": _hash_json(
            {name: _parameter_digest(name, tensor) for name, tensor in model_parameters.items()
             if name.startswith(LAYER11_PREFIX)}),
        "adapter_parameter_sha256": _hash_json(
            {name: _parameter_digest(name, tensor)
             for name, tensor in selected["adapters"].items()}),
        "adapter_init_policy": "all ones; identity at initialization",
        "reader_parameter_sha256": _hash_json(
            {name: _parameter_digest(name, tensor)
             for name, tensor in selected["reader_state"].items()}),
        "normalizer_sha256": _hash_json(result["normalizer"]),
        "normalizer": result["normalizer"],
        # state_dict() already carries adapter.0..5 index-aligned with
        # adapter_names; merging the canonical-name mapping in would inject
        # dotted adapter.* keys and break strict load_state_dict.
        "state_dict": dict(selected["model_state"]),
        "adapter_names": list(selected["adapter_names"]),
        "reader_state_dict": selected["reader_state"],
        "recipe": result["receipt"]["recipe"],
        "row_evidence": result["receipt"]["row_evidence"],
    }
    path = root / f"{ARM}.pt"
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
        stream.flush()
        os.fsync(stream.fileno())
    record = {
        "schema": "vey.eca2.adaptation-training.v1", **SCOPE,
        "arm": ARM, "protocol_sha256": EXPERIMENT.protocol()[1],
        "preregistration_sha256": PREREG_SHA256, "recipe": result["receipt"]["recipe"],
        "selection": "earliest minimum held-world validation objective",
        "selected_epoch": int(selected["epoch"]),
        "selected_validation_objective": float(selected["validation_objective"]),
        "initial_reader_sha256": result["initial_reader_sha256"],
        "trainable_surface": result["surface"],
        "component_denominators": {"train": result["train_den"], "validation": result["val_den"]},
        "encoder_counters": result["counters"], "elapsed_ns": result["seconds"],
        "immutability_checks": result["immutability"],
        "gradient_screen": result["gradient_screen"],
        "screen_receipts": result["receipt"]["screen_receipts"],
        "row_evidence": result["receipt"]["row_evidence"],
        "source_sha256": sources, "environment": environment,
        "environment_sha256": _hash_json(environment),
        "frozen_parameter_sha256": checkpoint["frozen_parameter_sha256"],
        "final_layer_parameter_sha256": checkpoint["final_layer_parameter_sha256"],
        "adapter_parameter_sha256": checkpoint["adapter_parameter_sha256"],
        "reader_parameter_sha256": checkpoint["reader_parameter_sha256"],
        "checkpoint": evaluator.artifact(path),
        "history": result["history"],
    }
    evaluator.write_json(root / f"{ARM}_history.json", record)
    return record


def load_adapted(root, device: str = "cpu", encoder=None) -> dict:
    """Reload the adapted surface; verify frozen bytes, adapter, epoch and protocol.

    The caller owns the FeatureEncoder so the CUDA lock is taken exactly once.
    """
    root = Path(root)
    history = json.loads((root / f"{ARM}_history.json").read_text(encoding="utf-8"))
    if evaluator.artifact(root / f"{ARM}.pt") != history["checkpoint"]:
        raise RuntimeError("the adaptation checkpoint changed after fitting")
    if history["screen_receipts"] != _screen_receipts(root):
        raise RuntimeError("the mandatory screen receipts changed after fitting")
    if history["source_sha256"] != _source_hashes():
        raise RuntimeError("the adaptation implementation changed after fitting")
    checkpoint = torch.load(root / f"{ARM}.pt", map_location="cpu", weights_only=False)
    if (checkpoint["arm"] != ARM or checkpoint.get("smoke") or
            checkpoint["protocol_sha256"] != EXPERIMENT.protocol()[1] or
            checkpoint["preregistration_sha256"] != PREREG_SHA256 or
            checkpoint["experiment_context"] != EXPERIMENT.context()):
        raise RuntimeError("the adaptation checkpoint protocol or preregistration mismatch")
    prereg = _preregistration()
    if _baseline_files(prereg) != checkpoint["baseline_checkpoint_files"]:
        raise RuntimeError("the pinned baseline checkpoint changed after fitting")
    with FeatureEncoder(device, EXPERIMENT.protocol_path) if encoder is None else _nullcontext(encoder):
        model, adapters = screens.attach_trainable_surface(torch, encoder, device)
        launch_digests = _verify_against_pinned_weight_file(model)
        if _frozen_subset_hash(launch_digests) != checkpoint["frozen_parameter_sha256"]:
            raise RuntimeError("frozen encoder parameter bytes differ from the fitted launch")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        # Adapter bytes are addressed by the canonical dotted names in the
        # returned mapping; state_dict carries them as adapter.0..5.
        adapter_digests = {name: _parameter_digest(name, tensor)
                           for name, tensor in adapters.items()}
        if _hash_json(adapter_digests) != checkpoint["adapter_parameter_sha256"]:
            raise RuntimeError("the adapter parameter hash mismatches at load")
        final_layer = {name: _parameter_digest(name, tensor)
                       for name, tensor in model.state_dict().items()
                       if name.startswith(LAYER11_PREFIX)}
        if _hash_json(final_layer) != checkpoint["final_layer_parameter_sha256"]:
            raise RuntimeError("the final-layer parameter hash mismatches at load")
        reader = readers.PageReader()
        reader.load_state_dict(checkpoint["reader_state_dict"], strict=True)
        reader_digests = {name: _parameter_digest(name, tensor)
                          for name, tensor in reader.state_dict().items()}
        if _hash_json(reader_digests) != checkpoint["reader_parameter_sha256"]:
            raise RuntimeError("the reader parameter hash mismatches at load")
    return {"model": model.to(device), "adapters": adapters, "reader": reader.to(device).eval(),
            "normalizer": checkpoint["normalizer"], "epoch": int(checkpoint["epoch"]),
            "checkpoint": checkpoint, "launch_digests": launch_digests, "encoder": encoder,
            "resources": encoder.priority}


# --------------------------------------------------------------------------- #
# prediction and evaluation-time ablations                                    #
# --------------------------------------------------------------------------- #

def _predict(state, data, *, intervention=None) -> readers.ReaderOutput:
    model, adapters, reader = state["model"], state["adapters"], state["reader"]
    device = reader.bk.device
    n, pmax = data["page_mask"].shape
    ix = np.arange(n, dtype=np.int64)
    fields = [[] for _ in readers.ReaderOutput._fields]
    counters = {"forward_calls": 0, "encoded_examples": 0, "backward_calls": 0}
    model.eval()
    reader.eval()
    tables = _stage_tables(data, model, state["encoder"], device, grad=False, counters=counters,
                          index_tables=data["_index_tables"])
    with torch.no_grad():
        for chunk in trainer._chunks(ix, CHUNK_SIZE):
            output = _reader_forward(reader, tables, data, chunk, state["normalizer"], device,
                                     intervention=intervention)
            for position, value in enumerate(output):
                data_array = value.detach().cpu().numpy()
                if data_array.ndim == 2 and data_array.shape[1] < pmax:
                    fill = -np.inf if position == 2 else 0
                    data_array = np.pad(data_array, ((0, 0), (0, pmax - data_array.shape[1])),
                                        constant_values=fill)
                fields[position].append(data_array)
    del tables
    state["encoder_counters"] = counters
    return readers.ReaderOutput(*(np.concatenate(values, axis=0) if values else
                                  np.empty((0,) if i < 2 else (0, pmax), dtype=np.float32)
                                  for i, values in enumerate(fields)))


def _snapshot_surface(state) -> dict:
    """Clone every adapted-encoder tensor, including adapter.0..5.

    state_dict() already carries the adapters, so a single key namespace makes
    snapshot and restore symmetric; mixing canonical adapter names in here would
    address tensors the model never looks up.
    """
    return {name: tensor.detach().clone()
            for name, tensor in state["model"].state_dict().items()}


def _restore_surface(state, snapshot: dict) -> None:
    with torch.no_grad():
        for name, tensor in state["model"].state_dict().items():
            if name not in snapshot:
                raise RuntimeError(f"the ablation snapshot is missing {name}")
            tensor.copy_(snapshot[name])




def _apply_ablation(state, name: str) -> dict:
    """Evaluation-time only: no refit, no reselection, no recalibration."""
    receipt = {"ablation": name, "evaluation_time_only": True,
               "encoder_parameters_modified": False, "reader_parameters_modified": False}
    if name == "adapter_blind":
        with torch.no_grad():
            for tensor in state["adapters"].values():
                tensor.fill_(1.0)
        receipt.update(effect="every adapter scalar forced to exactly 1.0",
                       encoder_parameters_modified=True)
    elif name == "layer11_frozen":
        from safetensors import safe_open
        model_state = state["model"].state_dict()
        with safe_open(str(screens.pinned_weight_path()), framework="pt") as handle, torch.no_grad():
            for key in model_state:
                if key.startswith(LAYER11_PREFIX):
                    model_state[key].copy_(handle.get_tensor(key))
        receipt.update(effect="final layer reverted to the pinned frozen weights; adapter retained",
                       encoder_parameters_modified=True)
    elif name in INTERVENTION_ABLATIONS:
        receipt.update(effect="frozen reader-space intervention; encoder and adapter untouched")
    else:
        raise ValueError(f"unknown ablation {name!r}")
    return receipt


# --------------------------------------------------------------------------- #
# mechanism claims                                                            #
# --------------------------------------------------------------------------- #

def _paired_deltas(adapted, baselines: dict, weights) -> dict:
    """Reuse the conditional paired-difference arithmetic, relabelled for this arm."""
    raw = conditional.paired_deltas(adapted, baselines, weights)
    return {key.replace("conditional_minus_", ARM + "_minus_"): value for key, value in raw.items()}


def _claims(adapted, targets: dict, weights) -> dict:
    result = {}
    for name in CLAIM_NAMES:
        other = targets[name]
        left, right = adapted.values[PRIMARY_METRIC], other.values[PRIMARY_METRIC]
        if set(left) != set(right):
            raise RuntimeError(f"paired comparison changed the fixed family population: {name}")
        samples = adapted.estimate(PRIMARY_METRIC, weights) - other.estimate(PRIMARY_METRIC, weights)
        valid = samples[np.isfinite(samples)]
        lower = float(np.quantile(valid, CLAIMS_ALPHA)) if len(valid) else None
        result[name] = {"metric": PRIMARY_METRIC,
                        "point": float(adapted.estimate(PRIMARY_METRIC) -
                                       other.estimate(PRIMARY_METRIC)),
                        "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                        "one_sided_Bonferroni_lower": lower, "alpha": CLAIMS_ALPHA,
                        "valid_bootstrap_draws": len(valid), "families": sorted(left),
                        "pass": lower is not None and lower > 0}
    result["familywise_alpha"] = 0.05
    result["claims"] = len(CLAIM_NAMES)
    result["pass"] = all(result[name]["pass"] for name in CLAIM_NAMES)
    result["unit"] = "whole world; variants clustered"
    result["bootstrap"] = {"draws": evaluator.BOOTSTRAPS, "seed": evaluator.BOOTSTRAP_SEED}
    return result


def _descriptive_deltas(adapted, ledgers: dict, weights) -> dict:
    """Preregistered descriptive-only contrasts; never part of the Bonferroni family."""
    result = {}
    for name, other in ledgers.items():
        if other is adapted:
            continue
        samples = adapted.estimate(PRIMARY_METRIC, weights) - other.estimate(PRIMARY_METRIC, weights)
        valid = samples[np.isfinite(samples)]
        result[name] = {"point": float(adapted.estimate(PRIMARY_METRIC) -
                                       other.estimate(PRIMARY_METRIC)),
                        "CI95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                        "one_sided95_lower": float(np.quantile(valid, .05)) if len(valid) else None,
                        "valid_bootstrap_draws": len(valid), "status": "descriptive_only"}
    return result


def _world_paired_summary(decisions: list[dict], baseline_path: Path) -> dict:
    """Descriptive paired-by-world contrast against the frozen corrected pages reader.

    Reuses the pinned baseline decision rows already verified by
    conditional._saved_baseline; no baseline model is run and no metric is refit.
    """
    base = {}
    with Path(baseline_path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                base[row["row_id"]] = row
    per_world = {}
    for row in decisions:
        if not evaluator.is_primary(row):
            continue
        other = base.get(row["row_id"])
        if other is None:
            raise RuntimeError("the frozen pages baseline lacks a primary development row")
        per_world.setdefault(row["world_id"], []).append(
            int(row["exact_winner_set_correct"]) - int(other["exact_winner_set_correct"]))
    if not per_world:
        raise RuntimeError("the paired-by-world contrast has no primary worlds")
    means = [sum(values) / len(values) for values in per_world.values()]
    return {"metric": PRIMARY_METRIC, "paired_by": "world", "unit": "whole world",
            "world_count": len(means),
            "mean_world_delta": float(np.mean(means)),
            "discordant_worlds": sum(1 for values in per_world.values() if any(values)),
            "improved_worlds": sum(1 for value in means if value > 0),
            "harmed_worlds": sum(1 for value in means if value < 0),
            "baseline_decisions": evaluator.artifact(baseline_path),
            "status": "descriptive_only",
            "note": "preregistered descriptive-only contrast; outside the Bonferroni family"}


# --------------------------------------------------------------------------- #
# calibration and development evaluation                                      #
# --------------------------------------------------------------------------- #

def _baseline_ledgers(phase_data, ir, worlds, families, baseline_manifest, baseline_calibration):
    ledgers, evidence = {}, {}
    for name in BASELINE_CONTROLS:
        ledgers[name], evidence[name] = conditional._saved_baseline(
            name, phase_data, ir, worlds, families, baseline_manifest,
            baseline_calibration["controls"][name], EXPERIMENT)
    return ledgers, evidence


def evaluate_adapted(root, device: str = "cuda") -> Path:
    """Calibrate on the corrected calibration worlds only; evaluate on development only."""
    prereg = _verify_preregistration()
    cfg, protocol_hash = EXPERIMENT.protocol()
    root = _safe_root(root)
    if any((root / name).exists() for name in ("calibration.json", "evaluation")):
        raise FileExistsError("refusing to replace adaptation evaluation artifacts")
    sources = _source_hashes()
    environment = _environment()
    baseline_root = conditional.baseline_root(EXPERIMENT)
    baseline_files = conditional._baseline_checkpoints(EXPERIMENT)
    baseline_calibration_path = baseline_root / "calibration.json"
    baseline_calibration = json.loads(baseline_calibration_path.read_text(encoding="utf-8"))
    if baseline_calibration["protocol_sha256"] != protocol_hash:
        raise RuntimeError("the baseline calibration protocol mismatch")
    baseline_manifest_path = baseline_root / "evaluation/development/evaluation.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    if (baseline_manifest["phase"] != "development" or
            baseline_manifest["protocol_sha256"] != protocol_hash or
            baseline_manifest["checkpoint_files"] != baseline_files):
        raise RuntimeError("the baseline development manifest mismatch")
    reports = {}
    with FeatureEncoder(device, EXPERIMENT.protocol_path) as encoder:
        state = load_adapted(root, device, encoder=encoder)
        calibration = None
        for phase in EVALUATION_PHASES:
            trainer._resource_guard(torch, encoder.initial_swap_mib)
            data = load_phase(phase)
            ir = evaluator.load_ir(phase, experiment=EXPERIMENT)
            if {record["row_id"] for record in data["records"]} != set(ir):
                raise RuntimeError("the adaptation capture and DecisionIR populations differ")
            worlds = sorted({record["world_id"] for record in data["records"]})
            families = cfg["corpus"]["families"]
            expected = set(families[:12 if phase == "calibration" else 16])
            observed = {family for row in ir.values() if row["metadata"]["variant"] == "base"
                        and row["metadata"]["query_kind"] == "atomic"
                        and evaluator.UNKNOWN not in row["gold"]
                        for family in row["metadata"]["family"]}
            if len(worlds) != cfg["corpus"]["worlds"][phase] or observed != expected:
                raise RuntimeError("the adaptation evaluation changed the frozen populations")
            directory = root / "evaluation" / phase
            directory.mkdir(parents=True, exist_ok=False)
            weights, bootstrap = evaluator.bootstrap_indices(directory, worlds)
            source = {"control": ARM, **SCOPE, "arm": ARM, "phase": phase,
                      "protocol_sha256": protocol_hash, "preregistration_sha256": PREREG_SHA256,
                      "source_sha256": sources,
                      "checkpoint": evaluator.artifact(root / f"{ARM}.pt"),
                      "selected_epoch": state["epoch"],
                      "adapter_names": sorted(state["adapters"]),
                      "reader_equations": "frozen rank-64 PageReader; byte-identical"}
            snapshot = _snapshot_surface(state)
            output = _predict(state, data)
            fields = evaluator.prediction_fields(output, data)
            if phase == "calibration":
                calibration = evaluator.calibrate(fields, data, ir)
            cal = calibration
            decisions = evaluator.aggregate(fields, data, ir, cal)
            for row in decisions:
                row["source"] = "semantic_adapted_final_layer"
                row["source_metadata"] = source
            annotated = {**data, "records": [{**record, "source_metadata": source}
                                             for record in data["records"]]}
            saved = evaluator.save_predictions(directory, ARM, fields, annotated, ir, decisions,
                                               cal["ordinal_sigma"])
            with (directory / f"{ARM}_reader_output.npz").open("xb") as stream:
                np.savez(stream, **{key: np.asarray(getattr(output, key))
                                    for key in readers.ReaderOutput._fields})
                stream.flush()
                os.fsync(stream.fileno())
            saved["reader_output"] = evaluator.artifact(directory / f"{ARM}_reader_output.npz")
            ledger, pairs = evaluator.metric_ledger("pages", fields, data, ir, decisions,
                                                    cal["ordinal_sigma"], families, worlds)
            summaries = ledger.summaries(weights)
            for prefix in ("raw_distribution_", "calibrated_distribution_",
                           "OOD_raw_distribution_", "OOD_calibrated_distribution_"):
                summaries[prefix + "ECE"] = evaluator.ece_summary(ledger, weights, prefix)
            saved["world_metric_ledger"] = evaluator.write_json(
                directory / f"{ARM}_world_metrics.json", ledger.ledger())
            with (directory / f"{ARM}_paired_interventions.jsonl").open("x", encoding="utf-8") as stream:
                for pair in pairs:
                    stream.write(json.dumps(evaluator.plain(pair), sort_keys=True,
                                            allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            saved["paired_interventions"] = evaluator.artifact(
                directory / f"{ARM}_paired_interventions.jsonl")
            del output, fields, annotated
            ablations, ablation_ledgers = {}, {}
            if phase == "development":
                for name in ABLATIONS:
                    _restore_surface(state, snapshot)
                    receipt = _apply_ablation(state, name)
                    result = _predict(state, data,
                                      intervention=name if name in INTERVENTION_ABLATIONS else None)
                    arm_fields = evaluator.prediction_fields(result, data)
                    arm_decisions = evaluator.aggregate(arm_fields, data, ir, cal)
                    ablation_source = {**source, "ablation": name, **receipt}
                    for row in arm_decisions:
                        row["source"] = "semantic_adapted_final_layer_ablation"
                        row["source_metadata"] = ablation_source
                    arm_data = {**data, "records": [{**record, "source_metadata": ablation_source}
                                                    for record in data["records"]]}
                    arm_saved = evaluator.save_predictions(directory, name, arm_fields, arm_data, ir,
                                                           arm_decisions, cal["ordinal_sigma"])
                    arm_ledger, _ = evaluator.metric_ledger("pages", arm_fields, data, ir,
                                                            arm_decisions, cal["ordinal_sigma"],
                                                            families, worlds)
                    arm_saved["world_metric_ledger"] = evaluator.write_json(
                        directory / f"{name}_world_metrics.json", arm_ledger.ledger())
                    ablations[name] = {**receipt, "metrics": arm_ledger.summaries(weights),
                                       "artifacts": arm_saved,
                                       "recording": "evaluation-time only; no refit, "
                                                    "reselection or recalibration"}
                    ablation_ledgers[name] = arm_ledger
                    del result, arm_fields, arm_decisions, arm_data
                _restore_surface(state, snapshot)
            claims, descriptive, baseline_pairs, baseline_evidence = None, {}, None, None
            if phase == "development":
                baseline_ledgers, baseline_evidence = _baseline_ledgers(
                    data, ir, worlds, families, baseline_manifest, baseline_calibration)
                targets = {"pages_frozen_baseline": baseline_ledgers["pages"],
                           "lexical": baseline_ledgers["lexical"],
                           "adapter_blind": ablation_ledgers["adapter_blind"],
                           "zero_pages": ablation_ledgers["zero_pages"]}
                claims = _claims(ledger, targets, weights)
                baseline_pairs = _paired_deltas(ledger, baseline_ledgers, weights)
                descriptive = _descriptive_deltas(ledger, {
                    "layer11_frozen": ablation_ledgers["layer11_frozen"],
                    "zero_question": ablation_ledgers["zero_question"],
                    "uniform_attention": ablation_ledgers["uniform_attention"]}, weights)
                descriptive.update(baseline_pairs)
                descriptive[ARM + "_paired_by_world"] = _world_paired_summary(
                    decisions, conditional.baseline_root(EXPERIMENT) /
                    "evaluation/development/pages_decisions.jsonl")
            report = {
                "schema": "vey.eca2.adaptation-evaluation.v1", **SCOPE,
                "phase": phase, "gate_status": "development_screen",
                "source_metadata": source, "calibration": calibration, "metrics": summaries,
                "gate_screens": {ARM: evaluator.gate_report(summaries, cfg)},
                "ablations": ablations, "mechanism_claims": claims,
                "descriptive_deltas": descriptive or None,
                "paired_deltas": baseline_pairs,
                "artifacts": saved, "bootstrap": bootstrap,
                "phase_evidence": _phase_evidence(phase, data, np.arange(len(data["records"]))),
                "baseline_saved_output_evidence": baseline_evidence,
                "baseline_checkpoint_files": baseline_files,
                "baseline_development_manifest": evaluator.artifact(baseline_manifest_path),
                "baseline_calibration": evaluator.artifact(baseline_calibration_path),
                "controls_refitted": 0, "controls_recalibrated": 0, "controls_reselected": 0,
                "environment": environment, "resources": encoder.priority,
                "screen_receipts": _screen_receipts(root),
                "encoder_counters_this_phase": state.get("encoder_counters"),
                "statistical_scope": "development mechanism decomposition conditioned on the "
                                     "fixed authored inventory; no promotion inference",
                "final_pool_access": False,
            }
            reports[phase] = evaluator.write_json(directory / "evaluation.json", report)
            if phase == "calibration":
                evaluator.write_json(root / "calibration.json", {
                    "schema": "vey.eca2.adaptation-calibration.v1", **SCOPE,
                    "protocol_sha256": protocol_hash, "arm": ARM, "calibration": calibration,
                    "fit_split": "calibration",
                    "knownness_grid": "unchanged frozen 0.05..0.95 step 0.05",
                    "ordinal_sigma_grid": list(evaluator.SIGMAS),
                    "development_used_for_calibration": False,
                    "checkpoint": evaluator.artifact(root / f"{ARM}.pt"),
                    "evaluation": reports[phase]})
            del data, ir, ledger, decisions, snapshot
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
        if conditional._baseline_checkpoints(EXPERIMENT) != baseline_files:
            raise RuntimeError("a frozen control changed during the adaptation evaluation")
    return evaluator.write_json(root / "evaluation_manifest.json", {
        "schema": "vey.eca2.adaptation-audit.v1", **SCOPE,
        "protocol_sha256": protocol_hash, "preregistration_sha256": PREREG_SHA256,
        "reports": reports, "source_sha256": _source_hashes(),
        "checkpoint": evaluator.artifact(root / f"{ARM}.pt"),
        "training_history": evaluator.artifact(root / f"{ARM}_history.json"),
        "launch_receipt": evaluator.artifact(root / "launch_receipt.json")})


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--screen", choices=("smoke", "full"), default="full",
                        help="smoke is owned by ephemeral_pages_adaptation_screens; full runs "
                             "the development fit and evaluation")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.screen == "smoke" and args.evaluate_only:
        parser.error("smoke checkpoints are not eligible for evaluation")
    if args.screen == "smoke":
        parser.error("run A/B/C0/C1 with ephemeral_pages_adaptation_screens and the bounded "
                     "train-only prerequisite with ephemeral_pages_adaptation_prerequisite; "
                     "this driver runs the full development fit")
    root = _safe_root(args.run_root)
    if not args.evaluate_only:
        root = train_adapted(args.run_root, args.device)
    evaluate_adapted(root, args.device)
    history = json.loads((root / f"{ARM}_history.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    development = json.loads(Path(manifest["reports"]["development"]["path"]).read_text(encoding="utf-8"))
    gates = development["gate_screens"][ARM]
    claims = development["mechanism_claims"]
    surface = history["trainable_surface"]
    print(json.dumps({
        "run_root": str(root), "arm": ARM, "device": args.device,
        "experiment_context": EXPERIMENT.context(),
        "preregistration_sha256": PREREG_SHA256, "protocol_sha256": EXPERIMENT.protocol()[1],
        "source_sha256": _source_hashes(),
        "screen_receipts": {name: entry["sha256"] for name, entry in history["screen_receipts"].items()},
        "recipe": history["recipe"],
        "trainable_surface": {key: surface[key] for key in
                              ("encoder_side_trainable_scalars", "combined_trainable_scalars",
                               "trainable_fraction_encoder_side", "trainable_fraction_combined",
                               "bound_encoder_side_fraction", "bound_combined_fraction")},
        "selected_epoch": history["selected_epoch"],
        "epochs_run": len(history["history"]),
        "frozen_parameter_sha256": history["frozen_parameter_sha256"],
        "final_layer_parameter_sha256": history["final_layer_parameter_sha256"],
        "adapter_parameter_sha256": history["adapter_parameter_sha256"],
        "immutability_checks": len(history["immutability_checks"]),
        "checkpoint": history["checkpoint"],
        "gate_status": development["gate_status"],
        "gate_pass": {name: value.get("pass") for name, value in sorted(gates.items())
                      if name != "exact_literals"},
        "mechanism_claims": {name: {"pass": claims[name]["pass"],
                                    "one_sided_Bonferroni_lower":
                                        claims[name]["one_sided_Bonferroni_lower"],
                                    "point": claims[name]["point"]} for name in CLAIM_NAMES},
        "mechanism_claim_set_pass": claims["pass"],
        "ablations": sorted(development["ablations"]),
        "controls_refitted": development["controls_refitted"],
        "final_pool_access": False, "promotion": False, "B_STEF_allowed": False,
        "endgame_complete": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())