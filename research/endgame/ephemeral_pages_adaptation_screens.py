#!/usr/bin/env python3
"""ECA-2 minimal final-layer encoder adaptation: mandatory screens A, B, C0, C1.

Implements, in order, the fail-closed screens of
``ephemeral_pages_adaptation_preregistration.json`` as corrected by
``ephemeral_pages_adaptation_screen_c_amendment.json``:

  A smoke fit      ``--screen smoke --run-root PATH``
  B immutability   ``--screen immutability --run-root PATH``
  C0 identity      ``--screen c0-identity --run-root PATH --features-root PATH``
  C1 adapted ceiling  ``--screen ceiling --run-root PATH --features-root PATH``

Every screen writes exactly one receipt with exclusive-create (``open("x")``)
semantics under the run root.  A protocol violation raises ``RuntimeError``
after the failing receipt is retained; a legitimate negative screen outcome
(C1 below its predeclared threshold) is written as ``"verdict": "fail"`` and
returned with exit code 3 instead of raising.  The process exits 0 only for a
passing screen.  C1 requires a passing C0 receipt in the same run root.

Screens never open the ECA-2 final pool: every phase load is restricted to
train/validation/calibration/development and the pinned atomic corpus/feature
manifests are hash-checked against the preregistration before use.

C0 is CPU-only, takes zero optimizer steps, builds no reader and makes no
quality claim.  It encodes the unique supervised train and development page
texts with the adapted final layer at its init identity (adapter exactly 1.0,
final layer at the pinned frozen weights), requires that this reproduces the
frozen captured page features within 1e-5 absolute, and requires that a
deliberately perturbed adapter demonstrably moves those features, which
separates an adapter-path failure from ordinary FP32 CPU/GPU drift.

C1 is CPU-only and builds no reader.  It first runs one bounded FP64
closed-form bias-free adaptation of layer 11 on the unique supervised train
page texts only, with no selection, no early stopping, no checkpoint choice
and no tuning.  Layer 11 computes ``LayerNorm(W . activation + residual)``
and a masked mean is linear, so the exact masked mean of that pre-norm sum is
``W . masked_mean(activation) + masked_mean(residual)``; solving for ``W``
therefore changes the representation rather than the readout.  It then encodes
the development unique page texts with the adapted layer and runs the same
finite bias-free FP64 least-squares procedure as
``ephemeral_pages_components.run_linear``, emitting pass or fail against the
pinned development unique-text family-macro threshold.  Validation is never
opened in C0 or C1 and can neither satisfy nor fail either screen.

All arithmetic that already exists in the pinned modules is imported rather
than reimplemented: ``features.FeatureEncoder`` for the pinned load, tokenizer
and resource policy; ``trainer`` for optimizer selection, the train-only
normalizer, chunking, target batches, component counts and denominators;
``readers`` for the frozen rank-64 equations and the equal-weight objective;
``components`` for the supervised unique-text population, the FP64 design and
the solver diagnostics; ``conditional`` for the child-local truth guard.

Metadata-only stdout; authored text, questions, pages and per-example values
are never printed.
"""
from __future__ import annotations

import os

# Process policy before NumPy, Torch or BLAS libraries are imported.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_HOME", "/home/fazinahamed/Documents/vey-data/decisionmix/hf-home")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import linalg  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
CANONICAL_ROOT = Path("/home/fazinahamed/Documents/vey")
if str(CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(CANONICAL_ROOT))

from vey_u.semantic.models import masked_mean  # noqa: E402

import ephemeral_pages_capture as capture  # noqa: E402
import ephemeral_pages_components as components  # noqa: E402
import ephemeral_pages_conditional as conditional  # noqa: E402
import ephemeral_pages_evaluate as evaluator  # noqa: E402
import ephemeral_pages_features as features  # noqa: E402
import ephemeral_pages_model as readers  # noqa: E402
import ephemeral_pages_train as trainer  # noqa: E402

PREREGISTRATION_PATH = HERE / "ephemeral_pages_adaptation_preregistration.json"
PREREGISTRATION_SHA256 = "2aa64bc77c236192681368a8c83751aeac185414d0cd0660578b2ada9cd7731c"
SCREEN_C_DEFECT_PATH = HERE / "ephemeral_pages_adaptation_screen_c_defect.json"
SCREEN_C_DEFECT_SHA256 = "7e494efc3a09e630c1f052b07e5ffa93dddb61edaf324f6b7e37bd9bf9eb57a5"
SCREEN_C_AMENDMENT_PATH = HERE / "ephemeral_pages_adaptation_screen_c_amendment.json"
SCREEN_C_AMENDMENT_SHA256 = "81a76d9eeed58745a0215106dfb7f6e1cf245607d10f4b9644d9a4b165304e2c"

# Pinned implementation sources: byte-identical to the preregistration pins.
PINNED_SOURCES = {
    "ephemeral_pages_model.py": "faabad5a6d024ca28faa16b70f66057aa2fd469268cbe13c3386a6ac2c087c14",
    "ephemeral_pages_features.py": "0b11703840366161f16a8755f15657612ffdf9d58c5101dcab2f0cf0b3ee31f3",
    "ephemeral_pages_train.py": "3fd63dee39e6a265e6bb087d7d209ea687f5da83154b47daad8e46343795130a",
    "ephemeral_pages_evaluate.py": "5d55d0066035c19b899b5b89479d30cdf58b2e5d65f76a596187ecf5723f5d4b",
    "ephemeral_pages_capture.py": "3749020ae76f5b45939f9672468f6228cc6ac7643397eda57fe5954bdc986be6",
    "ephemeral_pages_components.py": "c54cd718ab17f8d0510988b6bea18de943433d5e94b56e762d844dbb0e1df160",
    "ephemeral_pages_conditional.py": "fd04eeafe443ce3a7f55d3e85ce31e2f68e2abeed03b122df6706b413efa6810",
    "ephemeral_pages_verify.py": "79632aaed00f974b74a1614bfd352b35494672ee99ecaac2c560eabc175e3ad6",
}

ATOMIC_CORPUS_PINS = {
    "train": "2a200ea08068ba6d7029db75236b615d4bb294c3975c8d12df7d1b5721da0a38",
    "validation": "8c8c89cf89d5f0c2face02b8eef3f30f3d208ed932b5663de0c7a365b337922a",
    "calibration": "81a82aec29079f9709965c72e168e9f7a1dc3172be20ff2de6c3646ffb990dcb",
    "development": "c40c4d7c7838bae4f66d887e531cdb5ace387ac8bb17b4c0f41ba4c2615fea6c",
}
ATOMIC_FEATURE_MANIFEST_PINS = {
    "train": "cc84e1ba2d5028d9e7de8389d26ef9c9bdc8295b175b1ae62b7259ac92bc7ff8",
    "validation": "ca32ebfaaeed898fdfb7c62cf23fb6a11262bef8d5d7bffe6e9f2327de3ec79a",
    "calibration": "b383d8cbac991ef77eb2925129fdf9bd6f15bb5469f3f2440d83c374ba1e20e2",
    "development": "79afa10a043a50e5605d25c19bc4144cf1a5a08b9cc9b4d61f2a9634760eedba",
}
ATOMIC_FEATURES_ROOT = capture.ATOMIC_ROOT / "features"
PAGES_CHECKPOINT = capture.ATOMIC_ROOT / "runs" / "seed7" / "pages.pt"
PAGES_CHECKPOINT_SHA256 = "052fcf1657bcdfb21ac1d78914b8452002446df768bbc9d670ca7946756bbdcf"

ENCODER_REPO = "cross-encoder/nli-deberta-v3-xsmall"
ENCODER_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
ENCODER_WEIGHT_SHA256 = "4e4fc4977f8d29d2a164255c8f69b9d6c158deeb309bb5e70445b94666ccd9e9"
SAFETENSORS_HEADER_SHA256 = "1bb797470a42527cc3b5fcf4ddb6fd7cb5e1827eaf62824fe64e7a54b9e40068"
SAFETENSORS_HEADER_BYTES = 24648
PINNED_ENCODER_PARAMETERS_SHA256 = "6176abe0b5f5be31cb3201078e71af2a639a0cf32d808aa48f98b530780912a8"

WIDTH = features.WIDTH
MAX_TOKENS = features.MAX_TOKENS
BATCH_SIZE = features.BATCH_SIZE
# The preregistration names parameters in the safetensors namespace
# ("deberta.encoder.layer.11.").  ``AutoModel.from_pretrained`` strips that
# prefix, so the live module spells the same tensors "encoder.layer.11.".
# ``AdaptedEncoder`` registers the base model as ``self.deberta``, which
# restores the preregistered spelling for the adapted surface.
LAYER11_PREFIX = "deberta.encoder.layer.11."
LAYER11_LIVE_PREFIX = "encoder.layer.11."
WRAPPER_NAMESPACE = "deberta."
ADAPTER_PREFIX = "adapter."
LAYER11_TENSOR_COUNT = 16
LAYER11_SCALARS = 1774464
ADAPTER_SCALARS = 1769472
ENCODER_SIDE_TRAINABLE_SCALARS = 3543936
READER_TRAINABLE_SCALARS = 74116
COMBINED_TRAINABLE_SCALARS = 3618052
DENOMINATOR_ENCODER = 70682112
DENOMINATOR_COMBINED = 70756228
BOUND_ENCODER_SIDE_FRACTION = 0.055
BOUND_COMBINED_FRACTION = 0.06

# Ascending lexicographic adapter order over the six rank>=2 weight matrices.
ADAPTER_NAMES = (
    "adapter.deberta.encoder.layer.11.attention.output.dense.weight",
    "adapter.deberta.encoder.layer.11.attention.self.key_proj.weight",
    "adapter.deberta.encoder.layer.11.attention.self.query_proj.weight",
    "adapter.deberta.encoder.layer.11.attention.self.value_proj.weight",
    "adapter.deberta.encoder.layer.11.intermediate.dense.weight",
    "adapter.deberta.encoder.layer.11.output.dense.weight",
)
ADAPTER_TARGETS = tuple(name[len(ADAPTER_PREFIX):] for name in ADAPTER_NAMES)
# The safetensors-spelled targets above are how the preregistration and the
# ledger name them; the live loaded module is addressed without the prefix.
ADAPTER_TARGETS_LIVE = tuple(name[len(ADAPTER_PREFIX) + len(WRAPPER_NAMESPACE):]
                             for name in ADAPTER_NAMES)

READER_LR = 0.01
ENCODER_LR = 0.0001
ADAPTER_LR = 0.0001
RATIO_ENCODER_TO_READER = 0.01
WEIGHT_DECAY = 0.0001
SEED = 7
CHUNK_SIZE = 32

SMOKE_RECORDS_BOUND = 64
SMOKE_EPOCHS_BOUND = 2
IDENTITY_TOLERANCE = 1e-5
SCREEN_C_THRESHOLD = 0.3332298906765268
SCREEN_C_FROZEN_MAE = 0.3832298906765268

ALLOWED_PHASES = ("train", "validation", "calibration", "development")
SCREEN_C_PHASES = ("train", "development")

RECEIPT_NAMES = {
    "smoke": "screen_a_smoke_receipt.json",
    "immutability": "screen_b_immutability_receipt.json",
    "c0_identity": "screen_c0_identity_receipt.json",
    "c1_adapted_ceiling": "screen_c1_adapted_ceiling_receipt.json",
}
SCREEN_ORDER = ("smoke", "immutability", "c0_identity", "c1_adapted_ceiling")
SMOKE_CHECKPOINT_NAME = "screen_a_smoke_checkpoint.pt"

_PHASE_CACHE: dict[tuple[str, str], tuple[dict, dict]] = {}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def write_receipt(path: str | Path, payload: dict) -> dict:
    """Persist one receipt with exclusive-create semantics."""
    target = Path(path)
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite existing receipt: {target}") from exc
    return {"path": str(target.resolve()), "sha256": features.sha256_file(target)}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to("cpu", copy=True).contiguous().numpy()
    return memoryview(array).cast("B").tobytes()


def _entry_bytes(name: str, parameter: torch.Tensor) -> bytes:
    return (name.encode("utf-8") + str(parameter.dtype).encode("ascii")
            + repr(tuple(int(dim) for dim in parameter.shape)).encode("ascii")
            + _tensor_bytes(parameter))


def pinned_weight_path() -> Path:
    """Resolve the pinned local encoder weight file and verify its digest."""
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(ENCODER_REPO, "model.safetensors", revision=ENCODER_REVISION,
                                local_files_only=True))
    digest = features.sha256_file(path)
    _require(digest == ENCODER_WEIGHT_SHA256,
             f"pinned encoder weight hash mismatch: {digest}")
    return path


def safetensors_ledger(path: str | Path | None = None) -> dict:
    """Metadata-only parameter ledger read from the pinned safetensors header."""
    weight_path = pinned_weight_path() if path is None else Path(path)
    _require(features.sha256_file(weight_path) == ENCODER_WEIGHT_SHA256,
             "safetensors ledger opened a non-pinned weight file")
    with weight_path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        raw = stream.read(header_length)
    header = json.loads(raw)
    header_sha = hashlib.sha256(struct.pack("<Q", header_length) + raw).hexdigest()
    _require(header_sha == SAFETENSORS_HEADER_SHA256, "safetensors header digest changed")
    _require(8 + header_length == SAFETENSORS_HEADER_BYTES,
             "safetensors header length changed")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    f32 = {name: value for name, value in tensors.items() if value["dtype"] == "F32"}

    def scalars(value: dict) -> int:
        count = 1
        for dim in value["shape"]:
            count *= int(dim)
        return count

    layer11 = {name: value for name, value in tensors.items() if name.startswith(LAYER11_PREFIX)}
    lower = {name: value for name, value in tensors.items()
             if any(name.startswith(f"deberta.encoder.layer.{index}.") for index in range(11))}
    targets = tuple(sorted(name for name, value in layer11.items() if len(value["shape"]) >= 2))
    ledger = {
        "path": str(weight_path.resolve()),
        "sha256": features.sha256_file(weight_path),
        "header_sha256": header_sha,
        "header_bytes": 8 + header_length,
        "stored_tensors": len(tensors),
        "stored_f32_tensors": len(f32),
        "stored_f32_scalars": sum(scalars(value) for value in f32.values()),
        "stored_non_f32_scalars": sum(scalars(value) for name, value in tensors.items()
                                      if value["dtype"] != "F32"),
        "layer_11_tensors": len(layer11),
        "layer_11_scalars": sum(scalars(value) for value in layer11.values()),
        "layers_0_10_tensors": len(lower),
        "layers_0_10_scalars": sum(scalars(value) for value in lower.values()),
        "non_layer_11_f32_scalars": sum(scalars(value) for name, value in f32.items()
                                        if not name.startswith(LAYER11_PREFIX)),
        "adapter_targets": list(targets),
        "adapter_target_scalars": sum(scalars(layer11[name]) for name in targets),
    }
    _require(ledger["stored_tensors"] == 203 and ledger["stored_f32_tensors"] == 202
             and ledger["stored_f32_scalars"] == 70831107
             and ledger["stored_non_f32_scalars"] == 512,
             "safetensors metadata ledger does not reconcile with the preregistration")
    _require(ledger["layer_11_tensors"] == LAYER11_TENSOR_COUNT
             and ledger["layer_11_scalars"] == LAYER11_SCALARS
             and ledger["layers_0_10_tensors"] == 176
             and ledger["layers_0_10_scalars"] == 19519104
             and ledger["non_layer_11_f32_scalars"] == 69056643,
             "safetensors layer ledger does not reconcile with the preregistration")
    _require(targets == ADAPTER_TARGETS,
             "rank>=2 final-layer weight matrices changed; refusing to widen the adapter")
    _require(ledger["adapter_target_scalars"] == ADAPTER_SCALARS,
             "adapter scalar count does not reconcile with the preregistration")
    return ledger


def _check_pinned_sources() -> dict:
    hashes = {}
    for name, expected in PINNED_SOURCES.items():
        actual = features.sha256_file(HERE / name)
        _require(actual == expected, f"pinned implementation source changed: {name}")
        hashes[name] = actual
    return hashes


def _experiment(features_root: str | Path | None) -> capture.Experiment:
    experiment = capture.resolve_experiment(atomic=True, features_root=features_root)
    for phase in ALLOWED_PHASES:
        _require(features.sha256_file(experiment.corpus_root / f"{phase}.jsonl")
                 == ATOMIC_CORPUS_PINS[phase],
                 f"pinned corrected corpus changed: {phase}")
        _require(features.sha256_file(experiment.cache_root / f"{phase}_manifest.json")
                 == ATOMIC_FEATURE_MANIFEST_PINS[phase],
                 f"pinned corrected feature manifest changed: {phase}")
    return experiment


def _load_phase(phase: str, experiment: capture.Experiment) -> tuple[dict, dict]:
    """Load one open phase through the child-local truth guard; never final."""
    _require(phase in ALLOWED_PHASES, f"forbidden phase requested: {phase}")
    key = (phase, str(experiment.cache_root))
    cached = _PHASE_CACHE.get(key)
    if cached is not None:
        return cached
    data = conditional.load_phase(phase, experiment)
    receipt = conditional.leaf_truth_guard(data, phase, experiment)
    _require(data.get("_leaf_truth_verified") is True,
             "child-local truth guard did not certify the phase")
    evidence = {"phase": phase, "truth_receipt": receipt,
                "feature_manifest": evaluator.artifact(experiment.cache_root / f"{phase}_manifest.json"),
                "corpus": evaluator.artifact(experiment.corpus_root / f"{phase}.jsonl"),
                "capture_counters": data["counters"], "feature_lineage": data["lineage"]}
    _PHASE_CACHE[key] = (data, evidence)
    return data, evidence


def _safe_root(run_root: str | Path, experiment: capture.Experiment) -> Path:
    root = Path(run_root).resolve()
    protected = (experiment.corpus_root.resolve(), experiment.cache_root.resolve(),
                 PAGES_CHECKPOINT.parent.resolve())
    for path in protected:
        _require(root != path and path not in root.parents,
                 f"screen output may not modify pinned data: {root}")
    _require("final" not in root.parts, "screen output may not live under a final artifact path")
    root.mkdir(parents=True, exist_ok=True)
    return root


class AdaptedEncoder(torch.nn.Module):
    """Pinned encoder whose final layer carries one diagonal adapter per weight.

    ``weight_effective = weight * adapter`` for the six rank>=2 weight matrices
    of ``deberta.encoder.layer.11``; every other parameter and every other layer
    is untouched.  The base weight stays the original ``nn.Parameter`` at its
    original name, and each adapter is a separate parameter in ``adapter.*``.
    Pooling is the canonical ``masked_mean`` used by the frozen capture.
    """

    def __init__(self, encoder: torch.nn.Module, adapters: "torch.nn.ParameterList"):
        super().__init__()
        self.deberta = encoder
        self.adapter = adapters
        self.adapter_names = tuple(ADAPTER_NAMES)

    def adapters(self) -> dict:
        return {name: parameter for name, parameter in zip(self.adapter_names, self.adapter)}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean(output.last_hidden_state, attention_mask)


def _module_for(base: torch.nn.Module, dotted: str) -> torch.nn.Module:
    """Resolve a dotted path to its owning module.

    Adapter targets and layer names are spelled as parameter names and so end
    in ".weight".  Walking that suffix reaches the bare Parameter instead of
    the nn.Linear that owns it, so the trailing parameter name is dropped and
    resolved separately.
    """
    parts = dotted.split(".")
    if parts[-1].endswith("weight") or parts[-1].endswith("bias"):
        parts = parts[:-1]
    module = base
    for part in parts:
        _require(hasattr(module, part), f"pinned encoder has no module {dotted}")
        module = getattr(module, part)
    return module


def attach_trainable_surface(torch_module, encoder, device: str) -> tuple[AdaptedEncoder, dict]:
    """Attach the preregistered trainable surface to a loaded pinned encoder.

    ``encoder`` is a ``features.FeatureEncoder`` (already frozen and moved to
    ``device``).  Returns the adapted encoder and a live ``{adapter name:
    Parameter}`` mapping; the adapters also appear in ``state_dict()`` under
    ``adapter.<index>`` in ``ADAPTER_NAMES`` order.
    """
    base = encoder.model
    _require(isinstance(base, torch.nn.Module), "attach requires the loaded pinned encoder")
    for name, parameter in base.named_parameters():
        parameter.requires_grad_(name.startswith(LAYER11_LIVE_PREFIX))
    adapters = torch.nn.ParameterList()
    for target in ADAPTER_TARGETS_LIVE:
        module = _module_for(base, target)
        _require(isinstance(module, torch.nn.Linear),
                 f"adapted weight is not an nn.Linear: {target}")
        adapter = torch.nn.Parameter(torch.ones_like(module.weight, dtype=torch.float32))
        _require(bool(torch.equal(adapter.detach(), torch.ones_like(adapter))),
                 "adapter must initialize to exactly 1.0")
        adapters.append(adapter)
        _install_scaled_forward(module, adapter)
    adapted = AdaptedEncoder(base, adapters).to(device)
    adapted.eval()
    mapping = adapted.adapters()
    _require(tuple(mapping) == ADAPTER_NAMES, "adapter naming order changed")
    _require(sum(parameter.numel() for parameter in mapping.values()) == ADAPTER_SCALARS,
             "adapter scalar count changed")
    return adapted, mapping


def _install_scaled_forward(module: torch.nn.Linear, adapter: torch.nn.Parameter) -> None:
    """Apply ``F.linear(x, weight * adapter, bias)`` at every call site."""
    def forward(input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(input, module.weight * adapter, module.bias)
    module.forward = forward  # instance attribute; nn.Module.__call__ resolves self.forward


def module_hashes(module: torch.nn.Module, adapters: Mapping[str, torch.nn.Parameter] | None = None,
                  prefix: str = "") -> dict:
    """(name, dtype, shape, raw bytes) hashes over a module's parameters.

    ``prefix`` is prepended to every name so an adapted wrapper hashes in the
    preregistered ``deberta.`` namespace while the bare loaded module hashes
    under its own names.  Both spellings cover the same tensors.
    """
    mapping = {} if adapters is None else dict(adapters)
    full = hashlib.sha256()
    outside = hashlib.sha256()
    final = hashlib.sha256()
    adapter_digest = hashlib.sha256()
    outside_names, final_names = [], []
    outside_scalars = final_scalars = 0
    per_name = {}
    for name, parameter in sorted(module.named_parameters()):
        named = prefix + name
        payload = _entry_bytes(named, parameter)
        per_name[named] = hashlib.sha256(payload).hexdigest()
        full.update(payload)
        if name.startswith(LAYER11_LIVE_PREFIX):
            final.update(payload)
            final_names.append(named)
            final_scalars += int(parameter.numel())
        else:
            outside.update(payload)
            outside_names.append(named)
            outside_scalars += int(parameter.numel())
    for name in sorted(mapping):
        payload = _entry_bytes(name, mapping[name])
        per_name[name] = hashlib.sha256(payload).hexdigest()
        adapter_digest.update(payload)
    return {
        "full_sha256": full.hexdigest(),
        "outside_layer11_sha256": outside.hexdigest(),
        "layer11_sha256": final.hexdigest(),
        "adapters_sha256": adapter_digest.hexdigest(),
        "outside_layer11_parameter_count": len(outside_names),
        "outside_layer11_scalars": outside_scalars,
        "layer11_parameter_count": len(final_names),
        "layer11_scalars": final_scalars,
        "adapter_count": len(mapping),
        "adapter_scalars": sum(int(parameter.numel()) for parameter in mapping.values()),
        "outside_layer11_names": outside_names,
        "layer11_names": final_names,
        "adapter_names": sorted(mapping),
        "per_name_sha256": per_name,
    }


def parameter_hashes(adapted: AdaptedEncoder, adapters: Mapping[str, torch.nn.Parameter] | None = None) -> dict:
    """Hashes for the adapted encoder in the preregistered ``deberta.`` namespace."""
    mapping = adapted.adapters() if adapters is None else dict(adapters)
    return module_hashes(adapted.deberta, mapping, prefix=WRAPPER_NAMESPACE)


def trainable_report(adapted: AdaptedEncoder, reader: torch.nn.Module) -> dict:
    """Live trainable counts, fractions and the enforced preregistered bounds."""
    encoder_side = [(name, parameter) for name, parameter in adapted.named_parameters()
                    if parameter.requires_grad]
    unexpected = [name for name, _ in encoder_side
                  if not (name.startswith(LAYER11_PREFIX) or name.startswith(ADAPTER_PREFIX))]
    _require(not unexpected,
             "trainable surface escaped the final layer plus adapter: " + ", ".join(unexpected))
    reader_parameters = [(name, parameter) for name, parameter in reader.named_parameters()
                         if parameter.requires_grad]
    encoder_scalars = sum(int(parameter.numel()) for _, parameter in encoder_side)
    reader_scalars = sum(int(parameter.numel()) for _, parameter in reader_parameters)
    combined = encoder_scalars + reader_scalars
    loaded = sum(int(parameter.numel()) for parameter in adapted.deberta.parameters())
    _require(loaded == DENOMINATOR_ENCODER,
             f"live loaded encoder parameter count differs from the ledger: {loaded}")
    _require(encoder_scalars == ENCODER_SIDE_TRAINABLE_SCALARS,
             f"encoder-side trainable scalars changed: {encoder_scalars}")
    _require(reader_scalars == READER_TRAINABLE_SCALARS,
             f"reader trainable scalars changed: {reader_scalars}")
    _require(combined == COMBINED_TRAINABLE_SCALARS,
             f"combined trainable scalars changed: {combined}")
    encoder_fraction = encoder_scalars / DENOMINATOR_ENCODER
    combined_fraction = combined / DENOMINATOR_COMBINED
    _require(encoder_fraction <= BOUND_ENCODER_SIDE_FRACTION,
             f"encoder-side trainable fraction exceeds the bound: {encoder_fraction}")
    _require(combined_fraction <= BOUND_COMBINED_FRACTION,
             f"combined trainable fraction exceeds the bound: {combined_fraction}")
    return {
        "loaded_encoder_parameters": loaded,
        "layer11_trainable_scalars": sum(int(parameter.numel()) for name, parameter in encoder_side
                                         if name.startswith(LAYER11_PREFIX)),
        "adapter_trainable_scalars": sum(int(parameter.numel()) for name, parameter in encoder_side
                                         if name.startswith(ADAPTER_PREFIX)),
        "encoder_side_trainable_scalars": encoder_scalars,
        "reader_trainable_scalars": reader_scalars,
        "combined_trainable_scalars": combined,
        "denominator_encoder": DENOMINATOR_ENCODER,
        "denominator_combined": DENOMINATOR_COMBINED,
        "encoder_side_fraction": encoder_fraction,
        "combined_fraction": combined_fraction,
        "bound_encoder_side_fraction": BOUND_ENCODER_SIDE_FRACTION,
        "bound_combined_fraction": BOUND_COMBINED_FRACTION,
        "within_bounds": True,
        "trainable_names": [name for name, _ in encoder_side] + [f"reader.{name}" for name, _ in reader_parameters],
    }


class TextPlan:
    """Deduplicated token arrays plus the per-record question/page layout."""

    def __init__(self, arrays: dict, indices) -> None:
        self._slots: dict[str, int] = {}
        self.texts: list[str] = []
        self.question: dict[int, int] = {}
        self.pages: dict[int, tuple[int, ...]] = {}
        for index in (int(value) for value in indices):
            record = arrays["records"][index]
            self.question[index] = self._slot(record["question"])
            valid = np.flatnonzero(np.asarray(arrays["page_mask"][index]))
            self.pages[index] = tuple(self._slot(record["pages"][int(slot)]["text"]) for slot in valid)
        self.ids: np.ndarray | None = None
        self.masks: np.ndarray | None = None

    def _slot(self, text: str) -> int:
        key = features.sha256_bytes(text.encode("utf-8"))
        if key not in self._slots:
            self._slots[key] = len(self.texts)
            self.texts.append(text)
        return self._slots[key]

    def tokenize(self, encoder) -> "TextPlan":
        self.ids, self.masks = encoder.tokenize(self.texts)
        _require(self.ids.shape == (len(self.texts), MAX_TOKENS),
                 "tokenization shape does not match the pinned 128-token contract")
        return self

    def unique_text_count(self) -> int:
        return len(self.texts)



def encode_records(adapted: AdaptedEncoder, plan: TextPlan, indices, device: str,
                   page_slots: int, counters: dict, *, gradient: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the questions and owned pages of one record chunk through the adapted encoder."""
    _require(plan.ids is not None, "tokenization plan was not prepared")
    index_list = [int(value) for value in indices]
    needed = sorted({plan.question[index] for index in index_list}
                    | {slot for index in index_list for slot in plan.pages[index]})
    # ``plan`` slots index the whole text plan, while ``table`` holds only the
    # rows this chunk needs, so global slots must be translated to table
    # offsets before any gather.  Without this the gather reads past the table.
    offset = {slot: position for position, slot in enumerate(needed)}
    parts = []
    context = torch.enable_grad() if gradient else torch.no_grad()
    with context:
        for start in range(0, len(needed), BATCH_SIZE):
            part = needed[start:start + BATCH_SIZE]
            ids = torch.as_tensor(plan.ids[part], dtype=torch.long, device=device)
            masks = torch.as_tensor(plan.masks[part], dtype=torch.long, device=device)
            values = adapted(input_ids=ids, attention_mask=masks).to(torch.float32)
            _require(torch.isfinite(values).all(), "adapted encoder produced nonfinite features")
            parts.append(values)
            counters["encoder_forward_batches"] += 1
            counters["encoder_texts_encoded"] += len(part)
        table = (torch.cat(parts, dim=0) if parts
                 else torch.zeros((0, WIDTH), device=device, dtype=torch.float32))
        question_index = torch.as_tensor([offset[plan.question[index]] for index in index_list],
                                         dtype=torch.long, device=device)
        questions = table.index_select(0, question_index)
        rows = len(index_list)
        page_index = torch.full((rows, page_slots), -1, dtype=torch.long, device=device)
        for row, index in enumerate(index_list):
            for column, slot in enumerate(plan.pages[index]):
                page_index[row, column] = offset[slot]
        gathered = table.index_select(0, page_index.clamp_min(0).reshape(-1)).reshape(rows, page_slots, WIDTH)
        pages = torch.where((page_index >= 0).unsqueeze(-1), gathered,
                            torch.zeros((), device=device, dtype=gathered.dtype))
    return questions, pages


def _adapted_objective(adapted: AdaptedEncoder, reader: torch.nn.Module, plan: TextPlan,
                       arrays: dict, indices, normalizer: dict, denominators: dict,
                       page_slots: int, device: str, baseline: int, counters: dict) -> dict:
    """The frozen four-component objective on adapted features, one backward per chunk."""
    totals = dict.fromkeys(trainer.COMPONENTS, 0.0)
    stats = normalizer["pages"]
    mean = torch.as_tensor(np.asarray(stats["mean"]), device=device)
    std = torch.as_tensor(np.asarray(stats["std"]), device=device)
    for chunk in trainer._chunks(indices, CHUNK_SIZE):
        features._resource_guard(torch, baseline)
        targets = trainer._batch(arrays, chunk, device)
        mask = targets["page_mask"].to(torch.bool)
        raw_questions, raw_pages = encode_records(adapted, plan, chunk, device, page_slots,
                                                  counters, gradient=True)
        questions = (raw_questions - mean) / std
        pages = (raw_pages - mean) / std
        output = reader(questions, pages, mask, raw_questions, raw_pages, uniform_attention=False)
        losses = readers.eca_loss(output, targets, directed_grade=False)
        counts = trainer._counts(targets, "pages")
        weighted = []
        for key in trainer.COMPONENTS:
            if counts[key] and denominators[key]:
                term = losses[key] * (counts[key] / denominators[key])
                _require(bool(torch.isfinite(term)), f"nonfinite pages {key} objective")
                totals[key] += float(term.detach())
                weighted.append(term)
        if weighted:
            sum(weighted).backward()
            counters["loss_backward_calls"] += 1
        counters["reader_forward_chunks"] += 1
    totals["total"] = sum(totals.values())
    _require(math.isfinite(totals["total"]) and not math.isnan(totals["total"]),
             "nonfinite total smoke objective")
    return totals


def gradient_report(adapted: AdaptedEncoder, reader: torch.nn.Module) -> dict:
    """Per-parameter gradient finiteness, plus the finite gradient norm."""
    entries = [(name, parameter) for name, parameter in adapted.named_parameters()
               if parameter.requires_grad]
    entries += [(f"reader.{name}", parameter) for name, parameter in reader.named_parameters()
                if parameter.requires_grad]
    finite: dict[str, bool] = {}
    missing: list[str] = []
    total = 0.0
    group_norms = {"encoder_side": 0.0, "reader": 0.0}
    for name, parameter in entries:
        if parameter.grad is None:
            finite[name] = False
            missing.append(name)
            continue
        finite[name] = bool(torch.isfinite(parameter.grad).all())
        squared = float(parameter.grad.detach().to(torch.float64).pow(2).sum())
        total += squared
        group = "reader" if name.startswith("reader.") else "encoder_side"
        group_norms[group] += squared
    norm = math.sqrt(total)
    return {
        "gradient_finite_by_parameter": finite,
        "missing_gradient_parameters": sorted(missing),
        "all_trainable_gradients_finite": all(finite.values()) and not missing,
        "gradient_norm": norm,
        "gradient_norm_finite": math.isfinite(norm) and not math.isnan(norm),
        "group_norms": {name: math.sqrt(value) for name, value in group_norms.items()},
        "trainable_parameters_checked": len(entries),
    }


def load_pinned_reader(device: str) -> tuple[torch.nn.Module, dict | None, dict]:
    """Frozen rank-64 PageReader, initialized from the pinned checkpoint when present."""
    reader = readers.PageReader()
    scalars = sum(int(parameter.numel()) for parameter in reader.parameters())
    _require(scalars == READER_TRAINABLE_SCALARS,
             f"reader architecture scalar count changed: {scalars}")
    if not PAGES_CHECKPOINT.exists():
        return reader.to(device), None, {"source": "fresh_seed7", "path": None, "sha256": None}
    digest = features.sha256_file(PAGES_CHECKPOINT)
    _require(digest == PAGES_CHECKPOINT_SHA256,
             f"pinned pages checkpoint hash mismatch: {digest}")
    payload = torch.load(PAGES_CHECKPOINT, map_location="cpu", weights_only=False)
    reader.load_state_dict(payload["state_dict"], strict=True)
    return (reader.to(device), payload.get("normalizer"),
            {"source": "pinned_pages_checkpoint", "path": str(PAGES_CHECKPOINT.resolve()),
             "sha256": digest, "epoch": int(payload.get("epoch", 0))})


def _optimizer(adapted: AdaptedEncoder, reader: torch.nn.Module) -> torch.optim.AdamW:
    encoder_side = [parameter for name, parameter in adapted.named_parameters()
                    if parameter.requires_grad]
    reader_side = [parameter for name, parameter in reader.named_parameters()
                   if parameter.requires_grad]
    _require(encoder_side and reader_side, "optimizer needs both trainable groups")
    return torch.optim.AdamW(
        [{"params": encoder_side, "lr": ENCODER_LR},
         {"params": reader_side, "lr": READER_LR}],
        weight_decay=WEIGHT_DECAY)


def _open_encoder(device: str):
    return features.FeatureEncoder(device=device, protocol_path=features.ATOMIC_PROTOCOL_PATH)

def _runtime_policy() -> dict:
    """The inherited process policy every screen runs under."""
    return {
        "thread_env": {name: os.environ.get(name) for name in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")},
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "GPU_LOCK": str(features.GPU_LOCK),
        "nice_target": 10,
        "ionice_target": "2/7",
        "torch_threads": 4,
        "ram_swap_guard": "features._resource_guard",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _base_payload(screen: str, run_root: Path, experiment: capture.Experiment,
                  source_hashes: dict, ledger: dict) -> dict:
    return {
        "runtime_policy": _runtime_policy(),
        "schema": f"vey.eca2.adaptation-screen-{screen}.v1",
        "screen": screen,
        "verdict": "fail",
        "preregistration": evaluator.artifact(PREREGISTRATION_PATH),
        "pinned_sources": source_hashes,
        "safetensors_ledger": ledger,
        "experiment_context": experiment.context(),
        "run_root": str(run_root),
        "seed": SEED,
        "chunk_size": CHUNK_SIZE,
        "learning_rates": {"reader": READER_LR, "encoder": ENCODER_LR, "adapter": ADAPTER_LR,
                           "ratio_encoder_to_reader": RATIO_ENCODER_TO_READER,
                           "search_forbidden": True},
        "promotion": False,
        "B_STEF_allowed": False,
        "final_pool_touched": False,
    }


def screen_a(run_root, features_root=None, device: str = "cuda",
             records: int = SMOKE_RECORDS_BOUND, epochs: int = SMOKE_EPOCHS_BOUND) -> dict:
    """Screen A: bounded smoke fit with gradient-finiteness and counter evidence."""
    _require(1 <= records <= SMOKE_RECORDS_BOUND, "smoke record bound is at most 64")
    _require(1 <= epochs <= SMOKE_EPOCHS_BOUND, "smoke epoch bound is at most 2")
    source_hashes = _check_pinned_sources()
    ledger = safetensors_ledger()
    experiment = _experiment(features_root)
    root = _safe_root(run_root, experiment)
    payload = _base_payload("A-smoke-fit", root, experiment, source_hashes, ledger)
    payload.update({"device": device, "records_requested": records, "epochs_requested": epochs,
                    "records_bound": SMOKE_RECORDS_BOUND, "epochs_bound": SMOKE_EPOCHS_BOUND})
    receipt_path = root / RECEIPT_NAMES["smoke"]
    try:
        train, evidence = _load_phase("train", experiment)
        selected = trainer.optimizer_indices(train, "train", experiment)[:records]
        _require(len(selected) > 0, "smoke selection is empty")
        normalizer = trainer.fit_normalizer(train, selected, CHUNK_SIZE)
        denominators = trainer._denominators(train, selected, "pages", CHUNK_SIZE)
        page_slots = int(train["page_mask"].shape[1])
        plan = TextPlan(train, selected)
        counters = {"encoder_forward_batches": 0, "encoder_texts_encoded": 0,
                    "reader_forward_chunks": 0, "loss_backward_calls": 0, "optimizer_steps": 0}
        with _open_encoder(device) as encoder:
            plan.tokenize(encoder)
            trainer._seed()
            adapted, adapters = attach_trainable_surface(torch, encoder, device)
            reader, _checkpoint_normalizer, reader_origin = load_pinned_reader(device)
            report = trainable_report(adapted, reader)
            optimizer = _optimizer(adapted, reader)
            history, gradient_reports = [], []
            for epoch in range(1, epochs + 1):
                reader.train()
                optimizer.zero_grad(set_to_none=True)
                totals = _adapted_objective(adapted, reader, plan, train, selected, normalizer,
                                            denominators, page_slots, device,
                                            encoder.initial_swap_mib, counters)
                gradients = gradient_report(adapted, reader)
                _require(gradients["all_trainable_gradients_finite"],
                         "a trainable parameter had a missing or nonfinite gradient")
                _require(gradients["gradient_norm_finite"], "gradient norm was nonfinite or NaN")
                optimizer.step()
                counters["optimizer_steps"] += 1
                history.append({"epoch": epoch, "train": totals})
                gradient_reports.append({"epoch": epoch, **gradients})
            surface_state = {name: parameter.detach().to("cpu").clone()
                             for name, parameter in adapted.state_dict().items()}
            checkpoint = {
                "schema": "vey.eca2.adaptation-smoke-checkpoint.v1",
                "surface_state_dict": surface_state,
                "reader_state_dict": {name: parameter.detach().to("cpu").clone()
                                      for name, parameter in reader.state_dict().items()},
                "normalizer": normalizer,
                "adapter_names": list(ADAPTER_NAMES),
                "seed": SEED, "epochs": epochs, "records": int(len(selected)),
                "protocol_sha256": experiment.protocol()[1],
                "experiment_context": experiment.context(),
                "eligible_for_calibration_or_development": False,
                "eligible_for_adaptation_stage": False,
                "smoke": True,
                "purpose": "gradient-finiteness and counter smoke only; never calibration or development",
            }
            checkpoint_path = root / SMOKE_CHECKPOINT_NAME
            with checkpoint_path.open("xb") as stream:
                torch.save(checkpoint, stream)
            payload.update({
                "counters": counters,
                "trainable": report,
                "reader_initialization": reader_origin,
                "feature_evidence": evidence,
                "selected_records": int(len(selected)),
                "selected_row_identity_sha256": components.record_identity_hash(train["records"], selected),
                "component_denominators": denominators,
                "history": history,
                "gradient_reports": gradient_reports,
                "per_parameter_gradient_finite": gradient_reports[-1]["gradient_finite_by_parameter"],
                "gradient_norm": gradient_reports[-1]["gradient_norm"],
                "encoder_module_mode": "eval (no dropout; deterministic feature path)",
                "checkpoint": {**evaluator.artifact(checkpoint_path),
                               "eligible_for_calibration_or_development": False,
                               "eligible_for_adaptation_stage": False},
                "verdict": "pass",
            })
    except RuntimeError as exc:
        payload["failure_reason"] = str(exc)
        write_receipt(receipt_path, payload)
        raise
    payload["receipt"] = write_receipt(receipt_path, payload)
    return payload


def screen_b(run_root, features_root=None, device: str = "cpu", state_checkpoint=None) -> dict:
    """Screen B: frozen-parameter immutability around the trainable surface."""
    source_hashes = _check_pinned_sources()
    ledger = safetensors_ledger()
    experiment = _experiment(features_root)
    root = _safe_root(run_root, experiment)
    payload = _base_payload("B-encoder-immutability", root, experiment, source_hashes, ledger)
    payload.update({"device": device, "encoder_forwards": 0, "optimizer_steps_in_screen": 0})
    receipt_path = root / RECEIPT_NAMES["immutability"]
    try:
        with _open_encoder(device) as encoder:
            # The pinned loader's own hash, which omits shape, is the identity
            # definition for the frozen base weights.
            pinned_hash = encoder.parameter_hash()
            _require(pinned_hash == PINNED_ENCODER_PARAMETERS_SHA256,
                     "loaded encoder does not match the pinned parameter hash")
            before = module_hashes(encoder.model, {})
            adapted, adapters = attach_trainable_surface(torch, encoder, device)
            attached = parameter_hashes(adapted, adapters)
            _require(before["full_sha256"] == attached["full_sha256"],
                     "attaching the trainable surface changed encoder bytes")
            _require(attached["outside_layer11_sha256"] == before["outside_layer11_sha256"],
                     "attaching the trainable surface changed a non-final-layer parameter")
            after_source = {"kind": "attached_surface", "path": None, "sha256": None}
            candidate = Path(state_checkpoint) if state_checkpoint is not None else root / SMOKE_CHECKPOINT_NAME
            if candidate.exists():
                payload_state = torch.load(candidate, map_location="cpu", weights_only=False)
                _require(payload_state.get("eligible_for_adaptation_stage") is not True,
                         "immutability screen refuses an adaptation-stage eligible state")
                with torch.no_grad():
                    for name, tensor in payload_state["surface_state_dict"].items():
                        target = dict(adapted.named_parameters()).get(name)
                        _require(target is not None,
                                 f"state checkpoint carries an unknown surface tensor: {name}")
                        target.copy_(tensor)
                after_source = {"kind": "surface_state_checkpoint",
                                "path": str(candidate.resolve()),
                                "sha256": features.sha256_file(candidate)}
            after = parameter_hashes(adapted, adapters)
            outside_changed = sorted(name for name, digest in after["per_name_sha256"].items()
                                     if name in before["per_name_sha256"]
                                     and digest != before["per_name_sha256"][name])
            allowed = set(after["layer11_names"]) | set(after["adapter_names"])
            unexpected = [name for name in outside_changed
                          if not (name.startswith(LAYER11_PREFIX) or name.startswith(ADAPTER_PREFIX))]
            _require(not unexpected,
                     "non-adapter parameter outside the final layer changed: " + ", ".join(unexpected))
            _require(after["outside_layer11_sha256"] == attached["outside_layer11_sha256"],
                     "frozen subset hash changed outside the final layer")
            changed_allowed = [name for name in outside_changed if name in allowed]
            payload.update({
                "before": {key: value for key, value in before.items()
                           if key != "per_name_sha256"},
                "attached": {key: value for key, value in attached.items()
                             if key != "per_name_sha256"},
                "after": {key: value for key, value in after.items()
                          if key != "per_name_sha256"},
                "compared_parameter_count": after["outside_layer11_parameter_count"],
                "compared_scalars": after["outside_layer11_scalars"],
                "changed_names": outside_changed,
                "changed_allowed_names": changed_allowed,
                "changed_set_is_subset_of_layer11_plus_adapters": not unexpected,
                "changed_set_equals_layer11_plus_adapters": sorted(outside_changed) == sorted(allowed),
                "frozen_subset_identical": after["outside_layer11_sha256"] == before["outside_layer11_sha256"],
                "pinned_encoder_identity": pinned_hash == PINNED_ENCODER_PARAMETERS_SHA256,
                "pinned_encoder_parameters_sha256": pinned_hash,
                "after_state_source": after_source,
                "identity_result": True,
                "enforced_invariant": "no non-adapter encoder parameter outside deberta.encoder.layer.11 changed",
                "verdict": "pass",
            })
    except RuntimeError as exc:
        payload["failure_reason"] = str(exc)
        write_receipt(receipt_path, payload)
        raise
    payload["receipt"] = write_receipt(receipt_path, payload)
    return payload


def linear_ceiling(design: np.ndarray, target: np.ndarray) -> dict:
    """The finite bias-free FP64 procedure of components.run_linear, with its checks."""
    coefficients, residuals, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    fitted = design @ coefficients
    error = fitted - target
    gram = np.zeros((design.shape[1], design.shape[1]), dtype=np.float64)
    cross = np.zeros(design.shape[1], dtype=np.float64)
    for vector, value in zip(design, target):
        gram += np.outer(vector, vector)
        cross += vector * value
    normal_residual = gram @ coefficients - cross
    direct_gradient = design.T @ error
    scale = max(1.0, float(np.linalg.norm(gram, ord=np.inf)
                           * np.linalg.norm(coefficients, ord=np.inf)
                           + np.linalg.norm(cross, ord=np.inf)))
    _require(np.linalg.norm(normal_residual, ord=np.inf) / scale <= 1e-10
             and np.linalg.norm(direct_gradient, ord=np.inf) / scale <= 1e-10,
             "independent normal-equation check failed")
    second, _, second_rank, second_singular = linalg.lstsq(
        design, target, cond=np.finfo(np.float64).eps * max(design.shape), lapack_driver="gelss")
    second_fitted = design @ second
    coefficient_scale = max(1.0, float(np.linalg.norm(coefficients)))
    coefficient_difference = float(np.linalg.norm(second - coefficients) / coefficient_scale)
    fit_difference = float(np.max(np.abs(second_fitted - fitted)))
    second_mse = float(np.mean(np.square(second_fitted - target)))
    mse = float(np.mean(np.square(error)))
    _require(second_rank == rank and coefficient_difference <= 1e-7
             and fit_difference <= 1e-8 and abs(second_mse - mse) <= 1e-12 * max(1, mse),
             "independent GELSS minimum-norm/fit residual check failed")
    return {
        "coefficients": coefficients,
        "rank": int(rank),
        "singular_values": singular,
        "train_MSE": mse,
        "solver_reported_residuals": residuals.tolist(),
        "retained_singular_condition": float(singular[0] / singular[rank - 1]) if rank else None,
        "singular_condition": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
        "full_column_rank": rank == design.shape[1],
        "unidentified_coefficient_directions": int(design.shape[1] - rank),
        "normal_equation": {
            "independent_gram_residual_inf": float(np.max(np.abs(normal_residual))),
            "direct_gradient_inf": float(np.max(np.abs(direct_gradient))),
            "normalization_scale": scale, "relative_tolerance": 1e-10, "pass": True},
        "second_solver": {
            "driver": "scipy.linalg.lstsq GELSS", "rank": int(second_rank), "train_MSE": second_mse,
            "relative_coefficient_difference": coefficient_difference,
            "maximum_train_prediction_difference": fit_difference,
            "coefficient_tolerance": 1e-7, "prediction_tolerance": 1e-8,
            "MSE_absolute_scaled_tolerance": 1e-12 * max(1, mse), "pass": True},
        "second_singular_values": second_singular,
    }


def _normalized_design(matrix: np.ndarray, stats: dict) -> np.ndarray:
    """FP64 per-coordinate transform, identical to components.normalized_design."""
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    design = np.empty((matrix.shape[0], WIDTH), dtype=np.float64)
    for index in range(matrix.shape[0]):
        np.subtract(matrix[index], mean, out=design[index])
        design[index] /= std
    return design


def evaluate_ceiling(train_features: np.ndarray, train_target: np.ndarray,
                     development_features: np.ndarray, development_target: np.ndarray,
                     stats: dict, development_families: Sequence[str]) -> dict:
    """Fit the finite bias-free ceiling and score it against the pinned threshold.

    Exposed so a fitted adapted state can be scored under exactly the same
    arithmetic, diagnostics and threshold as the preregistered screen C.
    """
    design = _normalized_design(train_features, stats)
    _require(np.isfinite(design).all() and np.isfinite(train_target).all(),
             "nonfinite design or target in the adapted ceiling")
    solution = linear_ceiling(design, train_target)
    coefficients = solution.pop("coefficients")
    solution.pop("singular_values")
    solution.pop("second_singular_values")
    train_prediction = design @ coefficients
    development_design = _normalized_design(development_features, stats)
    development_prediction = development_design @ coefficients
    error = np.abs(development_prediction - development_target)
    by_family: dict[str, list[float]] = {}
    for family, value in zip(development_families, error):
        by_family.setdefault(family, []).append(float(value))
    family_macro = float(np.mean([np.mean(values) for values in by_family.values()]))
    return {
        "design_shape": list(design.shape),
        "diagnostics": solution,
        "train_prediction_range": [float(train_prediction.min()), float(train_prediction.max())],
        "development": {
            "family_macro_unique_text_MAE": family_macro,
            "raw_unclipped_MAE_unique_page_text": float(error.mean()),
            "unique_page_texts": int(error.shape[0]),
            "raw_prediction_range": [float(development_prediction.min()),
                                     float(development_prediction.max())],
            "counts_MAE_by_family": {name: {"count": len(values), "MAE": float(np.mean(values))}
                                     for name, values in by_family.items()},
            "used_for_verdict": True,
        },
        "verdict": "pass" if family_macro < SCREEN_C_THRESHOLD else "fail",
    }


def _encode_unique_texts(adapted: AdaptedEncoder, encoder, texts: list[str], device: str,
                         counters: dict) -> np.ndarray:
    _require(texts, "no unique texts to encode")
    ids, masks = encoder.tokenize(texts)
    _require(ids.shape == (len(texts), MAX_TOKENS), "unique-text tokenization shape changed")
    output = np.empty((len(texts), WIDTH), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            stop = start + BATCH_SIZE
            batch_ids = torch.as_tensor(ids[start:stop], dtype=torch.long, device=device)
            batch_masks = torch.as_tensor(masks[start:stop], dtype=torch.long, device=device)
            values = adapted(input_ids=batch_ids, attention_mask=batch_masks).to(torch.float32)
            _require(torch.isfinite(values).all(), "adapted encoder produced nonfinite features")
            output[start:stop] = values.detach().to("cpu").numpy()
            counters["encoder_forward_batches"] += 1
            counters["encoder_texts_encoded"] += int(values.shape[0])
    return output


def _rows_with_texts(rows: list[dict], data: dict) -> tuple[list[str], list[int]]:
    texts, slots, lookup = [], [], {}
    for row in rows:
        record = data["records"][int(row["record_index"])]
        text = record["pages"][int(row["page_index"])]["text"]
        digest = features.sha256_bytes(text.encode("utf-8"))
        _require(digest == row["text_sha256"], "supervised page text hash changed")
        if digest not in lookup:
            lookup[digest] = len(texts)
            texts.append(text)
        slots.append(lookup[digest])
    return texts, slots


def _screen_c_material(experiment: capture.Experiment) -> dict:
    """Custody, populations and texts shared by C0 and C1.

    Opens train and development only.  Validation is never requested, so it can
    neither satisfy nor fail either screen.
    """
    train, train_evidence = _load_phase("train", experiment)
    development, development_evidence = _load_phase("development", experiment)
    catalogue = components.source_catalogue(experiment)
    properties = catalogue[2]
    initial_swap = features._parse_meminfo()[1]
    selected = trainer.optimizer_indices(train, "train", experiment)
    computed_normalizer = trainer.fit_normalizer(train, selected, CHUNK_SIZE)
    if PAGES_CHECKPOINT.exists():
        checkpoint = torch.load(PAGES_CHECKPOINT, map_location="cpu", weights_only=False)
        _require(components.fingerprint(checkpoint["normalizer"])
                 == components.fingerprint(computed_normalizer),
                 "common train q/page normalizer identity failed")
    train_rows, train_occurrences = components.supervised_unique(
        train, selected, properties, initial_swap)
    development_indices = np.arange(len(development["records"]), dtype=np.int64)
    development_rows, development_occurrences = components.supervised_unique(
        development, development_indices, properties, initial_swap)
    train_texts = {row["text_sha256"] for row in train_rows}
    _require(not train_texts.intersection(row["text_sha256"] for row in development_rows),
             "held page texts overlap the fitting population")
    train_text_list, train_slots = _rows_with_texts(train_rows, train)
    development_text_list, development_slots = _rows_with_texts(development_rows, development)
    return {
        "train": train, "development": development,
        "train_evidence": train_evidence, "development_evidence": development_evidence,
        "selected": selected, "stats": computed_normalizer["pages"],
        "train_rows": train_rows, "development_rows": development_rows,
        "train_occurrences": train_occurrences,
        "development_occurrences": development_occurrences,
        "train_text_list": train_text_list, "train_slots": train_slots,
        "development_text_list": development_text_list,
        "development_slots": development_slots,
    }


def _screen_c_base(screen: str, root: Path, experiment: capture.Experiment,
                   source_hashes: dict, ledger: dict) -> dict:
    payload = _base_payload(screen, root, experiment, source_hashes, ledger)
    payload.update({
        "device": "cpu", "optimizer_steps": 0, "reader_forwards": 0,
        "validation_used_for_verdict": False,
        "validation_phase_opened": False,
        "phases_opened": list(SCREEN_C_PHASES),
        "screen_order": list(SCREEN_ORDER),
        "screen_c_defect": evaluator.artifact(SCREEN_C_DEFECT_PATH),
        "screen_c_amendment": evaluator.artifact(SCREEN_C_AMENDMENT_PATH),
        "scope": "CPU-only screens; no final pool, no promotion, no certificate",
    })
    return payload


def screen_c0_identity(run_root, features_root, device: str = "cpu") -> dict:
    """Screen C0: identity precondition and adapter liveness, with no quality claim.

    Verdict is decided only by the 1e-5 identity tolerance and the liveness
    control.  No MAE, no threshold and no quality statement is produced here.
    """
    _require(features_root is not None, "screen C0 requires --features-root")
    _require(device == "cpu", "screen C0 is CPU-only")
    source_hashes = _check_pinned_sources()
    ledger = safetensors_ledger()
    experiment = _experiment(features_root)
    root = _safe_root(run_root, experiment)
    payload = _screen_c_base("C0-identity-precondition", root, experiment, source_hashes, ledger)
    payload["identity_tolerance"] = IDENTITY_TOLERANCE
    payload["quality_claim"] = False
    payload["threshold"] = None
    receipt_path = root / RECEIPT_NAMES["c0_identity"]
    try:
        material = _screen_c_material(experiment)
        train, development = material["train"], material["development"]
        counters = {"encoder_forward_batches": 0, "encoder_texts_encoded": 0}
        control_counters = {"encoder_forward_batches": 0, "encoder_texts_encoded": 0}
        with _open_encoder("cpu") as encoder:
            adapted, adapters = attach_trainable_surface(torch, encoder, "cpu")
            encoded = _encode_unique_texts(adapted, encoder,
                                           material["train_text_list"]
                                           + material["development_text_list"],
                                           "cpu", counters)
            # Liveness control: a deliberately perturbed adapter must move the
            # features. This separates an adapter-path failure from ordinary
            # FP32 CPU/GPU reproduction drift, which the 1e-5 tolerance alone
            # cannot distinguish.
            probe = list(adapters.values())[0]
            with torch.no_grad():
                probe.add_(torch.full_like(probe, 0.01))
            control = _encode_unique_texts(adapted, encoder, material["train_text_list"],
                                           "cpu", control_counters)
            with torch.no_grad():
                probe.sub_(torch.full_like(probe, 0.01))
            restored = _encode_unique_texts(adapted, encoder, material["train_text_list"],
                                           "cpu", control_counters)
        train_count = len(material["train_text_list"])
        adapted_train = encoded[:train_count][np.asarray(material["train_slots"], dtype=np.int64)]
        adapted_development = encoded[train_count:][np.asarray(material["development_slots"],
                                                              dtype=np.int64)]
        frozen_train = np.asarray([train["pages"][int(row["record_index"]), int(row["page_index"])]
                                   for row in material["train_rows"]], dtype=np.float32)
        frozen_development = np.asarray([development["pages"][int(row["record_index"]),
                                                              int(row["page_index"])]
                                         for row in material["development_rows"]], dtype=np.float32)
        train_difference = float(np.max(np.abs(adapted_train.astype(np.float64)
                                               - frozen_train.astype(np.float64))))
        development_difference = float(np.max(np.abs(adapted_development.astype(np.float64)
                                                     - frozen_development.astype(np.float64))))
        control_difference = float(np.max(np.abs(control.astype(np.float64)
                                                 - adapted_train.astype(np.float64))))
        restored_difference = float(np.max(np.abs(restored.astype(np.float64)
                                                 - adapted_train.astype(np.float64))))
        _require(control_difference > IDENTITY_TOLERANCE,
                 "the adapter is not live: perturbing it did not change the encoded features")
        _require(restored_difference <= IDENTITY_TOLERANCE,
                 "restoring the adapter did not return the identity features")
        identity_ok = max(train_difference, development_difference) <= IDENTITY_TOLERANCE
        payload.update({
            "identity_precondition": {
                "train_max_abs_difference": train_difference,
                "development_max_abs_difference": development_difference,
                "tolerance": IDENTITY_TOLERANCE,
                "compared_train_texts": len(material["train_rows"]),
                "compared_development_texts": len(material["development_rows"]),
                "adapter_liveness_control_max_difference": control_difference,
                "adapter_liveness_control": "pass",
                "adapter_restore_max_difference": restored_difference,
                "pass": identity_ok,
            },
            "counters": counters,
            "liveness_counters": control_counters,
            "train_capture": material["train_evidence"],
            "development_capture": material["development_evidence"],
            "optimizer_row_identity_sha256": components.record_identity_hash(
                material["train"]["records"], material["selected"]),
            "verdict": "pass" if identity_ok else "fail",
        })
    except RuntimeError as exc:
        payload["failure_reason"] = str(exc)
        write_receipt(receipt_path, payload)
        raise
    payload["receipt"] = write_receipt(receipt_path, payload)
    return payload


class Layer11Probe:
    """Capture the layer-11 GELU activation and the pre-norm residual.

    Layer 11 computes ``LayerNorm(W . activation + residual)`` and a masked mean
    is linear, so the masked mean of that pre-norm sum is exactly
    ``W . masked_mean(activation) + masked_mean(residual)``.  Those two means
    are the exact closed-form design and offset for the layer-11 output weight.
    The hooks record raw activations per batch; the encode path owns the
    attention mask and performs the masked mean.
    """

    def __init__(self, adapted: AdaptedEncoder):
        layer = _module_for(adapted.deberta, LAYER11_LIVE_PREFIX.rstrip("."))
        self.activation: list[np.ndarray] = []
        self.residual: list[np.ndarray] = []
        self._handles = [
            layer.output.dense.register_forward_pre_hook(self._residual_hook),
            layer.intermediate.register_forward_hook(self._activation_hook),
        ]

    def _residual_hook(self, _module, inputs):
        self.residual.append(inputs[0].detach().to("cpu").to(torch.float32).numpy())

    def _activation_hook(self, _module, _inputs, output):
        self.activation.append(output.detach().to("cpu").to(torch.float32).numpy())

    def reset(self) -> None:
        self.activation = []
        self.residual = []

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _probe_encode(adapted: AdaptedEncoder, ids: np.ndarray, masks: np.ndarray,
                  probe: Layer11Probe) -> np.ndarray:
    """Encode while the layer-11 probe captures its pre-norm activations."""
    parts = []
    with torch.no_grad():
        for start in range(0, int(ids.shape[0]), BATCH_SIZE):
            stop = start + BATCH_SIZE
            batch_ids = torch.as_tensor(ids[start:stop], dtype=torch.long, device="cpu")
            batch_mask = torch.as_tensor(masks[start:stop], dtype=torch.long, device="cpu")
            values = adapted(input_ids=batch_ids, attention_mask=batch_mask).to(torch.float32)
            _require(torch.isfinite(values).all(), "probe encode produced nonfinite features")
            parts.append(values.numpy())
    _require(parts, "probe encode received no texts")
    return np.concatenate(parts, axis=0)


def _masked_mean_per_batch(activations: list[np.ndarray], masks: np.ndarray) -> np.ndarray:
    """Masked mean of each captured batch, matching the canonical FP32 pooling."""
    _require(activations and len(activations) == len(masks),
             "layer-11 probe captured a different number of batches than encoded")
    outputs = []
    for activation, mask in zip(activations, masks):
        _require(activation.shape[:2] == mask.shape,
                 "captured layer-11 activation does not match the attention mask shape")
        weights = np.asarray(mask, dtype=np.float32).reshape(activation.shape[0], -1, 1)
        # The canonical masked_mean takes a per-row denominator clamped at 1.
        denominator = np.maximum(weights.sum(axis=1), 1.0)
        outputs.append((activation * weights).sum(axis=1) / denominator)
    return np.concatenate(outputs, axis=0).astype(np.float64)


def adapt_layer11_closed_form(activation_mean: np.ndarray, residual_mean: np.ndarray,
                              target: np.ndarray, adapted: AdaptedEncoder) -> dict:
    """Bounded FP64 bias-free least-squares adaptation of the layer-11 output.

    Solves ``W . activation_mean + residual_mean == target`` for the layer-11
    output weight with the same ``numpy.linalg.lstsq(rcond=None)`` procedure,
    then writes the solution into the live weight and pins that weight's
    adapter to 1.0 so the effective weight equals the solve exactly.  One
    solve, no selection, no early stopping, no checkpoint choice, no tuning.
    The design is the exact masked mean of the layer-11 activation, so this
    changes the representation rather than the readout.
    """
    lhs = np.asarray(activation_mean, dtype=np.float64)
    rhs = np.asarray(target, dtype=np.float64) - np.asarray(residual_mean, dtype=np.float64)
    _require(lhs.ndim == 2 and lhs.shape[0] == rhs.shape[0],
             "closed-form adaptation design and target disagree on rows")
    coefficients, _residuals, rank, singular = np.linalg.lstsq(lhs, rhs, rcond=None)
    fitted = lhs @ coefficients
    error = fitted - rhs
    gram = np.zeros((lhs.shape[1], lhs.shape[1]), dtype=np.float64)
    cross = np.zeros(lhs.shape[1], dtype=np.float64)
    for vector, value in zip(lhs, rhs):
        gram += np.outer(vector, vector)
        cross += vector * value
    direct_gradient = lhs.T @ error
    scale = max(1.0, float(np.linalg.norm(gram, ord=np.inf)
                           * np.linalg.norm(coefficients, ord=np.inf)
                           + np.linalg.norm(cross, ord=np.inf)))
    _require(np.isfinite(coefficients).all(),
             "closed-form adaptation produced nonfinite coefficients")
    _require(np.linalg.norm(direct_gradient, ord=np.inf) / scale <= 1e-10,
             "closed-form adaptation normal-equation check failed")
    mse = float(np.mean(np.square(error)))
    second, _residual2, second_rank, _singular2 = linalg.lstsq(
        lhs, rhs, cond=np.finfo(np.float64).eps * max(lhs.shape), lapack_driver="gelss")
    coefficient_scale = max(1.0, float(np.linalg.norm(coefficients)))
    coefficient_difference = float(np.linalg.norm(second - coefficients) / coefficient_scale)
    _require(second_rank == rank and coefficient_difference <= 1e-7,
             "closed-form adaptation second-solver agreement failed")
    output_module = _module_for(adapted.deberta, LAYER11_LIVE_PREFIX + "output")
    weight = output_module.weight
    solved = np.asarray(coefficients).T
    _require(tuple(weight.shape) == tuple(solved.shape),
             "solved layer-11 weight shape does not match the live weight")
    with torch.no_grad():
        weight.copy_(torch.as_tensor(solved, dtype=torch.float32, device=weight.device))
    # Both attention.output.dense and output.dense end with "output.dense.weight",
    # so match the solved weight's own canonical adapter name exactly.
    solved_adapter = ADAPTER_PREFIX + LAYER11_PREFIX + "output.dense.weight"
    for name, parameter in adapted.adapters().items():
        if name == solved_adapter:
            with torch.no_grad():
                parameter.fill_(1.0)
    return {
        "diagnostics": {
            "rank": int(rank),
            "retained_singular_condition": float(singular[0] / singular[rank - 1]) if rank else None,
            "singular_condition": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
            "train_MSE": mse,
            "unidentified_coefficient_directions": int(lhs.shape[1] - rank),
            "normal_equation_direct_gradient_inf": float(np.max(np.abs(direct_gradient))),
            "normalization_scale": scale,
            "second_solver_rank": int(second_rank),
            "relative_coefficient_difference": coefficient_difference,
            "solver": "numpy.linalg.lstsq(rcond=None)",
            "fit": "single FP64 closed-form bias-free solve for the layer-11 output weight",
            "finite": True,
        },
    }


def screen_c1_adapted_ceiling(run_root, features_root, device: str = "cpu") -> dict:
    """Screen C1: adapted ceiling after one bounded closed-form layer-11 adaptation.

    Fits the adaptation on unique supervised train page texts only, encodes the
    development unique page texts with the adapted layer, and scores the frozen
    FP64 least-squares ceiling against the pinned development threshold.
    """
    _require(features_root is not None, "screen C1 requires --features-root")
    _require(device == "cpu", "screen C1 is CPU-only")
    source_hashes = _check_pinned_sources()
    ledger = safetensors_ledger()
    experiment = _experiment(features_root)
    root = _safe_root(run_root, experiment)
    payload = _screen_c_base("C1-adapted-ceiling", root, experiment, source_hashes, ledger)
    payload.update({
        "threshold": SCREEN_C_THRESHOLD,
        "threshold_derivation": "frozen page-only development unique-text family macro MAE minus 0.05",
        "frozen_reference_MAE": SCREEN_C_FROZEN_MAE,
        "adaptation_state": "bounded FP64 closed-form bias-free layer-11 adaptation on unique "
                            "supervised train page texts; no selection, no early stopping, no tuning",
        "quality_claim": True,
    })
    receipt_path = root / RECEIPT_NAMES["c1_adapted_ceiling"]
    try:
        c0_path = root / RECEIPT_NAMES["c0_identity"]
        _require(c0_path.exists(),
                 f"screen C1 requires a passing C0 identity receipt: {c0_path}")
        c0 = json.loads(c0_path.read_text(encoding="utf-8"))
        _require(c0.get("verdict") == "pass",
                 f"screen C0 did not pass; C1 is forbidden: {c0.get('verdict')}")
        payload["predecessor_c0_receipt"] = {
            **evaluator.artifact(c0_path), "verdict": c0["verdict"]}
        material = _screen_c_material(experiment)
        train_rows, development_rows = material["train_rows"], material["development_rows"]
        train_target = np.asarray([row["target"] for row in train_rows], dtype=np.float64)
        development_target = np.asarray([row["target"] for row in development_rows],
                                        dtype=np.float64)
        counters = {"encoder_forward_batches": 0, "encoder_texts_encoded": 0}
        with _open_encoder("cpu") as encoder:
            adapted, _adapters = attach_trainable_surface(torch, encoder, "cpu")
            # Bounded closed-form adaptation of layer 11 on the train
            # population only. Layer 11 computes LayerNorm(W . activation +
            # residual) and a masked mean is linear, so solving
            # W . masked_mean(activation) + masked_mean(residual) == raw extent
            # changes the representation on the unique supervised train page
            # texts alone. One solve, no selection, no early stopping, no
            # checkpoint choice, no tuning.
            probe = Layer11Probe(adapted)
            try:
                train_ids, train_masks = encoder.tokenize(material["train_text_list"])
                probe.reset()
                activation = _probe_encode(adapted, train_ids, train_masks, probe)
                activation_mean = _masked_mean_per_batch(probe.activation, train_masks)
                residual_mean = _masked_mean_per_batch(probe.residual, train_masks)
            finally:
                probe.close()
            _require(activation.shape[0] == len(material["train_text_list"]),
                     "probe encode returned a different row count than the train population")
            update = adapt_layer11_closed_form(activation_mean, residual_mean,
                                              train_target, adapted)
            adapted_train_unique = _encode_unique_texts(adapted, encoder,
                                                        material["train_text_list"],
                                                        "cpu", counters)
            adapted_development_unique = _encode_unique_texts(adapted, encoder,
                                                              material["development_text_list"],
                                                              "cpu", counters)
        adapted_train = adapted_train_unique[np.asarray(material["train_slots"], dtype=np.int64)]
        adapted_development = adapted_development_unique[np.asarray(material["development_slots"],
                                                                     dtype=np.int64)]
        metrics = evaluate_ceiling(adapted_train, train_target, adapted_development,
                                   development_target, material["stats"],
                                   [row["family"] for row in development_rows])
        payload.update({
            "unique_supervised_train_page_texts": len(train_rows),
            "unique_supervised_development_page_texts": len(development_rows),
            "supervised_train_occurrences": material["train_occurrences"],
            "supervised_development_occurrences": material["development_occurrences"],
            "counters": counters,
            "solver": "numpy.linalg.lstsq(rcond=None)",
            "bias": False,
            "dtype": "float64",
            "normalizer_policy": "existing train-only joint q/page statistics with the 0.01 floor",
            "adaptation": {**update["diagnostics"],
                           "population": "unique supervised train page texts only",
                           "selection": "none", "early_stopping": False, "tuning": False},
            "train_capture": material["train_evidence"],
            "development_capture": material["development_evidence"],
            "optimizer_row_identity_sha256": components.record_identity_hash(
                material["train"]["records"], material["selected"]),
            **metrics,
            "scope": "finite pinned bias-free linear function class on the adapted final-layer "
                     "representation; not a ranking bound and not proof that semantic "
                     "information is absent",
            "verdict": metrics["verdict"],
        })
    except RuntimeError as exc:
        payload["failure_reason"] = str(exc)
        write_receipt(receipt_path, payload)
        raise
    payload["receipt"] = write_receipt(receipt_path, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", required=True,
                        choices=("smoke", "immutability", "c0-identity", "ceiling"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--state-checkpoint", type=Path)
    parser.add_argument("--records", type=int, default=SMOKE_RECORDS_BOUND)
    parser.add_argument("--epochs", type=int, default=SMOKE_EPOCHS_BOUND)
    args = parser.parse_args(argv)

    if args.screen in ("c0-identity", "ceiling"):
        _require(args.device in (None, "cpu"),
                 "screens C0 and C1 are CPU-only; --device cuda is forbidden")
        _require(args.features_root is not None,
                 f"screen {args.screen} requires --features-root")
        # Scope the device mask to this CLI run only. Masking at module import
        # once hid a live GPU from every other module importing these helpers.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        result = (screen_c0_identity(args.run_root, args.features_root, device="cpu")
                  if args.screen == "c0-identity" else
                  screen_c1_adapted_ceiling(args.run_root, args.features_root, device="cpu"))
    elif args.screen == "smoke":
        device = args.device or "cuda"
        result = screen_a(args.run_root, args.features_root, device=device,
                          records=args.records, epochs=args.epochs)
    else:
        device = args.device or "cpu"
        result = screen_b(args.run_root, args.features_root, device=device,
                          state_checkpoint=args.state_checkpoint)

    summary = {"screen": result["screen"], "verdict": result["verdict"],
               "receipt": result["receipt"]["path"], "receipt_sha256": result["receipt"]["sha256"]}
    print(json.dumps(summary, sort_keys=True))
    return 0 if result["verdict"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
