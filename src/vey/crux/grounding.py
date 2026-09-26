"""CRUX grounder: frozen NLI predicate grounder + trained antisymmetric ordinal
comparator, behind a single Grounder interface.

    predicate(texts, concept)  -> per-candidate bipolar margin (>0 satisfies)
    ordinal(texts, axis)       -> per-candidate permutation-invariant potential
    evidence(texts, axis)      -> per-candidate 1.0 (supported) / 0.0 (no evidence)

Artifact contract (self-contained; no private paths):

  * The NLI backbone (``VEY_CRUX_NLI``, default ``tasksource/ModernBERT-base-nli``)
    downloads from the public Hugging Face Hub on first use.
  * The trained ordinal-comparator weights resolve in order:
      1. ``VEY_CRUX_COMPARATOR``: a local ``comparator.safetensors`` (or a
         legacy ``.pt`` state file), else
      2. ``VEY_CRUX_COMPARATOR_HF``: a ``repo_id[@revision]`` on the Hub, else
      3. the published default ``fazinahamed/vey`` at a pinned revision.
    If the file cannot be resolved or fetched, the CRUX lane FAILS CLOSED with an
    actionable error; it never silently degrades. The structured lane needs no
    artifact.

The safetensors file holds ``enc.*`` and ``head.*`` tensors with metadata
``{"model_id": str, ...}``. The legacy ``.pt`` is
``{"model_id": str, "enc": state_dict, "head": state_dict}``.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

DEFAULT_NLI = "tasksource/ModernBERT-base-nli"

_PREDICATE_PROPS = {
    "permitted": ("This option is permitted and satisfies the stated constraint.",
                  "This option is not permitted or violates the stated constraint."),
    "tool_support": ("This option supports the required tools.",
                     "This option does not support the required tools."),
    "quality_floor": ("This option meets the required quality floor.",
                      "This option falls below the required quality floor."),
    "terminal": ("This option completes the objective immediately.",
                 "This option does not complete the objective."),
}


def predicate_props(concept: str) -> tuple[str, str]:
    if concept in _PREDICATE_PROPS:
        return _PREDICATE_PROPS[concept]
    return (f"This option satisfies the {concept} requirement.",
            f"This option does not satisfy the {concept} requirement.")


@runtime_checkable
class Grounder(Protocol):
    def predicate(self, texts: list[str], concept: str, threshold=None) -> list[float]: ...
    def ordinal(self, texts: list[str], axis: str) -> list[float]: ...
    def evidence(self, texts: list[str], axis: str) -> list[float]: ...


class CruxArtifactError(RuntimeError):
    """Raised when the CRUX comparator artifact cannot be resolved (fail-closed)."""


DEFAULT_COMPARATOR_HF = "fazinahamed/vey"
# Pinned published revision of the comparator artifact (exact commit for
# reproducibility; set to None to track the repo default branch).
DEFAULT_COMPARATOR_REV: str | None = "04d22c8b3cff843d7e64c61ab6550ba412898c94"
COMPARATOR_FILENAME = "comparator.safetensors"


def _resolve_comparator_path() -> str:
    local = os.environ.get("VEY_CRUX_COMPARATOR")
    if local:
        if not os.path.exists(local):
            raise CruxArtifactError(f"VEY_CRUX_COMPARATOR={local!r} does not exist.")
        return local
    hf = os.environ.get("VEY_CRUX_COMPARATOR_HF")
    if hf:
        repo, _, rev = hf.partition("@")
        rev = rev or None
    else:
        repo, rev = DEFAULT_COMPARATOR_HF, DEFAULT_COMPARATOR_REV
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=COMPARATOR_FILENAME, revision=rev)
    except Exception as e:  # noqa: BLE001 - fail closed with an actionable message
        raise CruxArtifactError(
            f"Could not fetch the CRUX comparator ({COMPARATOR_FILENAME}) from "
            f"{repo!r}: {e}. Set VEY_CRUX_COMPARATOR to a local file, or "
            "VEY_CRUX_COMPARATOR_HF to 'repo_id[@revision]'. The structured lane "
            "runs without any artifact.") from e


def _load_comparator_state(path: str, device: str):
    """Return (model_id, enc_state_dict, head_state_dict) from a safetensors or
    legacy .pt comparator file."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        enc, head = {}, {}
        with safe_open(path, framework="pt", device="cpu") as f:
            model_id = (f.metadata() or {}).get("model_id")
            for k in f.keys():
                if k.startswith("enc."):
                    enc[k[4:]] = f.get_tensor(k)
                elif k.startswith("head."):
                    head[k[5:]] = f.get_tensor(k)
        if not model_id:
            raise CruxArtifactError(f"{path}: missing 'model_id' metadata.")
        return model_id, enc, head
    import torch
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck["model_id"], ck["enc"], ck["head"]


class SageGrounder:
    """Lazy-loaded CRUX grounder. Models load on first use; fails closed if the
    comparator artifact is unavailable."""

    ordinal_resolution = 0.15   # Semantic Resolution band for qualitative axes

    def __init__(self, device: str = "cpu", nli_model: str | None = None):
        self.device = device
        self.nli_model = nli_model or os.environ.get("VEY_CRUX_NLI", DEFAULT_NLI)
        self._nli = None
        self._cmp = None

    def _ensure(self):
        if self._nli is not None:
            return
        from .backbone import NLIGrounder
        from .ordinal import OrdinalComparator
        path = _resolve_comparator_path()        # fail-closed before any model load
        model_id, enc_sd, head_sd = _load_comparator_state(path, self.device)
        self._nli = NLIGrounder(self.nli_model, device=self.device)
        self._cmp = OrdinalComparator(model_id, device=self.device)
        self._cmp.load_state(enc_sd, head_sd)

    def predicate(self, texts, concept, threshold=None):
        self._ensure()
        pos, neg = predicate_props(concept)
        return [float(x) for x in self._nli.predicate(texts, pos, neg)]

    def ordinal(self, texts, axis):
        self._ensure()
        return [float(x) for x in self._cmp.ordinal_values(texts, axis)]

    def evidence(self, texts, axis):
        self._ensure()
        states, _ = self._nli.support_state(texts, axis)
        return [0.0 if s == "UNKNOWN" else 1.0 for s in states]
