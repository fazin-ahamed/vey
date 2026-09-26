"""CRUX grounder — frozen NLI predicate grounder + trained antisymmetric ordinal
comparator, behind a single Grounder interface.

    predicate(texts, concept)  -> per-candidate bipolar margin (>0 satisfies)
    ordinal(texts, axis)       -> per-candidate permutation-invariant potential
    evidence(texts, axis)      -> per-candidate 1.0 (supported) / 0.0 (no evidence)

Artifact contract (self-contained; no private paths):

  * The NLI backbone (``VEY_CRUX_NLI``, default ``tasksource/ModernBERT-base-nli``)
    downloads from the public Hugging Face Hub on first use.
  * The trained ordinal-comparator weights are loaded from, in order:
      1. ``VEY_CRUX_COMPARATOR`` — a local path to a ``.pt`` state file, or
      2. ``VEY_CRUX_COMPARATOR_HF`` — a ``repo_id[@revision]`` on the Hub,
         fetched via ``huggingface_hub`` and cached.
    If neither resolves, the CRUX lane FAILS CLOSED with an actionable error;
    it never silently degrades to a different decision. The structured lane
    needs no artifact.

The state file is ``{"model_id": str, "enc": state_dict, "head": state_dict}``.
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


def _resolve_comparator_path() -> str:
    local = os.environ.get("VEY_CRUX_COMPARATOR")
    if local:
        if not os.path.exists(local):
            raise CruxArtifactError(
                f"VEY_CRUX_COMPARATOR={local!r} does not exist.")
        return local
    hf = os.environ.get("VEY_CRUX_COMPARATOR_HF")
    if hf:
        from huggingface_hub import hf_hub_download
        repo, _, rev = hf.partition("@")
        return hf_hub_download(repo_id=repo, filename="comparator.pt",
                               revision=rev or None)
    raise CruxArtifactError(
        "The CRUX lane needs the trained ordinal comparator, but no artifact is "
        "configured. Set VEY_CRUX_COMPARATOR to a local comparator.pt, or "
        "VEY_CRUX_COMPARATOR_HF to a 'repo_id[@revision]' on the Hugging Face "
        "Hub. The structured lane runs without any artifact.")


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
        import torch
        from .backbone import NLIGrounder
        from .ordinal import OrdinalComparator
        path = _resolve_comparator_path()        # fail-closed before any download
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self._nli = NLIGrounder(self.nli_model, device=self.device)
        self._cmp = OrdinalComparator(ck["model_id"], device=self.device)
        self._cmp.load_state(ck["enc"], ck["head"])

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
