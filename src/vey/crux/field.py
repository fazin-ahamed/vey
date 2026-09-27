"""O(K) semantic field for axes the public runtime has validated.

The pairwise comparator scores K candidates with K*(K-1) forward passes. For
the axes listed in FIELD_AXES the same ordering is reproduced by one scalar
head: one forward pass per candidate, conditioned on the axis. Anything not in
that registry stays on the comparator, so an unvalidated axis never silently
rides a field that was not measured for it.

Artifact contract mirrors the comparator's: ``VEY_CRUX_FIELD`` points at a
local ``field.safetensors``, else ``VEY_CRUX_FIELD_HF`` names a Hub repo, else
the published default. Missing artifact fails closed.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModel, AutoTokenizer


# Axes the field was trained and checked against the pairwise teacher on.
# Keys are exactly what compile_instruction emits for the shipped examples.
FIELD_AXES = ("food progress", "open space", "headroom")

DEFAULT_FIELD_HF = "fazinahamed/vey"
DEFAULT_FIELD_REV = "3eb1460a82c98fc99c350032b9cf29a74075a4a6"
FIELD_FILENAME = "field.safetensors"


class FieldArtifactError(RuntimeError):
    """Raised when the field artifact cannot be resolved (fail-closed)."""


def _resolve_field_path() -> str:
    local = os.environ.get("VEY_CRUX_FIELD")
    if local:
        if os.path.isfile(local):
            return local
        raise FieldArtifactError(f"VEY_CRUX_FIELD={local} is not a file.")
    spec = os.environ.get("VEY_CRUX_FIELD_HF", DEFAULT_FIELD_HF)
    repo, _, rev = spec.partition("@")
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, FIELD_FILENAME, revision=rev or DEFAULT_FIELD_REV)
    except Exception as e:  # noqa: BLE001 - fail closed with the cause attached
        raise FieldArtifactError(
            "The CRUX field artifact could not be fetched. Set VEY_CRUX_FIELD to a "
            "local field.safetensors, or VEY_CRUX_FIELD_HF to a repo. The structured "
            "lane runs without any artifact.") from e


class FieldScorer:
    """Axis-conditioned scalar field. ``score`` returns one potential per text,
    higher meaning more of the axis, in a single batched forward pass."""

    def __init__(self, device: str = "cpu"):
        from safetensors.torch import load_file
        path = _resolve_field_path()
        tensors = load_file(path, device="cpu")
        meta_axes = None
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                meta_axes = f.metadata().get("axes") if f.metadata() else None
        except Exception:  # noqa: BLE001 - metadata is advisory
            meta_axes = None
        self.axes = tuple(meta_axes.split(",")) if meta_axes else FIELD_AXES
        model_id = "microsoft/deberta-v3-xsmall"
        rev = "4b419818330868dff6a60ad3e6b1c730f8b8c0c6"
        self.base_revision = rev
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, revision=rev)
        self.enc = AutoModel.from_config(cfg)
        enc_state = {k[len("enc."):]: v for k, v in tensors.items() if k.startswith("enc.")}
        head_state = {k[len("head.net."):] if k.startswith("head.net.") else k[len("head."):]: v
                      for k, v in tensors.items() if k.startswith("head.")}
        self.enc.load_state_dict(enc_state)
        h = self.enc.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(),
                                        torch.nn.Linear(h, 1))
        self.head.load_state_dict(head_state)
        self.device = device
        self.enc.to(device).eval()
        self.head.to(device).eval()

    def handles(self, axis: str) -> bool:
        return axis in self.axes

    @torch.no_grad()
    def ordinal(self, texts: list[str], axis: str) -> list[float]:
        """One potential per text, higher meaning more of the axis."""
        return self.ordinal_many(texts, [axis])[axis]

    @torch.no_grad()
    def ordinal_many(self, texts: list[str], axes: list[str]) -> dict[str, list[float]]:
        """Score every axis in one encoder pass. Returns one potential list per
        axis, each aligned to ``texts``. The transformer still sees one sequence
        per axis-candidate pair; the saving is a single invocation instead of one
        per axis."""
        seqs = [f"Axis: {a}. Candidate: {t}" for a in axes for t in texts]
        outs = []
        for i in range(0, len(seqs), 32):
            e = self.tok(seqs[i:i + 32], padding=True, truncation=True, max_length=96,
                         return_tensors="pt").to(self.device)
            cls = self.enc(**e).last_hidden_state[:, 0]
            outs.append(self.head(cls).squeeze(-1).cpu())
        vals = [round(float(x), 6) for x in torch.cat(outs)]
        k = len(texts)
        return {a: vals[i * k:(i + 1) * k] for i, a in enumerate(axes)}
