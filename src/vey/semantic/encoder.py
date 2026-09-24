"""Semantic lane: a frozen 24M sentence encoder with two heads.

The default English backbone is ``mixedbread-ai/mxbai-embed-xsmall-v1``
(24.1M parameters, Apache-2.0), loaded from the Hugging Face hub at a pinned
revision and never fine-tuned by Vey.

    PooledClassifier  fixed, known label set: a trained softmax head over the
                      frozen encoder. Highest accuracy on trained labels.
    Embedder          open/novel candidates: masked-mean, L2-normalized, cached
                      by text, scored by cosine. Caching by text is what makes a
                      candidate's score independent of the candidate set.

The encoder stays frozen. Vey trains heads and selectors, not the backbone.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-xsmall-v1"
DEFAULT_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"  # informational; mxbai is a different model

MXBAI_MODEL = "mixedbread-ai/mxbai-embed-xsmall-v1"


def _autocast(device: str):
    import contextlib

    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def load_encoder(model: str = MXBAI_MODEL, device: str = "cpu", revision: str | None = None):
    """Load the frozen encoder.

    FP32 is pinned deliberately: the upstream checkpoint ships FP16 weights, and
    FP16 on CPU is emulated and roughly 100x slower. On GPU the autocast context
    below keeps compute in FP16 anyway.
    """
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    enc = AutoModel.from_pretrained(model, revision=revision, dtype=torch.float32).to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return tok, enc


class Embedder:
    """Masked-mean, L2-normalized embeddings cached by exact text."""

    def __init__(self, tok, enc, device: str, max_len: int = 96):
        self.tok, self.enc, self.device, self.max_len = tok, enc, device, max_len
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _encode(self, texts: list[str], batch: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ein = self.tok(
                chunk, padding=True, truncation=True,
                max_length=self.max_len, return_tensors="pt",
            ).to(self.device)
            with _autocast(self.device):
                h = self.enc(input_ids=ein["input_ids"],
                              attention_mask=ein["attention_mask"]).last_hidden_state
            m = ein["attention_mask"].unsqueeze(-1).float()
            v = (h.float() * m).sum(1) / m.sum(1).clamp_min(1.0)
            out.append(F.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.enc.config.hidden_size), np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, v in zip(missing, self._encode(missing)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])


class PooledClassifier(nn.Module):
    """Trained softmax head over a frozen encoder, for a fixed label set."""

    def __init__(self, encoder, num_labels: int, hidden: int = 256):
        super().__init__()
        from .encoder import EncoderPool

        self.encoder = EncoderPool(encoder)
        dim = int(encoder.config.hidden_size)
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, num_labels))

    def forward(self, input_ids, attention_mask):
        return self.head(self.encoder(input_ids, attention_mask))


class EncoderPool(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return masked_mean(out.last_hidden_state, attention_mask)


__all__ = [
    "MXBAI_MODEL",
    "Embedder",
    "PooledClassifier",
    "EncoderPool",
    "masked_mean",
    "load_encoder",
]
