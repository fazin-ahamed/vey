"""Semantic lane: fixed-label classification and open-candidate matching.

Two interchangeable embedders feed :class:`CandidateMatcher`:

    Embedder       PyTorch (``pip install vey[torch]``), the reference.
    OnnxEmbedder   ONNX Runtime, no PyTorch. Same contract, same pooling.

The torch-backed names (``Embedder``, ``PooledClassifier``, ``load_encoder``, ...)
load on first access, so ``import vey.semantic`` stays torch-free.
"""
from .candidates import CandidateMatcher
from .onnx_encoder import OnnxEmbedder

_TORCH_NAMES = ("MXBAI_MODEL", "Embedder", "EncoderPool", "PooledClassifier", "load_encoder", "masked_mean")

__all__ = [
    "MXBAI_MODEL",
    "Embedder",
    "EncoderPool",
    "PooledClassifier",
    "OnnxEmbedder",
    "load_encoder",
    "masked_mean",
    "CandidateMatcher",
]


def __getattr__(name):
    if name in _TORCH_NAMES:
        try:
            from . import encoder
        except ModuleNotFoundError as e:  # torch / transformers missing
            raise ImportError(
                f"vey.semantic.{name} needs PyTorch. Install it with `pip install 'vey[torch]'`, "
                "or use OnnxEmbedder, which does not need it."
            ) from e
        return getattr(encoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
