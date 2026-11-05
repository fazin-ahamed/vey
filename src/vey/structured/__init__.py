"""Structured decisions over typed state (the R0.5 lane).

The smallest lane in Vey: an exact, auditable decision over a typed action set,
with online per-candidate reliability and exact recent-history predicates. It
consumes structured input, not natural language.

``SuccessHead``, ``score_candidates`` and ``decide`` are backed by PyTorch and
load on first access, so importing :mod:`vey` (and the exact, structured lane)
does not import torch.
"""
from .fact_memory import FactMemory
from .trigger_bank import TriggerBank

_TORCH_NAMES = ("SuccessHead", "score_candidates", "decide")

__all__ = [
    "FactMemory",
    "TriggerBank",
    "SuccessHead",
    "score_candidates",
    "decide",
]


def __getattr__(name):
    if name in _TORCH_NAMES:
        try:
            from . import success_head
        except ModuleNotFoundError as e:
            raise ImportError(f"vey.structured.{name} needs PyTorch: install it with `pip install 'vey[torch]'`") from e
        return getattr(success_head, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
