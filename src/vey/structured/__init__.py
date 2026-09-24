"""Structured decisions over typed state (the R0.5 lane).

The smallest lane in Vey: an exact, auditable decision over a typed action set,
with online per-candidate reliability and exact recent-history predicates. It
consumes structured input, not natural language.
"""
from .fact_memory import FactMemory
from .success_head import SuccessHead, decide, score_candidates
from .trigger_bank import TriggerBank

__all__ = [
    "FactMemory",
    "TriggerBank",
    "SuccessHead",
    "score_candidates",
    "decide",
]
