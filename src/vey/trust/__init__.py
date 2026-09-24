"""Calibrated trust and abstention.

The runtime estimates *risk*, never an action. :mod:`vey.trust.policy` turns
exact schema state and a calibrated risk estimate into an explicit decision
(``ACT`` / ``ASK_FOR_INFO`` / ``RERUN_CHEAP_LANE`` / ``ESCALATE`` / ``ABSTAIN``)
using rules a reader can audit; no model ever emits those labels directly.
"""
from .observation import Observation, from_scores
from .policy import (
    ABSTAIN,
    ACT,
    ASK_FOR_INFO,
    ESCALATE,
    REJECT,
    RERUN_CHEAP_LANE,
    PolicyConfig,
    decide,
)

__all__ = [
    "Observation",
    "from_scores",
    "PolicyConfig",
    "decide",
    "ACT",
    "ASK_FOR_INFO",
    "REJECT",
    "ESCALATE",
    "ABSTAIN",
    "RERUN_CHEAP_LANE",
]
