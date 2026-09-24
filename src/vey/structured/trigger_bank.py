"""TriggerBank: structured experience memory.

{STATE_SIGNATURE, ACTION, OUTCOME, FUTURE_TRIGGERS, EVIDENCE, AGE, RELIABILITY}

Retrieval returns experiences as an EVIDENCE SOURCE. It never selects an
action: a stored experience can be stale, and Vey is expected to disagree with
its own memory when the world has moved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Experience:
    state_signature: str
    action: int
    outcome_success: bool
    outcome_quality: float
    future_triggers: tuple[str, ...] = ()
    evidence: tuple[int, ...] = ()
    age_s: float = 0.0
    reliability: float = 1.0
    stored_at: float = field(default_factory=time.time)


def _signature(features: np.ndarray, buckets: int = 16) -> str:
    """Quantize a feature vector into a stable signature. Nearby states share
    a signature; distant states do not."""
    clipped = np.clip(features, 0.0, 1.0)
    idx = np.minimum((clipped * buckets).astype(int), buckets - 1)
    return ",".join(str(i) for i in idx)


class TriggerBank:
    def __init__(self, half_life_s: float = 86400.0):
        self.half_life_s = half_life_s
        self._by_signature: dict[str, list[Experience]] = {}

    def store(self, features: np.ndarray, action: int, success: bool,
              quality: float, triggers: tuple[str, ...] = (),
              evidence: tuple[int, ...] = ()) -> Experience:
        exp = Experience(
            state_signature=_signature(features),
            action=action,
            outcome_success=success,
            outcome_quality=quality,
            future_triggers=triggers,
            evidence=evidence,
        )
        self._by_signature.setdefault(exp.state_signature, []).append(exp)
        return exp

    def _decay(self, exp: Experience, now: float) -> float:
        age = now - exp.stored_at
        return exp.reliability * 0.5 ** (age / self.half_life_s)

    def retrieve(self, features: np.ndarray, now: float | None = None,
                 min_reliability: float = 0.1) -> list[Experience]:
        """Matching experiences, decayed by age, below the floor dropped.

        Returns evidence. The caller decides; this function does not.
        """
        now = time.time() if now is None else now
        found = []
        for exp in self._by_signature.get(_signature(features), []):
            weight = self._decay(exp, now)
            if weight >= min_reliability:
                exp.age_s = now - exp.stored_at
                exp.reliability = weight
                found.append(exp)
        return found

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_signature.values())
