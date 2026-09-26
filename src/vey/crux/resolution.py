"""Semantic Resolution — a grounded value carries both a magnitude and a
resolution (epsilon). Natural language yields intervals, not exact floats, so two
grounded values within epsilon are treated as semantically TIED and the decision
defers to the next priority. This is the first-class version of the
epsilon-lexicographic rule validated in D6/CRUX-P.
"""
from __future__ import annotations

from dataclasses import dataclass

# default resolutions per grounded quantity kind, in grounded-margin units
DEFAULT_RESOLUTION = {
    "ordinal": 0.15,     # graded qualitative axes (room, progress, quality, ...)
    "predicate": 0.0,    # categorical: sign decides, no tie band
    "numeric": 0.0,      # exact numbers compare exactly
}


@dataclass(frozen=True)
class Resolved:
    """A grounded value with its semantic resolution."""
    value: float
    resolution: float = 0.0

    def clearly_greater(self, other: "Resolved") -> bool:
        eps = max(self.resolution, other.resolution)
        return self.value - other.value > eps

    def tied_with(self, other: "Resolved") -> bool:
        eps = max(self.resolution, other.resolution)
        return abs(self.value - other.value) <= eps


def quantize(x: float, decimals: int = 3) -> float:
    """Absorb ~1e-6 batched-float reduction noise so a decision is a pure
    function of the option-content set (permutation/rename exactness)."""
    return round(float(x), decimals)
