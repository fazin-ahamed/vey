"""Structured success head.

The smallest learned component in Vey: an 8-feature observable action vector
plus one online reliability term into a two-layer MLP that predicts the
probability a candidate succeeds. The policy applies hard eligibility masks
*before* this head, scores only legal candidates, and escalates when nothing
clears the confidence floor.

The head is deliberately tiny. Every larger alternative tried (learned state
registers, fact-recency features, risk networks) either failed to beat the
simpler control or failed an intervention test, so none of them shipped.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SuccessHead(nn.Module):
    """P(success | observable action features, online reliability)."""

    def __init__(self, n_feat: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat + 1, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def score_candidates(head: SuccessHead, feature_rows, reliability: list[float],
                     device: str = "cpu") -> np.ndarray:
    """Score candidate rows. `reliability` is the per-candidate online estimate."""
    rows = [list(f) + [r] for f, r in zip(feature_rows, reliability)]
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        return head(x).cpu().numpy()


def decide(success: np.ndarray, floor: float = 0.7):
    """Constraint-first policy over already-legal candidates.

    Returns the index of the best candidate, or ``None`` to escalate when no
    legal candidate clears the floor. The caller applies eligibility masks
    before this point; an empty candidate list is also an escalation.
    """
    if len(success) == 0:
        return None
    best = int(np.argmax(success))
    return None if float(success[best]) < floor else best


__all__ = ["SuccessHead", "score_candidates", "decide"]
