"""Open-candidate matching: independent, order-free scoring.

A candidate's score depends only on its own meaning. Because candidate
embeddings are cached by text and the query is encoded once, permuting,
inserting, or removing other candidates cannot change an existing candidate's
score: structural invariance is a property of the representation, not something
a set-transformer has to learn. Measured drift: exactly 0.0.

This is what lets Vey answer against a candidate set it never trained on, and
why no DeepSets / set-transformer / candidate self-attention layer is needed.
"""
from __future__ import annotations

import numpy as np

from .encoder import Embedder


class CandidateMatcher:
    """Cosine matcher over cached candidate representations."""

    def __init__(self, emb: Embedder):
        self.emb = emb

    def prepare(self, candidates: dict[str, str]) -> None:
        """Pre-encode candidate descriptions once (the deployed configuration)."""
        self.emb.embed(list(candidates.values()))

    def score(self, state: str, candidates: dict[str, str]) -> dict[str, float]:
        """Cosine of the state embedding against every candidate description."""
        q = self.emb.embed([state])[0]
        vecs = self.emb.embed(list(candidates.values()))
        return {cid: float(v @ q) for cid, v in zip(candidates, vecs)}

    def select(self, state: str, candidates: dict[str, str]) -> tuple[str, float]:
        scores = self.score(state, candidates)
        best = max(scores, key=scores.get)
        return best, scores[best]

    @staticmethod
    def margin(scores: dict[str, float]) -> float:
        """Top-1 minus top-2 margin; a small margin means the set is ambiguous."""
        vals = sorted(scores.values(), reverse=True)
        return float(vals[0] - vals[1]) if len(vals) > 1 else float("inf")


__all__ = ["CandidateMatcher"]
