"""Retrieval lane: BM25, semantic scoring, and reciprocal-rank fusion.

Document retrieval runs both lanes and fuses their rankings with a single fixed
Reciprocal Rank Fusion rule (k=60). The fusion is not a learned reranker: on
out-of-domain technical retrieval the two lanes fail differently, and a fixed
rank fusion beat either lane alone on every FreshStack topic.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

_WORD = re.compile(r"\w+")
RRF_K = 60


class BM25:
    """Small in-memory BM25. Candidate sets per query are small, so a per-query
    index is cheaper than a global one and keeps the lane dependency-free."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [_WORD.findall(d.lower()) for d in documents]
        self.n = len(self.tokens)
        self.avgdl = max(1.0, sum(len(t) for t in self.tokens) / max(1, self.n))
        self.df: dict[str, int] = defaultdict(int)
        for t in self.tokens:
            for w in set(t):
                self.df[w] += 1

    def scores(self, query: str) -> np.ndarray:
        q = _WORD.findall(query.lower())
        out = np.zeros(self.n)
        for i, t in enumerate(self.tokens):
            tf: dict[str, int] = defaultdict(int)
            for w in t:
                tf[w] += 1
            dl = max(1, len(t))
            for w in q:
                if w not in tf:
                    continue
                idf = np.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                out[i] += idf * (tf[w] * (self.k1 + 1)) / (
                    tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
        return out


def rrf(score_a: np.ndarray, score_b: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion of two score vectors. One fixed rule, no tuning."""
    n = len(score_a)
    pos_a = np.empty(n, dtype=np.float64)
    pos_b = np.empty(n, dtype=np.float64)
    pos_a[np.argsort(-score_a)] = np.arange(n)
    pos_b[np.argsort(-score_b)] = np.arange(n)
    return 1.0 / (k + pos_a) + 1.0 / (k + pos_b)


__all__ = ["BM25", "rrf", "RRF_K"]
