"""Common evaluation metrics for Vey decision lanes."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def normalize_probs(probs: Sequence[float]) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return p / total


def multiclass_brier(probs: Sequence[float], gold_index: int) -> float:
    p = normalize_probs(probs)
    if not 0 <= gold_index < len(p):
        raise IndexError("gold index out of range")
    y = np.zeros_like(p)
    y[gold_index] = 1.0
    return float(np.square(p - y).sum())


def multiclass_nll(probs: Sequence[float], gold_index: int, eps: float = 1e-12) -> float:
    p = normalize_probs(probs)
    if not 0 <= gold_index < len(p):
        raise IndexError("gold index out of range")
    return float(-math.log(max(float(p[gold_index]), eps)))


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int],
    bins: int = 15,
) -> float:
    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correctness, dtype=np.float64)
    if c.shape != y.shape or c.ndim != 1:
        raise ValueError("confidences and correctness must be equally sized vectors")
    if np.any((c < 0) | (c > 1)):
        raise ValueError("confidence must be in [0, 1]")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(c))
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (c >= lo) & (c < hi if i + 1 < bins else c <= hi)
        n = int(mask.sum())
        if not n:
            continue
        ece += n / total * abs(float(c[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def risk_coverage_curve(risk_scores: Sequence[float], failures: Sequence[bool | int]):
    """Every exact prefix, safest first.

    Smaller risk_score means more trusted. Returns one row for each non-empty
    coverage prefix so callers cannot accidentally hide a narrow operating
    region behind a coarse grid.
    """
    score = np.asarray(risk_scores, dtype=np.float64)
    fail = np.asarray(failures, dtype=np.float64)
    if score.shape != fail.shape or score.ndim != 1:
        raise ValueError("risk_scores and failures must be equally sized vectors")
    if not len(score):
        return []
    order = np.argsort(score, kind="stable")
    ordered_fail = fail[order]
    cumulative = np.cumsum(ordered_fail)
    n = len(score)
    return [
        {
            "k": k,
            "coverage": k / n,
            "risk": float(cumulative[k - 1] / k),
            "threshold": float(score[order[k - 1]]),
        }
        for k in range(1, n + 1)
    ]


def risk_at_coverage(curve, coverage: float) -> float:
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1]")
    if not curve:
        raise ValueError("curve is empty")
    row = min(curve, key=lambda x: abs(x["coverage"] - coverage))
    return float(row["risk"])


def coverage_at_risk(curve, max_risk: float) -> float:
    feasible = [row["coverage"] for row in curve if row["risk"] <= max_risk]
    return float(max(feasible)) if feasible else 0.0


def reorder_agreement(reference_ids: Iterable[str], reordered_ids: Iterable[str]) -> float:
    a = list(reference_ids)
    b = list(reordered_ids)
    if len(a) != len(b):
        raise ValueError("prediction sequences must have equal length")
    if not a:
        raise ValueError("prediction sequences are empty")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _dcg(rels):
    import math
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(ranked_ids, relevance, k=10):
    """ranked_ids best-first; relevance maps doc id -> gain (0 if absent)."""
    gains = [float(relevance.get(doc, 0.0)) for doc in ranked_ids[:k]]
    ideal = sorted((float(v) for v in relevance.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hit = sum(1 for doc in ranked_ids[:k] if doc in relevant)
    return hit / len(relevant)


def mrr_at_k(ranked_ids, relevant_ids, k=10):
    relevant = set(relevant_ids)
    for i, doc in enumerate(ranked_ids[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0
